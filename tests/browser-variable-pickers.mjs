/** Search every dataset-variable selector through the actual browser controls. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {chooseChart, chooseSignals, waitForChart} from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5175/';
const artifacts = await mkdtemp(path.join(tmpdir(), 'tsplotter-variable-pickers-'));
const rows = 120, signals = Array.from({length: 233}, (_, i) => `Signal_${String(i).padStart(3, '0')}`);
const filename = 'private-many-variable-search.csv';
const file = {name: filename, mimeType: 'text/csv', buffer: Buffer.from(
  ['x', 'time', 'category', ...signals].join(',') + '\n' + Array.from({length: rows}, (_, row) =>
    [row, new Date(Date.UTC(2026, 0, 1, 0, 0, row)).toISOString(), row % 2 ? 'north' : 'south',
      ...signals.map((_, column) => (column + 1) * (row + 1) / 10 + column % 3)].join(',')).join('\n'))};
const report = {result: 'RUNNING', baseURL, sourceRows: rows, numericVariables: signals.length + 1,
  checks: [], errors: [], writes: [], external: [], graphicsWarnings: [], artifacts};
const browser = await chromium.launch({headless: true,
  ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
page.setDefaultTimeout(60000);
page.on('pageerror', error => report.errors.push(String(error)));
page.on('request', request => {
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) report.writes.push(request.url());
  if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) report.external.push(request.url());
});
page.on('console', message => {
  if (/(?:webgl.*(?:lost|too many|limit)|too many.*(?:webgl|context)|context.*lost)/i.test(message.text())) report.graphicsWarnings.push(message.text());
});
const pass = text => {report.checks.push(text); console.log(text);};
const picker = selector => page.locator(selector).locator('xpath=following-sibling::div[contains(concat(" ", @class, " "), " signal-picker ")][1]');
const selected = selector => page.locator(selector).evaluate(select => [...select.selectedOptions].map(option => option.value));
async function update(view) {await page.locator('#render-button').click(); await waitForChart(page); await retained(view);}
async function retained(view) {
  assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), rows);
  const traces = await page.locator('#chart').evaluate(graph => (graph.data || [])
    .filter(trace => trace.x?.some(value => value != null) && !trace.meta?.rollingBand)
    .map(trace => ({x: trace.x.length, y: trace.y.length, total: trace.type === 'bar' ? trace.y.reduce((sum, value) => sum + value, 0) : undefined})));
  assert.ok(traces.length);
  for (const trace of traces) {
    if (view === 'Distribution') assert.equal(trace.total, rows);
    else {assert.equal(trace.x, rows); assert.equal(trace.y, rows);}
  }
}
async function searchPick(selector, value, {multiple = false, keyboard = false} = {}) {
  const control = picker(selector);
  await control.locator('.picker-trigger').click();
  assert.ok(await control.locator('.picker-search').isVisible());
  assert.equal(await control.getByRole('listbox').getAttribute('aria-multiselectable'), String(multiple));
  const label = await page.locator(selector).locator('option').evaluateAll((options, target) => options.find(option => option.value === target)?.label, value);
  assert.ok(label !== undefined);
  await control.locator('.picker-search').fill(label);
  const option = control.getByRole('option', {name: label, exact: true});
  assert.ok(await option.isVisible());
  if (keyboard) {await control.locator('.picker-search').press('ArrowDown'); await page.keyboard.press('Enter');}
  else await option.click();
  if (multiple) {
    assert.equal(await control.locator('.picker-trigger').getAttribute('aria-expanded'), 'true');
    assert.ok((await selected(selector)).includes(value));
    await page.keyboard.press('Escape');
  } else {
    assert.deepEqual(await selected(selector), [value]);
    assert.ok(!await control.locator('.picker-clear').isVisible());
    assert.equal(await control.locator('[data-picker-remove]').count(), 0);
  }
  assert.equal(await control.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
  assert.ok(await control.locator('.picker-trigger').evaluate(element => element === document.activeElement));
}
async function noMatch(selector) {
  const before = await selected(selector), control = picker(selector);
  await control.locator('.picker-trigger').click();
  await control.locator('.picker-search').fill('certainly_missing_variable_987654');
  assert.ok(await control.locator('.picker-empty').isVisible());
  assert.equal(await control.locator('.picker-option:visible').count(), 0);
  await control.locator('.picker-search').press('Enter');
  assert.deepEqual(await selected(selector), before);
  await page.keyboard.press('Escape');
  assert.ok(await control.locator('.picker-trigger').evaluate(element => element === document.activeElement));
}
async function nativeSetting(selector) {
  assert.ok(await page.locator(selector).evaluate(select => !select.hasAttribute('data-variable') && !select.classList.contains('native-control') && !select.nextElementSibling?.classList.contains('signal-picker')));
}
async function saved() {
  const event = page.waitForEvent('download'); await page.locator('#save-config').click();
  const download = await event, text = await readFile(await download.path(), 'utf8'); await download.delete();
  assert.ok(!text.includes(filename)); const config = JSON.parse(text); assert.equal(config.version, 4); return config;
}
async function noOverflow() {
  assert.ok(await page.evaluate(() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) <= innerWidth + 1));
}

try {
  await page.goto(baseURL, {waitUntil: 'networkidle'}); await page.setInputFiles('#file-input', file); await waitForChart(page);
  await nativeSetting('[data-key="setup_time_mode"]');
  await page.selectOption('[data-key="setup_time_mode"]', 'Relative time');
  await searchPick('#x-column', 'x', {keyboard: true}); await noMatch('#x-column');
  pass('The X-axis variable selector searches all 236 columns, supports keyboard selection and preserves a selection on no match');

  await chooseSignals(page, [signals[229]]);
  await noMatch('#y-columns');
  await searchPick('#y-columns', signals[228], {multiple: true, keyboard: true});
  assert.deepEqual(await selected('#y-columns'), [signals[228], signals[229]]);
  await page.locator('#add-trend-subplot').click();
  await chooseSignals(page, [signals[230]], 'select[data-trend-subplot="1"]');
  await searchPick('select[data-trend-subplot="1"]', signals[231], {multiple: true});
  await update('Trends');
  assert.equal(await page.locator('#chart').evaluate(graph => graph.data.length), 4);
  pass('Every added Trend subplot has searchable variables, keeps multi-selection open and retains all 120 source rows');

  await chooseChart(page, 'Distribution'); await chooseSignals(page, [signals[232]]);
  await nativeSetting('[data-panel-key="binMode"]'); await update('Distribution');
  pass('Distribution variable search still supports late signals while bin settings remain native');

  await chooseChart(page, 'Scatter');
  await searchPick('#scatter-x', signals[228], {keyboard: true});
  await searchPick('#y-columns', signals[229]);
  await searchPick('[data-panel-key="color"]', 'category', {keyboard: true});
  await searchPick('[data-panel-key="marker"]', 'category');
  await searchPick('[data-panel-key="marker"]', '(none)', {keyboard: true});
  await searchPick('[data-panel-key="color"]', '(none)');
  await searchPick('[data-panel-key="color"]', signals[230]);
  await nativeSetting('[data-panel-key="colorMode"]'); await update('Scatter');
  pass('Scatter X, Y, color and marker variables all search; single choices close and refocus after rerenders, including explicit none');

  await chooseChart(page, 'Rolling window'); await searchPick('#y-columns', signals[231], {keyboard: true}); await update('Rolling window');
  await chooseChart(page, 'Derivatives'); await searchPick('#y-columns', signals[232]);
  await nativeSetting('[data-panel-key="timeUnit"]'); await update('Derivatives');
  pass('Rolling and Derivative signal selectors search independently without changing analysis rows or ordinary settings');

  await page.selectOption('[data-key="setup_time_mode"]', 'Absolute time');
  await searchPick('#x-column', 'time', {keyboard: true}); await update('Derivatives');
  pass('Absolute time columns use the same searchable selector and preserve every valid timestamp');

  await page.locator('#filter-panel > summary').click(); await page.locator('#add-filter').click();
  await searchPick('[data-key="flt2_var_0"]', signals[200]);
  assert.ok(await page.locator('[data-key="flt2_low_num_0"]').isVisible());
  await searchPick('[data-key="flt2_var_0"]', 'category', {keyboard: true});
  assert.ok(await page.locator('[data-membership-filter="0"]').isVisible()); await nativeSetting('[data-key="flt2_op_0"]');
  await searchPick('[data-key="flt2_var_0"]', 'time');
  assert.ok(await page.locator('[data-date-bound="low"][data-filter-index="0"]').isVisible());
  await searchPick('[data-key="flt2_var_0"]', '(time index)', {keyboard: true});
  await searchPick('[data-key="flt2_var_0"]', signals[200]); await update('Derivatives');
  pass('Filter column search survives numeric, categorical and time-field rerenders, restores focus and keeps empty conditions inactive');

  await chooseChart(page, 'Trends');
  const config = await saved();
  assert.deepEqual(config.settings.trend_panels[0].subplots.map(row => row.columns), [[signals[228], signals[229]], [signals[230], signals[231]]]);
  assert.equal(config.settings.scatter_panels[0].color, signals[230]); assert.equal(config.settings.scatter_panels[0].marker, '(none)');
  await chooseSignals(page, [signals[0]]); await update('Trends');
  await page.setInputFiles('#config-input', {name: 'variable-search-settings.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config))});
  await waitForChart(page); const restored = await saved();
  // Switching filter kinds leaves inactive draft fields which configuration
  // normalization may discard; verify the active choices and all chart panels.
  for (const key of ['trend_panels', 'distribution_panels', 'scatter_panels', 'rolling_panels', 'derivative_panels',
    'setup_time_mode', 'setup_dt_col', 'setup_index', 'flt2_n', 'flt2_var_0', 'flt2_low_num_0', 'flt2_up_num_0']) {
    assert.deepEqual(restored.settings[key], config.settings[key]);
  }
  assert.deepEqual(await selected('select[data-trend-subplot="1"]'), [signals[230], signals[231]]);
  await noMatch('select[data-trend-subplot="1"]'); await retained('Trends');
  pass('Portable settings reload every searched variable and reinitialize searchable subplot controls without storing source bytes');

  await page.setViewportSize({width: 390, height: 844}); await noOverflow();
  await page.locator('#settings-button').click(); assert.ok(await page.locator('body').evaluate(element => element.classList.contains('sidebar-open')));
  await picker('#x-column').locator('.picker-trigger').click(); await picker('#x-column').locator('.picker-search').fill('time');
  assert.ok(await picker('#x-column').getByRole('option', {name: 'time', exact: true}).isVisible()); await noOverflow();
  await page.screenshot({path: path.join(artifacts, 'mobile-axis-search.png'), fullPage: true});
  await page.keyboard.press('Escape'); assert.equal(await picker('#x-column').locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
  assert.ok(await page.locator('body').evaluate(element => element.classList.contains('sidebar-open')));
  await page.keyboard.press('Escape'); assert.ok(!await page.locator('body').evaluate(element => element.classList.contains('sidebar-open')));
  assert.ok(await page.locator('#settings-button').evaluate(element => element === document.activeElement));
  const mobileTrend = picker('select[data-trend-subplot="1"]');
  await mobileTrend.locator('.picker-trigger').click(); await mobileTrend.locator('.picker-search').fill(signals[232]);
  assert.ok(await mobileTrend.getByRole('option', {name: signals[232], exact: true}).isVisible()); await noOverflow();
  await page.screenshot({path: path.join(artifacts, 'mobile-trend-search.png'), fullPage: true});
  await page.keyboard.press('Escape'); await retained('Trends');
  pass('At 390px every search remains usable without overflow; Escape closes the picker before its settings drawer');

  assert.deepEqual(report.errors, []); assert.deepEqual(report.writes, []); assert.deepEqual(report.external, []); assert.deepEqual(report.graphicsWarnings, []);
  report.result = 'PASS';
} catch (error) {
  report.result = 'FAIL'; report.error = String(error); report.status = await page.locator('#status').textContent().catch(() => '');
  await page.screenshot({path: path.join(artifacts, 'failure.png'), fullPage: true}).catch(() => {}); process.exitCode = 1;
} finally {
  await writeFile(path.join(artifacts, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2)); await browser.close();
}
