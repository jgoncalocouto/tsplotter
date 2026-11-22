/** Real-browser regression coverage for independently configured distribution cards. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp, readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import * as XLSX from 'xlsx';
import {chooseSignals} from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5173/';
const rowCount = Number(process.env.TEST_ROWS || 2401);
assert.ok(Number.isInteger(rowCount) && rowCount >= 201, 'TEST_ROWS must be at least 201');
const artifacts = await mkdtemp(path.join(tmpdir(), 'tsplotter-distribution-panels-'));
const fixture = {name: 'independent-distributions.csv', mimeType: 'text/csv', buffer: Buffer.from(
  'x,signal,aux,sparse,category\n' + Array.from({length: rowCount}, (_, i) =>
    `${i},${2 * i + 1},${i % 101},${i < 50 ? i : ''},${i % 2 ? 'odd' : 'even'}`).join('\n'))};
const browser = await chromium.launch({headless: true,
  ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const checks = [], errors = [], writes = [], external = [];
const pass = message => { checks.push(message); console.log(message); };
function observe(target) {
  target.setDefaultTimeout(60000);
  target.on('pageerror', error => errors.push(String(error)));
  target.on('request', request => {
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) writes.push(`${request.method()} ${request.url()}`);
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) external.push(request.url());
  });
}
observe(page);
const chartSelector = (index, mode = 'Distribution') => index === 0 ? '#chart' : `#${mode === 'Distribution' ? 'distribution' : 'scatter'}-chart-${index}`;
async function ready(target, count, mode = 'Distribution') {
  await target.waitForFunction(({count, mode}) => {
    const status = document.querySelector('#status');
    if (status?.classList.contains('error')) return true;
    if (document.querySelector('#render-button')?.disabled || !/ready|settings loaded/i.test(status?.textContent || '')) return false;
    if (count === undefined) return Array.isArray(document.querySelector('#chart')?.data);
    // Empty histograms intentionally have no Plotly data; status changes after all cards finish.
    const kind = mode === 'Distribution' ? 'distribution' : 'scatter';
    return document.querySelectorAll(`[data-${kind}-card]`).length === count;
  }, {count, mode}, {timeout: 120000});
  assert.ok(!await target.locator('#status').evaluate(el => el.classList.contains('error')), await target.locator('#status').textContent());
}
async function update(target, count) {
  await target.locator('#render-button').click();
  await ready(target, count);
}
async function selectField(target, key, value) {
  await target.locator(`[data-panel-key="${key}"]`).selectOption(value);
}
async function fillField(target, key, value) {
  const field = target.locator(`[data-panel-key="${key}"]`);
  await field.fill(String(value));
  await field.press('Tab');
}
async function edit(target, index) {
  await target.locator(`[data-distribution-panel="${index}"]`).click();
  assert.equal(await target.locator(`[data-distribution-panel="${index}"]`).getAttribute('aria-pressed'), 'true');
}
async function download(target, selector, binary = false) {
  const event = target.waitForEvent('download');
  await target.locator(selector).click();
  const result = await event;
  return await readFile(await result.path(), binary ? undefined : 'utf8');
}
async function saved(target) {
  const openDrawer = await target.locator('#sidebar').evaluate(el => el.inert);
  if (openDrawer) await target.locator('#settings-button').click();
  const text = await download(target, '#save-config');
  if (openDrawer) await target.locator('#close-sidebar').click();
  const config = JSON.parse(text);
  assert.equal(config.app, 'tsplotter');
  assert.equal(config.version, 3);
  for (const key of ['trend_panels', 'rolling_panels', 'derivative_panels']) assert.ok(config.settings[key].length >= 1);
  assert.ok(!text.includes(fixture.name), 'Configuration contains settings, not the dataset source');
  assert.ok(Buffer.byteLength(text) < 256 * 1024);
  return config;
}
async function uploadConfig(target, config) {
  await target.setInputFiles('#config-input', {name: 'distribution-settings.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config))});
  if (await target.locator('#load-config').count()) await target.locator('#load-config').click();
}
async function load(target, config, count, mode = 'Distribution') {
  await uploadConfig(target, config);
  await ready(target, count, mode);
}
async function histogram(target, index) {
  return await target.locator(chartSelector(index)).evaluate(chart => (chart.data || []).filter(trace => trace.type === 'bar').map(trace => ({
    name: trace.name, x: Array.from(trace.x), y: Array.from(trace.y),
    width: typeof trace.width === 'number' ? trace.x.map(() => trace.width) : Array.from(trace.width),
    edges: Array.from(trace.customdata, pair => Array.from(pair)),
  })));
}
function total(trace) { return trace.y.reduce((sum, value) => sum + Number(value), 0); }
function numericClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(Number(actual[i]) - Number(expected[i])) <= 1e-10 * Math.max(1, Math.abs(Number(expected[i]))), `Numeric mismatch at ${i}: ${actual[i]} vs ${expected[i]}`);
}
function table(text) {
  const workbook = XLSX.read(text, {type: 'string', raw: true});
  return XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]], {defval: '', raw: true});
}
async function exportsDisabled(target) {
  for (const button of await target.locator('[data-distribution-export], [data-distribution-image]').all()) assert.ok(await button.isDisabled());
}
async function noOverflow(target) {
  const dimensions = await target.evaluate(() => ({width: innerWidth, scroll: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)}));
  assert.ok(dimensions.scroll <= dimensions.width + 1, `Horizontal overflow: ${JSON.stringify(dimensions)}`);
  for (const card of await target.locator('[data-distribution-card]').all()) {
    const rect = await card.boundingBox();
    assert.ok(rect && rect.x >= -1 && rect.x + rect.width <= dimensions.width + 1, `Distribution card exceeds viewport: ${JSON.stringify(rect)}`);
  }
}
const withoutTitle = panel => Object.fromEntries(Object.entries(panel).filter(([key]) => key !== 'title'));

try {
  await page.goto(baseURL, {waitUntil: 'networkidle'});
  await page.setInputFiles('#file-input', fixture);
  await ready(page);

  // A separately configured scatter must survive all distribution operations and config loads.
  await page.getByRole('tab', {name: 'Scatter', exact: true}).click();
  await ready(page, 1, 'Scatter');
  await fillField(page, 'title', 'Retained scatter');
  await selectField(page, 'x', 'aux');
  await selectField(page, 'y', 'signal');
  await selectField(page, 'color', 'category');
  await page.locator('#render-button').click();
  await ready(page, 1, 'Scatter');
  const scatterSettings = (await saved(page)).settings.scatter_panels;

  await page.getByRole('tab', {name: 'Distribution', exact: true}).click();
  await ready(page, 1);
  await fillField(page, 'title', 'Signal distribution');
  await chooseSignals(page, ['signal']);
  await selectField(page, 'binMode', 'Target #bins');
  await fillField(page, 'binCount', 20);
  await update(page, 1);
  const signal = await histogram(page, 0);
  assert.deepEqual(signal.map(trace => trace.name), ['signal']);
  assert.equal(signal[0].y.length, 20);
  assert.equal(total(signal[0]), rowCount);
  assert.equal(signal[0].edges[0][0], 1);
  assert.equal(signal[0].edges.at(-1)[1], 2 * rowCount - 1);
  pass(`A titled distribution with 20 bins represents all ${rowCount.toLocaleString()} signal readings`);

  await page.locator('#add-distribution').click();
  assert.equal(await page.locator('[data-distribution-panel]').count(), 2);
  await fillField(page, 'title', 'Auxiliary distribution');
  await chooseSignals(page, ['aux']);
  await selectField(page, 'binMode', 'Fixed width');
  await fillField(page, 'binWidth', 10);
  await update(page, 2);
  assert.deepEqual(await histogram(page, 0), signal);
  const auxiliary = await histogram(page, 1);
  assert.deepEqual(auxiliary.map(trace => trace.name), ['aux']);
  assert.equal(total(auxiliary[0]), rowCount);
  assert.ok(auxiliary[0].width.every(width => width === 10));
  assert.equal(auxiliary[0].edges[0][0], 0);
  assert.equal(auxiliary[0].edges.at(-1)[1], 100);
  pass('Two distribution cards have independent series, bin edges, widths and counts');

  await page.locator('[data-edit-distribution="0"]').click();
  assert.ok(!await page.locator('#dirty-indicator').isVisible());
  assert.deepEqual(await histogram(page, 1), auxiliary);
  await fillField(page, 'binCount', 25);
  assert.ok(await page.locator('#dirty-indicator').isVisible());
  await exportsDisabled(page);
  assert.deepEqual(await histogram(page, 1), auxiliary);
  await update(page, 2);
  assert.equal((await histogram(page, 0))[0].y.length, 25);
  assert.deepEqual(await histogram(page, 1), auxiliary);
  pass('Editor switching stays clean; changing bins disables stale exports and leaves other cards unchanged');

  const original = await saved(page);
  assert.equal(original.settings.distribution_panels.length, 2);
  await page.locator('#duplicate-distribution').click();
  await update(page, 3);
  const duplicated = await saved(page);
  assert.deepEqual(withoutTitle(duplicated.settings.distribution_panels[2]), withoutTitle(original.settings.distribution_panels[0]));
  await fillField(page, 'title', 'Independent comparison');
  await chooseSignals(page, ['x', 'signal']);
  await selectField(page, 'binMode', 'Auto (width)');
  await update(page, 3);
  const changed = await saved(page);
  assert.deepEqual(changed.settings.distribution_panels[0], original.settings.distribution_panels[0]);
  assert.deepEqual(changed.settings.distribution_panels[1], original.settings.distribution_panels[1]);
  assert.deepEqual(new Set(changed.settings.distribution_panels[2].columns), new Set(['x', 'signal']));
  for (const trace of await histogram(page, 2)) assert.equal(total(trace), rowCount);
  await edit(page, 0);
  await page.locator('#remove-distribution').click();
  await update(page, 2);
  const twoPanels = await saved(page);
  assert.deepEqual(twoPanels.settings.distribution_panels, changed.settings.distribution_panels.slice(1));
  assert.deepEqual((await histogram(page, 0)).map(trace => trace.name), ['aux']);
  pass('Duplicates keep independent signal arrays; removal renumbers cards while preserving survivors');

  await page.getByRole('tab', {name: 'Scatter', exact: true}).click();
  await ready(page, 1, 'Scatter');
  assert.deepEqual((await saved(page)).settings.scatter_panels, scatterSettings);
  assert.deepEqual((await saved(page)).settings.distribution_panels, twoPanels.settings.distribution_panels);
  await load(page, twoPanels, 2);
  const restored = await saved(page);
  assert.equal(restored.settings.plot_type, 'Distribution');
  assert.deepEqual(restored.settings.distribution_panels, twoPanels.settings.distribution_panels);
  assert.deepEqual(restored.settings.scatter_panels, scatterSettings);
  pass('Version 3 round-trip restores every distribution and retains independently configured scatter panels');

  const legacyV2 = structuredClone(twoPanels);
  legacyV2.version = 2;
  for (const key of ['trend_panels', 'rolling_panels', 'derivative_panels', 'comparison_link_x', 'comparison_link_y', 'distribution_shared_bins']) delete legacyV2.settings[key];
  for (const panel of legacyV2.settings.scatter_panels) {delete panel.colorBinScope; delete panel.markerBinScope;}
  for (const panel of legacyV2.settings.distribution_panels) delete panel.binScope;
  await load(page, legacyV2, 2);
  assert.deepEqual((await saved(page)).settings.distribution_panels, twoPanels.settings.distribution_panels);
  pass('Version 2 distribution panels retain their signals, bins and default filtered scope');

  await load(page, {app: 'tsplotter', version: 1, settings: {
    plot_type: 'Distribution', setup_time_mode: 'Relative time', setup_index: 'x',
    hist_cols_main: ['aux', 'signal'], hist_binmode: 'Fixed width', hist_width_input: 100, hist_bins_main: 31,
  }}, 1);
  const migrated = (await saved(page)).settings.distribution_panels;
  assert.equal(migrated.length, 1);
  assert.deepEqual(new Set(migrated[0].columns), new Set(['aux', 'signal']));
  assert.equal(migrated[0].binMode, 'Fixed width');
  assert.equal(migrated[0].binWidth, 100);
  for (const trace of await histogram(page, 0)) assert.equal(total(trace), rowCount);
  pass('Legacy histogram settings migrate into one distribution with the selected signals and fixed width');

  await load(page, twoPanels, 2);
  if (!await page.locator('#filter-panel').evaluate(el => el.open)) await page.locator('#filter-panel > summary').click();
  await page.locator('#add-filter').click();
  await page.selectOption('[data-key="flt2_var_0"]', 'x');
  await page.fill('[data-key="flt2_low_num_0"]', '100');
  await page.fill('[data-key="flt2_up_num_0"]', '199');
  await update(page, 2);
  assert.equal(await page.locator('#row-count').textContent(), '100');
  for (let i = 0; i < 2; i++) {
    const traces = await histogram(page, i);
    for (const trace of traces) assert.equal(total(trace), 100);
    const rows = table(await download(page, `[data-distribution-export="${i}"]`));
    await ready(page, 2);
    assert.equal(rows.length, traces[0].y.length);
    numericClose(rows.map(row => Number(row.bin_center)), traces[0].x);
    numericClose(rows.map(row => Number(row.bin_left)), traces[0].edges.map(pair => pair[0]));
    numericClose(rows.map(row => Number(row.bin_right)), traces[0].edges.map(pair => pair[1]));
    for (const trace of traces) assert.deepEqual(rows.map(row => Number(row[trace.name])), trace.y);
  }
  const png = await download(page, '[data-distribution-image="1"]', true);
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.ok(png.length > 1000);
  pass('Shared filters affect every card; independent CSV bins/counts and a real PNG match the rendered analysis');

  await edit(page, 0);
  await chooseSignals(page, ['sparse']);
  await update(page, 2);
  assert.equal(await page.locator('#chart').evaluate(el => el.data?.length || 0), 0);
  assert.deepEqual(await page.locator('[data-distribution-card="0"] [data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger)),
    {source: rowCount, matched: 100, filteredOut: rowCount - 100, invalidTime: 0, usable: 0, excludedValues: 100});
  assert.ok(await page.locator('[data-distribution-export="0"]').isDisabled());
  assert.ok(await page.locator('[data-distribution-image="0"]').isDisabled());
  for (const trace of await histogram(page, 1)) assert.equal(total(trace), 100);
  assert.ok(await page.locator('[data-distribution-export="1"]').isEnabled());
  await page.fill('[data-key="flt2_low_num_0"]', String(rowCount + 100));
  await page.fill('[data-key="flt2_up_num_0"]', String(rowCount + 200));
  await update(page, 2);
  assert.equal(await page.locator('#row-count').textContent(), '0');
  for (let i = 0; i < 2; i++) assert.equal(await page.locator(chartSelector(i)).evaluate(el => el.data?.length || 0), 0);
  await exportsDisabled(page);
  await page.locator('#clear-filters').click();
  await ready(page, 2);
  assert.equal(total((await histogram(page, 0))[0]), 50);
  for (const trace of await histogram(page, 1)) assert.equal(total(trace), rowCount);
  pass('An all-null filtered panel stays empty without blocking others; zero matches clear all cards and clearing filters recovers');

  await load(page, twoPanels, 2);
  const busyGuard = await page.locator('#render-button').evaluate(button => {
    button.click();
    return {busy: button.disabled, exportsDisabled: [...document.querySelectorAll('[data-distribution-export], [data-distribution-image]')].every(item => item.disabled)};
  });
  assert.deepEqual(busyGuard, {busy: true, exportsDisabled: true});
  await ready(page, 2);
  for (let i = 2; i < 6; i++) await page.locator('#add-distribution').click();
  assert.equal(await page.locator('[data-distribution-panel]').count(), 6);
  assert.ok(await page.locator('#add-distribution').isDisabled());
  assert.ok(await page.locator('#duplicate-distribution').isDisabled());
  await update(page, 6);
  for (let i = 0; i < 6; i++) for (const trace of await histogram(page, i)) assert.equal(total(trace), trace.name === 'sparse' ? 50 : rowCount);
  for (let i = 6; i > 1; i--) await page.locator('#remove-distribution').click();
  assert.equal(await page.locator('[data-distribution-panel]').count(), 1);
  assert.ok(await page.locator('#remove-distribution').isDisabled());
  await update(page, 1);
  pass('Busy exports and six-panel/one-panel bounds are enforced without dropping counts');

  // Rendering one view validates that view's panels; saving still validates the whole workspace.
  async function rejectInvalidWorkspaceSave() {
    const downloads = [];
    const observeDownload = item => downloads.push(item.suggestedFilename());
    page.on('download', observeDownload);
    try {
      await page.locator('#save-config').click();
      await page.waitForFunction(() => document.querySelector('#status')?.classList.contains('error'));
      assert.match(await page.locator('#status').textContent(), /invalid setting|greater than|at least/i);
      assert.deepEqual(downloads, [], 'An invalid inactive panel must prevent saving the complete configuration');
    } finally {
      page.off('download', observeDownload);
    }
  }
  await load(page, twoPanels, 2);
  await edit(page, 0);
  await selectField(page, 'binMode', 'Fixed width');
  await fillField(page, 'binWidth', 0);
  await page.getByRole('tab', {name: 'Scatter', exact: true}).click();
  await ready(page, 1, 'Scatter');
  assert.equal(await page.locator('#chart').evaluate(el => el.data.find(trace => trace.type === 'scattergl').x.length), rowCount);
  await rejectInvalidWorkspaceSave();
  await load(page, twoPanels, 2);
  await page.getByRole('tab', {name: 'Scatter', exact: true}).click();
  await ready(page, 1, 'Scatter');
  await selectField(page, 'color', 'aux');
  await selectField(page, 'colorBinMode', 'Bin width');
  await fillField(page, 'colorBinWidth', 0);
  await page.getByRole('tab', {name: 'Distribution', exact: true}).click();
  await ready(page, 2);
  for (let i = 0; i < 2; i++) for (const trace of await histogram(page, i)) assert.equal(total(trace), rowCount);
  await rejectInvalidWorkspaceSave();
  await load(page, twoPanels, 2);
  pass('Invalid inactive panels do not block another view, while saving the full workspace still rejects them');

  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true, acceptDownloads: true});
  observe(mobile);
  await mobile.goto(baseURL, {waitUntil: 'networkidle'});
  await uploadConfig(mobile, twoPanels);
  await mobile.waitForFunction(() => /settings.*loaded|loaded.*settings|choose.*dataset/i.test(document.querySelector('#status')?.textContent || ''));
  assert.ok(await mobile.locator('#workspace').evaluate(el => el.classList.contains('hidden')));
  await mobile.setInputFiles('#file-input', fixture);
  await ready(mobile, 2);
  assert.equal(await mobile.locator('#chart-type').inputValue(), 'Distribution');
  await noOverflow(mobile);
  for (let i = 0; i < 2; i++) for (const trace of await histogram(mobile, i)) assert.equal(total(trace), rowCount);
  assert.deepEqual((await saved(mobile)).settings.distribution_panels, twoPanels.settings.distribution_panels);
  await mobile.screenshot({path: path.join(artifacts, 'mobile-distributions.png'), fullPage: true});
  await load(page, twoPanels, 2);
  await page.screenshot({path: path.join(artifacts, 'desktop-distributions.png'), fullPage: true});
  await mobile.close();
  pass('Settings loaded before a file restore both cards at 390px with no horizontal overflow or missing counts');

  assert.deepEqual(errors, [], 'No uncaught browser errors');
  assert.deepEqual(writes, [], 'Local processing must not upload data');
  assert.deepEqual(external, [], 'Application dependencies must remain local');
  console.log(JSON.stringify({result: 'PASS', baseURL, rowCount, checks, artifacts}, null, 2));
} catch (error) {
  await page.screenshot({path: path.join(artifacts, 'failure.png'), fullPage: true}).catch(() => {});
  console.error(JSON.stringify({result: 'FAIL', message: String(error), status: await page.locator('#status').textContent().catch(() => ''), checks, errors, writes, external, artifacts}, null, 2));
  process.exitCode = 1;
} finally {
  await browser.close();
}
