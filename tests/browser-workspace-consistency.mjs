/** Real browser checks for consistent panels, workspace state, comparisons, and row accounting. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp, readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import * as XLSX from 'xlsx';
import {chooseSignals} from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5175/';
const artifacts = await mkdtemp(path.join(tmpdir(), 'tsplotter-workspace-consistency-'));
const rows = 300;
const file = {name: 'workspace-private-fixture.csv', mimeType: 'text/csv', buffer: Buffer.from(
  'x,signal,shifted,aux,category,payload\n' + Array.from({length: rows}, (_, i) =>
    `${i},${i + 1},${1000 + 2 * i},${i % 20},${i < 100 ? 'A' : i < 200 ? 'B' : 'C'},private-record-${i}`).join('\n'))};
const recoveryKey = 'tsplotter.workspace.v3';
const browser = await chromium.launch({headless: true,
  ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const checks = [], errors = [], writes = [], external = [];
const pass = message => { checks.push(message); console.log(message); };
const slugs = {'Trends': 'trend', 'Distribution': 'distribution', 'Scatter': 'scatter', 'Rolling window': 'rolling', 'Derivatives': 'derivative'};
const keys = {'Trends': 'trend_panels', 'Distribution': 'distribution_panels', 'Scatter': 'scatter_panels', 'Rolling window': 'rolling_panels', 'Derivatives': 'derivative_panels'};
function observe(target) {
  target.setDefaultTimeout(60000);
  target.on('pageerror', error => errors.push(String(error)));
  target.on('request', request => {
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) writes.push(`${request.method()} ${request.url()}`);
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) external.push(request.url());
  });
}
async function instrument(target) {
  observe(target);
  await target.addInitScript(() => {
    window.__workspaceEngineCalls = [];
    const original = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, ...args) {
      if (message?.method) window.__workspaceEngineCalls.push({method: message.method, mode: message.method === 'query' ? message.args?.[0]?.mode : undefined});
      return original.call(this, message, ...args);
    };
  });
}
await instrument(page);
const chart = (slug, index) => index === 0 ? '#chart' : `#${slug}-chart-${index}`;
async function ready(target, mode, count = 1) {
  await target.waitForFunction(({slug, count}) => {
    const status = document.querySelector('#status');
    return status?.classList.contains('error') || !document.querySelector('#render-button')?.disabled &&
      /ready|settings loaded/i.test(status?.textContent || '') && document.querySelectorAll(`[data-${slug}-card]`).length === count;
  }, {slug: slugs[mode], count}, {timeout: 120000});
  assert.ok(!await target.locator('#status').evaluate(el => el.classList.contains('error')), await target.locator('#status').textContent());
}
async function settled(target) {
  await target.waitForFunction(() => document.querySelector('#status')?.classList.contains('error') ||
    !document.querySelector('#render-button')?.disabled && /ready/i.test(document.querySelector('#status')?.textContent || ''));
  assert.ok(!await target.locator('#status').evaluate(el => el.classList.contains('error')), await target.locator('#status').textContent());
}
async function mode(target, name, count = 1) {
  await target.getByRole('tab', {name, exact: true}).click();
  await ready(target, name, count);
}
async function update(target, name, count = 1) {
  await target.locator('#render-button').click();
  await ready(target, name, count);
}
async function field(target, key, value) {
  const element = target.locator(`[data-panel-key="${key}"]`);
  if (!await element.isVisible()) {
    const details = element.locator('xpath=ancestor::details[1]');
    if (!await details.evaluate(el => el.open)) await details.locator('summary').first().click();
  }
  if (await element.evaluate(el => el.tagName === 'SELECT')) await element.selectOption(value);
  else if (await element.evaluate(el => el.type === 'checkbox')) await element.setChecked(Boolean(value));
  else { await element.fill(String(value)); await element.press('Tab'); }
}
async function edit(target, slug, index) {
  await target.locator(`[data-${slug}-panel="${index}"]`).click();
  assert.equal(await target.locator(`[data-${slug}-panel="${index}"]`).getAttribute('aria-pressed'), 'true');
}
async function checkbox(target, selector, value) {
  const element = target.locator(selector);
  if (!await element.isVisible()) {
    const details = element.locator('xpath=ancestor::details[1]');
    if (!await details.evaluate(el => el.open)) await details.locator('summary').first().click();
  }
  await element.setChecked(value);
}
async function download(target, selector) {
  const event = target.waitForEvent('download');
  await target.locator(selector).click();
  return await readFile(await (await event).path(), 'utf8');
}
async function saved(target) {
  const config = JSON.parse(await download(target, '#save-config'));
  assert.equal(config.app, 'tsplotter');
  assert.equal(config.version, 3);
  return config;
}
async function load(target, config, name, count = 1) {
  await target.setInputFiles('#config-input', {name: 'workspace-settings.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config))});
  if (await target.locator('#load-config').count()) await target.locator('#load-config').click();
  await ready(target, name, count);
}
async function traces(target, slug, index) {
  return await target.locator(chart(slug, index)).evaluate(graph => (graph.data || []).filter(trace => trace.x?.some(value => value !== null)).map(trace => ({
    name: trace.name, type: trace.type, x: Array.from(trace.x || []), y: Array.from(trace.y || []),
    color: Array.isArray(trace.marker?.color) || ArrayBuffer.isView(trace.marker?.color) ? Array.from(trace.marker.color) : trace.marker?.color,
    edges: trace.type === 'bar' ? Array.from(trace.customdata || [], pair => Array.from(pair)) : undefined,
  })));
}
function csvTable(text) {
  const workbook = XLSX.read(text, {type: 'string', raw: true});
  return XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]], {raw: true, defval: ''});
}
async function filterRange(target, low, high) {
  if (!await target.locator('#filter-panel').evaluate(el => el.open)) await target.locator('#filter-panel > summary').click();
  if (!await target.locator('[data-key="flt2_var_0"]').count()) await target.locator('#add-filter').click();
  await target.selectOption('[data-key="flt2_var_0"]', 'x');
  await target.fill('[data-key="flt2_low_num_0"]', String(low));
  await target.fill('[data-key="flt2_up_num_0"]', String(high));
}
async function noOverflow(target, slug) {
  const bounds = await target.evaluate(() => ({width: innerWidth, scroll: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)}));
  assert.ok(bounds.scroll <= bounds.width + 1, `Horizontal page overflow: ${JSON.stringify(bounds)}`);
  for (const card of await target.locator(`[data-${slug}-card]`).all()) {
    const rect = await card.boundingBox();
    assert.ok(rect && rect.x >= -1 && rect.x + rect.width <= bounds.width + 1, `Card exceeds viewport: ${JSON.stringify(rect)}`);
  }
}
const withoutTitle = panel => Object.fromEntries(Object.entries(panel).filter(([key]) => key !== 'title'));
const total = trace => trace.y.reduce((sum, value) => sum + Number(value), 0);

try {
  await page.goto(baseURL, {waitUntil: 'networkidle'});
  await page.setInputFiles('#file-input', file);
  await ready(page, 'Trends');

  // The formerly single-instance views now follow the same panel lifecycle.
  for (const [name, first, second] of [
    ['Trends', {columns: ['signal']}, {columns: ['shifted']}],
    ['Rolling window', {y: 'signal', window: 5}, {y: 'shifted', window: 11}],
    ['Derivatives', {y: 'signal', smoothWindow: 1}, {y: 'shifted', smoothWindow: 3}],
  ]) {
    const slug = slugs[name];
    await mode(page, name);
    await field(page, 'title', `${name} first`);
    for (const [key, value] of Object.entries(first)) key === 'columns' ? await chooseSignals(page, value) : await field(page, key, value);
    await page.locator(`#add-${slug}`).click();
    await field(page, 'title', `${name} second`);
    for (const [key, value] of Object.entries(second)) key === 'columns' ? await chooseSignals(page, value) : await field(page, key, value);
    await update(page, name, 2);
    for (let index = 0; index < 2; index++) assert.ok((await traces(page, slug, index)).some(trace => trace.x.length === rows));
    const before = (await saved(page)).settings[keys[name]];
    await page.locator(`#duplicate-${slug}`).click();
    await update(page, name, 3);
    const copied = (await saved(page)).settings[keys[name]];
    assert.deepEqual(withoutTitle(copied[2]), withoutTitle(before[1]));
    await page.locator(`#remove-${slug}`).click();
    await update(page, name, 2);
    assert.deepEqual((await saved(page)).settings[keys[name]], before);
    await edit(page, slug, 0);
    assert.ok(!await page.locator('#dirty-indicator').isVisible());
  }
  pass('Trends, rolling windows and derivatives share independent Add/Duplicate/Remove panels and preserve every row');

  await mode(page, 'Scatter');
  await field(page, 'x', 'x'); await field(page, 'y', 'signal'); await field(page, 'color', 'category');
  await page.locator('#add-scatter').click();
  await field(page, 'x', 'x'); await field(page, 'y', 'signal');
  await update(page, 'Scatter', 2);
  assert.ok(!await page.locator('#comparison-link-x').isChecked());
  const originalRange = await page.locator('#scatter-chart-1').evaluate(graph => [...graph.layout.xaxis.range]);
  await page.locator('#chart').evaluate(graph => graph.emit('plotly_relayout', {'xaxis.range[0]': 100, 'xaxis.range[1]': 150}));
  assert.deepEqual(await page.locator('#scatter-chart-1').evaluate(graph => [...graph.layout.xaxis.range]), originalRange);
  await checkbox(page, '#comparison-link-x', true); await checkbox(page, '#comparison-link-y', true);
  await update(page, 'Scatter', 2);
  const queryCountBeforeZoom = await page.evaluate(() => window.__workspaceEngineCalls.length);
  await page.locator('#chart').evaluate(graph => graph.emit('plotly_relayout', {'xaxis.range[0]': 100, 'xaxis.range[1]': 150, 'yaxis.range[0]': 110, 'yaxis.range[1]': 130}));
  await page.waitForFunction(() => {
    const graph = document.querySelector('#scatter-chart-1');
    return Number(graph?.layout?.xaxis?.range?.[0]) === 100 && Number(graph?.layout?.xaxis?.range?.[1]) === 150 &&
      Number(graph?.layout?.yaxis?.range?.[0]) === 110 && Number(graph?.layout?.yaxis?.range?.[1]) === 130;
  });
  assert.equal(await page.evaluate(() => window.__workspaceEngineCalls.length), queryCountBeforeZoom);
  pass('Linked compatible X/Y ranges are optional, propagate Plotly zoom events and do not rerun analysis');

  const originalColors = (await traces(page, 'scatter', 0))[0];
  await filterRange(page, 100, 199); await update(page, 'Scatter', 2);
  const filteredColors = (await traces(page, 'scatter', 0))[0];
  assert.equal(filteredColors.x.length, 100);
  assert.deepEqual(filteredColors.color, filteredColors.x.map(x => originalColors.color[originalColors.x.indexOf(x)]));
  assert.equal(await page.locator('#row-count').textContent(), '100');
  pass('Removing earlier categories with global filters does not reassign surviving colors');
  await page.locator('#clear-filters').click(); await ready(page, 'Scatter', 2);

  await edit(page, 'scatter', 0);
  await field(page, 'color', 'signal'); await field(page, 'colorBinMode', 'Number of bins');
  await field(page, 'colorBinCount', 4); await field(page, 'colorBinScope', 'Full dataset');
  await field(page, 'marker', 'shifted'); await field(page, 'markerBinMode', 'Number of bins');
  await field(page, 'markerBinCount', 4); await field(page, 'markerBinScope', 'Filtered data');
  await update(page, 'Scatter', 2);
  const unfilteredBins = csvTable(await download(page, '[data-scatter-export="0"]'));
  const colorColumn = Object.keys(unfilteredBins[0]).find(key => key.startsWith('__color_group'));
  const markerColumn = Object.keys(unfilteredBins[0]).find(key => key.startsWith('__marker_group'));
  assert.ok(colorColumn && markerColumn);
  const binByX = new Map(unfilteredBins.map(row => [Number(row.x), row]));
  await filterRange(page, 100, 199); await update(page, 'Scatter', 2);
  const selectedBins = csvTable(await download(page, '[data-scatter-export="0"]'));
  assert.equal(selectedBins.length, 100);
  assert.deepEqual(selectedBins.map(row => row[colorColumn]), selectedBins.map(row => binByX.get(Number(row.x))[colorColumn]));
  assert.ok(selectedBins.some(row => row[markerColumn] !== binByX.get(Number(row.x))[markerColumn]));
  pass('Color and marker bin scopes operate independently: dataset intervals stay fixed while filtered intervals adapt');
  await page.locator('#clear-filters').click(); await ready(page, 'Scatter', 2);

  await mode(page, 'Distribution');
  await chooseSignals(page, ['signal']);
  await field(page, 'binMode', 'Fixed width'); await field(page, 'binWidth', 7);
  await field(page, 'binMode', 'Target #bins'); await field(page, 'binCount', 10);
  await field(page, 'binScope', 'Full dataset');
  await page.locator('#add-distribution').click(); await chooseSignals(page, ['shifted']);
  await field(page, 'binMode', 'Target #bins'); await field(page, 'binCount', 10); await field(page, 'binScope', 'Filtered data');
  await update(page, 'Distribution', 2);
  const fullEdges = (await traces(page, 'distribution', 0))[0].edges;
  const otherFullEdges = (await traces(page, 'distribution', 1))[0].edges;
  await filterRange(page, 100, 199); await update(page, 'Distribution', 2);
  assert.deepEqual((await traces(page, 'distribution', 0))[0].edges, fullEdges);
  assert.notDeepEqual((await traces(page, 'distribution', 1))[0].edges, otherFullEdges);
  for (let index = 0; index < 2; index++) assert.equal(total((await traces(page, 'distribution', index))[0]), 100);
  await field(page, 'binScope', 'Full dataset');
  await checkbox(page, '#distribution-shared-bins', true); await update(page, 'Distribution', 2);
  assert.deepEqual((await traces(page, 'distribution', 0))[0].edges, (await traces(page, 'distribution', 1))[0].edges);
  const comparisonConfig = await saved(page);
  assert.equal(comparisonConfig.settings.distribution_panels[0].binWidth, 7, 'A valid inactive width remains reusable');
  pass('Full-dataset bins stay fixed across filters, filtered bins adapt, shared bins align comparisons and v3 preserves inactive settings');

  // Data inspection is independent from chart validation and never clears unapplied chart edits.
  await edit(page, 'distribution', 0); await field(page, 'binMode', 'Fixed width'); await field(page, 'binWidth', 0);
  assert.ok(await page.locator('#dirty-indicator').isVisible());
  const staleCharts = [await traces(page, 'distribution', 0), await traces(page, 'distribution', 1)];
  const callStart = await page.evaluate(() => window.__workspaceEngineCalls.length);
  if (!await page.locator('#preview-panel').evaluate(el => el.open)) await page.locator('#preview-panel > summary').click();
  await page.locator('#preview-button').click(); await settled(page);
  assert.equal(await page.locator('#preview-table tbody tr').count(), 100);
  if (!await page.locator('#summary-panel').evaluate(el => el.open)) await page.locator('#summary-panel > summary').click();
  await page.locator('#summary-button').click(); await settled(page);
  assert.ok(await page.locator('#summary-table tbody tr').count());
  const exported = csvTable(await download(page, '#data-export')); await settled(page);
  assert.equal(exported.length, 100); assert.equal(Number(exported[0].x), 100); assert.equal(Number(exported.at(-1).x), 199);
  const dataCalls = await page.evaluate(start => window.__workspaceEngineCalls.slice(start), callStart);
  assert.ok(!dataCalls.some(call => ['trends', 'scatter', 'distribution', 'rolling', 'derivatives'].includes(call.mode)), JSON.stringify(dataCalls));
  assert.ok(await page.locator('#dirty-indicator').isVisible());
  assert.deepEqual([await traces(page, 'distribution', 0), await traces(page, 'distribution', 1)], staleCharts);
  await field(page, 'binWidth', 10); await update(page, 'Distribution', 2);
  assert.ok(!await page.locator('#dirty-indicator').isVisible());
  assert.equal(await page.locator('#settings-indicator').getAttribute('data-unsaved'), 'true');
  await saved(page);
  assert.equal(await page.locator('#settings-indicator').getAttribute('data-unsaved'), 'false');
  pass('Preview, summary and full CSV work with invalid chart options without chart queries or clearing edits; plot and file-save states remain distinct');

  // A draft survives reload, but the local dataset never does.
  await field(page, 'title', 'Recovered analysis'); await update(page, 'Distribution', 2);
  assert.equal(await page.locator('#settings-indicator').getAttribute('data-unsaved'), 'true');
  await page.waitForFunction(key => localStorage.getItem(key)?.includes('Recovered analysis'), recoveryKey);
  const recovery = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), recoveryKey);
  assert.equal(typeof recovery.config, 'string'); assert.equal(typeof recovery.savedConfig, 'string');
  assert.ok(!JSON.stringify(recovery).includes(file.name)); assert.ok(!JSON.stringify(recovery).includes('private-record-'));
  await page.reload({waitUntil: 'networkidle'});
  assert.ok(await page.locator('#workspace').evaluate(el => el.classList.contains('hidden')));
  await page.setInputFiles('#file-input', file); await ready(page, 'Distribution', 2);
  assert.equal(await page.locator('[data-panel-key="title"]').inputValue(), 'Recovered analysis');
  assert.equal(await page.locator('#row-count').textContent(), '100');
  assert.equal(await page.locator('#settings-indicator').getAttribute('data-unsaved'), 'true');
  const beforeClear = [await traces(page, 'distribution', 0), await traces(page, 'distribution', 1)];
  await page.locator('#clear-recovery').click();
  assert.equal(await page.evaluate(key => localStorage.getItem(key), recoveryKey), null);
  assert.deepEqual([await traces(page, 'distribution', 0), await traces(page, 'distribution', 1)], beforeClear);
  await page.reload({waitUntil: 'networkidle'});
  assert.equal(await page.evaluate(key => localStorage.getItem(key), recoveryKey), null);
  assert.ok(await page.locator('#workspace').evaluate(el => el.classList.contains('hidden')));
  await page.setInputFiles('#file-input', file); await ready(page, 'Trends');
  pass('Local recovery restores unsaved workspace choices only after reopening a dataset; clearing recovery preserves the current analysis');

  await load(page, {app: 'tsplotter', version: 1, settings: {plot_type: 'Rolling window', setup_time_mode: 'Relative time', setup_index: 'x', rw_y: 'signal', rw_window: 7, rw_include_current: false}}, 'Rolling window');
  assert.equal(await page.locator('[data-panel-key="y"]').inputValue(), 'signal');
  assert.equal(await page.locator('[data-panel-key="window"]').inputValue(), '7');
  assert.equal(await page.locator('[data-panel-key="includeCurrent"]').isChecked(), false);
  const legacy = await saved(page); assert.equal(legacy.settings.rolling_panels.length, 1);
  await load(page, {app: 'tsplotter', version: 2, settings: {plot_type: 'Derivatives', setup_time_mode: 'Relative time', setup_index: 'x', der_y: 'shifted', der_smooth: 3, der_tunit: 'minutes'}}, 'Derivatives');
  assert.equal(await page.locator('[data-panel-key="y"]').inputValue(), 'shifted');
  assert.equal(await page.locator('[data-panel-key="smoothWindow"]').inputValue(), '3');
  assert.equal(await page.locator('[data-panel-key="timeUnit"]').inputValue(), 'minutes');
  assert.equal((await saved(page)).settings.derivative_panels.length, 1);
  pass('Version 1 rolling and version 2 derivative configurations migrate into the common version 3 panel model');

  const ledgerPage = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
  await instrument(ledgerPage);
  await ledgerPage.goto(baseURL, {waitUntil: 'networkidle'});
  const ledgerCSV = 'time,x,signal\n2026-01-01T00:00:00Z,0,1\n2026-01-02T00:00:00Z,1,\nbad,2,3\n2026-01-04T00:00:00Z,3,4\n2026-01-05T00:00:00Z,4,Infinity\n2026-01-06T00:00:00Z,5,6\nbad,6,7\n2026-01-08T00:00:00Z,7,8\n';
  await ledgerPage.setInputFiles('#file-input', {name: 'row-ledger.csv', mimeType: 'text/csv', buffer: Buffer.from(ledgerCSV)});
  await ready(ledgerPage, 'Trends');
  await mode(ledgerPage, 'Scatter'); await field(ledgerPage, 'x', 'x'); await field(ledgerPage, 'y', 'signal');
  await ledgerPage.selectOption('[data-key="setup_time_mode"]', 'Absolute time');
  await ledgerPage.selectOption('#x-column', 'time');
  await filterRange(ledgerPage, 1, 6); await update(ledgerPage, 'Scatter');
  assert.equal(await ledgerPage.locator('#row-count').textContent(), '6');
  const ledger = await ledgerPage.locator('[data-scatter-card="0"] [data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger));
  assert.deepEqual(Object.fromEntries(['source', 'matched', 'filteredOut', 'invalidTime', 'usable', 'excludedValues'].map(key => [key, Number(ledger[key])])),
    {source: 8, matched: 6, filteredOut: 2, invalidTime: 2, usable: 2, excludedValues: 2});
  const plotted = (await traces(ledgerPage, 'scatter', 0))[0];
  assert.deepEqual(plotted.x, [3, 5]); assert.deepEqual(plotted.y, [4, 6]);
  await ledgerPage.screenshot({path: path.join(artifacts, 'row-ledger.png'), fullPage: true});
  await ledgerPage.close();
  pass('Global matching, invalid timestamps and unusable selected values have separate accurate row ledgers');

  const axisPage = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
  await instrument(axisPage);
  await axisPage.goto(baseURL, {waitUntil: 'networkidle'});
  const missingCSV = 'x,signal,partial\n' + Array.from({length: rows}, (_, i) =>
    `${i},${i + 1},${i >= 50 && i < 250 ? i + 1 : ''}`).join('\n');
  await axisPage.setInputFiles('#file-input', {name: 'different-usable-extents.csv', mimeType: 'text/csv', buffer: Buffer.from(missingCSV)});
  await ready(axisPage, 'Trends'); await mode(axisPage, 'Scatter');
  await field(axisPage, 'x', 'x'); await field(axisPage, 'y', 'signal');
  await axisPage.locator('#add-scatter').click(); await field(axisPage, 'x', 'x'); await field(axisPage, 'y', 'partial');
  await update(axisPage, 'Scatter', 2);
  const ranges = () => axisPage.evaluate(() => ['#chart', '#scatter-chart-1'].map(selector => [...document.querySelector(selector)._fullLayout.xaxis.range].map(Number)));
  assert.notDeepEqual(...await ranges(), 'Different selected-value omissions must create different independent axis extents');
  assert.deepEqual((await traces(axisPage, 'scatter', 0))[0].x, Array.from({length: rows}, (_, i) => i));
  assert.deepEqual((await traces(axisPage, 'scatter', 1))[0].x, Array.from({length: 200}, (_, i) => i + 50));
  await checkbox(axisPage, '#comparison-link-x', true); await update(axisPage, 'Scatter', 2);
  const linkedRanges = await ranges(); assert.deepEqual(linkedRanges[0], linkedRanges[1]);
  const callsBeforeZoom = await axisPage.evaluate(() => window.__workspaceEngineCalls.length);
  await axisPage.locator('#chart').hover();
  await axisPage.locator('#chart .modebar-btn[data-title="Zoom in"]').click();
  await axisPage.waitForFunction(span => {
    const [first, second] = ['#chart', '#scatter-chart-1'].map(selector => document.querySelector(selector)._fullLayout.xaxis.range.map(Number));
    return first[0] === second[0] && first[1] === second[1] && first[1] - first[0] < span;
  }, linkedRanges[0][1] - linkedRanges[0][0]);
  // The un-applied checkbox edit must not alter the behavior of the existing plots.
  await checkbox(axisPage, '#comparison-link-x', false);
  assert.ok(await axisPage.locator('#dirty-indicator').isVisible());
  await axisPage.locator('#scatter-chart-1').hover();
  await axisPage.locator('#scatter-chart-1 .modebar-btn[data-title="Reset axes"]').click();
  await axisPage.waitForFunction(() => {
    const [first, second] = ['#chart', '#scatter-chart-1'].map(selector => document.querySelector(selector)._fullLayout.xaxis.range.map(Number));
    return first[0] === second[0] && first[1] === second[1] && first[0] <= 0 && first[1] >= 299;
  });
  assert.equal(await axisPage.evaluate(() => window.__workspaceEngineCalls.length), callsBeforeZoom);
  await axisPage.screenshot({path: path.join(artifacts, 'linked-axis-reset.png'), fullPage: true});
  await axisPage.close();
  pass('Real Plotly zoom and reset preserve linked union ranges despite different missingness and unapplied link edits');

  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true, acceptDownloads: true});
  await instrument(mobile);
  await mobile.goto(baseURL, {waitUntil: 'networkidle'});
  await mobile.setInputFiles('#file-input', file); await ready(mobile, 'Trends');
  for (const name of Object.keys(slugs)) {
    const slug = slugs[name]; await mode(mobile, name);
    await mobile.locator(`#add-${slug}`).click(); await update(mobile, name, 2);
    await noOverflow(mobile, slug);
    assert.equal(await mobile.locator('[data-panel-key="binMode"]').count(), name === 'Distribution' ? 1 : 0);
    assert.equal(await mobile.locator('[data-panel-key="color"]').count(), name === 'Scatter' ? 1 : 0);
    assert.equal(await mobile.locator('[data-panel-key="window"]').count(), name === 'Rolling window' ? 1 : 0);
    assert.equal(await mobile.locator('[data-panel-key="timeUnit"]').count(), name === 'Derivatives' ? 1 : 0);
    for (const card of await mobile.locator(`[data-${slug}-card]`).all()) {
      const ledger = await card.locator('[data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger));
      assert.equal(Number(ledger.source), rows); assert.equal(Number(ledger.matched), rows); assert.equal(Number(ledger.usable), rows);
    }
    await mobile.screenshot({path: path.join(artifacts, `mobile-${slug}.png`), fullPage: true});
  }
  const mobileRecovery = await mobile.evaluate(key => localStorage.getItem(key), recoveryKey);
  assert.ok(mobileRecovery && !mobileRecovery.includes(file.name) && !mobileRecovery.includes('private-record-'));
  await mobile.reload({waitUntil: 'networkidle'});
  assert.ok(await mobile.locator('#workspace').evaluate(el => el.classList.contains('hidden')));
  await mobile.setInputFiles('#file-input', file); await ready(mobile, 'Derivatives', 2);
  await noOverflow(mobile, 'derivative');
  await mobile.close();
  pass('All five views keep two scoped-control cards and accurate ledgers within390px; mobile recovery stores no dataset');

  await page.screenshot({path: path.join(artifacts, 'workspace.png'), fullPage: true});
  assert.deepEqual(errors, []); assert.deepEqual(writes, []); assert.deepEqual(external, []);
  console.log(JSON.stringify({result: 'PASS', baseURL, checks, artifacts}, null, 2));
} catch (error) {
  await page.screenshot({path: path.join(artifacts, 'failure.png'), fullPage: true}).catch(() => {});
  console.error(JSON.stringify({result: 'FAIL', message: String(error), status: await page.locator('#status').textContent().catch(() => ''), checks, errors, writes, external, artifacts}, null, 2));
  process.exitCode = 1;
} finally { await browser.close(); }
