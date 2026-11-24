/** Exercise independent scatter panels through the actual browser UI and exports. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp, readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import * as XLSX from 'xlsx';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5173/';
const rowCount = Number(process.env.TEST_ROWS || 2401);
assert.ok(Number.isInteger(rowCount) && rowCount >= 201, 'TEST_ROWS must be at least 201');
const artifacts = await mkdtemp(path.join(tmpdir(), 'tsplotter-scatter-panels-'));
const csv = 'x,signal,aux,category\n' + Array.from({length: rowCount}, (_, i) =>
  `${i},${2 * i + 1},${i % 101},${i % 2 ? 'odd' : 'even'}`).join('\n');
const file = {name: 'independent-scatter.csv', mimeType: 'text/csv', buffer: Buffer.from(csv)};
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
const chartSelector = index => index === 0 ? '#chart' : `#scatter-chart-${index}`;

async function ready(target, panels = undefined) {
  await target.waitForFunction(expected => {
    const status = document.querySelector('#status');
    if (status?.classList.contains('error')) return true;
    if (document.querySelector('#render-button')?.disabled || !/ready|settings loaded/i.test(status?.textContent || '')) return false;
    if (expected === undefined) return Array.isArray(document.querySelector('#chart')?.data);
    if (document.querySelectorAll('[data-scatter-card]').length !== expected) return false;
    return Array.from({length: expected}, (_, i) => document.querySelector(i === 0 ? '#chart' : `#scatter-chart-${i}`))
      .every(chart => Array.isArray(chart?.data));
  }, panels, {timeout: 120000});
  assert.ok(!await target.locator('#status').evaluate(el => el.classList.contains('error')), await target.locator('#status').textContent());
}
async function update(target, count) {
  await target.locator('#render-button').click();
  await ready(target, count);
}
async function reveal(field) {
  if (!await field.isVisible()) {
    const details = field.locator('xpath=ancestor::details[1]');
    assert.ok(await details.count(), 'A hidden panel control must have an expandable container');
    if (!await details.evaluate(el => el.open)) await details.locator('summary').first().click();
  }
}
async function selectPanelField(target, key, value) {
  const field = target.locator(`[data-panel-key="${key}"]`);
  await reveal(field);
  await field.selectOption(value);
}
async function fillPanelField(target, key, value) {
  const field = target.locator(`[data-panel-key="${key}"]`);
  await reveal(field);
  await field.fill(String(value));
  await field.press('Tab');
}
async function edit(target, index) {
  await target.locator(`[data-scatter-panel="${index}"]`).click();
  assert.equal(await target.locator(`[data-scatter-panel="${index}"]`).getAttribute('aria-pressed'), 'true');
}
async function download(target, selector) {
  const event = target.waitForEvent('download');
  await target.locator(selector).click();
  const result = await event;
  return await readFile(await result.path(), 'utf8');
}
async function saved(target) {
  const text = await download(target, '#save-config');
  const document = JSON.parse(text);
  assert.equal(document.app, 'tsplotter');
  assert.equal(document.version, 4);
  for (const key of ['trend_panels', 'rolling_panels', 'derivative_panels']) assert.ok(document.settings[key].length >= 1);
  for (const panel of document.settings.trend_panels) {
    assert.ok(panel.subplots.length >= 1);
    assert.equal(Object.hasOwn(panel, 'columns'), false);
  }
  assert.ok(!text.includes(file.name), 'Configurations must exclude the dataset source');
  assert.ok(Buffer.byteLength(text) < 256 * 1024);
  return document;
}
async function load(target, document, count) {
  await target.setInputFiles('#config-input', {name: 'scatter-settings.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(document))});
  if (await target.locator('#load-config').count()) await target.locator('#load-config').click();
  await ready(target, count);
}
async function points(target, index) {
  return await target.locator(chartSelector(index)).evaluate(chart => {
    const trace = chart.data.find(item => item.type === 'scattergl' && item.x?.length && item.x.some(value => value !== null));
    if (!trace) return null;
    const arrayOrScalar = value => Array.isArray(value) || ArrayBuffer.isView(value) ? Array.from(value) : value;
    return {x: Array.from(trace.x), y: Array.from(trace.y), color: arrayOrScalar(trace.marker?.color),
      symbol: arrayOrScalar(trace.marker?.symbol), size: trace.marker?.size, opacity: trace.marker?.opacity,
      coloraxis: trace.marker?.coloraxis,
      legend: chart.data.filter(item => item.showlegend).map(item => ({name: item.name, group: item.legendgroup, color: item.marker?.color, symbol: item.marker?.symbol}))};
  });
}
function table(text) {
  const workbook = XLSX.read(text, {type: 'string', raw: true});
  return XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]], {defval: '', raw: true});
}
async function noOverflow(target) {
  const dimensions = await target.evaluate(() => ({width: innerWidth, scroll: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)}));
  assert.ok(dimensions.scroll <= dimensions.width + 1, `Horizontal overflow: ${JSON.stringify(dimensions)}`);
  for (const card of await target.locator('[data-scatter-card]').all()) {
    const rect = await card.boundingBox();
    assert.ok(rect && rect.x >= -1 && rect.x + rect.width <= dimensions.width + 1, `Scatter card exceeds viewport: ${JSON.stringify(rect)}`);
  }
}
const withoutTitle = panel => Object.fromEntries(Object.entries(panel).filter(([key]) => key !== 'title'));

try {
  await page.goto(baseURL, {waitUntil: 'networkidle'});
  await page.setInputFiles('#file-input', file);
  await ready(page);
  await page.getByRole('tab', {name: 'Scatter', exact: true}).click();
  await ready(page, 1);
  assert.equal(await page.getByRole('tab', {name: 'Grouped scatter', exact: true}).count(), 0);
  assert.equal(await page.getByRole('tab').count(), 5);
  assert.equal(await page.locator('#chart-type').inputValue(), 'Scatter');
  assert.equal(await page.locator('[data-panel-key="palette"], [data-panel-key="colormap"], [data-panel-key="markerPreset"]').count(), 0,
    'Plain scatter appearance should show only relevant controls');
  pass('One Scatter view exposes color and marker grouping without a duplicate grouped-scatter tab');

  await fillPanelField(page, 'title', 'Continuous signal');
  await selectPanelField(page, 'x', 'x');
  await selectPanelField(page, 'y', 'signal');
  await selectPanelField(page, 'color', 'aux');
  await selectPanelField(page, 'marker', 'category');
  await selectPanelField(page, 'colorMode', 'continuous');
  assert.equal(await page.locator('[data-panel-key="palette"]').count(), 0);
  assert.equal(await page.locator('[data-panel-key="colormap"]').count(), 1);
  await update(page, 1);
  const continuous = await points(page, 0);
  assert.equal(continuous.x.length, rowCount);
  assert.deepEqual([continuous.x[0], continuous.x.at(-1), continuous.y[0], continuous.y.at(-1)], [0, rowCount - 1, 1, 2 * rowCount - 1]);
  assert.ok(continuous.color.every(value => typeof value === 'number'));
  assert.equal(new Set(continuous.symbol).size, 2);
  pass(`Continuous numeric colors coexist with category markers and retain all ${rowCount.toLocaleString()} points`);

  await page.locator('#add-scatter').click();
  assert.equal(await page.locator('[data-scatter-panel]').count(), 2);
  await fillPanelField(page, 'title', 'Grouped auxiliary');
  await selectPanelField(page, 'x', 'signal');
  await selectPanelField(page, 'y', 'aux');
  await selectPanelField(page, 'color', 'aux');
  await selectPanelField(page, 'marker', 'aux');
  await selectPanelField(page, 'colorMode', 'discrete');
  await selectPanelField(page, 'colorBinMode', 'Number of bins');
  await fillPanelField(page, 'colorBinCount', 4);
  await selectPanelField(page, 'markerBinMode', 'Bin width');
  await fillPanelField(page, 'markerBinWidth', 25);
  await selectPanelField(page, 'palette', 'D3');
  await selectPanelField(page, 'markerPreset', 'Crosshair Set');
  await fillPanelField(page, 'size', 11);
  await fillPanelField(page, 'opacity', .6);
  await update(page, 2);
  assert.deepEqual(await points(page, 0), continuous, 'Adding another panel must preserve the first panel');
  const grouped = await points(page, 1);
  assert.equal(grouped.x.length, rowCount);
  assert.deepEqual([grouped.x[0], grouped.x.at(-1)], [1, 2 * rowCount - 1]);
  assert.equal(new Set(grouped.color).size, 4);
  assert.equal(new Set(grouped.symbol).size, 4);
  assert.equal(grouped.size, 11);
  assert.equal(grouped.opacity, .6);
  assert.ok(grouped.color.every(value => typeof value === 'string'));
  pass('Two cards retain independent axes, palette, color bins, marker bins and display settings');

  const colorGuide = grouped.legend[0];
  const colorLegend = page.locator('#scatter-chart-1 .legend .traces').filter({hasText: colorGuide.name}).first();
  await colorLegend.click();
  const hiddenCount = grouped.color.filter(color => color !== colorGuide.color).length;
  await page.waitForFunction(count => document.querySelector('#scatter-chart-1').data[0].x.length === count, hiddenCount);
  assert.deepEqual(await points(page, 0), continuous, 'Legend selection must affect only its own card');
  const hidden = await points(page, 1);
  assert.ok(hidden.color.every(color => color !== colorGuide.color));
  await page.waitForTimeout(350); // A second single click, outside Plotly's double-click interval.
  await colorLegend.click();
  await page.waitForFunction(count => document.querySelector('#scatter-chart-1').data[0].x.length === count, rowCount);
  assert.deepEqual(await points(page, 1), grouped, 'Showing a group restores its original points and encodings');
  const markerGuide = continuous.legend[0];
  const markerLegend = page.locator('#chart .legend .traces').filter({hasText: markerGuide.name});
  await markerLegend.click();
  const remaining = continuous.symbol.filter(symbol => symbol !== markerGuide.symbol).length;
  await page.waitForFunction(count => document.querySelector('#chart').data[0].x.length === count, remaining);
  assert.deepEqual(await page.locator('#chart').evaluate(chart => [chart.data[0].marker.cmin, chart.data[0].marker.cmax]), [0, 100]);
  await page.waitForTimeout(350);
  await markerLegend.click();
  await page.waitForFunction(count => document.querySelector('#chart').data[0].x.length === count, rowCount);
  assert.deepEqual(await points(page, 0), continuous);
  const pngEvent = page.waitForEvent('download');
  await page.locator('[data-scatter-image="1"]').click();
  const png = await readFile(await (await pngEvent).path());
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.ok(png.readUInt32BE(16) > 0 && png.readUInt32BE(20) > 0);
  pass('Card legends hide and restore only their groups, continuous colors keep a fixed scale, and each card exports a PNG');

  await page.locator('[data-edit-scatter="0"]').click();
  assert.ok(!await page.locator('#dirty-indicator').isVisible(), 'Switching the editor alone must stay clean');
  assert.deepEqual(await points(page, 1), grouped);
  await fillPanelField(page, 'size', 9);
  assert.ok(await page.locator('#dirty-indicator').isVisible());
  for (const button of await page.locator('[data-scatter-export], [data-scatter-image]').all()) assert.ok(await button.isDisabled());
  assert.deepEqual(await points(page, 1), grouped, 'Editing one panel must leave the other rendered panel intact');
  await update(page, 2);
  assert.deepEqual(await points(page, 1), grouped);
  assert.equal((await points(page, 0)).size, 9);
  pass('Selecting an editor is clean; changing one panel preserves other panels and disables stale exports');

  const originalConfig = await saved(page);
  assert.equal(originalConfig.settings.scatter_panels.length, 2);
  await page.locator('#duplicate-scatter').click();
  assert.equal(await page.locator('[data-scatter-panel]').count(), 3);
  await update(page, 3);
  const duplicated = await saved(page);
  assert.deepEqual(withoutTitle(duplicated.settings.scatter_panels[2]), withoutTitle(originalConfig.settings.scatter_panels[0]));
  await fillPanelField(page, 'title', 'Independent copy');
  await selectPanelField(page, 'colorMode', 'discrete');
  await selectPanelField(page, 'colorBinMode', 'Number of bins');
  await fillPanelField(page, 'colorBinCount', 3);
  await selectPanelField(page, 'palette', 'Set2');
  await update(page, 3);
  const changedCopy = await saved(page);
  assert.deepEqual(changedCopy.settings.scatter_panels[0], originalConfig.settings.scatter_panels[0]);
  assert.deepEqual(changedCopy.settings.scatter_panels[1], originalConfig.settings.scatter_panels[1]);
  assert.equal(changedCopy.settings.scatter_panels[2].colorBinCount, 3);
  await edit(page, 0);
  await page.locator('#remove-scatter').click();
  await update(page, 2);
  const twoPanels = await saved(page);
  assert.deepEqual(twoPanels.settings.scatter_panels, changedCopy.settings.scatter_panels.slice(1));
  assert.equal((await points(page, 0)).x[0], 1, 'The surviving grouped card becomes panel zero');
  pass('Duplicates copy settings independently; removing a panel renumbers cards without changing survivors');

  await page.getByRole('tab', {name: 'Trends', exact: true}).click();
  await ready(page);
  await load(page, twoPanels, 2);
  assert.equal(await page.locator('#chart-type').inputValue(), 'Scatter');
  assert.deepEqual((await saved(page)).settings.scatter_panels, twoPanels.settings.scatter_panels);
  pass('Version 4 configuration round-trip restores the active Scatter view and every panel');

  const legacyV2 = structuredClone(twoPanels);
  legacyV2.version = 2;
  for (const key of ['trend_panels', 'rolling_panels', 'derivative_panels', 'comparison_link_x', 'comparison_link_y', 'distribution_shared_bins']) delete legacyV2.settings[key];
  for (const panel of legacyV2.settings.scatter_panels) {delete panel.colorBinScope; delete panel.markerBinScope;}
  for (const panel of legacyV2.settings.distribution_panels) delete panel.binScope;
  await load(page, legacyV2, 2);
  assert.deepEqual((await saved(page)).settings.scatter_panels, twoPanels.settings.scatter_panels);
  pass('Version 2 canonical scatter panels load with their settings and default filtered bin scopes intact');

  const legacy = {app: 'tsplotter', version: 1, settings: {plot_type: 'Grouped scatter',
    setup_time_mode: 'Relative time', setup_index: 'x', sc2_x: 'signal', sc2_y: 'aux',
    sc2_g1: 'aux', sc2_g2: 'category', sc2_color_bin_mode: 'Number of bins', sc2_color_bin_count: 3,
    sc2_marker_bin_mode: 'Individual values', sc2_palette: 'D3', sc2_marker_preset: 'Crosshair Set', sc2_msize: 9, sc2_mop: .65}};
  await load(page, legacy, 1);
  const migrated = (await saved(page)).settings;
  assert.equal(migrated.plot_type, 'Scatter');
  assert.equal(migrated.scatter_panels.length, 1);
  assert.deepEqual(Object.fromEntries(['x', 'y', 'color', 'marker', 'colorBinMode', 'colorBinCount', 'palette', 'markerPreset', 'size', 'opacity']
    .map(key => [key, migrated.scatter_panels[0][key]])),
  {x: 'signal', y: 'aux', color: 'aux', marker: 'category', colorBinMode: 'Number of bins', colorBinCount: 3, palette: 'D3', markerPreset: 'Crosshair Set', size: 9, opacity: .65});
  assert.equal((await points(page, 0)).x.length, rowCount);
  pass('Legacy grouped-scatter files migrate into one unified panel with markers and bin settings');

  await load(page, twoPanels, 2);
  if (!await page.locator('#filter-panel').evaluate(el => el.open)) await page.locator('#filter-panel > summary').click();
  await page.locator('#add-filter').click();
  await page.selectOption('[data-key="flt2_var_0"]', 'x');
  await page.fill('[data-key="flt2_low_num_0"]', '100');
  await page.fill('[data-key="flt2_up_num_0"]', '199');
  await update(page, 2);
  assert.equal(await page.locator('#row-count').textContent(), '100');
  for (let i = 0; i < 2; i++) assert.deepEqual(await page.locator(`[data-scatter-card="${i}"] [data-row-ledger]`).evaluate(el => JSON.parse(el.dataset.ledger)),
    {source: rowCount, matched: 100, filteredOut: rowCount - 100, invalidTime: 0, usable: 100, excludedValues: 0});
  for (let i = 0; i < 2; i++) {
    const trace = await points(page, i);
    assert.equal(trace.x.length, 100);
    const exported = table(await download(page, `[data-scatter-export="${i}"]`));
    await ready(page, 2);
    assert.equal(exported.length, 100);
    const panel = twoPanels.settings.scatter_panels[i];
    assert.deepEqual(exported.map(row => Number(row[panel.x])), trace.x);
    assert.deepEqual(exported.map(row => Number(row[panel.y])), trace.y);
  }
  pass('Shared filters affect every panel; each card CSV matches its own plotted axes and all 100 filtered rows');

  await page.locator('#clear-filters').click();
  await ready(page, 2);
  assert.equal((await points(page, 0)).x.length, rowCount);
  const busyGuard = await page.locator('#render-button').evaluate(button => {
    button.click();
    return {busy: button.disabled, exportsDisabled: [...document.querySelectorAll('[data-scatter-export], [data-scatter-image]')].every(item => item.disabled)};
  });
  assert.deepEqual(busyGuard, {busy: true, exportsDisabled: true});
  await ready(page, 2);
  pass('Updating all panels disables every card export while analysis is busy');

  for (let i = 2; i < 6; i++) await page.locator('#add-scatter').click();
  assert.equal(await page.locator('[data-scatter-panel]').count(), 6);
  assert.ok(await page.locator('#add-scatter').isDisabled());
  assert.ok(await page.locator('#duplicate-scatter').isDisabled());
  await update(page, 6);
  for (let i = 0; i < 6; i++) assert.equal((await points(page, i)).x.length, rowCount);
  for (let i = 6; i > 1; i--) await page.locator('#remove-scatter').click();
  assert.equal(await page.locator('[data-scatter-panel]').count(), 1);
  assert.ok(await page.locator('#remove-scatter').isDisabled());
  await update(page, 1);
  pass('The six-panel maximum and one-panel minimum are enforced without losing plotted rows');

  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true, acceptDownloads: true});
  observe(mobile);
  await mobile.goto(baseURL, {waitUntil: 'networkidle'});
  await mobile.setInputFiles('#config-input', {name: 'before-dataset.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(twoPanels))});
  await mobile.waitForFunction(() => /Open your dataset/.test(document.querySelector('#status')?.textContent || ''));
  await mobile.setInputFiles('#file-input', file);
  await ready(mobile, 2);
  await noOverflow(mobile);
  for (let i = 0; i < 2; i++) assert.equal((await points(mobile, i)).x.length, rowCount);
  await mobile.screenshot({path: path.join(artifacts, 'mobile-panels.png'), fullPage: true});
  await page.screenshot({path: path.join(artifacts, 'desktop-panels.png'), fullPage: true});
  await mobile.close();
  pass('Settings load before the dataset; independent cards stack at 390px without horizontal overflow or missing points');

  assert.deepEqual(errors, [], 'No uncaught browser errors');
  assert.deepEqual(writes, [], 'Local analysis must not upload data');
  assert.deepEqual(external, [], 'Application assets must remain local');
  console.log(JSON.stringify({result: 'PASS', baseURL, rowCount, checks, artifacts}, null, 2));
} catch (error) {
  await page.screenshot({path: path.join(artifacts, 'failure.png'), fullPage: true}).catch(() => {});
  console.error(JSON.stringify({result: 'FAIL', message: String(error), status: await page.locator('#status').textContent().catch(() => ''), checks, errors, writes, external, artifacts}, null, 2));
  process.exitCode = 1;
} finally {
  await browser.close();
}
