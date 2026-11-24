/** Run against the locally served build, or set BASE_URL to its deployed URL. */
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { chooseChart, chooseSignals, openImportOptions } from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5173/';
const rows = Number(process.env.TEST_ROWS || 35001);
const workdir = await mkdtemp(path.join(tmpdir(), 'plotter-browser-qa-'));
const fixture = path.join(workdir, 'full-points.csv');
await writeFile(fixture, 'x,signal,aux,category\n' + Array.from({ length: rows }, (_, i) => `${i},${2 * i + 1},${i % 101},${i % 2 ? 'odd' : 'even'}`).join('\n'));

const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
page.setDefaultTimeout(90000);
const scriptErrors = [];
const writes = [];
const failedAssets = [];
const thirdPartyRequests = [];
page.on('pageerror', error => scriptErrors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') console.error('Browser console:', message.text()); });
page.on('request', request => {
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) writes.push(`${request.method()} ${request.url()}`);
  if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) thirdPartyRequests.push(request.url());
});
page.on('requestfailed', request => failedAssets.push(`${request.url()}: ${request.failure()?.errorText}`));

async function rendered() {
  await page.waitForFunction(() => {
    const chart = document.querySelector('#chart');
    const status = document.querySelector('#status');
    const render = document.querySelector('#render-button');
    return status?.classList.contains('error') || Array.isArray(chart?.data) && chart.data.length > 0 && !render?.disabled && /ready/i.test(status?.textContent || '');
  }, undefined, { timeout: 120000 });
  assert.ok(!await page.locator('#status').evaluate(el => el.classList.contains('error')), await page.locator('#status').textContent());
}

async function draw(mode) {
  await chooseChart(page, mode);
  await rendered();
  await chooseSignals(page, 'signal');
  await page.click('#render-button');
  await rendered();
}

async function traces() {
  return await page.locator('#chart').evaluate(chart => chart.data.map(trace => ({
    name: trace.name, type: trace.type,
    length: trace.x?.length ?? trace.y?.length ?? 0,
    firstX: trace.x?.[0], lastX: trace.x?.[trace.x.length - 1],
    firstY: trace.y?.[0], lastY: trace.y?.[trace.y.length - 1],
    total: trace.type === 'bar' ? Array.from(trace.y || []).reduce((sum, x) => sum + Number(x), 0) : undefined,
  })));
}

const checks = [];
const pass = message => { checks.push(message); console.log(message); };
try {
  await page.goto(baseURL, { waitUntil: 'networkidle' });
  await page.setInputFiles('#file-input', fixture);
  await page.waitForFunction(count => document.querySelector('#status')?.classList.contains('error') || Number((document.querySelector('#row-count')?.textContent || '').replace(/[^0-9]/g, '')) === count, rows);
  assert.ok(!await page.locator('#status').evaluate(el => el.classList.contains('error')), await page.locator('#status').textContent());
  if (await page.locator('#x-column option[value="x"]').count()) await chooseSignals(page, ['x'], '#x-column');
  await chooseSignals(page, ['signal']);

  assert.equal(await page.getByRole('tab').count(), 5);
  assert.equal(await page.getByRole('tab', {name: 'Grouped scatter', exact: true}).count(), 0);
  for (const mode of ['Trends', 'Distribution', 'Scatter', 'Rolling window', 'Derivatives']) {
    await draw(mode);
    const data = await traces();
    if (mode === 'Distribution') {
      assert.ok(data.some(trace => trace.type === 'bar' && trace.total === rows), `${mode}: histogram counts must cover every row (${JSON.stringify(data)})`);
    } else {
      assert.ok(data.some(trace => trace.length === rows), `${mode}: every usable point must be retained (${JSON.stringify(data)})`);
      assert.ok(data.some(trace => trace.firstY === 1 && trace.lastY === 2 * (rows - 1) + 1), `${mode}: full original signal endpoints must survive`);
    }
    pass(`${mode}: full ${rows.toLocaleString()} rows represented`);
  }

  // Configuration round trip must restore the view, never embed dataset rows.
  await draw('Scatter');
  const downloadEvent = page.waitForEvent('download');
  await page.click('#save-config');
  const download = await downloadEvent;
  const saved = await readFile(await download.path(), 'utf8');
  const config = JSON.parse(saved);
  assert.equal(config.app, 'tsplotter');
  assert.equal(config.version, 4);
  assert.equal(config.settings.plot_type, 'Scatter');
  assert.equal(config.settings.scatter_panels.length, 1);
  assert.equal(config.settings.scatter_panels[0].y, 'signal');
  for (const key of ['trend_panels', 'distribution_panels', 'rolling_panels', 'derivative_panels']) {
    assert.equal(config.settings[key].length, 1, `${key} is saved independently`);
  }
  assert.deepEqual(config.settings.trend_panels[0].subplots, [{columns: ['signal']}]);
  assert.equal(Object.hasOwn(config.settings.trend_panels[0], 'columns'), false);
  assert.equal(config.settings.rolling_panels[0].y, 'signal');
  assert.equal(config.settings.derivative_panels[0].y, 'signal');
  assert.ok(!saved.includes('full-points.csv'), 'Configuration must not include dataset source names');
  assert.ok(saved.length < 256 * 1024, 'Configuration must contain settings rather than dataset rows');
  await draw('Trends');
  await page.setInputFiles('#config-input', { name: 'round-trip.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  if (await page.locator('#load-config').count()) await page.click('#load-config');
  await page.waitForFunction(() => document.querySelector('#chart-type')?.value === 'Scatter');
  await page.click('#render-button');
  await rendered();
  assert.ok((await traces()).some(trace => trace.length === rows), 'Config restore retains all points');
  pass('Configuration save/load restores the chart without serializing data');

  // Filters, preview and exports must share the exact same complete row set.
  await page.locator('#filter-panel > summary').click();
  await page.click('#add-filter');
  await chooseSignals(page, ['x'], '[data-key="flt2_var_0"]');
  await page.fill('[data-key="flt2_low_num_0"]', '100');
  await page.fill('[data-key="flt2_up_num_0"]', '199');
  await page.click('#render-button');
  await rendered();
  assert.equal(await page.locator('#row-count').textContent(), '100');
  assert.ok((await traces()).some(trace => trace.length === 100));
  await page.locator('#preview-panel > summary').click();
  await page.click('#preview-button');
  await rendered();
  assert.equal(await page.locator('#preview-table tbody tr').count(), 100);
  const filteredDownload = page.waitForEvent('download');
  await page.click('#data-export');
  const filteredCSV = await readFile(await (await filteredDownload).path(), 'utf8');
  assert.equal(filteredCSV.trim().split(/\r?\n/).length, 101);
  assert.match(filteredCSV, /100,201/);
  assert.match(filteredCSV, /199,399/);
  await rendered();
  pass('Inclusive filters, preview and full CSV export contain the same 100 rows');

  await page.click('#clear-filters');
  await rendered();
  assert.equal(Number((await page.locator('#row-count').textContent()).replaceAll(',', '')), rows);
  await page.click('#add-filter');
  await chooseSignals(page, ['x'], '[data-key="flt2_var_0"]');
  await page.fill('[data-key="flt2_low_num_0"]', String(rows + 100));
  await page.click('#render-button');
  await page.waitForFunction(() => document.querySelector('#row-count')?.textContent === '0' && !document.querySelector('#render-button')?.disabled);
  assert.equal(await page.locator('#chart').evaluate(el => el.data?.length || 0), 0, 'Zero-match filters must clear the previous chart');
  await page.click('#clear-filters');
  await rendered();
  pass('Zero-match filters clear stale charts and recover after clearing filters');
  await draw('Scatter');
  await chooseSignals(page, ['aux'], '[data-panel-key="color"]');
  await chooseSignals(page, ['aux'], '[data-panel-key="marker"]');
  await page.selectOption('[data-panel-key="colorMode"]', 'discrete');
  await page.selectOption('[data-panel-key="colorBinMode"]', 'Number of bins');
  await page.fill('[data-panel-key="colorBinCount"]', '5');
  await page.selectOption('[data-panel-key="markerBinMode"]', 'Bin width');
  await page.fill('[data-panel-key="markerBinWidth"]', '25');
  await page.click('#render-button');
  await rendered();
  const groupedRows = (await traces()).filter(trace => trace.type === 'scattergl').reduce((total, trace) => total + trace.length, 0);
  assert.equal(groupedRows, rows, 'Independent color/marker bins retain all rows');
  pass('Unified Scatter retains all points with independent numeric color/marker bins on the same source');

  // A separately generated Parquet fixture verifies the actual browser decoder.
  if (process.env.PARQUET_FIXTURE) {
    await page.setInputFiles('#file-input', process.env.PARQUET_FIXTURE);
    await rendered();
    assert.equal(await page.locator('#preview-table tbody tr').count(), 0, 'New imports must clear stale previews');
    await page.waitForFunction(count => Number((document.querySelector('#row-count')?.textContent || '').replace(/[^0-9]/g, '')) === count, rows);
    if (await page.locator('#x-column option[value="x"]').count()) await chooseSignals(page, ['x'], '#x-column');
    await chooseSignals(page, ['signal']);
    await draw('Trends');
    assert.ok((await traces()).some(trace => trace.length === rows), 'Parquet retains every decoded row');
    pass(`Parquet: all ${rows.toLocaleString()} rows decoded and plotted in browser`);
  }

  if (process.env.EXTRA_FORMATS !== '0') {
    const XLSX = await import('xlsx');
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([['ignore'], [999]]), 'First');
    XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([['x', 'signal'], [0, 1], [1, 3], [2, 5], [3, 7]]), 'Data');
    const excel = path.join(workdir, 'local-sheet.xlsx');
    await writeFile(excel, XLSX.write(workbook, { bookType: 'xlsx', type: 'buffer' }));
    await openImportOptions(page);
    await page.fill('#sheet-name', 'Data');
    await page.setInputFiles('#file-input', excel);
    await rendered();
    await draw('Trends');
    assert.equal(await page.locator('#row-count').textContent(), '4');
    assert.ok((await traces()).some(trace => trace.firstY === 1 && trace.lastY === 7));
    pass('Actual Excel workbook imports the explicitly selected local sheet');

    const h5 = await import('h5wasm/node');
    await h5.ready;
    const mat = path.join(workdir, 'local-numeric.mat');
    const hdf = new h5.File(mat, 'w');
    hdf.create_dataset({ name: 'x', data: new Float64Array([0, 1, 2, 3]) });
    hdf.create_dataset({ name: 'signal', data: new Float64Array([1, 3, 5, 7]) });
    hdf.close();
    await page.setInputFiles('#file-input', mat);
    await rendered();
    await draw('Trends');
    assert.equal(await page.locator('#row-count').textContent(), '4');
    assert.ok((await traces()).some(trace => trace.firstY === 1 && trace.lastY === 7));
    pass('Actual HDF5 numeric MAT vectors import locally through the browser WASM reader');
  }

  const datetimeCSV = 'date,signal,category\n01/02/2026 00:00:00,1,Day\n02/02/2026 00:00:00,3,Night\n03/02/2026 00:00:00,5,Day\nbad,7,Day\n';
  await page.setInputFiles('#file-input', { name: 'dmy.csv', mimeType: 'text/csv', buffer: Buffer.from(datetimeCSV) });
  await rendered();
  await page.selectOption('[data-key="setup_time_mode"]', 'Absolute time');
  await chooseSignals(page, ['date'], '#x-column');
  await page.locator('[data-key="setup_profile"]').evaluate(el => el.closest('details').open = true);
  await page.selectOption('[data-key="setup_profile"]', 'DMY (dd/mm/yyyy)');
  await chooseSignals(page, 'signal');
  await page.click('#render-button');
  await rendered();
  assert.equal(await page.locator('#row-count').textContent(), '4', 'Matched rows include the row with an unusable timestamp');
  assert.equal(await page.locator('#filtered-count').textContent(), '0');
  assert.equal(await page.locator('#invalid-time-count').textContent(), '1');
  assert.deepEqual(await page.locator('#primary-plot [data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger)),
    {source: 4, matched: 4, filteredOut: 0, invalidTime: 1, usable: 3, excludedValues: 0});
  assert.ok((await traces()).some(trace => trace.length === 3), 'All three valid timestamp rows are plotted');
  if (!await page.locator('#filter-panel').evaluate(el => el.open)) await page.locator('#filter-panel > summary').click();
  await page.click('#add-filter');
  await chooseSignals(page, ['(time index)'], '[data-key="flt2_var_0"]');
  await page.fill('[data-date-bound="low"][data-filter-index="0"]', '2026-02-02T00:00:00');
  await page.fill('[data-date-bound="up"][data-filter-index="0"]', '2026-02-03T00:00:00');
  await page.click('#render-button');
  await rendered();
  assert.equal(await page.locator('#row-count').textContent(), '2');
  assert.ok((await traces()).some(trace => trace.firstY === 3 && trace.lastY === 5));
  if (!await page.locator('#preview-panel').evaluate(el => el.open)) await page.locator('#preview-panel > summary').click();
  const dateDownload = page.waitForEvent('download');
  await page.click('#data-export');
  const dateCSV = await readFile(await (await dateDownload).path(), 'utf8');
  await rendered();
  assert.equal(dateCSV.trim().split(/\r?\n/).length, 3, 'Time-filter export contains the same two valid rows');
  await page.locator('#summary-button').evaluate(el => el.closest('details').open = true);
  await page.click('#summary-button');
  await rendered();
  const summary = await page.locator('#summary-table tbody tr').first().locator('td').allTextContents();
  assert.equal(summary[1], '2', 'Summary uses the same prepared, time-filtered rows');
  pass('DMY parsing, invalid timestamps, inclusive time-index filters, full CSV and summary use the same rows');

  const screenshot = path.join(workdir, 'browser-verified.png');
  await page.screenshot({ path: screenshot, fullPage: true });

  assert.deepEqual(writes, [], 'Dataset processing must not send HTTP writes/uploads');
  assert.deepEqual(thirdPartyRequests, [], 'All application dependencies must be served locally');
  assert.deepEqual(scriptErrors, [], 'No uncaught browser exceptions');
  assert.deepEqual(failedAssets, [], 'All application assets must load successfully');
  pass('No HTTP upload/write requests, failed assets or uncaught browser errors');
  console.log(JSON.stringify({ baseURL, rows, checks, screenshot, result: 'PASS' }, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(workdir, 'failure.png'), fullPage: true }).catch(() => {});
  console.error(JSON.stringify({ result: 'FAIL', message: String(error), checks, scriptErrors, writes, failedAssets, artifactDirectory: workdir }, null, 2));
  process.exitCode = 1;
} finally {
  await browser.close();
}
