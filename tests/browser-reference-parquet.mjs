/** Profile and verify the user-supplied Parquet through the actual browser app. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp, stat, writeFile, readFile, open} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createInterface} from 'node:readline';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {chooseSignals} from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:5173/';
const fixture = path.resolve(process.env.REFERENCE_PARQUET || 'tests/MasterLogD4_2_decoded_registers.parquet');
const phase = process.env.REFERENCE_PHASE || 'full';
assert.ok(['baseline', 'full'].includes(phase), 'REFERENCE_PHASE must be baseline or full');
const artifacts = await mkdtemp(path.join(tmpdir(), 'tsplotter-reference-parquet-'));
const reference = {rows: 315034, columns: 234, numericColumns: 233, bytes: (await stat(fixture)).size};
async function optionalProfile(filename, explicit) {
  try { return JSON.parse(await readFile(filename, 'utf8')); }
  catch (error) { if (explicit || error.code !== 'ENOENT') throw error; return undefined; }
}
const nativePath = process.env.REFERENCE_NATIVE_PROFILE || process.env.REFERENCE_NATIVE;
const native = phase === 'full' ? await optionalProfile(nativePath || 'artifacts/reference-parquet-native.json', Boolean(nativePath)) : undefined;
let profile = phase === 'full' ? await optionalProfile(process.env.REFERENCE_PROFILE || 'artifacts/reference-parquet-profile.json', Boolean(process.env.REFERENCE_PROFILE)) : undefined;
const report = {phase, baseURL, reference, stages: [], pageErrors: [], workerErrors: [], consoleErrors: [], graphicsWarnings: [], writes: [], external: [], failedAssets: [], crashes: [], heap: {samples: [], peakUsedBytes: 0, peakTotalBytes: 0}, artifacts};
const browser = await chromium.launch({headless: true,
  ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
page.setDefaultTimeout(125000);
page.on('pageerror', error => report.pageErrors.push(String(error).slice(0, 1000)));
page.on('crash', () => report.crashes.push('Main renderer crashed'));
page.on('console', message => {
  const text = message.text().slice(0, 1600);
  if (message.type() === 'error') report.consoleErrors.push(text);
  if (/(?:webgl.*(?:lost|too many|limit)|too many.*(?:webgl|context)|context.*lost)/i.test(text)) report.graphicsWarnings.push(text);
});
page.on('request', request => {
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) report.writes.push(`${request.method()} ${request.url()}`);
  if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) report.external.push(request.url());
});
page.on('requestfailed', request => report.failedAssets.push(`${request.url()}: ${request.failure()?.errorText}`));
await page.addInitScript(() => {
  window.__referenceCalls = [];
  window.__referenceWorkerErrors = [];
  window.__referenceSummary = undefined;
  window.__referenceGroups = {};
  window.__referenceGL = new WeakMap();
  window.__referenceGLErrors = [];
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, ...args) {
    const context = getContext.call(this, type, ...args);
    if (context && String(type).includes('webgl') && !window.__referenceGL.has(this)) {
      window.__referenceGL.set(this, context);
      this.addEventListener('webglcontextlost', () => { if (this.isConnected) window.__referenceGLErrors.push('A connected plot canvas lost its WebGL context'); });
    }
    return context;
  };
  const original = Worker.prototype.postMessage;
  const observed = new WeakSet(), pending = new Map();
  Worker.prototype.postMessage = function(message, ...args) {
    if (!observed.has(this)) {
      observed.add(this);
      this.addEventListener('error', event => window.__referenceWorkerErrors.push(String(event.message || 'Worker error').slice(0, 1000)));
      this.addEventListener('message', ({data}) => {
        const call = pending.get(data?.id);
        if (!call) return;
        pending.delete(data.id);
        const result = data.result;
        if (call.method === 'summary' && result?.columns) window.__referenceSummary = result.columns;
        if (call.groupKey && result?.meta?.groups) window.__referenceGroups[call.groupKey] = Object.fromEntries(Object.entries(result.meta.groups).map(([encoding, group]) =>
          [encoding, {scope: group.scope, edges: group.edges, labels: group.labels, counts: group.counts}]));
        window.__referenceCalls.push({...call, elapsedMs: performance.now() - call.started,
          rowCount: result?.rowCount, totalRows: result?.totalRows,
          columnCount: Array.isArray(result?.columns) ? result.columns.length : result?.columns ? Object.keys(result.columns).length : undefined,
          ledger: result?.meta?.rows, sampling: result?.meta?.sampling,
          error: data.error ? String(data.error.message).slice(0, 1000) : undefined});
      });
    }
    if (message?.method && message?.id) {
      const spec = message.method === 'query' ? message.args?.[0] : undefined;
      pending.set(message.id, {id: message.id, method: message.method, mode: spec?.mode,
        groupKey: spec?.mode === 'scatter' ? JSON.stringify([spec.x, spec.y, spec.color, spec.marker]) : undefined,
        started: performance.now()});
    }
    return original.call(this, message, ...args);
  };
});
const cdp = await page.context().newCDPSession(page);
await cdp.send('Performance.enable');
let sampling = false;
async function heap(label) {
  if (sampling || page.isClosed()) return;
  sampling = true;
  try {
    const usage = await cdp.send('Runtime.getHeapUsage');
    const sample = {label, elapsedMs: performance.now(), usedBytes: usage.usedSize, totalBytes: usage.totalSize};
    report.heap.samples.push(sample);
    report.heap.peakUsedBytes = Math.max(report.heap.peakUsedBytes, sample.usedBytes);
    report.heap.peakTotalBytes = Math.max(report.heap.peakTotalBytes, sample.totalBytes);
  } catch { /* A crashed renderer is reported by the stage and page events. */ }
  finally { sampling = false; }
}
const sampler = setInterval(() => { void heap('during'); }, 1000);
async function idle(timeout = 120000) {
  await page.waitForFunction(() => !document.querySelector('#render-button')?.disabled &&
    (document.querySelector('#status')?.classList.contains('error') || /ready|settings loaded/i.test(document.querySelector('#status')?.textContent || '')), undefined, {timeout});
  const result = await page.locator('#status').evaluate(element => ({ok: !element.classList.contains('error'), status: element.textContent}));
  return result;
}
async function stage(name, action, {allowError = false} = {}) {
  const started = performance.now();
  await heap(`${name}:before`);
  const result = {name};
  try {
    Object.assign(result, await action());
    result.ok ??= true;
  } catch (error) {
    result.ok = false; result.error = String(error).slice(0, 1600);
    result.trace = String(error.stack || '').split('\n').filter(line => line.includes('browser-reference-parquet.mjs')).slice(0, 4);
  }
  result.elapsedMs = performance.now() - started;
  await heap(`${name}:after`);
  report.stages.push(result);
  console.log(JSON.stringify(result));
  if (!result.ok && !allowError) throw new Error(`${name} failed: ${result.error || result.status}`);
  return result;
}
async function openDetails(selector) {
  const details = page.locator(selector);
  if (!await details.evaluate(element => element.open)) await details.locator('summary').first().click();
}
const slugs = {Trends: 'trend', Distribution: 'distribution', Scatter: 'scatter', 'Rolling window': 'rolling', Derivatives: 'derivative'};
// Chrome's datetime-local accepts millisecond fractions; the native profile
// serializes those same instants with three additional, insignificant zeros.
const localDateTime = value => String(value).replace(' ', 'T').replace(/(?:Z|[+-]\d{2}:\d{2})$/, '').replace(/(\.\d{3})0+$/, '$1');
async function selectView(name, panels = 1) {
  await page.getByRole('tab', {name, exact: true}).click();
  const state = await idle(); assert.ok(state.ok, 'Changing views must complete without an engine error');
  assert.equal(await page.locator(`[data-${slugs[name]}-card]`).count(), panels);
}
async function update(name, panels = 1) {
  await page.locator('#render-button').click();
  const state = await idle(); assert.ok(state.ok, `${name} update must complete without an engine error`);
  assert.equal(await page.locator(`[data-${slugs[name]}-card]`).count(), panels);
}
async function panelField(key, value) {
  const field = page.locator(`[data-panel-key="${key}"]`);
  if (!await field.isVisible()) {
    const details = field.locator('xpath=ancestor::details[1]');
    if (!await details.evaluate(element => element.open)) await details.locator('summary').first().click();
  }
  if (await field.evaluate(element => element.tagName === 'SELECT')) await field.selectOption(value);
  else if (await field.evaluate(element => element.type === 'checkbox')) await field.setChecked(Boolean(value));
  else { await field.fill(String(value)); await field.press('Tab'); }
}
async function editPanel(slug, index) { await page.locator(`[data-${slug}-panel="${index}"]`).click(); }
async function checkbox(selector, value) {
  const field = page.locator(selector);
  if (!await field.isVisible()) {
    const details = field.locator('xpath=ancestor::details[1]');
    if (!await details.evaluate(element => element.open)) await details.locator('summary').first().click();
  }
  await field.setChecked(value);
}
async function grabDownload(selector) {
  const pending = page.waitForEvent('download', {timeout: 180000});
  await page.locator(selector).click();
  const result = await pending;
  await result.path();
  return result;
}
const splitCSV = line => {
  if (!line.includes('"')) return line.split(',');
  const values = []; let value = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '"') {
      if (quoted && line[i + 1] === '"') { value += '"'; i++; } else quoted = !quoted;
    } else if (line[i] === ',' && !quoted) { values.push(value); value = ''; }
    else value += line[i];
  }
  values.push(value); return values;
};
async function scanCSV(download, {expectedRows, expectedColumns, expectedNames, expectedStats, statColumns, onStatistics, recordColumns = [], onRecord} = {}) {
  const filename = await download.path();
  const lines = createInterface({input: createReadStream(filename), crlfDelay: Infinity});
  let header, rowCount = 0, first, last;
  const summaries = new Map((statColumns || (expectedStats || []).map(item => item.column)).map(column => [column, {count: 0, min: Infinity, max: -Infinity, mean: 0, m2: 0}]));
  for await (const line of lines) {
    if (!header) { header = splitCSV(line.replace(/^\uFEFF/, '')); continue; }
    rowCount++; first ??= line; last = line;
    if (summaries.size || onRecord) {
      const fields = splitCSV(line);
      for (const [column, summary] of summaries) {
        const raw = fields[header.indexOf(column)], value = raw === '' ? NaN : Number(raw);
        if (!Number.isFinite(value)) continue;
        summary.count++; summary.min = Math.min(summary.min, value); summary.max = Math.max(summary.max, value);
        const delta = value - summary.mean; summary.mean += delta / summary.count; summary.m2 += delta * (value - summary.mean);
      }
      onRecord?.(Object.fromEntries(recordColumns.map(column => {
        const raw = fields[header.indexOf(column)]; return [column, raw === '' ? null : column === 'timestamp' ? Date.parse(raw) : Number(raw)];
      })));
    }
  }
  assert.ok(header?.length, 'CSV must have a header');
  if (expectedRows !== undefined) assert.equal(rowCount, expectedRows);
  if (expectedColumns !== undefined) assert.equal(header.length, expectedColumns);
  if (expectedNames) assert.ok(JSON.stringify([...header].sort()) === JSON.stringify([...expectedNames].sort()), 'CSV header must contain every expected source column exactly once');
  if (first !== undefined) assert.equal(splitCSV(first).length, header.length, 'First CSV row must contain every exported column');
  if (last !== undefined) assert.equal(splitCSV(last).length, header.length, 'Last CSV row must contain every exported column');
  for (const item of expectedStats || []) {
    const actual = summaries.get(item.column);
    assert.equal(actual.count, item.valid, `CSV valid-count mismatch for ${item.column}`);
    for (const key of ['min', 'max', 'mean']) if (item[key] !== null && item.valid) {
      assert.ok(Math.abs(actual[key] - item[key]) <= 1e-9 * Math.max(1, Math.abs(item[key])), `CSV native ${key} mismatch for ${item.column}`);
    }
  }
  const bytes = (await stat(filename)).size;
  onStatistics?.([...summaries].map(([column, item]) => ({column, valid: item.count, min: item.count ? item.min : null, max: item.count ? item.max : null,
    mean: item.count ? item.mean : null, stddev: item.count > 1 ? Math.sqrt(item.m2 / (item.count - 1)) : null})));
  await download.delete();
  return {rows: rowCount, columns: header.length, bytes};
}
async function assertNativeSummary(expected) {
  const problems = await page.evaluate(items => {
    const data = window.__referenceSummary, issues = [];
    if (!data?.column) return ['No completed summary reply'];
    const fields = {valid: data.valid ? 'valid' : 'count', min: 'min', max: 'max', mean: 'mean', stddev: data.stddev ? 'stddev' : 'std'};
    for (const item of items) {
      const index = data.column.indexOf(item.column);
      if (index < 0) { issues.push(`${item.column}: missing`); continue; }
      for (const [nativeKey, actualKey] of Object.entries(fields)) {
        const actual = data[actualKey]?.[index], expected = item[nativeKey];
        if (expected == null) { if (actual != null) issues.push(`${item.column}: ${nativeKey}`); }
        else if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - expected) > 1e-8 * Math.max(1, Math.abs(expected))) issues.push(`${item.column}: ${nativeKey}`);
      }
    }
    return issues;
  }, expected);
  assert.ok(problems.length === 0, `Native summary mismatch: ${problems.join('; ')}`);
}
async function traceMetrics(slug, index = 0) {
  const selector = index === 0 ? '#chart' : `#${slug}-chart-${index}`;
  return await page.locator(selector).evaluate(graph => ({
    traces: (graph.data || []).filter(trace => trace.x?.some(value => value !== null)).map(trace => ({name: trace.name, type: trace.type,
      length: trace.x?.length ?? trace.y?.length ?? 0, finite: Array.from(trace.y || []).reduce((sum, value) => sum + Number(Number.isFinite(value)), 0),
      histogramTotal: trace.type === 'bar' ? Array.from(trace.y || []).reduce((sum, value) => sum + Number(value), 0) : undefined,
    })),
    ledger: graph.closest('[data-panel-kind]')?.querySelector('[data-row-ledger]')?.dataset.ledger ? JSON.parse(graph.closest('[data-panel-kind]').querySelector('[data-row-ledger]').dataset.ledger) : undefined,
  }));
}
async function savedConfig() {
  const download = await grabDownload('#save-config');
  const text = await readFile(await download.path(), 'utf8'); await download.delete();
  const config = JSON.parse(text);
  assert.equal(config.app, 'tsplotter'); assert.equal(config.version, 3);
  assert.ok(!text.includes(path.basename(fixture)));
  return config;
}
async function graphicsHealth(slug) {
  const states = await page.locator(`[data-${slug}-card] .plot-canvas`).evaluateAll(graphs => graphs.map(graph => {
    const contexts = [...graph.querySelectorAll('canvas')].map(canvas => window.__referenceGL.get(canvas)).filter(Boolean);
    return {webglRequired: (graph.data || []).some(trace => trace.type === 'scattergl' && trace.x?.length), contexts: contexts.length,
      lost: contexts.some(context => context.isContextLost()), sized: contexts.every(context => context.drawingBufferWidth > 0 && context.drawingBufferHeight > 0)};
  }));
  for (const state of states) {
    assert.ok(!state.lost && state.sized, 'Every rendered plot must retain a live, nonzero WebGL drawing buffer');
    if (state.webglRequired) assert.ok(state.contexts > 0, 'A WebGL trace must have an actual drawing context');
  }
  return {cards: states.length, liveWebGLContexts: states.reduce((sum, item) => sum + item.contexts, 0)};
}
async function pngCheck(slug) {
  const download = await grabDownload(`[data-${slug}-image="0"]`);
  const handle = await open(await download.path(), 'r'); const signature = Buffer.alloc(8); await handle.read(signature, 0, 8, 0); await handle.close();
  assert.deepEqual([...signature], [137,80,78,71,13,10,26,10]);
  const bytes = (await stat(await download.path())).size; assert.ok(bytes > 1000);
  await download.saveAs(path.join(artifacts, `${slug}-reference.png`)); await download.delete();
  await graphicsHealth(slug); return bytes;
}
async function loadConfig(config) {
  await page.setInputFiles('#config-input', {name: 'reference-settings.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config))});
  const state = await idle(); assert.ok(state.ok, 'Saved configuration must restore successfully');
}
async function setTimeFilter() {
  await openDetails('#filter-panel'); await page.locator('#add-filter').click();
  await page.selectOption('[data-key="flt2_var_0"]', '(time index)');
  await page.fill('[data-date-bound="low"][data-filter-index="0"]', localDateTime(profile.time.filterLow));
  await page.fill('[data-date-bound="up"][data-filter-index="0"]', localDateTime(profile.time.filterHigh));
  assert.ok(await page.locator('[data-date-bound="low"][data-filter-index="0"]').inputValue());
  assert.ok(await page.locator('[data-date-bound="up"][data-filter-index="0"]').inputValue());
}
async function runFullWorkflow() {
  const runtimeStatistics = await page.evaluate(() => {
    const data = window.__referenceSummary;
    return data.column.map((column, i) => ({column, valid: Number((data.valid || data.count)[i]), min: data.min[i], max: data.max[i], mean: data.mean[i], stddev: (data.stddev || data.std)?.[i]}));
  });
  const fullStatistics = native?.statistics || runtimeStatistics;
  const sourceNames = native?.schema?.map(column => column.name) || await page.locator('#x-column option').evaluateAll(options => options.map(option => option.value).filter(value => !value.startsWith('(')));
  if (!profile) {
    const time = await page.locator('#chart').evaluate(graph => {
      const values = graph.data[0].x, numeric = value => typeof value === 'number' ? value : Date.parse(value);
      const low = numeric(values[100]), high = numeric(values[1099]);
      let filteredRows = 0; for (const value of values) { const numeric = typeof value === 'number' ? value : Date.parse(value); if (numeric >= low && numeric <= high) filteredRows++; }
      return {filterLow: new Date(low).toISOString(), filterHigh: new Date(high).toISOString(), max: new Date(numeric(values.at(-1))).toISOString(), filteredRows};
    });
    profile = {signals: ['R2045_Inlet_Water_Temp','R2046_Outlet_Water_Temp','R2048_Ambient_Temp_AT','R2012_Unit_mode','R1011_ON_OFF'],
      partialColumn: 'R1159_Cooling_Target_Temp', timeColumn: 'timestamp', time};
  }
  report.nativeReference = Boolean(native); report.privateTimeReference = Boolean(process.env.REFERENCE_PROFILE);
  const [signal, second, ambient, group, constant] = profile.signals;
  const partial = profile.partialColumn;
  const statsByColumn = new Map(fullStatistics.map(item => [item.column, item]));
  for (const name of [signal, second, ambient, group, constant, partial]) assert.ok(statsByColumn.has(name), 'The private profile must cover every selected reference signal');
  await stage('Verify full summary and search/pagination', async () => {
    if (native) await assertNativeSummary(native.statistics);
    for (const item of runtimeStatistics) {
      assert.ok(item.valid >= 0 && item.valid <= reference.rows, 'Summary validity counts must be bounded by source rows');
      if (item.valid) assert.ok(Number.isFinite(item.mean) && item.min <= item.mean + 1e-8 && item.mean <= item.max + 1e-8, 'Summary values must satisfy numeric invariants');
    }
    assert.equal(await page.locator('#summary-table tbody tr').count(), 100);
    await page.locator('#summary-next').click(); assert.equal(await page.locator('#summary-table tbody tr').count(), 100);
    await page.locator('#summary-next').click(); assert.equal(await page.locator('#summary-table tbody tr').count(), 33);
    assert.ok(await page.locator('#summary-next').isDisabled());
    const calls = await page.evaluate(() => window.__referenceCalls.length);
    await page.locator('#summary-search').fill(fullStatistics.at(-1).column);
    assert.equal(await page.locator('#summary-table tbody tr').count(), 1);
    assert.equal(await page.evaluate(() => window.__referenceCalls.length), calls, 'Searching summary pages must not rerun the statistics');
    await page.locator('#summary-search').fill('');
    await page.locator('#summary-button').click(); assert.ok((await idle()).ok);
    assert.equal(await page.evaluate(() => window.__referenceCalls.length), calls, 'Reopening an unchanged summary must reuse its completed calculation');
    return {verifiedSignals: fullStatistics.length, independentNativeReference: Boolean(native), pageSizes: [100, 100, 33]};
  });
  await stage('Full dataset CSV, streamed validation', async () => {
    const download = await grabDownload('#data-export');
    const result = await scanCSV(download, {expectedRows: reference.rows, expectedColumns: reference.columns, expectedNames: sourceNames});
    const state = await idle(180000); assert.ok(state.ok, 'The complete CSV export must finish successfully');
    return result;
  });

  await stage('Pinned varied, constant and partial-null Trends', async () => {
    await chooseSignals(page, [signal, second, constant, partial]); await update('Trends');
    const metrics = await traceMetrics('trend');
    for (const name of [signal, second, constant, partial]) {
      const trace = metrics.traces.find(item => item.name === name);
      assert.ok(trace, 'Every selected trend must be represented');
      assert.equal(trace.length, reference.rows);
      assert.equal(trace.finite, statsByColumn.get(name).valid, `Native valid-count mismatch for ${name}`);
    }
    return {selectedSignals: metrics.traces.length, allPointsPerSeries: reference.rows,
      finiteCounts: metrics.traces.map(item => item.finite)};
  });
  await stage('Full plotted Trend CSV, streamed validation', async () => {
    return await scanCSV(await grabDownload('[data-trend-export="0"]'), {expectedRows: reference.rows, expectedColumns: 5, expectedNames: [profile.timeColumn, signal, second, constant, partial]});
  });
  await stage('Full Trend PNG and graphics context', async () => ({pngBytes: await pngCheck('trend')}));
  const expectedScatterRows = await page.locator('#chart').evaluate((graph, names) => {
    const [a, b] = names.map(name => graph.data.find(trace => trace.name === name).y);
    let count = 0; for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) count++;
    return count;
  }, [signal, second]);
  if (profile.jointCounts?.[`${signal}|${second}`] !== undefined) assert.equal(expectedScatterRows, profile.jointCounts[`${signal}|${second}`]);
  await stage('Two full Trends and repeated render', async () => {
    await page.locator('#add-trend').click(); await chooseSignals(page, [ambient]); await update('Trends', 2);
    const other = await traceMetrics('trend', 1);
    assert.equal(other.traces[0].length, reference.rows); assert.equal(other.traces[0].finite, statsByColumn.get(ambient).valid);
    await update('Trends', 2);
    return {panels: 2, sourceRows: reference.rows};
  });
  await stage('Full unfiltered WebGL step and linear rerender', async () => {
    await editPanel('trend', 0); await chooseSignals(page, [signal]); await panelField('shape', 'hv'); await update('Trends', 2);
    const step = await traceMetrics('trend'); assert.equal(step.traces[0].type, 'scattergl'); assert.equal(step.traces[0].length, reference.rows);
    await panelField('shape', 'linear'); await update('Trends', 2);
    assert.equal((await traceMetrics('trend')).traces[0].length, reference.rows);
    return {webglStepPoints: reference.rows, linearPoints: reference.rows};
  });
  await stage('Independent full-data Distributions and shared bins', async () => {
    await selectView('Distribution'); await chooseSignals(page, [signal]);
    await panelField('binMode', 'Target #bins'); await panelField('binCount', 30); await panelField('binScope', 'Full dataset');
    await page.locator('#add-distribution').click(); await chooseSignals(page, [constant]);
    await panelField('binMode', 'Auto (width)'); await update('Distribution', 2);
    assert.equal((await traceMetrics('distribution', 1)).traces[0].histogramTotal, statsByColumn.get(constant).valid);
    await chooseSignals(page, [second]); await panelField('binMode', 'Target #bins'); await panelField('binCount', 30);
    await panelField('binScope', 'Filtered data'); await checkbox('#distribution-shared-bins', true); await update('Distribution', 2);
    for (const [index, name] of [[0, signal], [1, second]]) assert.equal((await traceMetrics('distribution', index)).traces[0].histogramTotal, statsByColumn.get(name).valid);
    const sameEdges = await page.evaluate(() => JSON.stringify(document.querySelector('#chart').data[0].customdata) === JSON.stringify(document.querySelector('#distribution-chart-1').data[0].customdata));
    assert.ok(sameEdges, 'Shared histogram bins must align on the reference data');
    return {panels: 2, histogramTotals: [statsByColumn.get(signal).valid, statsByColumn.get(second).valid], pngBytes: await pngCheck('distribution')};
  });
  await stage('Full Scatter points, groups and legend controls', async () => {
    await selectView('Scatter'); await panelField('x', signal); await panelField('y', second);
    await panelField('color', group); await panelField('colorMode', 'discrete');
    await panelField('marker', constant); await update('Scatter');
    const metric = await traceMetrics('scatter'); assert.equal(metric.traces[0].length, expectedScatterRows);
    const calls = await page.evaluate(() => window.__referenceCalls.length);
    const guideName = await page.locator('#chart').evaluate(graph => graph.data.find(trace => trace.meta?.scatterGuide).name);
    const guide = page.locator('#chart .legend').getByText(guideName, {exact: true}).first().locator('..').locator('.legendtoggle');
    await guide.click();
    await page.waitForFunction(before => document.querySelector('#chart').data[0].x.length < before, expectedScatterRows);
    await page.waitForTimeout(350); // Separate single clicks from Plotly's double-click gesture.
    await guide.click();
    await page.waitForFunction(before => document.querySelector('#chart').data[0].x.length === before, expectedScatterRows);
    assert.equal(await page.evaluate(() => window.__referenceCalls.length), calls, 'Legend controls must not run another data query');
    await panelField('color', signal); await panelField('colorMode', 'continuous'); await update('Scatter');
    assert.ok(await page.locator('#chart').evaluate(graph => graph.data[0].marker.color.every(Number.isFinite) && graph.data[0].marker.showscale));
    await panelField('colorBinMode', 'Number of bins'); await panelField('colorBinCount', 4); await panelField('colorBinScope', 'Full dataset');
    await panelField('marker', second); await panelField('markerBinMode', 'Bin width'); await panelField('markerBinWidth', (statsByColumn.get(second).max - statsByColumn.get(second).min) / 4); await panelField('markerBinScope', 'Filtered data');
    await panelField('palette', 'D3'); await panelField('markerPreset', 'Crosshair Set'); await panelField('customSymbols', 'circle,square,diamond,x');
    await page.locator('#add-scatter').click(); await panelField('x', signal); await panelField('y', second);
    await checkbox('#comparison-link-x', true); await update('Scatter', 2);
    for (let i = 0; i < 2; i++) assert.equal((await traceMetrics('scatter', i)).traces[0].length, expectedScatterRows);
    return {panels: 2, usablePointsPerPanel: expectedScatterRows};
  });
  const groupKey = JSON.stringify([signal, [second], signal, second]);
  const fullGroups = await page.evaluate(key => window.__referenceGroups[key], groupKey);
  await stage('Real Scatter zoom/reset and PNG export', async () => {
    const calls = await page.evaluate(() => window.__referenceCalls.length);
    await page.locator('#chart').hover(); await page.locator('#chart .modebar-btn[data-title="Zoom in"]').click();
    await page.waitForFunction(() => JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range) === JSON.stringify(document.querySelector('#scatter-chart-1')._fullLayout.xaxis.range));
    assert.equal(await page.evaluate(() => window.__referenceCalls.length), calls);
    const manualRange = await page.locator('#chart').evaluate(graph => [...graph._fullLayout.xaxis.range]);
    await update('Scatter', 2);
    const repeatedRange = await page.locator('#chart').evaluate(graph => [...graph._fullLayout.xaxis.range]);
    assert.ok(JSON.stringify(manualRange) === JSON.stringify(repeatedRange), 'An unchanged Update must preserve the manual axis range');
    assert.equal((await traceMetrics('scatter')).traces[0].length, expectedScatterRows);
    const callsAfterRerender = await page.evaluate(() => window.__referenceCalls.length);
    await page.locator('#scatter-chart-1').hover(); await page.locator('#scatter-chart-1 .modebar-btn[data-title="Reset axes"]').click();
    await page.waitForFunction(() => JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range) === JSON.stringify(document.querySelector('#scatter-chart-1')._fullLayout.xaxis.range));
    assert.equal(await page.evaluate(() => window.__referenceCalls.length), callsAfterRerender);
    const bytes = await pngCheck('scatter');
    return {pngBytes: bytes, zoomPreservedAcrossUpdate: true, extraZoomQueries: 0};
  });
  await stage('Two full Rolling windows', async () => {
    await selectView('Rolling window'); await panelField('y', signal); await panelField('window', 25);
    await page.locator('#add-rolling').click(); await panelField('y', second); await panelField('window', 50); await panelField('includeCurrent', false);
    await update('Rolling window', 2);
    for (const [index, name] of [[0, signal], [1, second]]) {
      const raw = (await traceMetrics('rolling', index)).traces.find(trace => trace.name === name);
      assert.ok(raw); assert.equal(raw.length, reference.rows); assert.equal(raw.finite, statsByColumn.get(name).valid);
    }
    assert.ok(await page.locator('#chart').evaluate(graph => graph.data.some(trace => trace.meta?.rollingBand && trace.fill === 'toself')), 'The full-data rolling band must be rendered');
    return {panels: 2, allPointsPerPanel: reference.rows, pngBytes: await pngCheck('rolling')};
  });
  await stage('Two full Derivative analyses including partial-null signal', async () => {
    await selectView('Derivatives'); await panelField('y', signal); await panelField('timeUnit', 'seconds');
    await page.locator('#add-derivative').click(); await panelField('y', partial); await panelField('smoothWindow', 3);
    await update('Derivatives', 2);
    for (const [index, name] of [[0, signal], [1, partial]]) {
      const raw = (await traceMetrics('derivative', index)).traces.find(trace => trace.name === name);
      assert.ok(raw); assert.equal(raw.length, statsByColumn.get(name).valid); assert.equal(raw.finite, statsByColumn.get(name).valid);
    }
    return {panels: 2, allUsablePointsRetained: true, partialSignalFinite: statsByColumn.get(partial).valid, pngBytes: await pngCheck('derivative')};
  });
  const fullConfig = await savedConfig();
  await stage('Inclusive native-profile time filter', async () => {
    await setTimeFilter(); await update('Derivatives', 2);
    assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), profile.time.filteredRows);
    await openDetails('#preview-panel'); await page.locator('#preview-button').click(); assert.ok((await idle()).ok);
    assert.equal(await page.locator('#preview-table tbody tr').count(), 100);
    await selectView('Scatter', 2);
    const filteredGroups = await page.evaluate(key => window.__referenceGroups[key], groupKey);
    assert.ok(JSON.stringify(fullGroups.color.edges) === JSON.stringify(filteredGroups.color.edges), 'Dataset color bins must remain stable under a time filter');
    assert.ok(JSON.stringify(fullGroups.marker.edges) !== JSON.stringify(filteredGroups.marker.edges), 'Filtered marker bins must adapt to the selected time range');
    await selectView('Distribution', 2);
    for (let i = 0; i < 2; i++) assert.equal((await traceMetrics('distribution', i)).traces[0].histogramTotal, profile.time.filteredRows);
    await selectView('Derivatives', 2);
    return {matchedRows: profile.time.filteredRows, invalidTime: 0};
  });
  let filteredCSVStatistics; const boundedSamples = [];
  await stage('Filtered full-width CSV matches independent native statistics', async () => {
    const result = await scanCSV(await grabDownload('#data-export'), {expectedRows: profile.time.filteredRows, expectedColumns: reference.columns, expectedNames: sourceNames,
      expectedStats: profile.filteredStatistics, statColumns: profile.signals, onStatistics: value => { filteredCSVStatistics = value; },
      recordColumns: [...profile.signals, profile.timeColumn], onRecord: row => boundedSamples.push(row)});
    assert.ok((await idle()).ok); return result;
  });
  await stage('Filtered full summary remains complete and exact', async () => {
    await openDetails('#summary-panel'); await page.locator('#summary-button').click(); assert.ok((await idle()).ok);
    if (profile.filteredStatistics) await assertNativeSummary(profile.filteredStatistics);
    await assertNativeSummary(filteredCSVStatistics);
    const total = await page.evaluate(() => window.__referenceSummary.column.length); assert.equal(total, reference.numericColumns);
    return {signals: total, matchedRows: profile.time.filteredRows};
  });
  const boundedConfig = await savedConfig();
  await stage('Rolling and derivative values independently checked against reference rows', async () => {
    const close = (actual, expected, label) => {
      assert.equal(actual.length, expected.length, `${label}: row count`);
      for (let index = 0; index < expected.length; index++) {
        const a = actual[index], b = expected[index];
        assert.ok(b === null ? a === null : Number.isFinite(a) && Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b)), `${label}: row ${index}`);
      }
    };
    await selectView('Rolling window', 2); await editPanel('rolling', 0); await panelField('y', signal);
    await panelField('window', 13); await panelField('includeCurrent', false); await panelField('requireFull', true); await update('Rolling window', 2);
    const rolls = await page.locator('#chart').evaluate(graph => graph.data.slice(1,4).map(trace => Array.from(trace.y)));
    const expected = {min: [], max: [], mean: []};
    for (let index = 0; index < boundedSamples.length; index++) {
      const window = boundedSamples.slice(Math.max(0,index-13),index).map(row=>row[signal]).filter(Number.isFinite);
      for (const stat of ['min','max','mean']) expected[stat].push(window.length < 13 ? null : stat === 'mean' ? window.reduce((sum,value)=>sum+value,0)/window.length : Math[stat](...window));
    }
    for (const [index, stat] of ['min','max','mean'].entries()) close(rolls[index], expected[stat], `Rolling ${stat}`);
    await selectView('Derivatives', 2); await editPanel('derivative', 0); await panelField('y', signal);
    await panelField('timeUnit', 'minutes'); await panelField('smoothWindow', 3); await update('Derivatives', 2);
    const derivatives = await page.locator('#chart').evaluate(graph => graph.data.slice(1,3).map(trace => Array.from(trace.y)));
    const rows = boundedSamples.filter(row=>Number.isFinite(row[signal])&&Number.isFinite(row[profile.timeColumn]));
    const first = [], second = [];
    for (let index = 0; index < rows.length; index++) {
      const dx = index ? (rows[index][profile.timeColumn]-rows[index-1][profile.timeColumn])/60000 : 0;
      first.push(dx ? (rows[index][signal]-rows[index-1][signal])/dx : null);
      second.push(dx && first[index] !== null && first[index-1] !== null ? (first[index]-first[index-1])/dx : null);
    }
    const smooth = (values, leading) => values.map((value,index) => {
      if (index < leading) return null;
      const window=values.slice(Math.max(0,index-2),index+1).filter(Number.isFinite);
      return window.length ? window.reduce((sum,value)=>sum+value,0)/window.length : null;
    });
    close(derivatives[0],smooth(first,1),'First derivative'); close(derivatives[1],smooth(second,2),'Second derivative');
    await loadConfig(boundedConfig);
    return {sourceRowsCompared: boundedSamples.length, rollingStatistics: 3, derivatives: 2, trailingWindow: 13, smoothingWindow: 3, derivativeUnit: 'minutes'};
  });
  await stage('Bounded SVG/spline/step/log/marker/range-slider/spike options', async () => {
    await selectView('Trends', 2); await editPanel('trend', 0); await chooseSignals(page, [signal]);
    await panelField('markers', true); await panelField('rangeslider', false); await panelField('spikes', false); await panelField('ytype', 'log');
    for (const shape of ['linear','hv','vh','hvh','vhv','spline']) {
      await panelField('shape', shape); await update('Trends', 2);
      const graph = await page.locator('#chart').evaluate(element => ({type: element.data[0].type, shape: element.data[0].line.shape,
        points: element.data[0].x.length, scale: element._fullLayout.yaxis.type, slider: element._fullLayout.xaxis.rangeslider.visible, spikes: element._fullLayout.xaxis.showspikes,
        markers: element.data[0].mode.includes('markers')}));
      assert.equal(graph.points, profile.time.filteredRows); assert.equal(graph.scale, 'log'); assert.equal(graph.slider, false); assert.equal(graph.spikes, false); assert.ok(graph.markers);
      assert.equal(graph.type, shape === 'spline' ? 'scatter' : 'scattergl');
    }
    return {shapes: 6, allPointsPerSeries: profile.time.filteredRows};
  });
  await stage('Numeric OR conditions with a final global time restriction', async () => {
    const a = (profile.filteredStatistics || filteredCSVStatistics).find(item => item.column === signal);
    const b = (profile.filteredStatistics || filteredCSVStatistics).find(item => item.column === second);
    const low = (a.min + a.max) / 2, high = (b.min + b.max) / 2;
    const expected = boundedSamples.filter(row => Number.isFinite(row[signal]) && row[signal] >= low || Number.isFinite(row[second]) && row[second] <= high).length;
    await openDetails('#filter-panel'); await page.selectOption('[data-key="flt2_var_0"]', signal);
    await page.fill('[data-key="flt2_low_num_0"]', String(low)); await page.fill('[data-key="flt2_up_num_0"]', '');
    await page.locator('#add-filter').click(); await page.selectOption('[data-key="flt2_var_1"]', second); await page.selectOption('[data-key="flt2_logic_1"]', 'OR');
    await page.fill('[data-key="flt2_low_num_1"]', ''); await page.fill('[data-key="flt2_up_num_1"]', String(high));
    await page.locator('#add-filter').click(); await page.selectOption('[data-key="flt2_var_2"]', '(time index)'); await page.selectOption('[data-key="flt2_logic_2"]', 'AND');
    await page.fill('[data-date-bound="low"][data-filter-index="2"]', localDateTime(profile.time.filterLow));
    await page.fill('[data-date-bound="up"][data-filter-index="2"]', localDateTime(profile.time.filterHigh));
    await page.locator('#add-filter').click(); await page.selectOption('[data-key="flt2_var_3"]', constant);
    await page.fill('[data-key="flt2_low_num_3"]', String(statsByColumn.get(constant).min));
    await page.fill('[data-key="flt2_up_num_3"]', String(statsByColumn.get(constant).min));
    await update('Trends', 2);
    assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), expected);
    await loadConfig(boundedConfig);
    return {matchedRows: expected, conditions: 4, exactNumericMatch: true};
  });
  await stage('Zero matches and invalid chart-width recovery', async () => {
    const low = new Date(Date.parse(profile.time.max || profile.time.filterHigh) + 86400000).toISOString().replace('Z', '');
    const high = new Date(Date.parse(profile.time.max || profile.time.filterHigh) + 90000000).toISOString().replace('Z', '');
    await openDetails('#filter-panel');
    await page.fill('[data-date-bound="low"][data-filter-index="0"]', low); await page.fill('[data-date-bound="up"][data-filter-index="0"]', high);
    await update('Derivatives', 2); assert.equal(await page.locator('#row-count').textContent(), '0');
    assert.equal(await page.locator('#chart').evaluate(graph => graph.data?.length || 0), 0);
    await loadConfig(boundedConfig); await selectView('Distribution', 2); await editPanel('distribution', 0);
    await panelField('binMode', 'Fixed width'); await panelField('binWidth', 0); await page.locator('#render-button').click();
    assert.equal((await idle()).ok, false, 'Invalid active width must be rejected');
    const calls = await page.evaluate(() => window.__referenceCalls.length);
    await openDetails('#preview-panel'); await page.locator('#preview-button').click(); assert.ok((await idle()).ok);
    assert.equal(await page.locator('#preview-table tbody tr').count(), 100);
    assert.ok(await page.locator('#dirty-indicator').isVisible());
    const newModes = await page.evaluate(start => window.__referenceCalls.slice(start).map(item => item.mode), calls);
    assert.ok(!newModes.some(mode => ['trends','distribution','scatter','rolling','derivatives'].includes(mode)), 'Preview must not render an invalid chart');
    await panelField('binWidth', 1); await update('Distribution', 2);
    for (let i = 0; i < 2; i++) assert.equal((await traceMetrics('distribution', i)).traces[0].histogramTotal, profile.time.filteredRows);
    await loadConfig(boundedConfig);
    return {zeroMatchCleared: true, invalidWidthRecovered: true};
  });
  await stage('Six live cards in every view on the bounded reference window', async () => {
    const views = Object.keys(slugs); const contexts = {};
    for (const name of views) {
      const slug = slugs[name]; await selectView(name, 2);
      await page.locator(`#duplicate-${slug}`).click();
      while (await page.locator(`[data-${slug}-panel]`).count() < 6) await page.locator(`#add-${slug}`).click();
      assert.ok(await page.locator(`#add-${slug}`).isDisabled()); assert.ok(await page.locator(`#duplicate-${slug}`).isDisabled());
      for (let i = 0; i < 6; i++) {
        await editPanel(slug, i);
        if (name === 'Trends' || name === 'Distribution') await chooseSignals(page, [signal]);
        else { if (name === 'Scatter') await panelField('x', signal); await panelField('y', name === 'Scatter' ? second : signal); }
        if (name === 'Trends') { await panelField('shape', 'linear'); await panelField('markers', false); await panelField('ytype', 'linear'); }
      }
      await update(name, 6);
      for (let i = 0; i < 6; i++) {
        const metrics = await traceMetrics(slug, i); assert.ok(metrics.traces.length, 'Each maximum-panel card must have actual plotted traces');
        if (name === 'Distribution') assert.equal(metrics.traces[0].histogramTotal, profile.time.filteredRows);
        else assert.ok(metrics.traces.some(trace => trace.length === profile.time.filteredRows));
      }
      contexts[slug] = await graphicsHealth(slug);
      while (await page.locator(`[data-${slug}-panel]`).count() > 1) await page.locator(`#remove-${slug}`).click();
      assert.ok(await page.locator(`#remove-${slug}`).isDisabled());
    }
    await loadConfig(boundedConfig);
    return {views: views.length, panelsPerView: 6, sourceRowsInWindow: profile.time.filteredRows, contexts};
  });
  await stage('Version3 configuration and local recovery without dataset bytes', async () => {
    const config = await savedConfig();
    for (const key of ['trend_panels','distribution_panels','scatter_panels','rolling_panels','derivative_panels']) assert.equal(config.settings[key].length, 2);
    await selectView('Trends', 2); await loadConfig(config);
    assert.equal(await page.locator('#chart-type').inputValue(), 'Derivatives');
    await page.reload({waitUntil: 'networkidle'}); assert.ok(await page.locator('#workspace').evaluate(element => element.classList.contains('hidden')));
    const record = await page.evaluate(() => localStorage.getItem('tsplotter.workspace.v3'));
    assert.ok(record && !record.includes(path.basename(fixture)) && record.length < 256 * 1024);
    await page.setInputFiles('#file-input', fixture); assert.ok((await idle()).ok);
    assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), profile.time.filteredRows);
    assert.equal(await page.locator('[data-derivative-card]').count(), 2);
    await page.locator('#clear-recovery').click();
    assert.equal(await page.evaluate(() => localStorage.getItem('tsplotter.workspace.v3')), null);
    assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), profile.time.filteredRows);
    return {restoredPanels: 10, matchedRows: profile.time.filteredRows, recoveryBytes: record.length, clearedRecoveryWithoutClearingDataset: true};
  });
  await stage('Reimport resets stale tables and reuses the worker', async () => {
    await page.setInputFiles('#file-input', {name: 'reference-reimport-check.csv', mimeType: 'text/csv', buffer: Buffer.from('x,value\n0,1\n1,2\n2,3\n')});
    assert.ok((await idle()).ok); assert.equal(await page.locator('#row-count').textContent(), '3');
    assert.equal(await page.locator('#preview-table tbody tr').count(), 0); assert.equal(await page.locator('#summary-table tbody tr').count(), 0);
    await page.setInputFiles('#file-input', fixture); assert.ok((await idle()).ok);
    await loadConfig(fullConfig);
    assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g, '')), reference.rows);
    return {reimportedRows: reference.rows, stalePreviewRows: 0, staleSummaryRows: 0};
  });
  await stage('Cancel real reference analysis and reopen successfully', async () => {
    await openDetails('#summary-panel');
    await page.locator('#summary-button').evaluate(button => { button.click(); document.querySelector('#cancel-button').click(); });
    await page.waitForFunction(() => document.querySelector('#workspace').classList.contains('hidden') && /cancel/i.test(document.querySelector('#status').textContent));
    await page.setInputFiles('#file-input', fixture); assert.ok((await idle()).ok); await loadConfig(fullConfig);
    await openDetails('#summary-panel'); await page.locator('#summary-button').click(); assert.ok((await idle()).ok);
    if (native) await assertNativeSummary(native.statistics);
    return {reopenedRows: reference.rows, recheckedSummarySignals: reference.numericColumns};
  });
  report.coverage = {reference: ['all5views','all233statistics','315034x234CSV','fullchartCSV','PNGEachView','constantAndPartialNull','2fullPanelsEachView','6liveBoundedPanelsEachView','maxAndMinPanelGuards','fullWebGLstep','all6lineShapesBounded','rollingBand','independentRollingAndDerivativeValues','colorAndMarkerBins','binScopes','paletteAndCustomSymbols','legends','linkedZoomReset','unchangedUpdateZoom','numericORAndTimeAndExactFilters','zeroRows','invalidWidthRecovery','v3Config','localRecoveryAndDeviceClear','independentPreview','summarySearchPaginationAndCache','reimport','cancellation'],
    supplemental: ['Text/category contains/starts/ends/regex and missing-category matching require categorical source columns; the reference contains only numeric values and timestamps.','Additional legacy v1/v2 migration cases, mobile390px layouts and alternate file formats remain covered by the separate synthetic browser/unit suites.']};
  assert.deepEqual(report.pageErrors, []); assert.deepEqual(report.writes, []); assert.deepEqual(report.external, []); assert.deepEqual(report.crashes, []);
  assert.deepEqual(report.graphicsWarnings, []);
  assert.deepEqual(await page.evaluate(() => window.__referenceGLErrors), []);
}
let fatal;
try {
  await stage('Load application', async () => {
    await page.goto(baseURL, {waitUntil: 'networkidle'});
    return {ok: true};
  });
  await stage('Import reference and render default Trends', async () => {
    await page.setInputFiles('#file-input', fixture);
    const state = await idle();
    if (!state.ok) return state;
    const metadata = await page.evaluate(() => ({
      matched: Number(document.querySelector('#row-count').textContent.replace(/[^0-9]/g, '')),
      columns: Number(document.querySelector('#column-count').textContent.replace(/[^0-9]/g, '')),
      numeric: Number(document.querySelector('#numeric-count').textContent.replace(/[^0-9]/g, '')),
      traceLengths: (document.querySelector('#chart').data || []).map(trace => trace.x?.length ?? trace.y?.length ?? 0),
    }));
    assert.equal(metadata.matched, reference.rows);
    assert.equal(metadata.columns, reference.columns);
    assert.equal(metadata.numeric, reference.numericColumns);
    assert.ok(metadata.traceLengths.includes(reference.rows), 'The initial plot must retain every point');
    return {ok: true, ...metadata};
  });
  const summary = await stage('Full numeric summary', async () => {
    await openDetails('#summary-panel');
    await page.locator('#summary-button').click();
    const state = await idle(120000);
    return {...state, summaryRows: await page.locator('#summary-table tbody tr').count()};
  }, {allowError: phase === 'baseline'});
  if (!summary.ok && !page.isClosed() && await page.locator('#cancel-button').isVisible().catch(() => false)) {
    await page.locator('#cancel-button').click();
    report.stages.push({name: 'Cancel timed-out summary', ok: true});
  }
  await stage('Preview after summary', async () => {
    await openDetails('#preview-panel');
    if (!await page.locator('#preview-button').isEnabled()) return {ok: false, error: 'Preview is unavailable after summary; reimport is required'};
    await page.locator('#preview-button').click();
    const state = await idle(30000);
    const previewRows = await page.locator('#preview-table tbody tr').count();
    if (phase === 'full' && state.ok) assert.equal(previewRows, 100);
    return {...state, previewRows};
  }, {allowError: phase === 'baseline'});

  if (phase === 'full') { await runFullWorkflow(); report.result = 'PASS'; }
  else report.result = 'BASELINE_COMPLETE';
} catch (error) {
  fatal = String(error).slice(0, 1600);
  report.result = 'FAIL'; report.error = fatal; process.exitCode = 1;
} finally {
  clearInterval(sampler);
  await heap('final');
  try {
    report.workerCalls = await page.evaluate(() => window.__referenceCalls || []);
    report.finalUI = await page.evaluate(() => ({status: document.querySelector('#status')?.textContent,
      workspaceVisible: !document.querySelector('#workspace')?.classList.contains('hidden'),
      renderDisabled: document.querySelector('#render-button')?.disabled, selectedView: document.querySelector('#chart-type')?.value,
      rowCount: document.querySelector('#row-count')?.textContent, selectedFiles: document.querySelector('#file-input')?.files.length}));
    report.workerErrors = await page.evaluate(() => window.__referenceWorkerErrors || []);
    report.performance = (await cdp.send('Performance.getMetrics')).metrics.filter(metric => ['JSHeapUsedSize', 'JSHeapTotalSize', 'TaskDuration', 'ScriptDuration', 'LayoutCount', 'Nodes', 'Documents'].includes(metric.name));
    report.dom = await cdp.send('Memory.getDOMCounters');
    report.graphicsEvents = await page.evaluate(() => window.__referenceGLErrors || []);
  } catch { /* Failure diagnostics above remain usable after a renderer crash. */ }
  if (phase === 'full' && report.result === 'PASS' && (report.workerErrors.length || report.workerCalls?.some(call => call.error))) {
    report.result = 'FAIL'; report.error = 'An unexpected analysis-worker error was recorded'; process.exitCode = 1;
  }
  report.heap.scope = 'Main renderer JavaScript heap; excludes DuckDB Worker WASM memory and GPU allocations';
  await writeFile(path.join(artifacts, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({result: report.result, error: fatal, reference, stages: report.stages,
    peakMainHeapMiB: report.heap.peakUsedBytes / 1048576, pageErrors: report.pageErrors, workerErrors: report.workerErrors,
    consoleErrors: report.consoleErrors, graphicsWarnings: report.graphicsWarnings, graphicsEvents: report.graphicsEvents,
    writes: report.writes, external: report.external, nativeReference: report.nativeReference, coverage: report.coverage, artifacts}, null, 2));
  await browser.close();
}
