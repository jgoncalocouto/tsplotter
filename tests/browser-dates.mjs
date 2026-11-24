/** Focused real-browser checks for time preparation, filters, and stale data. */
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { chooseSignals } from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:8765/';
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
page.setDefaultTimeout(20000);
const errors = [], forbiddenRequests = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
page.on('request', request => {
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method()) || /^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(baseURL).origin) forbiddenRequests.push(`${request.method()} ${request.url()}`);
});
async function ready() {
  await page.waitForFunction(() => !document.querySelector('#render-button')?.disabled && /ready|settings loaded/i.test(document.querySelector('#status')?.textContent || '') || document.querySelector('#status')?.classList.contains('error'));
  assert.ok(!await page.locator('#status').evaluate(el => el.classList.contains('error')), await page.locator('#status').textContent());
}
async function update() { await page.click('#render-button'); await ready(); }
const checks = [];
const pass = message => { checks.push(message); console.log(message); };
try {
  await page.goto(baseURL, { waitUntil: 'networkidle' });
  const csv = 'date,signal,category\n01/02/2026 00:00:00,1,Day\n02/02/2026 00:00:00,3,Night\n03/02/2026 00:00:00,5,Day\nbad,7,Day\n';
  await page.setInputFiles('#file-input', { name: 'dmy.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) });
  await ready();
  await page.selectOption('[data-key="setup_time_mode"]', 'Absolute time');
  await chooseSignals(page, ['date'], '#x-column');
  await page.locator('[data-key="setup_profile"]').evaluate(el => el.closest('details').open = true);
  await page.selectOption('[data-key="setup_profile"]', 'DMY (dd/mm/yyyy)');
  await chooseSignals(page, 'signal');
  await update();
  assert.equal(await page.locator('#row-count').textContent(), '4');
  assert.equal(await page.locator('#filtered-count').textContent(), '0');
  assert.equal(await page.locator('#invalid-time-count').textContent(), '1');
  assert.deepEqual(await page.locator('#primary-plot [data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger)),
    {source: 4, matched: 4, filteredOut: 0, invalidTime: 1, usable: 3, excludedValues: 0});
  assert.deepEqual(await page.locator('#chart').evaluate(el => Array.from(el.data[0].y)), [1, 3, 5]);
  pass('DMY accounting distinguishes four matched rows, one invalid timestamp and three plotted rows');

  await page.locator('#filter-panel > summary').click();
  await page.click('#add-filter');
  await chooseSignals(page, ['(time index)'], '[data-key="flt2_var_0"]');
  await page.fill('[data-date-bound="low"][data-filter-index="0"]', '2026-02-02T00:00:00');
  await page.fill('[data-date-bound="up"][data-filter-index="0"]', '2026-02-03T00:00:00');
  await update();
  assert.equal(await page.locator('#row-count').textContent(), '2');
  assert.deepEqual(await page.locator('#primary-plot [data-row-ledger]').evaluate(el => JSON.parse(el.dataset.ledger)),
    {source: 4, matched: 2, filteredOut: 2, invalidTime: 0, usable: 2, excludedValues: 0});
  const points = await page.locator('#chart').evaluate(el => Array.from(el.data[0].y));
  assert.deepEqual(points, [3, 5]);
  pass('Inclusive time-index filter retains precisely the two expected points');

  await page.locator('#preview-panel > summary').click();
  await page.click('#preview-button'); await ready();
  assert.equal(await page.locator('#preview-table tbody tr').count(), 2);
  const download = page.waitForEvent('download');
  await page.click('#data-export');
  const exported = await readFile(await (await download).path(), 'utf8'); await ready();
  assert.equal(exported.trim().split(/\r?\n/).length, 3);
  assert.ok(!exported.includes('bad') && !exported.includes(',1,Day'));
  await page.locator('#summary-button').evaluate(el => el.closest('details').open = true);
  await page.click('#summary-button'); await ready();
  const summary = await page.locator('#summary-table tbody tr').first().locator('td').allTextContents();
  assert.equal(summary[1], '2');
  pass('Prepared preview, full CSV and summary all contain the same two valid rows');

  await page.fill('[data-date-bound="low"][data-filter-index="0"]', '2030-01-01T00:00:00');
  await page.fill('[data-date-bound="up"][data-filter-index="0"]', '2030-01-02T00:00:00');
  await page.click('#preview-button'); await ready();
  assert.equal(await page.locator('#preview-table tbody tr').count(), 0);
  assert.ok(await page.locator('#dirty-indicator').isVisible());
  assert.deepEqual(await page.locator('#chart').evaluate(el => Array.from(el.data[0].y)), [3, 5], 'Preview must not apply pending chart changes');
  await update();
  assert.equal(await page.locator('#row-count').textContent(), '0');
  assert.equal(await page.locator('#chart').evaluate(el => el.data?.length || 0), 0);
  assert.equal(await page.locator('#preview-table tbody tr').count(), 0);
  assert.equal(await page.locator('#summary-table tbody tr').count(), 0);
  await page.click('#clear-filters'); await ready();
  assert.equal(await page.locator('#row-count').textContent(), '4');
  assert.equal(await page.locator('#invalid-time-count').textContent(), '1');
  assert.deepEqual(await page.locator('#chart').evaluate(el => Array.from(el.data[0].y)), [1, 3, 5]);
  pass('Empty filters clear charts, preview and summary; clearing filters recovers');

  await page.setInputFiles('#file-input', { name: 'invalid.mat', mimeType: 'application/octet-stream', buffer: Buffer.from('not a MATLAB file') });
  await page.waitForFunction(() => document.querySelector('#status')?.classList.contains('error'));
  assert.ok(await page.locator('#workspace').evaluate(el => el.classList.contains('hidden')));
  await page.setInputFiles('#file-input', { name: 'recovery.csv', mimeType: 'text/csv', buffer: Buffer.from('x,signal\n0,1\n1,3\n2,5\n') });
  await ready();
  assert.equal(await page.locator('#row-count').textContent(), '3');
  assert.equal(await page.locator('#preview-table tbody tr').count(), 0);
  pass('Invalid import clears stale workspace and a valid file can be opened afterwards');
  assert.deepEqual(errors, []);
  assert.deepEqual(forbiddenRequests, []);
  await page.screenshot({ path: '/tmp/plotter-browser-final.png', fullPage: true });
  console.log(JSON.stringify({ result: 'PASS', checks, screenshot: '/tmp/plotter-browser-final.png' }, null, 2));
} catch (error) {
  await page.screenshot({ path: '/tmp/plotter-browser-date-failure.png', fullPage: true }).catch(() => {});
  console.error(JSON.stringify({ result: 'FAIL', error: String(error), status: await page.locator('#status').textContent().catch(() => ''), errors, forbiddenRequests, checks }, null, 2));
  process.exitCode = 1;
} finally { await browser.close(); }
