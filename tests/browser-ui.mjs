/** Visual and interaction verification for the redesigned desktop/mobile UI. */
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';
import { chooseChart, chooseSignals, waitForChart } from './browser-controls.mjs';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:8765/';
const screenshotPrefix = process.env.SCREENSHOT_PREFIX || '/tmp/plotter-redesign';
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'] });
const checks = [], errors = [], writes = [];
const pass = message => { checks.push(message); console.log(message); };
function observe(page) {
  page.setDefaultTimeout(30000);
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', request => { if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) writes.push(`${request.method()} ${request.url()}`); });
}
async function noHorizontalOverflow(page) {
  const dimensions = await page.evaluate(() => ({
    width: innerWidth,
    scrollWidth: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
  }));
  assert.ok(dimensions.scrollWidth <= dimensions.width + 1, `Page overflows horizontally: ${JSON.stringify(dimensions)}`);
}
async function legendDoesNotOverlapToolbar(page) {
  const overlaps = await page.locator('#chart').evaluate(chart => {
    const rects = selector => [...chart.querySelectorAll(selector)].map(element => {
      const rect = element.getBoundingClientRect();
      return {left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height};
    }).filter(rect => rect.width > 0 && rect.height > 0);
    const toolbar = rects('.modebar-btn');
    return rects('.legend .traces').flatMap(legend => toolbar.filter(button =>
      Math.min(legend.right, button.right) - Math.max(legend.left, button.left) > 1 &&
      Math.min(legend.bottom, button.bottom) - Math.max(legend.top, button.top) > 1
    ).map(button => ({legend, button})));
  });
  assert.deepEqual(overlaps, [], 'Plotly toolbar must not obscure legend labels');
}
async function loadExample(page) {
  await page.locator('#welcome-example').click();
  await waitForChart(page);
  assert.equal(await page.locator('#row-count').textContent(), '720');
}
async function pickerEscape(page) {
  const select = page.locator('#y-columns');
  const picker = select.locator('xpath=following-sibling::div[contains(concat(" ", @class, " "), " signal-picker ")][1]');
  const original = await select.evaluate(el => [...el.selectedOptions].map(option => option.value));
  await picker.locator('.picker-trigger').click();
  await picker.locator('.picker-search').fill('humidity');
  assert.equal(await picker.getByRole('option', { name: 'humidity', exact: true }).count(), 1);
  await page.keyboard.press('Escape');
  assert.equal(await picker.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
  assert.ok(await picker.locator('.picker-trigger').evaluate(el => el === document.activeElement));
  assert.deepEqual(await select.evaluate(el => [...el.selectedOptions].map(option => option.value)), original);
}

try {
  const desktopContext = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const desktop = await desktopContext.newPage(); observe(desktop);
  await desktop.goto(baseURL, { waitUntil: 'networkidle' });
  await desktop.screenshot({ path: `${screenshotPrefix}-desktop-welcome.png`, fullPage: true });
  await loadExample(desktop);
  assert.equal(await desktop.getByRole('tab').count(), 5);
  assert.equal(await desktop.getByRole('tab', {name: 'Grouped scatter', exact: true}).count(), 0);
  for (const tab of await desktop.getByRole('tab').all()) assert.ok(await tab.isVisible());
  await desktop.evaluate(() => window.scrollTo(0, 0));
  const chartTop = await desktop.locator('#chart').evaluate(el => el.getBoundingClientRect().top);
  assert.ok(chartTop >= 0 && chartTop < 600, `Desktop chart starts at ${chartTop}px; expected before 600px`);
  await noHorizontalOverflow(desktop);
  pass('Desktop exposes five focused chart tabs and starts the chart before 600px at 1440×1000');

  await pickerEscape(desktop);
  pass('Signal picker searches actual options; Escape closes it and restores trigger focus without changing data');
  const previousTraceNames = await desktop.locator('#chart').evaluate(el => el.data.map(trace => trace.name));
  await chooseSignals(desktop, ['temperature', 'humidity']);
  assert.ok(await desktop.locator('#dirty-indicator').isVisible());
  assert.ok(await desktop.locator('#chart-export').isDisabled());
  assert.ok(await desktop.locator('#image-export').isDisabled());
  assert.deepEqual(await desktop.locator('#chart').evaluate(el => el.data.map(trace => trace.name)), previousTraceNames);
  await desktop.locator('#render-button').click(); await waitForChart(desktop);
  assert.ok(!await desktop.locator('#dirty-indicator').isVisible());
  assert.ok(await desktop.locator('#chart-export').isEnabled());
  assert.ok(await desktop.locator('#image-export').isEnabled());
  assert.deepEqual(await desktop.locator('#chart').evaluate(el => el.data.map(trace => trace.x.length)), [720, 720]);
  pass('Unapplied signal changes disable chart/image exports until the full 720-point traces are updated');

  const trendsTab = desktop.getByRole('tab', { name: 'Trends', exact: true });
  await trendsTab.focus(); await trendsTab.press('ArrowRight');
  assert.equal(await desktop.getByRole('tab', { name: 'Distribution', exact: true }).evaluate(el => el === document.activeElement), true);
  await desktop.keyboard.press('Enter'); await waitForChart(desktop);
  assert.equal(await desktop.getByRole('tab', { name: 'Distribution', exact: true }).getAttribute('aria-selected'), 'true');
  await chooseChart(desktop, 'Trends');
  await desktop.evaluate(() => window.scrollTo(0, 0));
  await desktop.screenshot({ path: `${screenshotPrefix}-desktop-example.png`, fullPage: true });
  pass('Chart tabs support keyboard focus navigation and activation');
  await desktopContext.close();

  const mobileContext = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const mobile = await mobileContext.newPage(); observe(mobile);
  await mobile.goto(baseURL, { waitUntil: 'networkidle' });
  assert.equal(await mobile.locator('body').evaluate(el => el.classList.contains('sidebar-open')), false);
  assert.equal(await mobile.locator('#sidebar').evaluate(el => el.inert), true);
  assert.equal(await mobile.locator('#settings-button').getAttribute('aria-expanded'), 'false');
  await noHorizontalOverflow(mobile);
  await mobile.screenshot({ path: `${screenshotPrefix}-mobile-welcome.png`, fullPage: true });
  await mobile.locator('#settings-button').click();
  assert.equal(await mobile.locator('#sidebar').evaluate(el => el.inert), false);
  assert.equal(await mobile.locator('#settings-button').getAttribute('aria-expanded'), 'true');
  assert.equal(await mobile.locator('main').evaluate(el => el.inert), true);
  assert.equal(await mobile.locator('#close-sidebar').evaluate(el => el === document.activeElement), true);
  await mobile.screenshot({ path: `${screenshotPrefix}-mobile-settings.png`, fullPage: true });
  await mobile.locator('#close-sidebar').click();
  assert.equal(await mobile.locator('#sidebar').evaluate(el => el.inert), true);
  assert.equal(await mobile.locator('main').evaluate(el => el.inert), false);
  await mobile.locator('#settings-button').click(); await mobile.keyboard.press('Escape');
  assert.equal(await mobile.locator('#settings-button').getAttribute('aria-expanded'), 'false');
  assert.equal(await mobile.locator('#settings-button').evaluate(el => el === document.activeElement), true);
  pass('Mobile 390px starts with the settings drawer closed; open, close and Escape restore focus correctly');

  await loadExample(mobile);
  await noHorizontalOverflow(mobile);
  await pickerEscape(mobile);
  await chooseSignals(mobile, ['temperature', 'humidity']);
  await mobile.locator('#render-button').click(); await waitForChart(mobile);
  assert.deepEqual(await mobile.locator('#chart').evaluate(el => el.data.map(trace => trace.x.length)), [720, 720]);
  await noHorizontalOverflow(mobile);
  await legendDoesNotOverlapToolbar(mobile);
  await mobile.evaluate(() => window.scrollTo(0, 0));
  await mobile.screenshot({ path: `${screenshotPrefix}-mobile-example.png`, fullPage: true });
  pass('Mobile keeps every example point, searchable pickers and page content within the viewport');
  await mobileContext.close();
  assert.deepEqual(errors, []); assert.deepEqual(writes, []);
  console.log(JSON.stringify({ result: 'PASS', checks, screenshotPrefix }, null, 2));
} catch (error) {
  console.error(JSON.stringify({ result: 'FAIL', error: String(error), checks, errors, writes }, null, 2));
  process.exitCode = 1;
} finally { await browser.close(); }
