/** Real user controls shared by deferred end-to-end checks. No browser is started here. */
import assert from 'node:assert/strict';

export async function waitForChart(page) {
  await page.waitForFunction(() => {
    const status = document.querySelector('#status');
    return status?.classList.contains('error') || !document.querySelector('#render-button')?.disabled &&
      Array.isArray(document.querySelector('#chart')?.data) && /ready|settings loaded/i.test(status?.textContent || '');
  }, undefined, { timeout: 120000 });
  assert.ok(!await page.locator('#status').evaluate(el => el.classList.contains('error')), await page.locator('#status').textContent());
}

export async function chooseChart(page, mode) {
  await page.getByRole('tab', { name: mode, exact: true }).click();
  await waitForChart(page);
  assert.equal(await page.getByRole('tab', { name: mode, exact: true }).getAttribute('aria-selected'), 'true');
}

export async function chooseSignals(page, values, selector = '#y-columns') {
  const wanted = Array.isArray(values) ? values : [values];
  const select = page.locator(selector);
  const multiple = await select.evaluate(el => el.multiple);
  if (!multiple) {
    assert.equal(wanted.length, 1, 'Single signal selectors accept one choice');
    if (!await select.evaluate(el => el.hasAttribute('data-variable'))) {
      await select.selectOption(wanted[0]);
      return;
    }
  }
  const picker = select.locator('xpath=following-sibling::div[contains(concat(" ", @class, " "), " signal-picker ")][1]');
  await picker.locator('.picker-trigger').click();
  if (multiple && await picker.locator('.picker-clear').isEnabled()) await picker.locator('.picker-clear').click();
  for (const value of wanted) {
    const label = await select.locator('option').evaluateAll((options, target) => options.find(option => option.value === target)?.label, value);
    assert.ok(label !== undefined, `Signal ${value} must exist`);
    await picker.locator('.picker-search').fill(label);
    const option = picker.getByRole('option', { name: label, exact: true });
    if (await option.getAttribute('aria-selected') !== 'true') await option.click();
  }
  if (await picker.locator('.picker-trigger').getAttribute('aria-expanded') === 'true') await page.keyboard.press('Escape');
  assert.equal(await picker.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
  assert.deepEqual(await select.evaluate(el => [...el.selectedOptions].map(option => option.value)),
    await select.locator('option').evaluateAll((options, selected) => options.filter(option => selected.includes(option.value)).map(option => option.value), wanted));
}

export async function openImportOptions(page) {
  const panel = page.locator('#file-options');
  if (!await panel.evaluate(el => el.open)) await panel.locator('summary').click();
}
