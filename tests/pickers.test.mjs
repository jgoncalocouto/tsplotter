import test from 'node:test';
import assert from 'node:assert/strict';
import {filterPickerOptions, pickerSummary, nextPickerOption, readPickerOptions, updatePickerSelection, clearPickerSelection} from '../src/pickers.js';
const options = [
  {index: 0, value: 't', label: 'Temperature', selected: true},
  {index: 1, value: 'a', label: 'Água flow', selected: false},
  {index: 2, value: 'd', label: 'Disabled', disabled: true, selected: true},
  {index: 3, value: 'h', label: 'Hidden', hidden: true, selected: false},
  {index: 4, value: 'x', label: '<private>& name', selected: true},
];
function selectModel(initial) {
  const events = [];
  const select = {options: initial.map(option => ({...option})), disabled: false, dispatchEvent: event => events.push(event)};
  return {select, events};
}
test('search ignores case/accents and supports multiple terms without changing original labels', () => {
  assert.deepEqual(filterPickerOptions(options, 'agua FL'), [options[1]]);
  assert.deepEqual(filterPickerOptions(options, '  private  name  '), [options[4]]);
  assert.equal(filterPickerOptions(options).length, 4);
  assert.equal(options[1].label, 'Água flow');
});
test('selected chips preserve source order, literal labels and overflow count', () => {
  const result = pickerSummary(options, 2);
  assert.equal(result.count, 3); assert.deepEqual(result.chips.map(option => option.value), ['t', 'd']); assert.equal(result.remaining, 1);
  assert.equal(result.selected[2].label, '<private>& name');
  assert.deepEqual(pickerSummary([], 3), {selected: [], chips: [], remaining: 0, count: 0});
});
test('keyboard navigation wraps among visible enabled search matches only', () => {
  assert.equal(nextPickerOption(options, undefined, 1), 0); assert.equal(nextPickerOption(options, undefined, -1), 4);
  assert.equal(nextPickerOption(options, 1, 1), 4); assert.equal(nextPickerOption(options, 4, 1), 0); assert.equal(nextPickerOption(options, 0, -1), 4);
  assert.equal(nextPickerOption(filterPickerOptions(options, 'agua'), 0, 1), 1);
  assert.equal(nextPickerOption([{index: 0, disabled: true}], undefined, 1), undefined);
});
test('native selection updates emit one bubbling change and preserve independent choices', () => {
  const {select, events} = selectModel(options);
  assert.equal(updatePickerSelection(select, 1, true), true); assert.equal(select.options[1].selected, true); assert.equal(select.options[0].selected, true);
  assert.equal(events.length, 1); assert.equal(events[0].type, 'change'); assert.equal(events[0].bubbles, true);
  assert.equal(updatePickerSelection(select, 1, true), false); assert.equal(events.length, 1);
  assert.equal(updatePickerSelection(select, 99, true), false);
});
test('disabled select and disabled/optgroup choices are immutable', () => {
  const {select, events} = selectModel(options);
  assert.equal(updatePickerSelection(select, 2, false), false);
  select.options[1].parentElement = {tagName: 'OPTGROUP', disabled: true};
  assert.equal(updatePickerSelection(select, 1, true), false);
  select.disabled = true; assert.equal(updatePickerSelection(select, 0, false), false); assert.equal(clearPickerSelection(select), false); assert.equal(events.length, 0);
});
test('clear selection preserves disabled selected options and emits only one change', () => {
  const {select, events} = selectModel(options);
  assert.equal(clearPickerSelection(select), true); assert.equal(select.options[0].selected, false); assert.equal(select.options[4].selected, false); assert.equal(select.options[2].selected, true); assert.equal(events.length, 1);
  assert.equal(clearPickerSelection(select), false); assert.equal(events.length, 1);
});
test('batched selection can synchronize UI before the application gets its event', () => {
  const {select, events} = selectModel(options);
  assert.equal(updatePickerSelection(select, 1, true, {notify: false}), true); assert.equal(events.length, 0);
  assert.equal(clearPickerSelection(select, {notify: false}), true); assert.equal(events.length, 0);
});
test('native option adapter preserves original indices, literal values and disabled groups', () => {
  const {select} = selectModel([{value: 'a', label: '', textContent: 'Literal <x>', selected: true}, {value: 'b', label: 'Label', parentElement: {tagName: 'OPTGROUP', disabled: true}}]);
  assert.deepEqual(readPickerOptions(select), [{index: 0, value: 'a', label: 'Literal <x>', selected: true, hidden: false, disabled: false}, {index: 1, value: 'b', label: 'Label', selected: false, hidden: false, disabled: true}]);
});
test('actual browser picker keeps native selection, keyboard, search and busy-state in sync', {skip: process.env.PICKER_DOM_TESTS !== '1'}, async () => {
  const [{chromium}, {readFile}] = await Promise.all([import('@playwright/test'), import('node:fs/promises')]);
  const source = await readFile(new URL('../src/pickers.js', import.meta.url), 'utf8');
  const browser = await chromium.launch({headless: true, ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}), args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.setContent(`<style>.native-control{display:none}[hidden]{display:none!important}</style><label>Signals<select id="y-columns" data-key="mlc_cols_0" multiple><option selected value="a">Alpha</option><option value="b">Beta &lt;private&gt;</option><option value="c">Água flow</option><option disabled value="d">Disabled</option></select></label><button id="outside">Outside</button>`);
    await page.evaluate(async base64 => {window.pickerModule = await import(`data:text/javascript;base64,${base64}`); window.changes = 0; document.querySelector('select').addEventListener('change', () => window.changes++); window.pickerModule.enhancePickers();}, Buffer.from(source, 'utf8').toString('base64'));
    assert.equal(await page.locator('.signal-picker').count(), 1);
    await page.evaluate(() => window.pickerModule.enhancePickers()); assert.equal(await page.locator('.signal-picker').count(), 1);
    await page.locator('.picker-trigger').click(); assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'true');
    await page.locator('.picker-search').fill('beta'); assert.equal(await page.locator('.picker-option:visible').count(), 1);
    await page.locator('.picker-search').press('ArrowDown'); await page.keyboard.press('Space');
    assert.deepEqual(await page.locator('select').evaluate(select => [...select.selectedOptions].map(option => option.value)), ['a', 'b']);
    assert.equal(await page.evaluate(() => window.changes), 1); assert.match(await page.locator('.picker-trigger').getAttribute('aria-label'), /2 selected/);
    await page.keyboard.press('Escape'); assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'false'); assert.equal(await page.locator('.picker-trigger').evaluate(element => element === document.activeElement), true);
    await page.evaluate(() => {const select = document.querySelector('select'); for (const option of select.options) option.selected = option.value === 'c'; select.dispatchEvent(new Event('change', {bubbles: true}));});
    assert.match(await page.locator('.picker-trigger').textContent(), /Água flow/);
    await page.locator('.picker-trigger').press('ArrowDown'); await page.keyboard.press('End'); assert.equal(await page.locator('.picker-option:focus').textContent(), '✓Água flow');
    await page.locator('#outside').click(); assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
    await page.locator('.picker-trigger').click(); await page.locator('.picker-clear').click();
    assert.equal(await page.locator('select').evaluate(select => select.selectedOptions.length), 0);
    await page.evaluate(() => {document.querySelector('select').disabled = true;});
    await page.waitForFunction(() => document.querySelector('.picker-trigger').disabled);
    assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
    await page.evaluate(() => {document.querySelector('select').disabled = false;}); await page.waitForFunction(() => !document.querySelector('.picker-trigger').disabled);
    await page.locator('.picker-trigger').click(); await page.locator('.picker-search').fill('agua'); assert.equal(await page.locator('.picker-option:visible').count(), 1);
    await page.locator('.picker-search').press('Enter'); assert.deepEqual(await page.locator('select').evaluate(select => [...select.selectedOptions].map(option => option.value)), ['c']);
    await page.evaluate(() => window.pickerModule.closePickers()); assert.equal(await page.locator('.picker-popover').isVisible(), false);
    assert.equal(await page.locator('.signal-picker [data-key]').count(), 0);
  } finally {await browser.close();}
});

test('searchable single-variable pickers replace choices, close on selection and preserve optional none', {skip: process.env.PICKER_DOM_TESTS !== '1'}, async () => {
  const [{chromium}, {readFile}] = await Promise.all([import('@playwright/test'), import('node:fs/promises')]);
  const source = await readFile(new URL('../src/pickers.js', import.meta.url), 'utf8');
  const browser = await chromium.launch({headless: true, ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}), args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.setContent('<style>.native-control{display:none}[hidden]{display:none!important}</style><label>Color by<select id="color" data-variable><option value="(none)">(none)</option><option value="a">Alpha</option><option value="b">Beta</option><option disabled value="d">Disabled</option></select></label><label>Scale<select id="scale"><option>linear</option><option>log</option></select></label>');
    await page.evaluate(async base64 => {
      window.pickerModule = await import(`data:text/javascript;base64,${base64}`);
      window.changes = 0; window.escapes = 0;
      document.querySelector('#color').addEventListener('change', () => window.changes++);
      document.addEventListener('keydown', event => {if (event.key === 'Escape') window.escapes++;});
      window.pickerModule.enhancePickers();
    }, Buffer.from(source, 'utf8').toString('base64'));
    assert.equal(await page.locator('.signal-picker').count(), 1); assert.ok(await page.locator('#scale').isVisible());
    assert.equal(await page.locator('[role="listbox"]').getAttribute('aria-multiselectable'), 'false');
    assert.equal(await page.locator('.picker-chip-remove').count(), 0); assert.ok(!await page.locator('.picker-clear').isVisible());
    await page.locator('.picker-trigger').click(); assert.ok(await page.locator('.picker-search').evaluate(el => el === document.activeElement));
    await page.locator('.picker-search').fill('beta'); await page.locator('.picker-search').press('Enter');
    assert.equal(await page.locator('#color').inputValue(), 'b'); assert.equal(await page.evaluate(() => window.changes), 1);
    assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'false'); assert.ok(await page.locator('.picker-trigger').evaluate(el => el === document.activeElement));
    await page.locator('.picker-trigger').click(); await page.locator('.picker-search').fill('beta'); await page.locator('.picker-search').press('Enter');
    assert.equal(await page.evaluate(() => window.changes), 1); assert.equal(await page.locator('#color').inputValue(), 'b');
    await page.locator('.picker-trigger').press('Backspace'); assert.equal(await page.locator('#color').inputValue(), 'b');
    await page.locator('.picker-trigger').press('n'); await page.locator('.picker-search').fill('none'); await page.getByRole('option', {name: '(none)', exact: true}).click();
    assert.equal(await page.locator('#color').inputValue(), '(none)'); assert.equal(await page.evaluate(() => window.changes), 2);
    await page.locator('.picker-trigger').click(); await page.locator('.picker-search').press('Escape'); assert.equal(await page.evaluate(() => window.escapes), 0);
    await page.locator('.picker-trigger').press('Escape'); assert.equal(await page.evaluate(() => window.escapes), 1);
    await page.locator('.picker-trigger').click(); await page.locator('.picker-search').fill('missing'); assert.ok(await page.locator('.picker-empty').isVisible());
    assert.equal(await page.locator('#color').inputValue(), '(none)');
    await page.evaluate(() => {document.querySelector('#color').disabled = true;});
    await page.waitForFunction(() => document.querySelector('.picker-trigger').disabled);
    assert.equal(await page.locator('.picker-trigger').getAttribute('aria-expanded'), 'false');
  } finally {await browser.close();}
});
