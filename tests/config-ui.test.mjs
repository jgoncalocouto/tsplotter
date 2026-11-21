/** Exercise the actual UI adapters against portable configurations without a browser runtime. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const capture = source.slice(source.indexOf('function capture()'), source.indexOf('function renderAxis()'));
const renderFilters = source.slice(source.indexOf('function renderFilters()'), source.indexOf('function xColumn()'));
function captured(state, fields = [], dates = []) {
  const context = {state, document: {querySelectorAll: selector => selector === '[data-key]' ? fields : dates}};
  vm.runInNewContext(`${capture}; capture();`, context); return context.state;
}
test('editing imported categorical values replaces saved array; untouched literal comma values stay exact', () => {
  const state = {flt2_vals_0: ['last, first', 'other']};
  const field = {dataset: {key: 'flt2_vals_csv_0', membershipFilter: '0', originalValue: 'last, first, other'}, type: 'text', value: 'last, first, other'};
  captured(state, [field]); assert.deepEqual(state.flt2_vals_0, ['last, first', 'other']); assert.equal(state.flt2_vals_csv_0, undefined);
  field.value = 'different, second'; captured(state, [field]); assert.equal(state.flt2_vals_0, undefined); assert.equal(state.flt2_vals_csv_0, 'different, second');
});
test('time-index bound controls write idx settings while datetime columns write col settings', () => {
  const state = {};
  captured(state, [], [{dataset: {filterIndex: '0', dateBound: 'low', dateSuffix: 'idx'}, value: '2025-01-31T12:34:56.123456'}, {dataset: {filterIndex: '1', dateBound: 'up', dateSuffix: 'col'}, value: '2025-02-01T23:59'}]);
  assert.equal(state.flt2_use_lowdt_idx_0, true); assert.equal(state.flt2_low_date_idx_0, '2025-01-31'); assert.equal(state.flt2_low_time_idx_0, '12:34:56.123456');
  assert.equal(state.flt2_use_updt_col_1, true); assert.equal(state.flt2_up_date_col_1, '2025-02-01');
  captured(state, [], [{dataset: {filterIndex: '0', dateBound: 'low', dateSuffix: 'idx'}, value: ''}]); assert.equal(state.flt2_use_lowdt_idx_0, false);
});
test('filter renderer exposes saved time-index datetime bound controls', () => {
  const target = {innerHTML: ''};
  const state = {setup_time_mode: 'Absolute time', setup_dt_col: 'time', flt2_n: 1, flt2_var_0: '(time index)', flt2_use_lowdt_idx_0: true, flt2_low_date_idx_0: '2025-01-31', flt2_low_time_idx_0: '12:34:56.123456'};
  const context = {state, schema: {columns: [{name: 'time', kind: 'datetime'}]}, names: () => ['time'], $: () => target, esc: value => String(value), select: (label, key, options) => JSON.stringify({label, key, options}), input: () => '', check: () => ''};
  vm.runInNewContext(`${renderFilters}; renderFilters();`, context);
  assert.match(target.innerHTML, /Time index · time/); assert.match(target.innerHTML, /data-date-suffix="idx"/); assert.match(target.innerHTML, /2025-01-31T12:34:56.123456/); assert.doesNotMatch(target.innerHTML, /Value\(s\)/);
});
