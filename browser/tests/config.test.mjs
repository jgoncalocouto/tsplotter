import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeConfig, encodeConfig, validateSettings, normalizeConfiguration, defaultSettings, filtersFromSettings, MAX_CONFIG_BYTES, isConfigKey, ROW_INDEX} from '../src/config.js';
const document = (settings, extra = {}) => JSON.stringify({app: 'tsplotter', version: 1, settings, ...extra});
const schema = {rowCount: 10, columns: [{name: 'temperature', type: 'DOUBLE', kind: 'numeric'}, {name: 'humidity', type: 'DOUBLE', kind: 'numeric'}, {name: 'time', type: 'TIMESTAMP', kind: 'datetime'}, {name: 'shift', type: 'VARCHAR', kind: 'categorical'}]};

test('portable version 1 preserves chart, time, filter and independent grouping controls', () => {
  const settings = {plot_type: 'Grouped scatter', setup_time_mode: 'Absolute time', setup_dt_col: 'time', setup_out_tz: 'UTC', sc2_x: 'temperature', sc2_y: 'humidity', sc2_g1: 'temperature', sc2_g2: 'temperature', sc2_color_bin_mode: 'Number of bins', sc2_color_bin_count: 5, sc2_marker_bin_mode: 'Bin width', sc2_marker_bin_width: 2.5, sc2_msize: 9, sc2_mop: .65, sc2_palette: 'D3', mlc_cols_0: ['temperature', 'humidity'], mlc_n_subplots: 2, hist_binmode: 'Fixed width', hist_width_input: .125, rw_window: 100, der_smooth: 3, overview_cols: ['temperature'], flt2_n: 2, flt2_var_0: '(time index)', flt2_use_lowdt_idx_0: true, flt2_low_date_idx_0: '2025-01-31', flt2_low_time_idx_0: '12:34:56', flt2_logic_1: 'OR', flt2_var_1: 'shift', flt2_op_1: 'is one of', flt2_vals_1: ['Nuit', '日'], flt2_na_1: false};
  assert.deepEqual(decodeConfig(encodeConfig(settings)), {settings, warnings: []});
});
test('export excludes all data, files, handles, trigger state and unrecognized keys', () => {
  const state = {plot_type: 'Trends', raw_df: () => {}, dataset: {private: 10}, config_file: {}, sc_download: true, setup_anything: 'secret'};
  assert.deepEqual(decodeConfig(encodeConfig(state)).settings, {plot_type: 'Trends'});
});
test('decoding unsupported keys emits warnings without returning their values', () => {
  const result = decodeConfig(document({plot_type: 'Scatter', raw_df: [1, 2], sc_download: true}));
  assert.deepEqual(result.settings, {plot_type: 'Scatter'});
  assert.equal(result.warnings.length, 2);
});
for (const key of ['mlc_cols_6', 'mlc_cols_-1', 'flt2_var_10', 'flt2_var_0_extra', 'sc_color_bin_source', 'sc2_download', 'flt2_clear', 'raw_df', '__proto__']) test(`reject unsupported key ${key}`, () => assert.equal(isConfigKey(key), false));
for (const [key, value] of [['mlc_n_subplots', 0], ['mlc_n_subplots', true], ['mlc_n_subplots', 2.5], ['mlc_markers', 'false'], ['sc2_color_bin_width', 0], ['sc2_marker_bin_count', 101], ['sc_mop', .1], ['rw_window', 1], ['hist_bins_main', 201], ['hist_width_input', 1e-13], ['plot_type', 'Unknown'], ['overview_cols', 'temperature'], ['mlc_cols_0', [1]], ['flt2_n', 11], ['flt2_logic_0', 'NOT'], ['flt2_low_num_0', 12], ['flt2_low_date_idx_0', '2025-02-30'], ['flt2_low_date_idx_0', '2025-01-01T12:00:00'], ['flt2_low_time_idx_0', '25:00:00'], ['flt2_low_time_idx_0', '12:00:00Z']]) test(`validate ${key}=${JSON.stringify(value)}`, () => assert.throws(() => decodeConfig(document({plot_type: 'Scatter', [key]: value})), new RegExp(key)));
for (const data of ['not json', '', '[]', 'null', '42', '{"app":"tsplotter","version":1,"settings":{"x":1e999}}', '{"app":"tsplotter","version":1,"settings":{"a":NaN}}', '{"app":"tsplotter","version":1,"settings":{\u000bx:2}}']) test(`reject malformed/nonfinite config ${data.slice(0, 20)}`, () => assert.throws(() => decodeConfig(data)));
test('reject duplicate fields rather than applying ambiguous values', () => assert.throws(() => decodeConfig('{"app":"tsplotter","version":1,"settings":{"sc_msize":5,"sc_msize":8}}'), /Duplicate/));
test('reject duplicate escaped fields too', () => assert.throws(() => decodeConfig('{"app":"tsplotter","version":1,"settings":{"sc_msize":5,"sc_\\u006dsize":8}}'), /Duplicate/));
for (const version of [0, 2, true, '1', null]) test(`reject unsupported version ${version}`, () => assert.throws(() => decodeConfig(document({}, {version})), /version/));
test('handles UTF8 BOM, uint8 subarrays, and non-ASCII values', () => {
  const bytes = new TextEncoder().encode(`x\uFEFF${document({mlc_cols_0: ['日']})}`);
  assert.deepEqual(decodeConfig(bytes.subarray(1)).settings.mlc_cols_0, ['日']);
  assert.throws(() => decodeConfig(new Uint8Array([255])), /UTF|encoded|encoding/i);
});
test('limits import and export to 256KB', () => {
  assert.throws(() => decodeConfig(' '.repeat(MAX_CONFIG_BYTES + 1)), /too large/);
  assert.throws(() => encodeConfig({flt2_vals_0: Array(30).fill('x'.repeat(9999))}), /too large/);
});
test('does not alias lists or mutate originals', () => {
  const settings = {mlc_cols_0: ['temperature']};
  const result = validateSettings(settings); result.settings.mlc_cols_0.push('humidity');
  assert.deepEqual(settings.mlc_cols_0, ['temperature']);
});
test('inactive invalid numeric-bin width does not block export', () => {
  assert.deepEqual(decodeConfig(encodeConfig({sc_color_bin_mode: 'Individual values', sc_color_bin_width: 0})).settings, {sc_color_bin_mode: 'Individual values'});
});
test('schema defaults pick actual timestamps and numeric columns without sampling options', () => {
  const defaults = defaultSettings(schema);
  assert.equal(defaults.setup_dt_col, 'time'); assert.equal(defaults.setup_time_mode, 'Absolute time');
  assert.deepEqual(defaults.mlc_cols_0, ['temperature']); assert.equal(defaults.sc_y, 'humidity');
  assert.equal(Object.keys(defaults).some(key => key.endsWith('_maxpts')), false);
  assert.doesNotThrow(() => encodeConfig(defaults));
});
test('normalization skips missing numeric columns and falls back safely when time unavailable', () => {
  const result = normalizeConfiguration({setup_time_mode: 'Absolute time', setup_dt_col: 'missing', setup_index: 'missing', mlc_cols_0: ['temperature', 'missing', 'shift'], sc_y: 'missing'}, schema);
  assert.equal(result.settings.setup_time_mode, 'Relative time'); assert.equal(result.settings.setup_index, ROW_INDEX);
  assert.deepEqual(result.settings.mlc_cols_0, ['temperature']); assert.equal(result.settings.sc_y, undefined);
  assert.ok(result.warnings.length >= 3);
});
test('normalization keeps compatible filters and rebuilds contiguous clause indices', () => {
  const result = normalizeConfiguration({flt2_n: 3, flt2_var_0: 'missing', flt2_low_num_0: '2', flt2_var_1: 'temperature', flt2_low_num_1: '5', flt2_var_2: 'humidity', flt2_op_2: 'contains', flt2_patt_2: 'bad'}, schema);
  assert.equal(result.settings.flt2_n, 1); assert.equal(result.settings.flt2_var_0, 'temperature'); assert.equal(result.settings.flt2_low_num_0, '5');
  assert.equal(result.settings.flt2_var_1, undefined); assert.equal(result.warnings.length, 2);
});
test('normalization retains datetime-index filter when actual time column exists', () => {
  const result = normalizeConfiguration({setup_time_mode: 'Absolute time', setup_dt_col: 'time', flt2_n: 1, flt2_var_0: '(time index)', flt2_use_lowdt_idx_0: true, flt2_low_date_idx_0: '2025-01-31'}, schema);
  assert.equal(result.settings.flt2_n, 1); assert.equal(result.warnings.length, 0);
});
test('filter adapter preserves mixed logic, numeric, categorical and actual time mapping', () => {
  const settings = {setup_time_mode: 'Absolute time', setup_dt_col: 'time', flt2_n: 3, flt2_var_0: 'temperature', flt2_low_num_0: ' -2.5e1 ', flt2_up_num_0: '', flt2_var_1: 'shift', flt2_logic_1: 'OR', flt2_op_1: 'is one of', flt2_vals_csv_1: 'day, night', flt2_case_1: true, flt2_na_1: true, flt2_var_2: '(time index)', flt2_use_lowdt_idx_2: true, flt2_low_date_idx_2: '2025-01-31', flt2_low_time_idx_2: '12:34:56'};
  assert.deepEqual(filtersFromSettings(settings, schema), [
    {logic: 'AND', column: 'temperature', kind: 'numeric', low: -25, up: undefined},
    {logic: 'OR', column: 'shift', kind: 'categorical', op: 'is one of', values: ['day', 'night'], caseSensitive: true, naMatch: true},
    {logic: 'AND', column: 'time', kind: 'datetime', low: '2025-01-31T12:34:56'},
  ]);
});
test('empty filters are noops while invalid numeric bounds fail explicitly', () => {
  assert.deepEqual(filtersFromSettings({flt2_n: 1, flt2_var_0: 'shift', flt2_op_0: 'contains', flt2_patt_0: ''}, schema), []);
  for (const low of ['NaN', '0xff', 'Infinity', '2x']) assert.throws(() => filtersFromSettings({flt2_n: 1, flt2_var_0: 'temperature', flt2_low_num_0: low}, schema), /finite/);
});
test('upper datetime default includes whole day', () => {
  const [filter] = filtersFromSettings({flt2_n: 1, flt2_var_0: 'time', flt2_use_updt_col_0: true, flt2_up_date_col_0: '2025-01-31'}, schema);
  assert.equal(filter.up, '2025-01-31T23:59:59.999999');
});
test('intentional no-grouping placeholders retain valid inactive bins without warnings', () => {
  const original = {sc_group: '(none)', sc_color_bin_mode: 'Individual values', sc_color_bin_count: 5, sc_color_bin_width: 1, sc2_g1: '(none)', sc2_color_bin_mode: 'Number of bins', sc2_color_bin_count: 7, sc2_g2: '(none)', sc2_marker_bin_mode: 'Bin width', sc2_marker_bin_width: 2};
  const result = normalizeConfiguration(original, schema);
  assert.deepEqual(result.warnings, []);
  assert.deepEqual(result.settings, {...original, flt2_n: 0});
  assert.deepEqual(original.sc_color_bin_width, 1);
});
test('fresh default settings do not produce warnings for unused scatter groups', () => {
  const result = normalizeConfiguration(defaultSettings(schema), schema);
  assert.deepEqual(result.warnings, []);
  assert.equal(result.settings.sc2_g1, '(none)'); assert.equal(result.settings.sc2_marker_bin_count, 5);
});
