import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeConfig, encodeConfig, validateSettings, normalizeConfiguration, defaultSettings, filtersFromSettings, CONFIG_VERSION, MAX_CONFIG_BYTES, isConfigKey, ROW_INDEX, PLOT_TYPES, defaultScatterPanel, scatterPanels, MAX_SCATTER_PANELS, defaultDistributionPanel, distributionPanels, MAX_DISTRIBUTION_PANELS, defaultTrendPanel, defaultTrendSubplot, trendPanels, trendSubplots, MAX_TREND_SUBPLOTS, defaultRollingPanel, rollingPanels, defaultDerivativePanel, derivativePanels, MAX_ANALYSIS_PANELS} from '../src/config.js';
const document = (settings, extra = {}) => JSON.stringify({app: 'tsplotter', version: 1, settings, ...extra});
const schema = {rowCount: 10, columns: [{name: 'temperature', type: 'DOUBLE', kind: 'numeric'}, {name: 'humidity', type: 'DOUBLE', kind: 'numeric'}, {name: 'time', type: 'TIMESTAMP', kind: 'datetime'}, {name: 'shift', type: 'VARCHAR', kind: 'categorical'}]};

test('portable version 1 migrates grouped scatter while preserving time, filter and grouping controls', () => {
  const settings = {plot_type: 'Grouped scatter', setup_time_mode: 'Absolute time', setup_dt_col: 'time', setup_out_tz: 'UTC', sc2_x: 'temperature', sc2_y: 'humidity', sc2_g1: 'temperature', sc2_g2: 'temperature', sc2_color_bin_mode: 'Number of bins', sc2_color_bin_count: 5, sc2_marker_bin_mode: 'Bin width', sc2_marker_bin_width: 2.5, sc2_msize: 9, sc2_mop: .65, sc2_palette: 'D3', mlc_cols_0: ['temperature', 'humidity'], mlc_n_subplots: 2, hist_binmode: 'Fixed width', hist_width_input: .125, rw_window: 100, der_smooth: 3, overview_cols: ['temperature'], flt2_n: 2, flt2_var_0: '(time index)', flt2_use_lowdt_idx_0: true, flt2_low_date_idx_0: '2025-01-31', flt2_low_time_idx_0: '12:34:56', flt2_logic_1: 'OR', flt2_var_1: 'shift', flt2_op_1: 'is one of', flt2_vals_1: ['Nuit', '日'], flt2_na_1: false};
  const loaded = decodeConfig(document(settings));
  const {scatter_panels: panels, distribution_panels: distributions, trend_panels: trends, rolling_panels: rolling, derivative_panels: derivatives, ...legacy} = loaded.settings;
  assert.deepEqual(legacy, {...settings, plot_type: 'Scatter'});
  assert.deepEqual(loaded.warnings, []);
  assert.equal(panels.length, 1);
  assert.equal(panels[0].x, 'temperature'); assert.equal(panels[0].y, 'humidity');
  assert.equal(panels[0].color, 'temperature'); assert.equal(panels[0].marker, 'temperature');
  assert.equal(panels[0].colorMode, 'discrete'); assert.equal(panels[0].colorBinCount, 5);
  assert.equal(panels[0].markerBinWidth, 2.5); assert.equal(panels[0].size, 9);
  assert.equal(panels[0].opacity, .65); assert.equal(panels[0].palette, 'D3');
  assert.equal(distributions[0].binMode, 'Fixed width'); assert.equal(distributions[0].binWidth, .125);
  assert.equal(trends.length, 1); assert.deepEqual(trends[0].subplots, [{columns: ['temperature', 'humidity']}, {}]);
  assert.equal(rolling[0].window, 100); assert.equal(derivatives[0].smoothWindow, 3);
  const exported = encodeConfig(loaded.settings);
  assert.equal(JSON.parse(exported).version, 4);
  assert.deepEqual(decodeConfig(exported), loaded);
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
for (const version of [0, 5, true, '1', null]) test(`reject unsupported version ${version}`, () => assert.throws(() => decodeConfig(document({}, {version})), /version/));
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
  const {scatter_panels: panels, ...legacy} = decodeConfig(encodeConfig({sc_color_bin_mode: 'Individual values', sc_color_bin_width: 0})).settings;
  assert.deepEqual(legacy, {sc_color_bin_mode: 'Individual values'});
  assert.equal(panels[0].colorBinMode, 'Individual values');
  assert.equal(panels[0].colorBinWidth, 1);
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
  const {scatter_panels: panels, ...legacy} = result.settings;
  assert.deepEqual(legacy, {...original, flt2_n: 0});
  assert.equal(panels.length, 2);
  assert.equal(panels[1].markerBinWidth, 2);
  assert.deepEqual(original.sc_color_bin_width, 1);
});
test('fresh default settings do not produce warnings for unused scatter groups', () => {
  const result = normalizeConfiguration(defaultSettings(schema), schema);
  assert.deepEqual(result.warnings, []);
  assert.equal(result.settings.sc2_g1, '(none)'); assert.equal(result.settings.sc2_marker_bin_count, 5);
});

test('the chart navigation exposes one scatter view and independent default panels', () => {
  assert.deepEqual(PLOT_TYPES, ['Trends', 'Distribution', 'Scatter', 'Rolling window', 'Derivatives']);
  const first = defaultScatterPanel(schema), second = defaultScatterPanel(schema, 1);
  assert.equal(first.title, 'Scatter 1'); assert.equal(second.title, 'Scatter 2');
  assert.equal(first.x, 'temperature'); assert.equal(first.y, 'humidity');
  first.x = 'humidity'; assert.equal(second.x, 'temperature');
  assert.equal(defaultSettings(schema).scatter_panels.length, 1);
});

test('version 4 round-trips multiple independent scatter panels without data or UI state', () => {
  const panels = [
    {...defaultScatterPanel(schema), title: 'Conditions', color: 'temperature', colorBinMode: 'Number of bins', colorBinCount: 4, marker: 'shift', opacity: .45},
    {...defaultScatterPanel(schema, 1), title: 'Humidity response', x: 'humidity', y: 'temperature', color: 'humidity', colorMode: 'continuous', colormap: 'Cividis', marker: 'temperature', markerBinMode: 'Bin width', markerBinWidth: .25, markerPreset: 'Directional', size: 11},
  ];
  const original = structuredClone(panels);
  const encoded = encodeConfig({plot_type: 'Scatter', scatter_panels: panels, activeScatter: 1, dataset: {private: true}});
  const document = JSON.parse(encoded);
  assert.equal(document.version, 4);
  assert.deepEqual(Object.keys(document.settings).sort(), ['plot_type', 'scatter_panels']);
  const loaded = decodeConfig(encoded);
  assert.deepEqual(loaded.settings.scatter_panels, panels);
  assert.deepEqual(loaded.warnings, []);
  loaded.settings.scatter_panels[0].color = '(none)';
  assert.deepEqual(panels, original);
});

test('legacy simple scatter migrates automatic continuous color and its presentation', () => {
  const loaded = decodeConfig(document({plot_type: 'Scatter', sc_x: 'temperature', sc_y: 'humidity', sc_group: 'temperature', sc_cmap_cont: 'Plasma', sc_msize: 12, sc_mop: .55}));
  assert.equal(loaded.settings.plot_type, 'Scatter');
  assert.equal(loaded.settings.scatter_panels.length, 1);
  assert.equal(loaded.settings.scatter_panels[0].colorMode, 'auto');
  assert.equal(loaded.settings.scatter_panels[0].colormap, 'Plasma');
  assert.equal(loaded.settings.scatter_panels[0].size, 12);
  assert.equal(loaded.settings.scatter_panels[0].opacity, .55);
});

test('style-only version 1 settings survive default merging for simple and grouped scatter', () => {
  for (const [plotType, styles, expected] of [
    ['Scatter', {sc_msize: 12, sc_mop: .45, sc_cmap_disc: 'D3'}, {size: 12, opacity: .45, palette: 'D3', colorMode: 'auto'}],
    ['Grouped scatter', {sc2_msize: 10, sc2_marker_preset: 'Directional', sc2_marker_custom: 'x, diamond'}, {size: 10, markerPreset: 'Directional', customSymbols: 'x, diamond', colorMode: 'discrete'}],
  ]) {
    const loaded = decodeConfig(document({plot_type: plotType, ...styles}));
    const normalized = normalizeConfiguration(loaded.settings, schema);
    const merged = {...defaultSettings(schema), ...normalized.settings};
    assert.equal(merged.plot_type, 'Scatter');
    assert.equal(merged.scatter_panels.length, 1);
    assert.equal(merged.scatter_panels[0].x, 'temperature');
    assert.equal(merged.scatter_panels[0].y, 'humidity');
    for (const [field, value] of Object.entries(expected)) assert.equal(merged.scatter_panels[0][field], value);
  }
});

test('inactive legacy scatter style presets migrate while obsolete point caps do not create panels', () => {
  const loaded = decodeConfig(document({plot_type: 'Trends', sc2_palette: 'Dark24', sc2_marker_bin_mode: 'Number of bins', sc2_marker_bin_count: 4}));
  assert.equal(loaded.settings.plot_type, 'Trends');
  assert.equal(loaded.settings.scatter_panels.length, 1);
  assert.equal(loaded.settings.scatter_panels[0].palette, 'Dark24');
  assert.equal(loaded.settings.scatter_panels[0].markerBinCount, 4);
  assert.equal(decodeConfig(document({sc_maxpts: 1000, sc2_maxpts: 1000})).settings.scatter_panels, undefined);
});

test('different old scatter views become independent panels with the active view first', () => {
  const loaded = decodeConfig(document({plot_type: 'Grouped scatter', sc_x: 'temperature', sc_y: 'humidity', sc_group: '(none)', sc2_x: 'humidity', sc2_y: 'temperature', sc2_g1: 'shift', sc2_g2: 'temperature', sc2_marker_bin_mode: 'Number of bins', sc2_marker_bin_count: 3}));
  const panels = loaded.settings.scatter_panels;
  assert.equal(panels.length, 2); assert.equal(panels[0].x, 'humidity');
  assert.equal(panels[0].marker, 'temperature'); assert.equal(panels[0].markerBinCount, 3);
  assert.equal(panels[1].x, 'temperature'); assert.equal(panels[1].marker, '(none)');
  panels[0].markerBinCount = 9; assert.equal(panels[1].markerBinCount, 5);
});

test('equivalent old default scatter views collapse into one panel even on an inactive view', () => {
  const loaded = decodeConfig(document({plot_type: 'Trends', sc_x: 'temperature', sc_y: 'humidity', sc_group: '(none)', sc2_x: 'temperature', sc2_y: 'humidity', sc2_g1: '(none)', sc2_g2: '(none)'}));
  assert.equal(loaded.settings.plot_type, 'Trends');
  assert.equal(loaded.settings.scatter_panels.length, 1);
});

test('canonical panels take precedence over stale old view fields', () => {
  const panel = {...defaultScatterPanel(schema), x: 'humidity', color: 'shift'};
  const loaded = decodeConfig(document({plot_type: 'Grouped scatter', scatter_panels: [panel], sc2_x: 'temperature', sc2_y: 'temperature'}, {version: 2}));
  assert.equal(loaded.settings.plot_type, 'Scatter');
  assert.deepEqual(loaded.settings.scatter_panels, [panel]);
});

for (const value of [[], Array(MAX_SCATTER_PANELS + 1).fill({}), {}, [null], ['scatter']]) test(`reject invalid scatter panel collection ${JSON.stringify(value)}`, () => {
  assert.throws(() => decodeConfig(document({scatter_panels: value}, {version: 2})), /scatter_panels/);
});
for (const [key, value] of [['title', 'x'.repeat(81)], ['x', 3], ['colorMode', 'random'], ['colorBinMode', 'Quantiles'], ['colorBinCount', 0], ['markerBinCount', 101], ['colorBinWidth', 0], ['markerBinWidth', -1], ['palette', 'Unknown'], ['colormap', 'Unknown'], ['markerPreset', 'Unknown'], ['customSymbols', false], ['size', 2], ['opacity', .1]]) test(`validate scatter panel ${key}`, () => {
  assert.throws(() => decodeConfig(document({scatter_panels: [{[key]: value}]}, {version: 2})), new RegExp(key));
});

test('unknown nested panel fields are ignored without restoring dataset or prototype values', () => {
  const loaded = decodeConfig('{"app":"tsplotter","version":2,"settings":{"scatter_panels":[{"x":"temperature","y":"humidity","dataset":{"private":1},"__proto__":{"polluted":true}}]}}');
  assert.equal(loaded.warnings.length, 2);
  assert.equal(loaded.settings.scatter_panels[0].dataset, undefined);
  assert.equal({}.polluted, undefined);
  assert.equal(JSON.stringify(loaded.settings).includes('private'), false);
});

test('scatter normalization preserves raw axes, handles missing columns, and disables incompatible bins', () => {
  const original = {plot_type: 'Scatter', scatter_panels: [
    {...defaultScatterPanel(schema), x: 'missing-x', y: 'humidity', color: 'shift', colorMode: 'continuous', colorBinMode: 'Bin width', colorBinWidth: 2, marker: 'missing-group'},
    {...defaultScatterPanel(schema, 1), x: 'humidity', y: 'temperature', color: 'temperature', colorBinMode: 'Number of bins', colorBinCount: 4},
  ]};
  const result = normalizeConfiguration(original, schema);
  assert.equal(result.settings.scatter_panels[0].x, 'temperature');
  assert.equal(result.settings.scatter_panels[0].y, 'humidity');
  assert.equal(result.settings.scatter_panels[0].color, 'shift');
  assert.equal(result.settings.scatter_panels[0].colorBinMode, 'Individual values');
  assert.equal(result.settings.scatter_panels[0].colorMode, 'auto');
  assert.equal(result.settings.scatter_panels[0].marker, '(none)');
  assert.equal(result.settings.scatter_panels[1].colorBinCount, 4);
  assert.equal(original.scatter_panels[0].x, 'missing-x');
  assert.ok(result.warnings.length >= 3);
});

test('runtime scatter panel copies preserve transient inactive values without aliasing', () => {
  const original = {scatter_panels: [{...defaultScatterPanel(schema), colorBinWidth: 0}]};
  const runtime = scatterPanels(original, schema);
  assert.equal(runtime[0].colorBinWidth, 0);
  runtime[0].x = 'humidity'; assert.equal(original.scatter_panels[0].x, 'temperature');
  const loaded = decodeConfig(encodeConfig(original));
  assert.equal(loaded.settings.scatter_panels[0].colorBinWidth, 1);
});

test('distribution defaults select numeric columns and keep arrays independent', () => {
  const first = defaultDistributionPanel(schema), second = defaultDistributionPanel(schema, 1);
  assert.deepEqual(first, {title: 'Distribution 1', columns: ['temperature'], binMode: 'Auto (width)', binCount: 50, binWidth: 1, binScope: 'Filtered data'});
  assert.equal(second.title, 'Distribution 2');
  first.columns.push('humidity');
  assert.deepEqual(second.columns, ['temperature']);
  assert.deepEqual(defaultDistributionPanel({columns: schema.columns.slice(2)}).columns, []);
  const defaults = defaultSettings(schema);
  defaults.distribution_panels[0].columns.push('humidity');
  assert.deepEqual(defaults.hist_cols_main, ['temperature']);
  assert.deepEqual(defaultSettings(schema).distribution_panels[0].columns, ['temperature']);
});

test('version 4 round-trips independent distributions and scatter settings together', () => {
  const panels = [
    defaultDistributionPanel(schema),
    {...defaultDistributionPanel(schema, 1), title: 'Humidity spread', columns: ['humidity'], binMode: 'Fixed width', binWidth: .125},
    {...defaultDistributionPanel(schema, 2), title: 'Compare conditions', columns: ['temperature', 'humidity'], binMode: 'Target #bins', binCount: 75},
  ];
  const state = {plot_type: 'Distribution', distribution_panels: panels, scatter_panels: [defaultScatterPanel(schema)], activeDistribution: 2, dataset: {private: true}};
  const original = structuredClone(state);
  const exported = JSON.parse(encodeConfig(state));
  assert.equal(exported.version, 4);
  assert.deepEqual(Object.keys(exported.settings).sort(), ['distribution_panels', 'plot_type', 'scatter_panels']);
  const loaded = decodeConfig(JSON.stringify(exported));
  assert.deepEqual(loaded.settings.distribution_panels, panels);
  assert.deepEqual(loaded.settings.scatter_panels, state.scatter_panels);
  assert.deepEqual(loaded.warnings, []);
  loaded.settings.distribution_panels[0].columns.push('humidity');
  loaded.settings.distribution_panels[1].binWidth = 9;
  assert.deepEqual(state, original);
  assert.deepEqual(loaded.settings.distribution_panels[2].columns, ['temperature', 'humidity']);
});

test('current exports consistently use version 4 including legacy settings', () => {
  assert.equal(CONFIG_VERSION, 4);
  assert.equal(JSON.parse(encodeConfig({distribution_panels: [defaultDistributionPanel(schema)]})).version, 4);
  assert.equal(JSON.parse(encodeConfig({plot_type: 'Trends', mlc_width: 3})).version, 4);
});

test('legacy distribution columns and bins migrate from versions 1 and 2 before defaults merge', () => {
  for (const version of [1, 2]) {
    const loaded = decodeConfig(document({plot_type: 'Distribution', hist_cols_main: ['humidity', 'temperature'], hist_binmode: 'Fixed width', hist_width_input: .25, hist_bins_main: 80}, {version}));
    const normalized = normalizeConfiguration(loaded.settings, schema);
    const merged = {...defaultSettings(schema), ...normalized.settings};
    assert.deepEqual(merged.distribution_panels, [{title: 'Distribution 1', columns: ['humidity', 'temperature'], binMode: 'Fixed width', binWidth: .25, binCount: 80, binScope: 'Filtered data'}]);
    assert.deepEqual(normalized.warnings, []);
  }
});

test('each partial legacy distribution control migrates even while another chart is active', () => {
  for (const [legacy, field, value] of [
    [{hist_cols_main: ['humidity']}, 'columns', ['humidity']],
    [{hist_binmode: 'Target #bins'}, 'binMode', 'Target #bins'],
    [{hist_bins_main: 125}, 'binCount', 125],
    [{hist_width_input: .125}, 'binWidth', .125],
  ]) {
    const loaded = decodeConfig(document({plot_type: 'Trends', ...legacy}));
    const normalized = normalizeConfiguration(loaded.settings, schema);
    const merged = {...defaultSettings(schema), ...normalized.settings};
    assert.equal(merged.plot_type, 'Trends');
    assert.equal(merged.distribution_panels.length, 1);
    assert.deepEqual(merged.distribution_panels[0][field], value);
    assert.ok(merged.distribution_panels[0].columns.length);
    assert.deepEqual(normalized.warnings, []);
  }
  assert.equal(decodeConfig(document({plot_type: 'Distribution'})).settings.distribution_panels, undefined);
});

test('canonical distributions take precedence over stale histogram settings', () => {
  const panel = {...defaultDistributionPanel(schema), title: 'Saved panel', columns: ['humidity'], binMode: 'Target #bins', binCount: 100};
  const loaded = decodeConfig(document({distribution_panels: [panel], hist_cols_main: ['temperature'], hist_binmode: 'Fixed width', hist_width_input: 9}, {version: 2}));
  assert.deepEqual(loaded.settings.distribution_panels, [panel]);
  const copy = distributionPanels(loaded.settings, schema);
  copy[0].columns.push('temperature');
  assert.deepEqual(loaded.settings.distribution_panels[0].columns, ['humidity']);
});

for (const panels of [[], Array(MAX_DISTRIBUTION_PANELS + 1).fill({}), {}, [null], ['distribution']]) test(`reject invalid distribution panel collection ${JSON.stringify(panels)}`, () => {
  assert.throws(() => decodeConfig(document({distribution_panels: panels})), /distribution_panels/);
});

for (const [field, value] of [
  ['title', 'x'.repeat(81)], ['title', 5], ['columns', 'temperature'], ['columns', [1]], ['columns', Array(2001).fill('temperature')],
  ['binMode', 'Automatic'], ['binCount', 9], ['binCount', 201], ['binCount', 50.5], ['binCount', '50'],
  ['binWidth', 0], ['binWidth', 1e-13], ['binWidth', '1'],
]) test(`validate distribution panel ${field}=${JSON.stringify(value).slice(0, 40)}`, () => {
  assert.throws(() => decodeConfig(document({distribution_panels: [{[field]: value}]})), new RegExp(`distribution_panels\\[0\\].${field}`));
});

test('distribution validation accepts bounds, six panels, and drops unsupported nested fields safely', () => {
  const panels = Array.from({length: MAX_DISTRIBUTION_PANELS}, (_, index) => ({title: 'x'.repeat(80), columns: ['temperature'], binCount: index % 2 ? 10 : 200, binWidth: 1e-12}));
  const validated = validateSettings({distribution_panels: panels});
  assert.equal(validated.settings.distribution_panels.length, 6);
  validated.settings.distribution_panels[0].columns.push('humidity');
  assert.deepEqual(panels[0].columns, ['temperature']);
  assert.deepEqual(validated.settings.distribution_panels[1].columns, ['temperature']);
  const loaded = decodeConfig('{"app":"tsplotter","version":2,"settings":{"distribution_panels":[{"dataset":{"private":true},"__proto__":{"polluted":true}}]}}');
  assert.equal(loaded.warnings.length, 2);
  assert.equal(Object.hasOwn(loaded.settings.distribution_panels[0], 'dataset'), false);
  assert.equal(Object.hasOwn(loaded.settings.distribution_panels[0], '__proto__'), false);
  assert.equal({}.polluted, undefined);
});

test('distribution normalization keeps numeric signals and recovers defaults for unusable selections', () => {
  const original = {distribution_panels: [
    {...defaultDistributionPanel(schema), columns: ['humidity', 'missing', 'shift']},
    {...defaultDistributionPanel(schema, 1), columns: ['time', 'missing'], binMode: 'Fixed width', binWidth: .5},
    {...defaultDistributionPanel(schema, 2), columns: []},
  ]};
  const copy = structuredClone(original);
  const normalized = normalizeConfiguration(original, schema);
  assert.deepEqual(normalized.settings.distribution_panels.map(panel => panel.columns), [['humidity'], ['temperature'], ['temperature']]);
  assert.equal(normalized.settings.distribution_panels[1].binWidth, .5);
  assert.equal(normalized.warnings.length, 2);
  assert.match(normalized.warnings[1], /first numeric signal/);
  assert.deepEqual(original, copy);
  const noNumeric = normalizeConfiguration(original, {columns: schema.columns.slice(2)});
  assert.deepEqual(noNumeric.settings.distribution_panels.map(panel => panel.columns), [[], [], []]);
  assert.equal(noNumeric.warnings.length, 2);
});

test('runtime distribution copies preserve transient editor input and clone legacy arrays', () => {
  const legacy = {hist_cols_main: ['humidity'], hist_binmode: 'Target #bins', hist_bins_main: 90};
  const migrated = distributionPanels(legacy, schema);
  assert.deepEqual(migrated[0].columns, ['humidity']);
  assert.equal(migrated[0].binCount, 90);
  migrated[0].columns.push('temperature');
  assert.deepEqual(legacy.hist_cols_main, ['humidity']);
  const state = {distribution_panels: [{columns: [], binWidth: 0}, {columns: ['humidity']}]};
  const panels = distributionPanels(state, schema);
  assert.equal(panels[0].binWidth, 0); assert.deepEqual(panels[0].columns, []);
  assert.equal(panels[1].title, 'Distribution 2'); assert.equal(panels[1].binCount, 50);
  panels[0].columns.push('temperature');
  assert.deepEqual(state.distribution_panels[0].columns, []);
  assert.deepEqual(panels[1].columns, ['humidity']);
});

test('distribution export discards invalid inactive bin controls and validates active controls', () => {
  for (const [mode, fields, expected] of [
    ['Auto (width)', {binCount: 0, binWidth: 0}, {binCount: 50, binWidth: 1}],
    ['Fixed width', {binCount: 0, binWidth: .5}, {binCount: 50, binWidth: .5}],
    ['Target #bins', {binCount: 100, binWidth: 0}, {binCount: 100, binWidth: 1}],
  ]) {
    const state = {distribution_panels: [{...defaultDistributionPanel(schema), binMode: mode, ...fields}]};
    const original = structuredClone(state);
    const [saved] = decodeConfig(encodeConfig(state)).settings.distribution_panels;
    assert.equal(saved.binMode, mode);
    assert.equal(saved.binCount, expected.binCount); assert.equal(saved.binWidth, expected.binWidth);
    assert.deepEqual(state, original);
  }
  assert.throws(() => encodeConfig({distribution_panels: [{binMode: 'Fixed width', binWidth: 0}]}), /binWidth/);
  assert.throws(() => encodeConfig({distribution_panels: [{binMode: 'Target #bins', binCount: 0}]}), /binCount/);
  const legacy = decodeConfig(encodeConfig({hist_cols_main: ['humidity'], hist_binmode: 'Auto (width)', hist_width_input: 0, hist_bins_main: 0}));
  assert.equal(legacy.settings.distribution_panels[0].binWidth, 1);
  assert.equal(legacy.settings.distribution_panels[0].binCount, 50);
});

const analysisFamilies = [
  {key: 'trend_panels', create: defaultTrendPanel, read: trendPanels, prefix: 'mlc'},
  {key: 'rolling_panels', create: defaultRollingPanel, read: rollingPanels, prefix: 'rw'},
  {key: 'derivative_panels', create: defaultDerivativePanel, read: derivativePanels, prefix: 'der'},
];

test('all analysis views have independent defaults and shared comparison settings are opt-in', () => {
  const defaults = defaultSettings(schema);
  assert.equal(MAX_ANALYSIS_PANELS, 6);
  assert.equal(defaults.comparison_link_x, false); assert.equal(defaults.comparison_link_y, false);
  assert.equal(defaults.distribution_shared_bins, false);
  for (const family of analysisFamilies) {
    assert.equal(defaults[family.key].length, 1);
    assert.deepEqual(defaults[family.key][0], family.create(schema));
    assert.notEqual(defaults[family.key][0], defaultSettings(schema)[family.key][0]);
  }
  assert.equal(defaultTrendPanel(schema, 1).title, 'Trends 2');
  assert.deepEqual(defaultTrendPanel(schema, 1).subplots, [{columns: ['humidity']}]);
  assert.deepEqual(defaultTrendPanel(schema, 5).subplots, [{columns: ['temperature']}]);
  assert.equal(defaultRollingPanel(schema, 1).title, 'Rolling window 2');
  assert.equal(defaultDerivativePanel(schema, 1).title, 'Derivatives 2');
  assert.equal(defaultRollingPanel(schema).rangeslider, true);
  assert.equal(defaultDerivativePanel(schema).rangeslider, true);
  assert.equal(defaultRollingPanel({columns: schema.columns.slice(0, 2)}).rangeslider, false);
  defaults.trend_panels[0].subplots[0].columns.push('humidity');
  assert.deepEqual(defaults.mlc_cols_0, ['temperature']);
  assert.deepEqual(defaultSettings(schema).trend_panels[0].subplots, [{columns: ['temperature']}]);
  assert.deepEqual(normalizeConfiguration(defaultSettings(schema), schema).warnings, []);
});

test('version 4 independently round-trips every view, comparison options, and bin scopes', () => {
  const settings = {
    plot_type: 'Derivatives', comparison_link_x: true, comparison_link_y: true, distribution_shared_bins: true,
    trend_panels: [defaultTrendPanel(schema), {...defaultTrendPanel(schema, 1), title: 'Humidity history', subplots: [{columns: ['humidity', 'temperature']}], shape: 'hv', width: 4, ytype: 'log', markers: true, rangeslider: false, unified: false, spikes: false}],
    rolling_panels: [defaultRollingPanel(schema), {...defaultRollingPanel(schema, 1), title: 'Trailing humidity', y: 'humidity', window: 100, includeCurrent: false, requireFull: true, band: false, unified: false, rangeslider: false, spikes: false, size: 10, opacity: .4}],
    derivative_panels: [defaultDerivativePanel(schema), {...defaultDerivativePanel(schema, 1), title: 'Humidity rate', y: 'humidity', timeUnit: 'minutes', xMode: 'use numeric index values', smoothWindow: 3, markers: false, width: 5, unified: false, rangeslider: false, spikes: false}],
    distribution_panels: [{...defaultDistributionPanel(schema), binScope: 'Full dataset', binCount: 120, binWidth: .25}],
    scatter_panels: [{...defaultScatterPanel(schema), colorBinScope: 'Full dataset', markerBinScope: 'Full dataset', colorBinCount: 12, colorBinWidth: .125}],
  };
  const before = structuredClone(settings);
  const encoded = encodeConfig({...settings, activePanels: {Trends: 1}, raw_df: {private: true}});
  assert.equal(JSON.parse(encoded).version, 4);
  const loaded = decodeConfig(encoded);
  assert.deepEqual(loaded.settings, settings);
  assert.deepEqual(loaded.warnings, []);
  loaded.settings.trend_panels[1].subplots[0].columns.push('other');
  loaded.settings.rolling_panels[0].window = 20;
  loaded.settings.derivative_panels[0].smoothWindow = 5;
  assert.deepEqual(settings, before);
  assert.deepEqual(loaded.settings.trend_panels[0].subplots, [{columns: ['temperature']}]);
});

test('versions 1 and 2 migrate multiple trend subplots and all legacy analysis appearances', () => {
  const legacy = {
    plot_type: 'Trends', mlc_n_subplots: 2, mlc_cols_0: ['temperature'], mlc_cols_1: ['humidity'], mlc_cols_5: ['temperature'],
    mlc_shape: 'spline', mlc_width: 4, mlc_ytype: 'log', mlc_markers: true, mlc_rangeslider: false, mlc_unified: false, mlc_spikes: false,
    rw_y: 'humidity', rw_window: 75, rw_include_current: false, rw_full: true, rw_band: false, rw_unified: false, rw_rangeslider: true, rw_spikes: false, rw_msize: 11, rw_mop: .35,
    der_y: 'humidity', der_tunit: 'hours', der_xmode: 'use numeric index values', der_smooth: 5, der_markers: false, der_lw: 6, der_unified: false, der_rangeslider: true, der_spikes: false,
  };
  for (const version of [1, 2]) {
    const loaded = decodeConfig(document(legacy, {version}));
    const merged = {...defaultSettings(schema), ...normalizeConfiguration(loaded.settings, schema).settings};
    assert.equal(merged.trend_panels.length, 1);
    assert.deepEqual(merged.trend_panels[0].subplots, [{columns: ['temperature']}, {columns: ['humidity']}]);
    for (const panel of merged.trend_panels) {
      assert.equal(panel.shape, 'spline'); assert.equal(panel.width, 4); assert.equal(panel.ytype, 'log');
      assert.equal(panel.markers, true); assert.equal(panel.rangeslider, false); assert.equal(panel.unified, false); assert.equal(panel.spikes, false);
    }
    assert.deepEqual(merged.rolling_panels, [{title: 'Rolling window 1', y: 'humidity', window: 75, includeCurrent: false, requireFull: true, band: false, unified: false, rangeslider: true, spikes: false, size: 11, opacity: .35}]);
    assert.deepEqual(merged.derivative_panels, [{title: 'Derivatives 1', y: 'humidity', timeUnit: 'hours', xMode: 'use numeric index values', smoothWindow: 5, markers: false, width: 6, unified: false, rangeslider: true, spikes: false}]);
    assert.deepEqual(loaded.warnings, []);
    assert.deepEqual(decodeConfig(encodeConfig(loaded.settings)), loaded);
  }
});

test('partial legacy analysis settings migrate before defaults and obsolete point caps do not', () => {
  for (const [legacy, key, field, expected] of [
    [{mlc_width: 5}, 'trend_panels', 'width', 5],
    [{rw_window: 125}, 'rolling_panels', 'window', 125],
    [{rw_band: false}, 'rolling_panels', 'band', false],
    [{der_tunit: 'days'}, 'derivative_panels', 'timeUnit', 'days'],
    [{der_markers: false}, 'derivative_panels', 'markers', false],
  ]) {
    const decoded = decodeConfig(document({plot_type: 'Scatter', ...legacy}));
    const normalized = normalizeConfiguration(decoded.settings, schema);
    const merged = {...defaultSettings(schema), ...normalized.settings};
    assert.equal(merged[key][0][field], expected);
    assert.deepEqual(normalized.warnings, []);
    if (key !== 'trend_panels') assert.equal(merged[key][0].y, 'temperature');
    else assert.deepEqual(merged[key][0].subplots, [{columns: ['temperature']}]);
  }
  const inferred = decodeConfig(document({mlc_cols_2: ['humidity']}));
  assert.equal(inferred.settings.trend_panels.length, 1);
  assert.deepEqual(inferred.settings.trend_panels[0].subplots, [{}, {}, {columns: ['humidity']}]);
  const ignoredCaps = decodeConfig(document({mlc_maxpts: 1000, rw_maxpts: 1000, der_maxpts: 1000}));
  for (const family of analysisFamilies) assert.equal(ignoredCaps.settings[family.key], undefined);
});

test('canonical analysis arrays take precedence over stale legacy subplot counts and styles', () => {
  const canonical = {
    trend_panels: [{...defaultTrendPanel(schema), title: 'Kept trend', subplots: [{columns: ['humidity']}], width: 3}],
    rolling_panels: [{...defaultRollingPanel(schema), window: 150}],
    derivative_panels: [{...defaultDerivativePanel(schema), timeUnit: 'days'}],
  };
  const loaded = decodeConfig(document({...canonical, mlc_n_subplots: 6, mlc_width: 8, rw_window: 20, der_tunit: 'seconds'}, {version: 3}));
  for (const family of analysisFamilies) assert.deepEqual(loaded.settings[family.key], canonical[family.key]);
});

test('runtime analysis helpers return independent panel and signal arrays', () => {
  for (const family of analysisFamilies) {
    const panel = family.create(schema), state = {[family.key]: [panel, family.create(schema, 1)]};
    const copy = family.read(state, schema);
    copy[0].title = 'Changed';
    assert.notEqual(state[family.key][0].title, 'Changed');
    if (family.key === 'trend_panels') {
      copy[0].subplots[0].columns.push('humidity');
      assert.deepEqual(state.trend_panels[0].subplots, [{columns: ['temperature']}]);
      assert.deepEqual(copy[1].subplots, [{columns: ['humidity']}]);
      const legacy = {mlc_n_subplots: 1, mlc_cols_0: ['humidity']};
      family.read(legacy, schema)[0].subplots[0].columns.push('temperature');
      assert.deepEqual(legacy.mlc_cols_0, ['humidity']);
    }
  }
});

test('analysis normalization restores numeric signals and retains global filters', () => {
  const settings = {
    trend_panels: [{...defaultTrendPanel(schema), subplots: [{columns: ['humidity', 'shift', 'gone']}]}, {...defaultTrendPanel(schema, 1), subplots: [{columns: ['time']}]}],
    rolling_panels: [{...defaultRollingPanel(schema), y: 'shift'}, {...defaultRollingPanel(schema, 1), y: 'humidity'}],
    derivative_panels: [{...defaultDerivativePanel(schema), y: 'gone'}],
    flt2_n: 1, flt2_var_0: 'temperature', flt2_low_num_0: '20', comparison_link_x: true,
  };
  const before = structuredClone(settings);
  const result = normalizeConfiguration(settings, schema);
  assert.deepEqual(result.settings.trend_panels.map(panel => panel.subplots), [[{columns: ['humidity']}], [{columns: ['temperature']}]]);
  assert.deepEqual(result.settings.rolling_panels.map(panel => panel.y), ['temperature', 'humidity']);
  assert.equal(result.settings.derivative_panels[0].y, 'temperature');
  assert.equal(result.settings.comparison_link_x, true);
  assert.deepEqual(filtersFromSettings(result.settings, schema), [{logic: 'AND', column: 'temperature', kind: 'numeric', low: 20, up: undefined}]);
  assert.equal(result.warnings.length, 4);
  assert.deepEqual(settings, before);
  const empty = normalizeConfiguration({trend_panels: [{}], rolling_panels: [{}], derivative_panels: [{}]}, {columns: schema.columns.slice(2)});
  assert.deepEqual(empty.settings.trend_panels[0].subplots, [{columns: []}]);
  assert.equal(empty.settings.rolling_panels[0].y, ''); assert.equal(empty.settings.derivative_panels[0].y, '');
  assert.deepEqual(empty.warnings, []);
});

for (const family of analysisFamilies) {
  for (const value of [[], Array(MAX_ANALYSIS_PANELS + 1).fill({}), {}, [null], ['panel'], [new Date()]]) test(`reject invalid ${family.key} collection ${JSON.stringify(value).slice(0, 80)}`, () => {
    assert.throws(() => validateSettings({[family.key]: value}), new RegExp(family.key));
  });
  test(`${family.key} accepts six panels and validates independent plain objects`, () => {
    const panels = Array.from({length: MAX_ANALYSIS_PANELS}, (_, index) => family.create(schema, index));
    const result = validateSettings({[family.key]: panels});
    assert.deepEqual(result.settings[family.key], panels);
    assert.notEqual(result.settings[family.key][0], panels[0]);
    const inherited = Object.create({title: 'Inherited'});
    assert.throws(() => validateSettings({[family.key]: [inherited]}), /settings object/);
  });
  test(`${family.key} ignores per-panel filters and unsafe unknown fields`, () => {
    const loaded = decodeConfig(`{"app":"tsplotter","version":3,"settings":{"${family.key}":[{"title":"Safe","filters":[{"private":1}],"__proto__":{"polluted":true},"dataset":[1]}]}}`);
    assert.equal(loaded.settings[family.key][0].title, 'Safe');
    assert.equal(loaded.warnings.length, 3);
    for (const key of ['filters', '__proto__', 'dataset']) assert.equal(Object.hasOwn(loaded.settings[family.key][0], key), false);
    assert.equal({}.polluted, undefined);
  });
}

for (const [key, field, value] of [
  ['trend_panels', 'title', 'x'.repeat(81)], ['trend_panels', 'columns', 'temperature'], ['trend_panels', 'columns', [3]], ['trend_panels', 'shape', 'bezier'], ['trend_panels', 'width', 0], ['trend_panels', 'ytype', 'symlog'], ['trend_panels', 'markers', 1],
  ['rolling_panels', 'y', 3], ['rolling_panels', 'window', 1], ['rolling_panels', 'window', 200001], ['rolling_panels', 'window', 2.5], ['rolling_panels', 'includeCurrent', 'yes'], ['rolling_panels', 'requireFull', 1], ['rolling_panels', 'band', null], ['rolling_panels', 'size', 17], ['rolling_panels', 'opacity', .1],
  ['derivative_panels', 'timeUnit', 'weeks'], ['derivative_panels', 'xMode', 'automatic'], ['derivative_panels', 'smoothWindow', 0], ['derivative_panels', 'smoothWindow', 10000], ['derivative_panels', 'width', 9], ['derivative_panels', 'rangeslider', 'false'], ['derivative_panels', 'unified', 0], ['derivative_panels', 'spikes', null],
]) test(`validate ${key}.${field}=${JSON.stringify(value).slice(0, 30)}`, () => {
  assert.throws(() => decodeConfig(document({[key]: [{[field]: value}]}, {version: 3})), new RegExp(field));
});

for (const key of ['comparison_link_x', 'comparison_link_y', 'distribution_shared_bins']) test(`${key} requires an actual boolean`, () => {
  assert.throws(() => decodeConfig(document({[key]: 'true'}, {version: 3})), new RegExp(key));
  assert.equal(decodeConfig(document({[key]: true}, {version: 3})).settings[key], true);
});

test('v1 and v2 bins acquire filtered scope while explicit v3 scopes survive normalization', () => {
  for (const version of [1, 2, 3]) {
    const loaded = decodeConfig(document({distribution_panels: [{}], scatter_panels: [{}]}, {version}));
    assert.equal(loaded.settings.distribution_panels[0].binScope, 'Filtered data');
    assert.equal(loaded.settings.scatter_panels[0].colorBinScope, 'Filtered data');
    assert.equal(loaded.settings.scatter_panels[0].markerBinScope, 'Filtered data');
  }
  const normalized = normalizeConfiguration({distribution_panels: [{binScope: 'Full dataset'}], scatter_panels: [{colorBinScope: 'Full dataset', markerBinScope: 'Full dataset'}]}, schema);
  assert.equal(normalized.settings.distribution_panels[0].binScope, 'Full dataset');
  assert.equal(normalized.settings.scatter_panels[0].colorBinScope, 'Full dataset');
  assert.equal(normalized.settings.scatter_panels[0].markerBinScope, 'Full dataset');
  for (const [key, field] of [['distribution_panels', 'binScope'], ['scatter_panels', 'colorBinScope'], ['scatter_panels', 'markerBinScope']]) assert.throws(() => decodeConfig(document({[key]: [{[field]: 'Visible points'}]})), new RegExp(field));
});

test('saving preserves valid preferences for inactive bin modes including legacy settings', () => {
  const settings = {
    distribution_panels: [{...defaultDistributionPanel(schema), binCount: 125, binWidth: .25}],
    scatter_panels: [{...defaultScatterPanel(schema), colorBinCount: 12, colorBinWidth: .125, markerBinCount: 8, markerBinWidth: 2.5}],
    hist_binmode: 'Auto (width)', hist_bins_main: 125, hist_width_input: .25,
    sc_color_bin_mode: 'Individual values', sc_color_bin_count: 12, sc_color_bin_width: .125,
  };
  assert.deepEqual(decodeConfig(encodeConfig(settings)).settings, settings);
  const partial = decodeConfig(encodeConfig({hist_width_input: .125, sc_color_bin_count: 12}));
  assert.equal(partial.settings.distribution_panels[0].binWidth, .125);
  assert.equal(partial.settings.scatter_panels[0].colorBinCount, 12);
});

test('shared distribution bins ignore invalid unused follower values while preserving valid preferences', () => {
  const first = {...defaultDistributionPanel(schema), binMode: 'Fixed width', binWidth: .25};
  const second = {...defaultDistributionPanel(schema, 1), binMode: 'Fixed width', binWidth: 0, binCount: 125};
  const original = structuredClone(second);
  const saved = decodeConfig(encodeConfig({distribution_shared_bins: true, distribution_panels: [first, second]}));
  assert.equal(saved.settings.distribution_panels[1].binWidth, 1);
  assert.equal(saved.settings.distribution_panels[1].binCount, 125);
  assert.deepEqual(second, original);
  assert.throws(() => encodeConfig({distribution_shared_bins: false, distribution_panels: [first, second]}), /binWidth/);
  assert.throws(() => encodeConfig({distribution_shared_bins: true, distribution_panels: [{...first, binWidth: 0}, second]}), /binWidth/);
  const none = decodeConfig(encodeConfig({scatter_panels: [{color: '(none)', colorBinMode: 'Bin width', colorBinWidth: 0, markerBinMode: 'Number of bins', markerBinCount: 0}]}));
  assert.equal(none.settings.scatter_panels[0].colorBinWidth, 1);
  assert.equal(none.settings.scatter_panels[0].markerBinCount, 5);
  assert.throws(() => encodeConfig({scatter_panels: [{color: 'temperature', colorBinMode: 'Bin width', colorBinWidth: 0}]}), /colorBinWidth/);
});

test('nested Trends preserve independent instances, multiple signals, cleared rows and shared appearance', () => {
  const settings = {
    plot_type: 'Trends', trend_panels: [
      {...defaultTrendPanel(schema), title: 'Conditions', subplots: [{columns: ['temperature', 'humidity']}, {columns: []}, {columns: ['humidity']}], shape: 'spline', width: 5, ytype: 'log', markers: true, rangeslider: false, unified: false, spikes: false},
      {...defaultTrendPanel(schema, 1), title: 'Other view', subplots: [{columns: ['temperature']}, {columns: ['temperature', 'humidity']}]},
    ],
    flt2_n: 1, flt2_var_0: 'temperature', flt2_low_num_0: '5', comparison_link_x: true,
  };
  const before = structuredClone(settings);
  const encoded = encodeConfig({...settings, activeTrendSubplot: 2, dataset: {private: true}});
  assert.equal(JSON.parse(encoded).version, 4);
  const decoded = decodeConfig(encoded);
  assert.deepEqual(decoded.settings, settings);
  assert.deepEqual(normalizeConfiguration(decoded.settings, schema).settings, settings);
  assert.deepEqual(decoded.warnings, []);
  decoded.settings.trend_panels[0].subplots[0].columns.push('other');
  decoded.settings.trend_panels[0].subplots[1].columns.push('humidity');
  assert.deepEqual(settings, before);
  assert.deepEqual(decoded.settings.trend_panels[1].subplots[0].columns, ['temperature']);
});

test('old flat canonical Trends migrate to one subplot per existing instance for versions 1 through 3', () => {
  for (const version of [1, 2, 3]) {
    const loaded = decodeConfig(document({
      plot_type: 'Trends', mlc_n_subplots: 6, mlc_cols_0: ['humidity'],
      trend_panels: [{title: 'First instance', columns: ['temperature', 'humidity'], width: 4}, {title: 'Cleared instance', columns: [], shape: 'hv'}],
    }, {version}));
    assert.equal(loaded.settings.trend_panels.length, 2);
    assert.deepEqual(loaded.settings.trend_panels.map(panel => panel.subplots), [[{columns: ['temperature', 'humidity']}], [{columns: []}]]);
    assert.equal(loaded.settings.trend_panels[0].title, 'First instance');
    assert.equal(loaded.settings.trend_panels[0].width, 4);
    assert.equal(loaded.settings.trend_panels[1].title, 'Cleared instance');
    assert.equal(loaded.settings.trend_panels[1].shape, 'hv');
    assert.equal(Object.hasOwn(loaded.settings.trend_panels[0], 'columns'), false);
    assert.deepEqual(loaded.warnings, []);
    assert.deepEqual(normalizeConfiguration(loaded.settings, schema).settings.trend_panels[1].subplots, [{columns: []}]);
    assert.equal(JSON.parse(encodeConfig(loaded.settings)).version, 4);
  }
});

test('legacy subplot counts restore one Trend and preserve empty rows before and after default merging', () => {
  for (const version of [1, 2, 3]) {
    const loaded = decodeConfig(document({plot_type: 'Trends', mlc_n_subplots: 3, mlc_cols_0: ['temperature', 'humidity'], mlc_cols_1: [], mlc_cols_2: ['humidity'], mlc_markers: true, mlc_width: 6}, {version}));
    const merged = {...defaultSettings(schema), ...normalizeConfiguration(loaded.settings, schema).settings};
    assert.equal(merged.trend_panels.length, 1);
    assert.deepEqual(merged.trend_panels[0].subplots, [{columns: ['temperature', 'humidity']}, {columns: []}, {columns: ['humidity']}]);
    assert.equal(merged.trend_panels[0].width, 6); assert.equal(merged.trend_panels[0].markers, true);
    assert.deepEqual(decodeConfig(encodeConfig(merged)).settings.trend_panels, merged.trend_panels);
  }
});

test('missing Trend selections stay absent through schema-free migration while explicit empty rows stay empty', () => {
  for (const settings of [
    {mlc_n_subplots: 2, mlc_cols_1: [], mlc_width: 5},
    {trend_panels: [{width: 5, subplots: [{}, {columns: []}]}]},
  ]) {
    const loaded = decodeConfig(document(settings));
    assert.deepEqual(loaded.settings.trend_panels[0].subplots, [{}, {columns: []}]);
    const portable = decodeConfig(encodeConfig(loaded.settings));
    assert.deepEqual(portable.settings.trend_panels[0].subplots, [{}, {columns: []}]);
    const normalized = normalizeConfiguration(portable.settings, schema);
    assert.deepEqual(normalized.settings.trend_panels[0].subplots, [{columns: ['temperature']}, {columns: []}]);
    assert.equal(normalized.settings.trend_panels[0].width, 5);
    assert.deepEqual(normalized.warnings, []);
  }
  const styleOnly = decodeConfig(document({mlc_shape: 'hvh'}));
  assert.deepEqual(styleOnly.settings.trend_panels[0].subplots, [{}]);
  assert.deepEqual(normalizeConfiguration(styleOnly.settings, schema).settings.trend_panels[0].subplots, [{columns: ['temperature']}]);
});

test('nested canonical rows win over stale flat columns and legacy subplot controls', () => {
  const panel = {...defaultTrendPanel(schema), subplots: [{columns: ['humidity']}, {columns: []}]};
  const loaded = decodeConfig(document({trend_panels: [{...panel, columns: 'unused old field'}], mlc_n_subplots: 6, mlc_cols_0: ['temperature']}, {version: 4}));
  assert.deepEqual(loaded.settings.trend_panels, [panel]);
  assert.deepEqual(loaded.warnings, []);
  assert.deepEqual(trendSubplots({...panel, columns: ['temperature']}, schema), panel.subplots);
});

test('subplot helpers deep-copy nested arrays and resolve only missing selections', () => {
  assert.equal(MAX_TREND_SUBPLOTS, 6);
  assert.deepEqual(defaultTrendSubplot(schema), {columns: ['temperature']});
  assert.deepEqual(defaultTrendSubplot(schema, 1), {columns: ['humidity']});
  assert.deepEqual(defaultTrendSubplot(schema, 5), {columns: ['temperature']});
  const panel = {...defaultTrendPanel(schema), subplots: [{columns: ['temperature', 'humidity']}, {columns: []}, {}]};
  const before = structuredClone(panel), rows = trendSubplots(panel, schema);
  assert.deepEqual(rows, [{columns: ['temperature', 'humidity']}, {columns: []}, {columns: ['temperature']}]);
  rows[0].columns.push('other'); rows[1].columns.push('humidity');
  assert.deepEqual(panel, before);
  const copies = trendPanels({trend_panels: [panel, panel]}, schema);
  copies[0].subplots[0].columns.push('changed');
  assert.deepEqual(copies[1].subplots[0].columns, ['temperature', 'humidity']);
  assert.deepEqual(panel, before);
  const flat = {columns: ['humidity']};
  trendSubplots(flat, schema)[0].columns.push('temperature');
  assert.deepEqual(flat.columns, ['humidity']);
  const legacy = {mlc_n_subplots: 2, mlc_cols_0: ['humidity'], mlc_cols_1: []};
  const migrated = trendPanels(legacy, schema);
  assert.equal(migrated.length, 1); assert.deepEqual(migrated[0].subplots, [{columns: ['humidity']}, {columns: []}]);
  migrated[0].subplots[0].columns.push('temperature');
  assert.deepEqual(legacy.mlc_cols_0, ['humidity']);
});

test('six Trend instances each accept six independent subplot rows without aliasing', () => {
  const shared = {columns: ['temperature']};
  const panels = Array.from({length: MAX_ANALYSIS_PANELS}, (_, index) => ({...defaultTrendPanel(schema, index), subplots: Array(MAX_TREND_SUBPLOTS).fill(shared)}));
  const loaded = decodeConfig(encodeConfig({trend_panels: panels}));
  assert.equal(loaded.settings.trend_panels.length, 6);
  for (const panel of loaded.settings.trend_panels) assert.equal(panel.subplots.length, 6);
  loaded.settings.trend_panels[0].subplots[0].columns.push('humidity');
  assert.deepEqual(loaded.settings.trend_panels[0].subplots[1].columns, ['temperature']);
  assert.deepEqual(loaded.settings.trend_panels[1].subplots[0].columns, ['temperature']);
  assert.deepEqual(shared.columns, ['temperature']);
});

for (const subplots of [[], Array(MAX_TREND_SUBPLOTS + 1).fill({}), null, {}, ['row'], [null], [new Date(0)], [Object.create({columns: ['temperature']})]]) test(`reject invalid nested Trend subplot collection ${JSON.stringify(subplots).slice(0, 90)}`, () => {
  assert.throws(() => validateSettings({trend_panels: [{subplots}]}), /trend_panels\[0\].subplots/);
});

for (const columns of ['temperature', null, [1], Array(2001).fill('temperature'), ['x'.repeat(10001)]]) test(`validate nested Trend signals ${JSON.stringify(columns).slice(0, 45)}`, () => {
  assert.throws(() => validateSettings({trend_panels: [{subplots: [{columns}]}]}), /subplots\[0\].columns/);
});

test('nested Trend rows ignore private data, per-row styling, filters and prototype fields safely', () => {
  const loaded = decodeConfig('{"app":"tsplotter","version":4,"settings":{"trend_panels":[{"subplots":[{"columns":["temperature"],"width":8,"filters":[1],"dataset":{"private":1},"__proto__":{"polluted":true}}]}]}}');
  assert.deepEqual(loaded.settings.trend_panels[0].subplots, [{columns: ['temperature']}]);
  assert.equal(loaded.warnings.length, 4);
  assert.equal({}.polluted, undefined);
  assert.equal(loaded.settings.trend_panels[0].width, 2);
  assert.throws(() => decodeConfig('{"app":"tsplotter","version":4,"settings":{"trend_panels":[{"subplots":[{"columns":[],"columns":["temperature"]}]}]}}'), /Duplicate configuration field/);
});

test('nested schema normalization warns per row, recovers unavailable selections and preserves cleared rows', () => {
  const settings = {trend_panels: [{...defaultTrendPanel(schema), subplots: [{columns: ['humidity', 'shift', 'gone']}, {columns: []}, {columns: ['time']}, {}]}]};
  const before = structuredClone(settings);
  const result = normalizeConfiguration(settings, schema);
  assert.deepEqual(result.settings.trend_panels[0].subplots, [{columns: ['humidity']}, {columns: []}, {columns: ['temperature']}, {columns: ['temperature']}]);
  assert.equal(result.warnings.length, 2);
  assert.match(result.warnings[0], /subplot 1/); assert.match(result.warnings[1], /subplot 3/);
  assert.deepEqual(settings, before);
  const noNumeric = normalizeConfiguration(settings, {columns: schema.columns.slice(2)});
  assert.deepEqual(noNumeric.settings.trend_panels[0].subplots, [{columns: []}, {columns: []}, {columns: []}, {columns: []}]);
});
