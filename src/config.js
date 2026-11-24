/** Portable TSPlotter settings. Dataset bytes and browser file handles are never saved. */
export const CONFIG_APP = 'tsplotter';
export const CONFIG_VERSION = 4;
export const MAX_CONFIG_BYTES = 256 * 1024;
export const ROW_INDEX = '(row number 0..N-1)';
export const TIME_INDEX = '(use existing time index)';
export const FILTER_TIME_INDEX = '(time index)';
export const PALETTES = ['Plotly', 'D3', 'Set1', 'Set2', 'Set3', 'Pastel', 'Bold', 'Dark24', 'Light24'];
export const TIMEZONES = ['UTC', 'Europe/Lisbon', 'Europe/London', 'US/Eastern', 'US/Pacific', 'Asia/Tokyo'];
export const BIN_MODES = ['Individual values', 'Number of bins', 'Bin width'];
export const PLOT_TYPES = ['Trends', 'Distribution', 'Scatter', 'Rolling window', 'Derivatives'];
export const MAX_SCATTER_PANELS = 6;
export const MAX_DISTRIBUTION_PANELS = 6;
export const MAX_ANALYSIS_PANELS = 6;
export const MAX_TREND_SUBPLOTS = 6;
export const BIN_SCOPES = ['Filtered data', 'Full dataset'];
export const COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds'];
export const MARKER_PRESETS = ['Core Shapes', 'Crosshair Set', 'Directional', 'Starburst', 'Minimal Dots'];
const enumSpec = (...choices) => ({kind: 'string', choices});
const text = {kind: 'string'};
const bool = {kind: 'bool'};
const strings = {kind: 'strings'};
const int = (minimum, maximum) => ({kind: 'int', minimum, maximum});
const number = (minimum, maximum, exclusiveMinimum = false) => ({kind: 'number', minimum, maximum, exclusiveMinimum});
const specs = {
  plot_type: enumSpec(...PLOT_TYPES, 'Grouped scatter'), overview_cols: strings,
  scatter_panels: {kind: 'scatter-panels'},
  distribution_panels: {kind: 'distribution-panels'},
  trend_panels: {kind: 'analysis-panels'}, rolling_panels: {kind: 'analysis-panels'}, derivative_panels: {kind: 'analysis-panels'},
  comparison_link_x: bool, comparison_link_y: bool, distribution_shared_bins: bool,
  setup_time_mode: enumSpec('Absolute time', 'Relative time'), setup_index: text, setup_dt_col: text,
  setup_profile: enumSpec('Auto-detect', 'DMY (dd/mm/yyyy)', 'MDY (mm/dd/yyyy)', 'YMD (yyyy-mm-dd)', 'Unix epoch (seconds)', 'Unix epoch (milliseconds)', 'Custom strptime'),
  setup_format: text, setup_tz_mode: enumSpec('Treat as naive (no tz)', 'Assume timezone and localize', 'Input already has timezone'),
  setup_assume_tz: enumSpec(...TIMEZONES, 'Custom...'), setup_out_tz: enumSpec('(keep)', ...TIMEZONES, 'Custom...'),
  setup_custom_tz: text, setup_target_tz: text,
  mlc_n_subplots: int(1, 6), mlc_shape: enumSpec('linear', 'spline', 'hv', 'vh', 'hvh', 'vhv'), mlc_width: int(1, 8), mlc_ytype: enumSpec('linear', 'log'), mlc_maxpts: int(1000, 100000),
  hist_cols_main: strings, hist_binmode: enumSpec('Auto (width)', 'Fixed width', 'Target #bins'), hist_width_input: number(1e-12), hist_bins_main: int(10, 200),
  sc_x: text, sc_y: text, sc_group: text, sc_cmap_cont: enumSpec('Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds'), sc_cmap_disc: enumSpec(...PALETTES),
  sc2_x: text, sc2_y: text, sc2_g1: text, sc2_g2: text, sc2_palette: enumSpec(...PALETTES), sc2_marker_preset: enumSpec('Core Shapes', 'Crosshair Set', 'Directional', 'Starburst', 'Minimal Dots'), sc2_marker_custom: text,
  rw_y: text, rw_window: int(2, 200000), rw_msize: int(3, 16), rw_mop: number(.2, 1), rw_maxpts: int(1000, 200000),
  der_y: text, der_tunit: enumSpec('seconds', 'minutes', 'hours', 'days', 'none (sample index)'), der_xmode: enumSpec('index step (1)', 'use numeric index values'), der_lw: int(1, 8), der_smooth: int(1, 9999), der_maxpts: int(1000, 200000), flt2_n: int(0, 10),
};
for (const key of ['mlc_markers', 'mlc_rangeslider', 'mlc_unified', 'mlc_spikes', 'rw_include_current', 'rw_full', 'rw_band', 'rw_unified', 'rw_spikes', 'rw_rangeslider', 'der_markers', 'der_unified', 'der_rangeslider', 'der_spikes']) specs[key] = bool;
for (const prefix of ['sc', 'sc2']) {
  specs[`${prefix}_maxpts`] = int(1000, 200000);
  specs[`${prefix}_msize`] = int(3, 20);
  specs[`${prefix}_mop`] = number(.2, 1);
}
for (const prefix of ['sc_color', 'sc2_color', 'sc2_marker']) {
  specs[`${prefix}_bin_mode`] = enumSpec(...BIN_MODES);
  specs[`${prefix}_bin_count`] = int(1, 100);
  specs[`${prefix}_bin_width`] = number(0, undefined, true);
}
const dynamicSpecs = [
  [/^mlc_cols_[0-5]$/, strings], [/^flt2_logic_[0-9]$/, enumSpec('AND', 'OR')], [/^flt2_var_[0-9]$/, text],
  [/^flt2_use_(?:low|up)dt_(?:idx|col)_[0-9]$/, bool], [/^flt2_(?:low|up)_date_(?:idx|col)_[0-9]$/, {kind: 'date'}], [/^flt2_(?:low|up)_time_(?:idx|col)_[0-9]$/, {kind: 'time'}],
  [/^flt2_(?:low|up)_num_[0-9]$/, text], [/^flt2_op_[0-9]$/, enumSpec('is one of', 'is not one of', 'contains', 'not contains', 'starts with', 'ends with', 'regex')], [/^flt2_vals_[0-9]$/, strings], [/^flt2_(?:vals_csv|patt)_[0-9]$/, text], [/^flt2_(?:case|na)_[0-9]$/, bool],
];
const isObject = value => value !== null && typeof value === 'object' && [Object.prototype, null].includes(Object.getPrototypeOf(value));
const panelSpecs = {
  title: {...text, maxLength: 80}, x: text, y: text, color: text, marker: text,
  colorMode: enumSpec('auto', 'continuous', 'discrete'),
  colorBinMode: enumSpec(...BIN_MODES), colorBinCount: int(1, 100), colorBinWidth: number(0, undefined, true),
  markerBinMode: enumSpec(...BIN_MODES), markerBinCount: int(1, 100), markerBinWidth: number(0, undefined, true),
  colorBinScope: enumSpec(...BIN_SCOPES), markerBinScope: enumSpec(...BIN_SCOPES),
  palette: enumSpec(...PALETTES), colormap: enumSpec(...COLORMAPS), markerPreset: enumSpec(...MARKER_PRESETS),
  customSymbols: text, size: int(3, 20), opacity: number(.2, 1),
};
const distributionPanelSpecs = {
  title: {...text, maxLength: 80}, columns: strings,
  binMode: enumSpec('Auto (width)', 'Fixed width', 'Target #bins'), binCount: int(10, 200), binWidth: number(1e-12),
  binScope: enumSpec(...BIN_SCOPES),
};
const trendPanelSpecs = {
  title: {...text, maxLength: 80}, subplots: {kind: 'trend-subplots'}, columns: strings, shape: specs.mlc_shape, width: specs.mlc_width, ytype: specs.mlc_ytype,
  markers: bool, rangeslider: bool, unified: bool, spikes: bool,
};
const rollingPanelSpecs = {
  title: {...text, maxLength: 80}, y: text, window: specs.rw_window, includeCurrent: bool, requireFull: bool, band: bool,
  unified: bool, rangeslider: bool, spikes: bool, size: specs.rw_msize, opacity: specs.rw_mop,
};
const derivativePanelSpecs = {
  title: {...text, maxLength: 80}, y: text, timeUnit: specs.der_tunit, xMode: specs.der_xmode, smoothWindow: specs.der_smooth,
  markers: bool, width: specs.der_lw, unified: bool, rangeslider: bool, spikes: bool,
};
const numericColumns = schema => schemaColumns(schema).filter(column => columnKind(column) === 'numeric').map(column => column.name);
const hasTimeColumn = schema => schemaColumns(schema).some(column => columnKind(column) === 'datetime');

export function defaultTrendSubplot(schema = {}, index = 0) {
  const numeric = numericColumns(schema), first = numeric[index] || numeric[0];
  return {columns: first ? [first] : []};
}
export function defaultTrendPanel(schema = {}, index = 0) {
  return {title: `Trends ${index + 1}`, subplots: [defaultTrendSubplot(schema, index)], shape: 'linear', width: 2, ytype: 'linear', markers: false, rangeslider: true, unified: true, spikes: true};
}
/** Runtime rows always own their arrays; an explicit empty row stays empty. */
export function trendSubplots(panel = {}, schema = {}) {
  const rows = Array.isArray(panel.subplots) ? panel.subplots : Object.hasOwn(panel, 'columns') ? [{columns: panel.columns}] : [{}];
  return rows.map((row, index) => ({columns: [...(row.columns ?? defaultTrendSubplot(schema, index).columns)]}));
}
export function defaultRollingPanel(schema = {}, index = 0) {
  return {title: `Rolling window ${index + 1}`, y: numericColumns(schema)[0] || '', window: 50, includeCurrent: true, requireFull: false, band: true, unified: true, rangeslider: hasTimeColumn(schema), spikes: true, size: 7, opacity: .7};
}
export function defaultDerivativePanel(schema = {}, index = 0) {
  return {title: `Derivatives ${index + 1}`, y: numericColumns(schema)[0] || '', timeUnit: 'seconds', xMode: 'index step (1)', smoothWindow: 1, markers: true, width: 2, unified: true, rangeslider: hasTimeColumn(schema), spikes: true};
}

const trendLegacyFields = {shape: 'mlc_shape', width: 'mlc_width', ytype: 'mlc_ytype', markers: 'mlc_markers', rangeslider: 'mlc_rangeslider', unified: 'mlc_unified', spikes: 'mlc_spikes'};
const rollingLegacyFields = {y: 'rw_y', window: 'rw_window', includeCurrent: 'rw_include_current', requireFull: 'rw_full', band: 'rw_band', unified: 'rw_unified', rangeslider: 'rw_rangeslider', spikes: 'rw_spikes', size: 'rw_msize', opacity: 'rw_mop'};
const derivativeLegacyFields = {y: 'der_y', timeUnit: 'der_tunit', xMode: 'der_xmode', smoothWindow: 'der_smooth', markers: 'der_markers', width: 'der_lw', unified: 'der_unified', rangeslider: 'der_rangeslider', spikes: 'der_spikes'};
function mappedLegacyPanel(settings, defaults, fields) {
  return {...defaults, ...Object.fromEntries(Object.entries(fields).filter(([, key]) => settings[key] !== undefined).map(([field, key]) => [field, settings[key]]))};
}
function copyAnalysisPanels(panels, schema, create) {
  return panels.map((panel, index) => {
    const next = {...create(schema, index), ...panel};
    if (Array.isArray(next.columns)) next.columns = [...next.columns];
    return next;
  });
}
function legacyTrendPanels(settings, schema = {}, preserveMissing = false) {
  const indices = Object.keys(settings).flatMap(key => /^mlc_cols_([0-5])$/.test(key) ? [Number(key.slice(-1))] : []);
  const count = settings.mlc_n_subplots ?? (indices.length ? Math.max(...indices) + 1 : 1);
  const panel = mappedLegacyPanel(settings, defaultTrendPanel(schema), trendLegacyFields);
  panel.subplots = Array.from({length: count}, (_, index) => {
    const key = `mlc_cols_${index}`;
    return Object.hasOwn(settings, key) ? {columns: [...settings[key]]} : preserveMissing ? {} : defaultTrendSubplot(schema, index);
  });
  return [panel];
}
export function trendPanels(settings = {}, schema = {}) {
  if (!Array.isArray(settings.trend_panels)) return legacyTrendPanels(settings, schema);
  return settings.trend_panels.map((panel, index) => {
    const {columns, ...fields} = panel;
    const defaults = defaultTrendPanel(schema, index);
    return {...defaults, ...fields, subplots: trendSubplots(Object.hasOwn(panel, 'subplots') || Object.hasOwn(panel, 'columns') ? panel : defaults, schema)};
  });
}
export function rollingPanels(settings = {}, schema = {}) {
  if (Array.isArray(settings.rolling_panels)) return copyAnalysisPanels(settings.rolling_panels, schema, defaultRollingPanel);
  return [mappedLegacyPanel(settings, defaultRollingPanel(schema), rollingLegacyFields)];
}
export function derivativePanels(settings = {}, schema = {}) {
  if (Array.isArray(settings.derivative_panels)) return copyAnalysisPanels(settings.derivative_panels, schema, defaultDerivativePanel);
  return [mappedLegacyPanel(settings, defaultDerivativePanel(schema), derivativeLegacyFields)];
}

const analysisPanelDefinitions = {
  trend_panels: {label: 'Trends', fields: trendPanelSpecs, create: defaultTrendPanel, read: trendPanels, prefix: 'mlc'},
  rolling_panels: {label: 'Rolling window', fields: rollingPanelSpecs, create: defaultRollingPanel, read: rollingPanels, prefix: 'rw'},
  derivative_panels: {label: 'Derivatives', fields: derivativePanelSpecs, create: defaultDerivativePanel, read: derivativePanels, prefix: 'der'},
};
function validateAnalysisPanels(key, value, warnings) {
  const definition = analysisPanelDefinitions[key];
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_ANALYSIS_PANELS) throw new Error(`Invalid setting '${key}': use between 1 and ${MAX_ANALYSIS_PANELS} ${definition.label.toLowerCase()} panels.`);
  return value.map((panel, index) => {
    if (!isObject(panel)) throw new Error(`Invalid setting '${key}[${index}]': expected a settings object.`);
    const clean = {};
    for (const [field, value] of Object.entries(panel)) {
      if (!Object.hasOwn(definition.fields, field)) warnings.push(`Ignored unsupported setting: ${key}[${index}].${field.slice(0, 100)}.`);
      else if (key === 'trend_panels' && field === 'subplots') clean.subplots = validateTrendSubplots(value, `trend_panels[${index}].subplots`, warnings);
      else if (key === 'trend_panels' && field === 'columns') {
        if (!Object.hasOwn(panel, 'subplots')) clean.subplots = [{columns: validateValue(`trend_panels[${index}].columns`, value, strings)}];
      }
      else clean[field] = validateValue(`${key}[${index}].${field}`, value, definition.fields[field]);
    }
    // Missing selection is distinct from an intentionally cleared [] until a schema is known.
    return {...definition.create({}, index), ...(key === 'trend_panels' ? {subplots: [{}]} : {}), ...clean};
  });
}
function validateTrendSubplots(value, key, warnings) {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_TREND_SUBPLOTS) throw new Error(`Invalid setting '${key}': use between 1 and ${MAX_TREND_SUBPLOTS} subplots.`);
  return value.map((subplot, index) => {
    if (!isObject(subplot)) throw new Error(`Invalid setting '${key}[${index}]': expected a settings object.`);
    const clean = {};
    for (const [field, value] of Object.entries(subplot)) {
      if (field === 'columns') clean.columns = validateValue(`${key}[${index}].columns`, value, strings);
      else warnings.push(`Ignored unsupported setting: ${key}[${index}].${field.slice(0, 100)}.`);
    }
    return clean;
  });
}

export function defaultDistributionPanel(schema = {}, index = 0) {
  const numeric = schemaColumns(schema).filter(column => columnKind(column) === 'numeric').map(column => column.name);
  return {title: `Distribution ${index + 1}`, columns: numeric.slice(0, 1), binMode: 'Auto (width)', binCount: 50, binWidth: 1, binScope: 'Filtered data'};
}

function legacyDistributionPanel(settings, schema) {
  const defaults = defaultDistributionPanel(schema);
  return {
    ...defaults, columns: [...(settings.hist_cols_main ?? defaults.columns)],
    binMode: settings.hist_binmode ?? defaults.binMode,
    binCount: settings.hist_bins_main ?? defaults.binCount, binWidth: settings.hist_width_input ?? defaults.binWidth,
  };
}

/** Return independent runtime settings, including transient values from the editor. */
export function distributionPanels(settings = {}, schema = {}) {
  if (Array.isArray(settings.distribution_panels)) {
    return settings.distribution_panels.map((panel, index) => {
      const defaults = defaultDistributionPanel(schema, index);
      return {...defaults, ...panel, columns: [...(panel.columns ?? defaults.columns)]};
    });
  }
  return [legacyDistributionPanel(settings, schema)];
}

function validateDistributionPanels(value, warnings) {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_DISTRIBUTION_PANELS) throw new Error(`Invalid setting 'distribution_panels': use between 1 and ${MAX_DISTRIBUTION_PANELS} distribution panels.`);
  return value.map((panel, index) => {
    if (!isObject(panel)) throw new Error(`Invalid setting 'distribution_panels[${index}]': expected a settings object.`);
    const clean = {};
    for (const [key, field] of Object.entries(panel)) {
      if (!Object.hasOwn(distributionPanelSpecs, key)) warnings.push(`Ignored unsupported setting: distribution_panels[${index}].${key.slice(0, 100)}.`);
      else clean[key] = validateValue(`distribution_panels[${index}].${key}`, field, distributionPanelSpecs[key]);
    }
    return {...defaultDistributionPanel({}, index), ...clean};
  });
}

export function defaultScatterPanel(schema = {}, index = 0) {
  const columns = schemaColumns(schema);
  const numeric = columns.filter(column => columnKind(column) === 'numeric').map(column => column.name);
  const all = columns.map(column => column.name);
  return {
    title: `Scatter ${index + 1}`, x: numeric[0] || all[0] || '', y: numeric[1] || numeric[0] || all[1] || all[0] || '',
    color: '(none)', marker: '(none)', colorMode: 'auto',
    colorBinMode: 'Individual values', colorBinCount: 5, colorBinWidth: 1,
    markerBinMode: 'Individual values', markerBinCount: 5, markerBinWidth: 1,
    colorBinScope: 'Filtered data', markerBinScope: 'Filtered data',
    palette: 'Plotly', colormap: 'Viridis', markerPreset: 'Core Shapes', customSymbols: '', size: 7, opacity: .7,
  };
}

function legacyScatterPanel(settings, schema, grouped = false) {
  const defaults = defaultScatterPanel(schema);
  const prefix = grouped ? 'sc2' : 'sc';
  const get = (name, fallback) => settings[name] === undefined ? fallback : settings[name];
  const colorPrefix = grouped ? 'sc2_color' : 'sc_color';
  return {
    ...defaults, x: get(`${prefix}_x`, defaults.x), y: get(`${prefix}_y`, defaults.y),
    color: get(grouped ? 'sc2_g1' : 'sc_group', '(none)'), marker: grouped ? get('sc2_g2', '(none)') : '(none)',
    colorMode: grouped ? 'discrete' : 'auto',
    colorBinMode: get(`${colorPrefix}_bin_mode`, defaults.colorBinMode), colorBinCount: get(`${colorPrefix}_bin_count`, 5), colorBinWidth: get(`${colorPrefix}_bin_width`, 1),
    markerBinMode: grouped ? get('sc2_marker_bin_mode', defaults.markerBinMode) : defaults.markerBinMode,
    markerBinCount: grouped ? get('sc2_marker_bin_count', 5) : 5, markerBinWidth: grouped ? get('sc2_marker_bin_width', 1) : 1,
    palette: get(grouped ? 'sc2_palette' : 'sc_cmap_disc', defaults.palette), colormap: get('sc_cmap_cont', defaults.colormap),
    markerPreset: grouped ? get('sc2_marker_preset', defaults.markerPreset) : defaults.markerPreset,
    customSymbols: grouped ? get('sc2_marker_custom', '') : '', size: get(`${prefix}_msize`, defaults.size), opacity: get(`${prefix}_mop`, defaults.opacity),
  };
}

function hasLegacyScatterSettings(settings, prefix) {
  return Object.keys(settings).some(key => key.startsWith(`${prefix}_`) && key !== `${prefix}_maxpts`);
}

function legacyScatterPanels(settings, schema) {
  const hasSimple = hasLegacyScatterSettings(settings, 'sc');
  const hasGrouped = hasLegacyScatterSettings(settings, 'sc2');
  const groupedFirst = settings.plot_type === 'Grouped scatter' || !hasSimple && hasGrouped;
  const first = legacyScatterPanel(settings, schema, groupedFirst);
  const panels = [first];
  if (hasSimple && hasGrouped) {
    const second = legacyScatterPanel(settings, schema, !groupedFirst);
    const signature = panel => JSON.stringify({...panel, title: '', colorMode: panel.color === '(none)' ? 'auto' : panel.colorMode});
    if (signature(first) !== signature(second)) panels.push(second);
  }
  return panels.map((panel, index) => ({...panel, title: `Scatter ${index + 1}`}));
}

/** Read runtime panels without mutating state or rejecting transient editor input. */
export function scatterPanels(settings = {}, schema = {}) {
  if (Array.isArray(settings.scatter_panels)) {
    return settings.scatter_panels.map((panel, index) => ({...defaultScatterPanel(schema, index), ...panel}));
  }
  return legacyScatterPanels(settings, schema);
}

function validateScatterPanels(value, warnings) {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_SCATTER_PANELS) throw new Error(`Invalid setting 'scatter_panels': use between 1 and ${MAX_SCATTER_PANELS} scatter panels.`);
  return value.map((panel, index) => {
    if (!isObject(panel)) throw new Error(`Invalid setting 'scatter_panels[${index}]': expected a settings object.`);
    const clean = {};
    for (const [key, field] of Object.entries(panel)) {
      if (!Object.hasOwn(panelSpecs, key)) warnings.push(`Ignored unsupported setting: scatter_panels[${index}].${key.slice(0, 100)}.`);
      else clean[key] = validateValue(`scatter_panels[${index}].${key}`, field, panelSpecs[key]);
    }
    return {...defaultScatterPanel({}, index), ...clean};
  });
}
export function settingSpec(key) {
  if (typeof key !== 'string') return undefined;
  return Object.hasOwn(specs, key) ? specs[key] : dynamicSpecs.find(([pattern]) => pattern.test(key))?.[1];
}
export const isConfigKey = key => Boolean(settingSpec(key));
function validDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split('-').map(Number);
  if (year < 1 || month < 1 || month > 12 || day < 1) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  return day <= [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1];
}
function validateValue(key, value, spec) {
  const error = message => { throw new Error(`Invalid setting '${key}': ${message}`); };
  if (spec.kind === 'bool') {
    if (typeof value !== 'boolean') error('expected true or false.');
  } else if (spec.kind === 'int' || spec.kind === 'number') {
    if (typeof value !== 'number') error(`expected a ${spec.kind === 'int' ? 'whole number' : 'number'}.`);
    if (!Number.isFinite(value)) error('numbers must be finite.');
    if (spec.kind === 'int' && !Number.isSafeInteger(value)) error('expected a safe whole number.');
    if (spec.minimum !== undefined && (value < spec.minimum || (spec.exclusiveMinimum && value === spec.minimum))) error(`use a value ${spec.exclusiveMinimum ? 'greater than' : 'at least'} ${spec.minimum}.`);
    if (spec.maximum !== undefined && value > spec.maximum) error(`use a value no greater than ${spec.maximum}.`);
  } else if (spec.kind === 'strings') {
    if (!Array.isArray(value) || value.length > 2000) error('expected a list of at most 2000 names or values.');
    return value.map(item => validateValue(key, item, text));
  } else if (spec.kind === 'string') {
    const maxLength = spec.maxLength ?? 10000;
    if (typeof value !== 'string' || value.length > maxLength) error(`expected text of at most ${maxLength} characters.`);
    if (spec.choices && !spec.choices.includes(value)) error(`choose one of ${spec.choices.join(', ')}.`);
  } else if (spec.kind === 'date') {
    if (typeof value !== 'string' || !validDate(value)) error('expected an ISO date such as 2025-01-31.');
  } else if (spec.kind === 'time') {
    const parts = typeof value === 'string' && /^(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$/.exec(value);
    if (!parts || +parts[1] > 23 || +parts[2] > 59 || +(parts[3] || 0) > 59) error('expected a local ISO time such as 14:30:00.');
  }
  return value;
}
export function validateSettings(settings) {
  if (!isObject(settings)) throw new Error('Configuration settings must be a JSON object.');
  if (Object.keys(settings).length > 512) throw new Error('A configuration can contain at most 512 settings.');
  const clean = {}, warnings = [];
  for (const [key, value] of Object.entries(settings)) {
    const spec = settingSpec(key);
    if (!spec) warnings.push(`Ignored unsupported setting: ${key.slice(0, 100)}.`);
    else if (spec.kind === 'scatter-panels') clean[key] = validateScatterPanels(value, warnings);
    else if (spec.kind === 'distribution-panels') clean[key] = validateDistributionPanels(value, warnings);
    else if (spec.kind === 'analysis-panels') clean[key] = validateAnalysisPanels(key, value, warnings);
    else clean[key] = validateValue(key, value, spec);
  }
  if (!Object.hasOwn(clean, 'scatter_panels') && (hasLegacyScatterSettings(clean, 'sc') || hasLegacyScatterSettings(clean, 'sc2'))) clean.scatter_panels = legacyScatterPanels(clean);
  if (!Object.hasOwn(clean, 'distribution_panels') && Object.keys(clean).some(key => key.startsWith('hist_'))) clean.distribution_panels = [legacyDistributionPanel(clean)];
  for (const [key, definition] of Object.entries(analysisPanelDefinitions)) {
    if (!Object.hasOwn(clean, key) && Object.keys(clean).some(field => field.startsWith(`${definition.prefix}_`) && field !== `${definition.prefix}_maxpts`)) clean[key] = key === 'trend_panels' ? legacyTrendPanels(clean, {}, true) : definition.read(clean);
  }
  if (clean.plot_type === 'Grouped scatter') clean.plot_type = 'Scatter';
  return {settings: clean, warnings};
}
/** JSON.parse loses duplicate fields. A small strict reader rejects ambiguity before applying any settings. */
function parseStrictJSON(text) {
  let pos = 0;
  function whitespace() { while (/[ \t\r\n]/.test(text[pos] || '') && pos < text.length) pos++; }
  function value(depth = 0) {
    if (depth > 64) throw new Error('Configuration JSON is nested too deeply.');
    whitespace();
    const ch = text[pos];
    if (ch === '{') {
      pos++; whitespace(); const entries = [], keys = new Set();
      if (text[pos] === '}') {pos++; return {};}
      while (true) {
        whitespace(); if (text[pos] !== '"') throw new Error('Invalid JSON object.');
        const key = string(); whitespace();
        if (keys.has(key)) throw new Error(`Duplicate configuration field: ${key.slice(0, 100)}.`);
        keys.add(key); if (text[pos++] !== ':') throw new Error('Invalid JSON object.');
        entries.push([key, value(depth + 1)]); whitespace();
        const delimiter = text[pos++]; if (delimiter === '}') break;
        if (delimiter !== ',') throw new Error('Invalid JSON object.');
      }
      return Object.fromEntries(entries);
    }
    if (ch === '[') {
      pos++; whitespace(); const result = []; if (text[pos] === ']') {pos++; return result;}
      while (true) {result.push(value(depth + 1)); whitespace(); const delimiter = text[pos++]; if (delimiter === ']') break; if (delimiter !== ',') throw new Error('Invalid JSON array.');}
      return result;
    }
    if (ch === '"') return string();
    for (const [token, result] of [['true', true], ['false', false], ['null', null]]) {
      if (text.startsWith(token, pos)) {pos += token.length; return result;}
    }
    const numeric = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(pos));
    if (!numeric) throw new Error('This is not a valid JSON configuration file.');
    pos += numeric[0].length;
    const number = Number(numeric[0]); if (!Number.isFinite(number)) throw new Error('Configuration numbers must be finite.');
    return number;
  }
  function string() {
    const start = pos++;
    while (pos < text.length) {
      const ch = text[pos++];
      if (ch === '\\') {pos++; continue;}
      if (ch === '"') return JSON.parse(text.slice(start, pos));
    }
    throw new Error('Unterminated JSON text.');
  }
  const result = value(); whitespace(); if (pos !== text.length) throw new Error('Unexpected content after configuration.'); return result;
}
export function decodeConfig(data) {
  let raw;
  if (typeof data === 'string') raw = new TextEncoder().encode(data);
  else if (data instanceof ArrayBuffer) raw = new Uint8Array(data);
  else if (ArrayBuffer.isView(data)) raw = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  else throw new Error('Choose a JSON configuration file.');
  if (raw.length > MAX_CONFIG_BYTES) throw new Error('This configuration is too large. The limit is 256 KB.');
  let document;
  try {document = parseStrictJSON(new TextDecoder('utf-8', {fatal: true}).decode(raw).replace(/^\uFEFF/, ''));}
  catch (error) {throw new Error(`Could not read configuration: ${error.message}`);}
  if (!isObject(document)) throw new Error('A configuration must be a JSON object.');
  if (document.app !== CONFIG_APP) throw new Error('This file is not a TSPlotter configuration.');
  if (![1, 2, 3, CONFIG_VERSION].includes(document.version)) throw new Error(`Unsupported configuration version. This app supports versions 1, 2, 3 and ${CONFIG_VERSION}.`);
  if (!Object.hasOwn(document, 'settings')) throw new Error('The configuration is missing its settings object.');
  return validateSettings(document.settings);
}
export function encodeConfig(state) {
  const snapshot = Object.fromEntries(Object.entries(state).filter(([key]) => isConfigKey(key)));
  const keepBinValue = (object, key, spec, active) => {
    if (active || !Object.hasOwn(object, key)) return;
    try {validateValue(key, object[key], spec);} catch {delete object[key];}
  };
  if (Array.isArray(snapshot.distribution_panels)) snapshot.distribution_panels = snapshot.distribution_panels.map((panel, index) => {
    if (!isObject(panel)) return panel;
    const clean = {...panel};
    const sharedFollower = snapshot.distribution_shared_bins === true && index > 0;
    keepBinValue(clean, 'binWidth', distributionPanelSpecs.binWidth, !sharedFollower && clean.binMode === 'Fixed width');
    keepBinValue(clean, 'binCount', distributionPanelSpecs.binCount, !sharedFollower && clean.binMode === 'Target #bins');
    return clean;
  });
  if (Array.isArray(snapshot.scatter_panels)) snapshot.scatter_panels = snapshot.scatter_panels.map(panel => {
    if (!isObject(panel)) return panel;
    const clean = {...panel};
    for (const prefix of ['color', 'marker']) {
      const grouping = Boolean(clean[prefix] && clean[prefix] !== '(none)');
      keepBinValue(clean, `${prefix}BinWidth`, panelSpecs[`${prefix}BinWidth`], grouping && clean[`${prefix}BinMode`] === 'Bin width');
      keepBinValue(clean, `${prefix}BinCount`, panelSpecs[`${prefix}BinCount`], grouping && clean[`${prefix}BinMode`] === 'Number of bins');
    }
    return clean;
  });
  // Retain valid preferences for every mode, while ignoring unfinished unused inputs.
  for (const prefix of ['sc_color', 'sc2_color', 'sc2_marker']) {
    keepBinValue(snapshot, `${prefix}_bin_width`, specs[`${prefix}_bin_width`], !snapshot.scatter_panels && snapshot[`${prefix}_bin_mode`] === 'Bin width');
    keepBinValue(snapshot, `${prefix}_bin_count`, specs[`${prefix}_bin_count`], !snapshot.scatter_panels && snapshot[`${prefix}_bin_mode`] === 'Number of bins');
  }
  keepBinValue(snapshot, 'hist_width_input', specs.hist_width_input, !snapshot.distribution_panels && snapshot.hist_binmode === 'Fixed width');
  keepBinValue(snapshot, 'hist_bins_main', specs.hist_bins_main, !snapshot.distribution_panels && snapshot.hist_binmode === 'Target #bins');
  const {settings} = validateSettings(snapshot);
  const json = JSON.stringify({app: CONFIG_APP, version: CONFIG_VERSION, settings}, null, 2);
  if (new TextEncoder().encode(json).length > MAX_CONFIG_BYTES) throw new Error('This configuration is too large. Save fewer selected values.');
  return json;
}
function columnKind(column) {
  if (column.kind === 'numeric' || column.kind === 'number') return 'numeric';
  if (column.kind === 'datetime' || column.kind === 'temporal' || /^(DATE|TIMESTAMP|TIME)/i.test(column.type || '')) return 'datetime';
  return 'categorical';
}
const schemaColumns = schema => Array.isArray(schema) ? schema : schema?.columns || [];
export function normalizeConfiguration(settings, schema) {
  const {settings: clean, warnings} = validateSettings(settings);
  const columns = schemaColumns(schema);
  const names = new Set(columns.map(column => column.name));
  const numeric = new Set(columns.filter(column => columnKind(column) === 'numeric').map(column => column.name));
  const timeIndex = schema?.timeIndex;
  if (clean.setup_dt_col !== undefined && !names.has(clean.setup_dt_col) && !(timeIndex && clean.setup_dt_col === TIME_INDEX) && clean.setup_dt_col !== '') {
    const missing = clean.setup_dt_col; delete clean.setup_dt_col;
    if (clean.setup_time_mode === 'Absolute time') {
      clean.setup_time_mode = 'Relative time'; clean.setup_index = ROW_INDEX;
      warnings.push(`Saved time column '${missing}' is unavailable. Relative time with row numbers is being used.`);
    } else warnings.push(`Skipped unavailable inactive time column '${missing}'.`);
  }
  if (clean.setup_index !== undefined && !names.has(clean.setup_index) && ![ROW_INDEX, '__row__'].includes(clean.setup_index)) {
    warnings.push(`Saved relative X-axis column '${clean.setup_index}' is unavailable and has been reset to row numbers.`); clean.setup_index = ROW_INDEX;
  }
  for (const key of ['sc_x', 'sc_y', 'sc_group', 'sc2_x', 'sc2_y', 'sc2_g1', 'sc2_g2', 'rw_y', 'der_y']) {
    const allowed = ['rw_y', 'der_y'].includes(key) ? numeric : names;
    if (clean[key] !== undefined && !allowed.has(clean[key]) && !(['sc_group', 'sc2_g1', 'sc2_g2'].includes(key) && clean[key] === '(none)')) {
      warnings.push(`Saved column '${clean[key]}' for '${key}' is unavailable. Choose a replacement.`); delete clean[key];
    }
  }
  for (const key of ['hist_cols_main', 'overview_cols', ...Array.from({length: 6}, (_, i) => `mlc_cols_${i}`)]) {
    if (!Array.isArray(clean[key])) continue;
    const missing = clean[key].filter(name => !numeric.has(name));
    if (missing.length) warnings.push(`Unavailable numeric columns were skipped: ${missing.join(', ')}.`);
    clean[key] = clean[key].filter(name => numeric.has(name));
  }
  if (clean.distribution_panels) clean.distribution_panels = clean.distribution_panels.map((panel, index) => {
    const missing = panel.columns.filter(name => !numeric.has(name));
    let selected = panel.columns.filter(name => numeric.has(name));
    const needsDefault = !selected.length;
    if (!selected.length) selected = defaultDistributionPanel(schema, index).columns;
    if (missing.length) warnings.push(`Distribution ${index + 1}: unavailable numeric columns were skipped: ${missing.join(', ')}.${needsDefault && selected.length ? ' The first numeric signal is selected.' : ''}`);
    return {...panel, columns: selected};
  });
  for (const [key, definition] of Object.entries(analysisPanelDefinitions)) {
    if (!clean[key]) continue;
    clean[key] = clean[key].map((panel, index) => {
      const defaults = definition.create(schema, index);
      if (key === 'trend_panels') {
        const subplots = panel.subplots.map((subplot, row) => {
          const fallback = defaultTrendSubplot(schema, row).columns;
          if (!Object.hasOwn(subplot, 'columns')) return {columns: fallback};
          const missing = subplot.columns.filter(name => !numeric.has(name));
          const selected = subplot.columns.filter(name => numeric.has(name));
          if (missing.length) warnings.push(`Trends ${index + 1}, subplot ${row + 1}: unavailable numeric columns were skipped: ${missing.join(', ')}.${!selected.length && fallback.length ? ' A default signal is selected.' : ''}`);
          return {columns: missing.length && !selected.length ? fallback : selected};
        });
        return {...panel, subplots};
      }
      if (!numeric.has(panel.y)) {
        if (panel.y) warnings.push(`${definition.label} ${index + 1}: saved signal '${panel.y}' is unavailable or nonnumeric. A default signal is selected.`);
        return {...panel, y: defaults.y};
      }
      return {...panel};
    });
  }
  for (const [prefix, selector] of [['sc_color', 'sc_group'], ['sc2_color', 'sc2_g1'], ['sc2_marker', 'sc2_g2']]) {
    // No grouping is an intentional selection. Keep its inactive controls for later use.
    if (clean[selector] === '(none)' || numeric.has(clean[selector])) continue;
    const keys = ['mode', 'count', 'width'].map(suffix => `${prefix}_bin_${suffix}`);
    if (keys.some(key => clean[key] !== undefined)) warnings.push(`Saved numeric bins for '${clean[selector] || selector}' were skipped because its column is missing or nonnumeric.`);
    keys.forEach(key => delete clean[key]);
  }
  if (clean.scatter_panels) clean.scatter_panels = clean.scatter_panels.map((panel, index) => {
    const next = {...panel}, defaults = defaultScatterPanel(schema, index);
    for (const field of ['x', 'y']) {
      if (!names.has(next[field])) {
        if (next[field]) warnings.push(`Scatter ${index + 1}: saved ${field.toUpperCase()} column '${next[field]}' is unavailable. A default signal is selected.`);
        next[field] = defaults[field];
      }
    }
    for (const field of ['color', 'marker']) {
      if (next[field] !== '(none)' && !names.has(next[field])) {
        warnings.push(`Scatter ${index + 1}: saved ${field} column '${next[field]}' is unavailable. Grouping was turned off.`);
        next[field] = '(none)';
      }
      if (next[field] !== '(none)' && !numeric.has(next[field])) {
        if (next[`${field}BinMode`] !== 'Individual values') warnings.push(`Scatter ${index + 1}: numeric ${field} bins were skipped for nonnumeric column '${next[field]}'.`);
        next[`${field}BinMode`] = 'Individual values';
      }
    }
    if (next.colorMode === 'continuous' && next.color !== '(none)' && !numeric.has(next.color)) {
      next.colorMode = 'auto';
      warnings.push(`Scatter ${index + 1}: continuous color requires a numeric column. Automatic color was selected.`);
    }
    return next;
  });
  const fields = Object.fromEntries(Object.entries(clean).filter(([key]) => key.startsWith('flt2_')));
  for (const key of Object.keys(fields)) delete clean[key];
  let count = 0;
  for (let i = 0; i < (fields.flt2_n || 0); i++) {
    const variable = fields[`flt2_var_${i}`];
    const column = columns.find(column => column.name === variable);
    const isIndex = variable === FILTER_TIME_INDEX && (timeIndex || clean.setup_time_mode === 'Absolute time' && names.has(clean.setup_dt_col));
    if (!column && !isIndex) {warnings.push(`Skipped saved filter ${i + 1}: column '${variable}' is unavailable.`); continue;}
    const kind = isIndex ? 'datetime' : columnKind(column);
    const allowed = new Set(kind === 'datetime' ? ['logic', 'var', ...['use_lowdt', 'use_updt', 'low_date', 'up_date', 'low_time', 'up_time'].map(key => `${key}_${isIndex ? 'idx' : 'col'}`)] : kind === 'numeric' ? ['logic', 'var', 'low_num', 'up_num'] : ['logic', 'var', 'op', 'vals', 'vals_csv', 'patt', 'case', 'na']);
    const clause = Object.entries(fields).flatMap(([key, value]) => {
      const match = new RegExp(`^flt2_(.+)_${i}$`).exec(key);
      return match && allowed.has(match[1]) ? [[match[1], value]] : [];
    });
    if (!clause.some(([field]) => !['var', 'logic'].includes(field))) {warnings.push(`Skipped saved filter ${i + 1}: '${variable}' has a different data type in this dataset.`); continue;}
    for (const [field, value] of clause) clean[`flt2_${field}_${count}`] = value;
    count++;
  }
  clean.flt2_n = count;
  return {settings: clean, warnings};
}
function numericBound(value, label) {
  if (value === undefined || String(value).trim() === '') return undefined;
  const raw = String(value).trim();
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw)) throw new Error(`${label} must be a finite number.`);
  const number = Number(raw);
  if (!Number.isFinite(number)) throw new Error(`${label} must be a finite number.`);
  return number;
}
export function filtersFromSettings(settings, schema) {
  const columns = schemaColumns(schema);
  const result = [];
  for (let i = 0; i < (settings.flt2_n || 0); i++) {
    const get = field => settings[`flt2_${field}_${i}`];
    const variable = get('var'); if (!variable) continue;
    const isTimeIndex = variable === FILTER_TIME_INDEX;
    const column = isTimeIndex ? schema?.timeIndex || (settings.setup_time_mode === 'Absolute time' ? settings.setup_dt_col : undefined) : variable;
    if (!column) continue;
    const definition = columns.find(item => item.name === column);
    if (!definition && column !== '__row__') continue;
    const kind = isTimeIndex ? 'datetime' : definition ? columnKind(definition) : 'numeric';
    const condition = {logic: get('logic') || 'AND', column, kind};
    if (kind === 'numeric') {
      condition.low = numericBound(get('low_num'), `Filter ${i + 1} lower bound`);
      condition.up = numericBound(get('up_num'), `Filter ${i + 1} upper bound`);
      if (condition.low === undefined && condition.up === undefined) continue;
    } else if (kind === 'datetime') {
      const suffix = isTimeIndex ? 'idx' : 'col';
      for (const [side, target] of [['low', 'low'], ['up', 'up']]) {
        if (!get(`use_${side}dt_${suffix}`)) continue;
        const date = get(`${side}_date_${suffix}`); if (!date) continue;
        condition[target] = `${date}T${get(`${side}_time_${suffix}`) || (side === 'low' ? '00:00:00' : '23:59:59.999999')}`;
      }
      if (condition.low === undefined && condition.up === undefined) continue;
    } else {
      condition.op = get('op') || 'is one of';
      condition.values = ['is one of', 'is not one of'].includes(condition.op)
        ? get('vals') || (get('vals_csv') || '').split(',').map(value => value.trim()).filter(Boolean)
        : [get('patt') || ''];
      condition.caseSensitive = get('case') || false;
      condition.naMatch = get('na') || false;
      if (!condition.values.length || condition.values.length === 1 && condition.values[0] === '') continue;
    }
    result.push(condition);
  }
  return result;
}
export function defaultSettings(schema = {}) {
  const columns = schemaColumns(schema);
  const numeric = columns.filter(column => columnKind(column) === 'numeric').map(column => column.name);
  const time = columns.find(column => columnKind(column) === 'datetime')?.name;
  return {
    plot_type: 'Trends', overview_cols: numeric.slice(0, 4),
    scatter_panels: [defaultScatterPanel(schema)],
    distribution_panels: [defaultDistributionPanel(schema)],
    trend_panels: [defaultTrendPanel(schema)], rolling_panels: [defaultRollingPanel(schema)], derivative_panels: [defaultDerivativePanel(schema)],
    comparison_link_x: false, comparison_link_y: false, distribution_shared_bins: false,
    setup_time_mode: time ? 'Absolute time' : 'Relative time', setup_index: ROW_INDEX, setup_dt_col: time || '', setup_profile: 'Auto-detect', setup_format: '', setup_tz_mode: 'Treat as naive (no tz)', setup_assume_tz: 'UTC', setup_out_tz: '(keep)', setup_custom_tz: '', setup_target_tz: '',
    mlc_n_subplots: 1, ...Object.fromEntries(Array.from({length: 6}, (_, i) => [`mlc_cols_${i}`, numeric[i] ? [numeric[i]] : []])), mlc_shape: 'linear', mlc_width: 2, mlc_ytype: 'linear', mlc_markers: false, mlc_rangeslider: true, mlc_unified: true, mlc_spikes: true,
    hist_cols_main: numeric.slice(0, 1), hist_binmode: 'Auto (width)', hist_width_input: 1, hist_bins_main: 50,
    sc_x: numeric[0] || '', sc_y: numeric[1] || numeric[0] || '', sc_group: '(none)', sc_msize: 7, sc_mop: .7, sc_cmap_cont: 'Viridis', sc_cmap_disc: 'Plotly', sc_color_bin_mode: 'Individual values', sc_color_bin_count: 5, sc_color_bin_width: 1,
    sc2_x: numeric[0] || '', sc2_y: numeric[1] || numeric[0] || '', sc2_g1: '(none)', sc2_g2: '(none)', sc2_msize: 7, sc2_mop: .7, sc2_palette: 'Plotly', sc2_marker_preset: 'Core Shapes', sc2_marker_custom: '',
    sc2_color_bin_mode: 'Individual values', sc2_color_bin_count: 5, sc2_color_bin_width: 1, sc2_marker_bin_mode: 'Individual values', sc2_marker_bin_count: 5, sc2_marker_bin_width: 1,
    rw_y: numeric[0] || '', rw_window: 50, rw_msize: 7, rw_mop: .7, rw_include_current: true, rw_full: false, rw_band: true, rw_unified: true, rw_spikes: true, rw_rangeslider: Boolean(time),
    der_y: numeric[0] || '', der_tunit: 'seconds', der_xmode: 'index step (1)', der_lw: 2, der_smooth: 1, der_markers: true, der_unified: true, der_spikes: true, der_rangeslider: Boolean(time), flt2_n: 0,
  };
}
