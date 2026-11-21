/** Browser-only data engine. All SQL is generated from validated identifiers/values. */
export const ROW_COLUMN = '__row__';
const TABLE = 'plotter_source';
let internalRow = '__plotter_internal_row__';
let database, connection, databaseWorker, sourceMeta;
const loadedExtensions = new Set();
const NUMERIC = /^(U?TINYINT|U?SMALLINT|U?INTEGER|U?BIGINT|HUGEINT|UHUGEINT|FLOAT|DOUBLE|DECIMAL|REAL|INT\d*|UINT\d*)/i;
const DATETIME = /^(TIMESTAMP|DATE|TIME)/i;

export function quoteIdentifier(value) {
  if (typeof value !== 'string' || !value || value.includes('\0')) throw new Error('Invalid column name.');
  return `"${value.replaceAll('"', '""')}"`;
}
export function quoteValue(value) {
  if (value == null) return 'NULL';
  if (typeof value === 'boolean') return value ? 'TRUE' : 'FALSE';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Enter a finite numeric value.');
    return String(value);
  }
  if (typeof value === 'bigint') return value.toString();
  if (value instanceof Date) {
    if (!Number.isFinite(value.valueOf())) throw new Error('Invalid date.');
    value = value.toISOString();
  }
  return `'${String(value).replaceAll("'", "''")}'`;
}
function integer(value, fallback, min = 1, max = 200000) {
  const number = value == null ? fallback : Number(value);
  if (!Number.isInteger(number) || number < min || number > max) throw new Error(`Enter a whole number between ${min} and ${max}.`);
  return number;
}
export function columnKind(type) {
  return NUMERIC.test(type) ? 'numeric' : DATETIME.test(type) ? 'datetime' : 'categorical';
}
function field(name, columns, rowColumn = internalRow) {
  if (name === ROW_COLUMN) return quoteIdentifier(rowColumn);
  if (!columns.some(c => c.name === name)) throw new Error(`Column “${name}” is not in this dataset.`);
  return quoteIdentifier(name);
}
function numericField(name, columns, rowColumn) {
  return `TRY_CAST(${field(name, columns, rowColumn)} AS DOUBLE)`;
}
function finite(expression) { return `(${expression} IS NOT NULL AND isfinite(${expression}))`; }
function normalizedLogic(value) {
  const result = String(value ?? 'AND').toUpperCase();
  if (!['AND', 'OR'].includes(result)) throw new Error('Filters must use AND or OR.');
  return result;
}

/** Preserve the Python filter engine's left-to-right AND/OR composition. */
export function buildFilterSQL(conditions = [], columns = [], rowColumn = internalRow, axisOptions = {}) {
  let combined = '';
  for (const condition of conditions) {
    const name = field(condition.column, columns, rowColumn);
    const kind = condition.kind ?? condition.type ?? 'categorical';
    const logic = normalizedLogic(condition.logic);
    let expression;
    if (kind === 'numeric' || kind === 'datetime') {
      if (condition.low == null && condition.up == null) continue;
      const parsing = condition.timeAxis || (condition.column === (axisOptions.preparedAxis?.x || axisOptions.x) ? (axisOptions.preparedAxis?.timeAxis || axisOptions.timeAxis) : undefined);
      const preparedName = kind === 'datetime' && parsing ? buildTimeExpression(condition.column, columns, parsing, rowColumn) : name;
      const target = kind === 'numeric' ? `TRY_CAST(${name} AS DOUBLE)` : `TRY_CAST(${preparedName} AS TIMESTAMP)`;
      const bound = value => kind === 'numeric' ? quoteValue(Number(value)) : `TRY_CAST(${quoteValue(value)} AS TIMESTAMP)`;
      const terms = [];
      if (condition.low != null && condition.low !== '') terms.push(`${target} >= ${bound(condition.low)}`);
      if (condition.up != null && condition.up !== '') terms.push(`${target} <= ${bound(condition.up)}`);
      if (!terms.length) continue;
      expression = `COALESCE((${terms.join(' AND ')}), FALSE)`;
    } else if (kind === 'categorical') {
      const values = condition.values ?? (condition.value != null ? [condition.value] : []);
      const op = condition.op ?? 'is one of';
      if (!values.length || (!['is one of', 'is not one of'].includes(op) && values[0] === '')) continue;
      const target = `CAST(${name} AS VARCHAR)`;
      const insensitive = !condition.caseSensitive;
      const compared = insensitive ? `lower(${target})` : target;
      const pattern = quoteValue(insensitive ? String(values[0]).toLowerCase() : String(values[0]));
      const na = condition.naMatch ? 'TRUE' : 'FALSE';
      switch (op) {
        // Python isin deliberately keeps exact/case-sensitive membership.
        case 'is one of': expression = `COALESCE(${target} IN (${values.map(quoteValue).join(', ')}), FALSE)`; break;
        case 'is not one of': expression = `NOT COALESCE(${target} IN (${values.map(quoteValue).join(', ')}), FALSE)`; break;
        case 'contains': expression = `COALESCE(contains(${compared}, ${pattern}), ${na})`; break;
        case 'not contains': expression = `NOT COALESCE(contains(${compared}, ${pattern}), ${na})`; break;
        case 'starts with': expression = `COALESCE(starts_with(${compared}, ${pattern}), ${na})`; break;
        case 'ends with': expression = `COALESCE(ends_with(${compared}, ${pattern}), ${na})`; break;
        case 'regex': expression = `COALESCE(regexp_matches(${target}, ${quoteValue(values[0])}, ${quoteValue(insensitive ? 'i' : 'c')}), ${na})`; break;
        default: throw new Error(`Unsupported filter operation: ${op}`);
      }
      if (condition.naMatch) expression = `(${expression} OR ${name} IS NULL)`;
    } else throw new Error(`Unsupported filter kind: ${kind}`);
    combined = combined ? `(${combined} ${logic} (${expression}))` : `(${expression})`;
  }
  return combined || 'TRUE';
}

export function buildTimeExpression(name, columns, options = {}, rowColumn = internalRow) {
  const source = field(name, columns, rowColumn);
  if (!options || !Object.keys(options).length || name === ROW_COLUMN) return source;
  const profile = String(options.profile ?? '').toLowerCase();
  const epochUnits = {s: 1000, seconds: 1000, ms: 1, milliseconds: 1, us: 0.001, microseconds: 0.001, ns: 0.000001, nanoseconds: 0.000001};
  let expression = source;
  const fromProfile = profile.includes('milliseconds') ? 'milliseconds' : profile.includes('seconds') ? 'seconds' : undefined;
  const epochUnit = options.unit ?? options.epochUnit ?? (profile.includes('epoch') ? fromProfile || profile.split(/[ :()]+/).filter(Boolean)[1] : undefined);
  if (epochUnit && epochUnits[epochUnit] != null) expression = `epoch_ms(CAST(TRY_CAST(${source} AS DOUBLE) * ${epochUnits[epochUnit]} AS BIGINT))`;
  else if (options.format) expression = `try_strptime(CAST(${source} AS VARCHAR), ${quoteValue(options.format)})`;
  else if (profile.startsWith('dmy') || profile.startsWith('mdy')) {
    const base = profile.startsWith('dmy') ? ['%d/%m/%Y','%d-%m-%Y','%d.%m.%Y'] : ['%m/%d/%Y','%m-%d-%Y','%m.%d.%Y'];
    const formats = base.flatMap(format => [`${format} %H:%M:%S.%f`, `${format} %H:%M:%S`, `${format} %H:%M`, format]);
    expression = `try_strptime(CAST(${source} AS VARCHAR), [${formats.map(quoteValue).join(', ')}])`;
  } else if (profile && !['relative', 'preserve', 'raw', 'none'].includes(profile)) {
    // Aware timestamps preserve their instant; naive values are interpreted as UTC
    // unless an explicit source timezone is supplied below.
    expression = `TRY_CAST(${source} AS TIMESTAMPTZ)`;
  }
  const timezone = options.timezone;
  if (timezone && !['preserve', '(preserve)', '(keep)'].includes(timezone)) expression = `timezone(${quoteValue(timezone)}, CAST(${expression} AS TIMESTAMP))`;
  // Display timezone is applied to returned strings, never to derivative spacing.
  return expression;
}

export function inDisplayTimezone(value, timezone) {
  if (value == null || !timezone || ['preserve', '(preserve)', '(keep)'].includes(timezone)) return value;
  const date = new Date(value);
  if (!Number.isFinite(date.valueOf())) return value;
  const formatter = new Intl.DateTimeFormat('en-CA', {timeZone: timezone, year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23',timeZoneName:'longOffset'});
  const parts = Object.fromEntries(formatter.formatToParts(date).map(part => [part.type, part.value]));
  const zone = parts.timeZoneName.replace('GMT', '') || '+00:00';
  const milliseconds = String(date.getUTCMilliseconds()).padStart(3, '0');
  return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}.${milliseconds}${zone}`;
}

function selectedColumns(spec, columns) {
  const chosen = Array.isArray(spec.y) ? spec.y : spec.y ? [spec.y] : [];
  for (const name of chosen) field(name, columns);
  return [...new Set(chosen)];
}
function baseSQL(spec, columns, rowColumn) {
  const xName = spec.x || ROW_COLUMN;
  const xExpr = buildTimeExpression(xName, columns, spec.timeAxis, rowColumn);
  const filterWhere = buildFilterSQL(spec.filters, columns, rowColumn, spec);
  const prepared = spec.preparedAxis || (spec.timeAxis ? {x: xName, timeAxis: spec.timeAxis} : undefined);
  const preparedExpr = prepared ? buildTimeExpression(prepared.x, columns, prepared.timeAxis, rowColumn) : undefined;
  const where = preparedExpr ? `(${filterWhere}) AND ${preparedExpr} IS NOT NULL` : filterWhere;
  return {xName, xExpr, where, filterWhere, preparedExpr};
}

/** Build complete chart queries. LIMIT appears only in explicitly requested preview mode. */
export function buildQuerySQL(spec, columns, rowColumn = internalRow) {
  const mode = spec.mode ?? 'trends';
  const y = selectedColumns(spec, columns);
  const {xName, xExpr, where} = baseSQL(spec, columns, rowColumn);
  const row = quoteIdentifier(rowColumn);
  if (mode === 'export' || mode === 'preview') {
    const selected = spec.columns?.length ? spec.columns : columns.map(c => c.name);
    const sql = `SELECT ${selected.map(name => `${field(name, columns, rowColumn)} AS ${quoteIdentifier(name)}`).join(', ')} FROM ${TABLE} WHERE ${where} ORDER BY ${row}`;
    return mode === 'preview' ? `${sql} LIMIT ${integer(spec.limit, 100, 1, 1000)}` : sql;
  }
  if (!y.length) throw new Error('Choose at least one numeric signal.');
  const xProjection = `${xExpr} AS ${quoteIdentifier(xName)}`;
  const projections = y.filter(name => name !== xName).map(name => `${field(name, columns, rowColumn)} AS ${quoteIdentifier(name)}`);
  if (mode === 'trends') return `SELECT ${[xProjection, ...projections].join(', ')} FROM ${TABLE} WHERE ${where} AND ${xExpr} IS NOT NULL ORDER BY ${xExpr}, ${row}`;
  if (mode === 'scatter' || mode === 'grouped-scatter' || mode === 'grouped scatter') {
    const names = [...new Set([xName, ...y, spec.color, spec.marker].filter(Boolean))];
    const fields = names.map(name => name === xName ? xProjection : `${field(name, columns, rowColumn)} AS ${quoteIdentifier(name)}`);
    const complete = names.map(name => {
      const expr = name === xName ? xExpr : field(name, columns, rowColumn);
      const kind = name === ROW_COLUMN ? 'numeric' : columns.find(c => c.name === name)?.kind;
      return kind === 'numeric' ? finite(`TRY_CAST(${expr} AS DOUBLE)`) : `${expr} IS NOT NULL`;
    });
    return `SELECT ${fields.join(', ')} FROM ${TABLE} WHERE ${where} AND ${complete.join(' AND ')} ORDER BY ${row}`;
  }
  const signal = y[0];
  const value = numericField(signal, columns, rowColumn);
  const order = `${xExpr}, ${row}`;
  if (mode === 'rolling') {
    const window = integer(spec.window, 50);
    const minCount = spec.requireFull ? window : 1;
    const frame = spec.includeCurrent === false ? `ROWS BETWEEN ${window} PRECEDING AND 1 PRECEDING` : `ROWS BETWEEN ${window - 1} PRECEDING AND CURRENT ROW`;
    const over = `(ORDER BY ${order} ${frame})`;
    const rolls = ['min', 'mean', 'max'].map(stat => `CASE WHEN count(${value}) OVER ${over} >= ${minCount} THEN ${stat === 'mean' ? 'avg' : stat}(${value}) OVER ${over} END AS ${quoteIdentifier(`${signal}__roll_${stat}`)}`);
    return `SELECT ${[xProjection, `${value} AS ${quoteIdentifier(signal)}`, ...rolls].join(', ')} FROM ${TABLE} WHERE ${where} AND ${xExpr} IS NOT NULL ORDER BY ${order}`;
  }
  if (mode === 'derivatives') {
    const xKind = columns.find(c => c.name === xName)?.kind;
    const datetime = xKind === 'datetime' || Boolean(spec.timeAxis && Object.keys(spec.timeAxis).length && !['relative', 'preserve', 'raw', 'none'].includes(spec.timeAxis.profile));
    const unit = spec.derivativeUnit ?? 'sample';
    const divisors = {seconds: 1, minutes: 60, hours: 3600, days: 86400};
    let spacing = `CAST(row_number() OVER (ORDER BY ${order}) - 1 AS DOUBLE)`;
    if (unit !== 'sample' && unit !== 'none (sample index)' && unit !== 'index step (1)') {
      spacing = datetime ? `epoch(${xExpr}) / ${divisors[unit] ?? 1}` : `TRY_CAST(${xExpr} AS DOUBLE)`;
    }
    const smooth = integer(spec.smoothWindow, 1, 1, 9999);
    const d1Name = quoteIdentifier(`${signal}__d1`), d2Name = quoteIdentifier(`${signal}__d2`);
    const d1Expr = '(v - lag(v) OVER w) / nullif(dx - lag(dx) OVER w, 0)';
    const d2Expr = '(d1 - lag(d1) OVER w) / nullif(dx - lag(dx) OVER w, 0)';
    const orderDerived = 'ORDER BY __order';
    const result1 = smooth > 1 ? `CASE WHEN __order > 1 THEN avg(d1) OVER (${orderDerived} ROWS BETWEEN ${smooth - 1} PRECEDING AND CURRENT ROW) END` : 'd1';
    const result2 = smooth > 1 ? `CASE WHEN __order > 2 THEN avg(d2) OVER (${orderDerived} ROWS BETWEEN ${smooth - 1} PRECEDING AND CURRENT ROW) END` : 'd2';
    return `WITH base AS (SELECT ${xExpr} AS x, ${value} AS v, ${spacing} AS dx, row_number() OVER (ORDER BY ${order}) AS __order FROM ${TABLE} WHERE ${where} AND ${xExpr} IS NOT NULL AND ${finite(value)}), first_derivative AS (SELECT *, ${d1Expr} AS d1 FROM base WINDOW w AS (${orderDerived})), second_derivative AS (SELECT *, ${d2Expr} AS d2 FROM first_derivative WINDOW w AS (${orderDerived})) SELECT x AS ${quoteIdentifier(xName)}, v AS ${quoteIdentifier(signal)}, ${result1} AS ${d1Name}, ${result2} AS ${d2Name} FROM second_derivative ORDER BY __order`;
  }
  throw new Error(`Unknown chart mode: ${mode}`);
}

// Decimal construction avoids moving boundaries such as 0.3 to 0.30000000000000004.
function decimalParts(number) {
  const [mantissa, exponentText = '0'] = String(number).toLowerCase().split('e');
  const [whole, fraction = ''] = mantissa.split('.');
  const exponent = Number(exponentText) - fraction.length;
  return {coefficient: BigInt(whole + fraction), exponent};
}
export function makeBinEdges(low, high, settings = {}) {
  if (!Number.isFinite(low) || !Number.isFinite(high)) throw new Error('No finite numeric values are available.');
  if (low === high) return [low, high];
  const mode = settings.mode ?? 'count';
  let count, width;
  if (['width', 'Bin width'].includes(mode)) {
    width = Number(settings.width);
    if (!Number.isFinite(width) || width <= 0) throw new Error('Bin width must be greater than zero.');
    count = Math.ceil((high - low) / width - Number.EPSILON * 8);
  } else {
    count = integer(settings.count ?? settings.bins, 5, 1, 10000);
    width = (high - low) / count;
  }
  if (count > (settings.maxBins ?? 100)) throw new Error(`This setting creates more than ${settings.maxBins ?? 100} bins. Increase the width.`);
  const a = decimalParts(low), b = decimalParts(high), w = decimalParts(width);
  const exp = Math.min(a.exponent, b.exponent, w.exponent, -16);
  const scale = p => p.coefficient * 10n ** BigInt(p.exponent - exp);
  const lowInt = scale(a), highInt = scale(b), widthInt = scale(w);
  const edges = Array.from({length: count + 1}, (_, index) => {
    const numerator = ['width', 'Bin width'].includes(mode) ? lowInt + BigInt(index) * widthInt : lowInt + BigInt(index) * (highInt - lowInt) / BigInt(count);
    return Number(`${numerator}e${exp}`);
  });
  if (!['width', 'Bin width'].includes(mode)) edges[count] = high;
  if (edges.some((value, i) => !Number.isFinite(value) || (i > 0 && value <= edges[i - 1]))) throw new Error('These bins are too narrow to represent accurately. Use fewer bins or a larger width.');
  return edges;
}
export function binLabel(edges, index) {
  if (edges[0] === edges[edges.length - 1]) return String(edges[0]);
  return `[${edges[index]}, ${edges[index + 1]}${index === edges.length - 2 ? ']' : ')'}`;
}
export function assignBins(values, settings = {}) {
  const valid = values.filter(value => typeof value === 'number' && Number.isFinite(value));
  if (!valid.length) throw new Error('This grouping column has no finite numbers.');
  if (valid.some(value => Number.isInteger(value) && !Number.isSafeInteger(value))) throw new Error('These integer values are too large to bin accurately. Scale the column first.');
  let low = Infinity, high = -Infinity;
  for (const value of valid) { low = Math.min(low, value); high = Math.max(high, value); }
  const edges = makeBinEdges(low, high, settings);
  const labels = edges.slice(1).map((_, index) => binLabel(edges, index));
  const counts = labels.map(() => 0);
  const groups = values.map(value => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    let left = 0, right = edges.length - 1;
    while (left < right) { const mid = (left + right + 1) >> 1; if (edges[mid] <= value) left = mid; else right = mid - 1; }
    const index = Math.min(left, labels.length - 1); counts[index]++; return labels[index];
  });
  return {groups, edges, labels, counts, width: edges.length > 1 ? edges[1] - edges[0] : 0};
}

function progress(message, fraction) { self.postMessage({type: 'progress', message, fraction}); }
async function initialize(options = {}) {
  if (database) return;
  const duckdb = await import('@duckdb/duckdb-wasm');
  const base = new URL(options.assetBase || '../duckdb/', self.location.href).href;
  const bundles = options.bundles || {
    mvp: {mainModule: `${base}duckdb-mvp.wasm`, mainWorker: `${base}duckdb-browser-mvp.worker.js`},
    eh: {mainModule: `${base}duckdb-eh.wasm`, mainWorker: `${base}duckdb-browser-eh.worker.js`},
  };
  progress('Starting local analysis engine…', 0.05);
  const bundle = await duckdb.selectBundle(bundles);
  databaseWorker = new Worker(bundle.mainWorker);
  // No query logger: source contents never enter application logs.
  database = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), databaseWorker);
  await database.instantiate(bundle.mainModule, bundle.pthreadWorker);
  await database.open({path: ':memory:', query: {castBigIntToDouble: false, castDecimalToDouble: true, castTimestampToDate: true}});
  connection = await database.connect();
  // File operations can only use explicitly registered browser files. No remote scan SQL.
  await connection.query(`SET custom_extension_repository = ${quoteValue(`${base}extensions/duckdb-wasm`)}; SET autoinstall_known_extensions = false; SET autoload_known_extensions = false;`);
}
async function loadExtension(name) {
  if (loadedExtensions.has(name)) return;
  progress(`Preparing local ${name === 'icu' ? 'timezone' : name} support…`);
  await connection.query(`LOAD ${quoteIdentifier(name)}`);
  loadedExtensions.add(name);
}
function scalar(value, type = '') {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === 'bigint') return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString();
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    if (/^(TIMESTAMP|DATE)/i.test(type)) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? value : date.toISOString(); }
    return value;
  }
  if (ArrayBuffer.isView(value)) return Array.from(value, v => scalar(v));
  if (Array.isArray(value)) return value.map(v => scalar(v));
  if (typeof value === 'object') {
    if (typeof value.toJSON === 'function') return scalar(value.toJSON());
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, scalar(v)]));
  }
  return value;
}
function tableColumns(table) {
  const columns = {};
  for (const field of table.schema.fields) {
    const vector = table.getChild(field.name);
    columns[field.name] = Array.from({length: table.numRows}, (_, index) => scalar(vector.get(index), field.type.toString()));
  }
  return columns;
}
async function queryColumns(sql) { return tableColumns(await connection.query(sql)); }
async function queryScalar(sql) { const table = await connection.query(sql); return scalar(table.get(0)?.[table.schema.fields[0].name]); }
async function metadata(name) {
  const rows = (await connection.query(`DESCRIBE ${TABLE}`)).toArray();
  const columns = rows.filter(row => row.column_name !== internalRow).map(row => ({name: row.column_name, type: row.column_type, kind: columnKind(row.column_type)}));
  const rowCount = Number(await queryScalar(`SELECT count(*) FROM ${TABLE}`));
  sourceMeta = {sourceName: name, rowCount, columns};
  return sourceMeta;
}
async function importFile(file, options = {}) {
  if (!(file instanceof Blob)) throw new Error('Choose a local data file.');
  await initialize(options);
  const original = file.name || options.sourceName || 'dataset';
  sourceMeta = undefined;
  const format = (options.format || original.split('.').pop()).toLowerCase();
  if (!['parquet', 'pq', 'csv', 'tsv', 'json', 'jsonl', 'ndjson'].includes(format)) throw new Error('This file needs the Excel or MATLAB importer.');
  await connection.query(`DROP VIEW IF EXISTS ${TABLE}`);
  await database.dropFiles();
  const virtualName = `local-input.${format}`;
  const duckdb = await import('@duckdb/duckdb-wasm');
  progress('Reading your local file…', 0.15);
  // Browser FileReader keeps Parquet compressed and permits lazy column/range scans.
  const bom = new Uint8Array(await file.slice(0, 3).arrayBuffer());
  const encoding = options.encoding || (bom[0] === 255 && bom[1] === 254 ? 'utf-16le' : bom[0] === 254 && bom[1] === 255 ? 'utf-16be' : 'utf-8');
  if (['csv', 'tsv', 'json', 'jsonl', 'ndjson'].includes(format) && !/^utf-?8$/i.test(encoding)) {
    const decoded = new TextDecoder(encoding).decode(await file.arrayBuffer());
    await database.registerFileText(virtualName, decoded);
  } else await database.registerFileHandle(virtualName, file, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
  if (['parquet', 'pq'].includes(format)) await loadExtension('parquet');
  if (['json', 'jsonl', 'ndjson'].includes(format)) await loadExtension('json');
  let reader;
  if (['parquet', 'pq'].includes(format)) reader = `read_parquet(${quoteValue(virtualName)})`;
  else if (['json', 'jsonl', 'ndjson'].includes(format)) reader = `read_json_auto(${quoteValue(virtualName)}, format='auto', maximum_object_size=67108864)`;
  else {
    const settings = ['auto_detect=true', 'sample_size=-1', 'ignore_errors=false', 'null_padding=false'];
    if (options.delimiter || format === 'tsv') settings.push(`delim=${quoteValue(options.delimiter || '\t')}`);
    if (typeof options.header === 'boolean') settings.push(`header=${options.header}`);
    if (options.decimalSeparator) settings.push(`decimal_separator=${quoteValue(options.decimalSeparator)}`);
    reader = `read_csv(${quoteValue(virtualName)}, ${settings.join(', ')})`;
  }
  // Discover source names before selecting a collision-free internal row identifier.
  const description = (await connection.query(`DESCRIBE SELECT * FROM ${reader}`)).toArray();
  const sourceNames = new Set(description.map(row => row.column_name));
  internalRow = '__plotter_internal_row__';
  while (sourceNames.has(internalRow)) internalRow += '_';
  if (sourceNames.has(ROW_COLUMN)) throw new Error(`The source contains the reserved column ${ROW_COLUMN}. Rename that column before import.`);
  await connection.query(`CREATE VIEW ${TABLE} AS SELECT row_number() OVER () - 1 AS ${quoteIdentifier(internalRow)}, * FROM ${reader}`);
  progress('Inspecting the dataset…', 0.8);
  return metadata(original);
}
async function importRows(rows, options = {}) {
  if (!Array.isArray(rows) || !rows.length) throw new Error('The selected sheet has no rows.');
  const text = JSON.stringify(rows, (_, value) => typeof value === 'bigint' ? value.toString() : value);
  const file = new File([text], `${options.sourceName || 'dataset'}.json`, {type: 'application/json'});
  const result = await importFile(file, {...options, format: 'json'});
  result.sourceName = options.sourceName || 'Imported worksheet';
  return result;
}
async function summary(filters = [], options = {}) {
  if (!sourceMeta) throw new Error('Import a dataset first.');
  if (options.preparedAxis?.timeAxis?.timezone || options.timeAxis?.timezone || filters.some(condition => condition.timeAxis?.timezone)) await loadExtension('icu');
  const {where} = baseSQL({...options, filters}, sourceMeta.columns, internalRow);
  const totalRows = Number(await queryScalar(`SELECT count(*) FROM ${TABLE} WHERE ${where}`));
  const numeric = sourceMeta.columns.filter(col => col.kind === 'numeric');
  const columns = {column: [], count: [], valid: [], min: [], max: [], mean: [], stddev: []};
  if (numeric.length) {
    const aggregates = numeric.flatMap((col, index) => {
      const value = numericField(col.name, sourceMeta.columns);
      return [`count(${value}) FILTER (WHERE isfinite(${value})) AS s${index}_valid`, `min(${value}) FILTER (WHERE isfinite(${value})) AS s${index}_min`, `max(${value}) FILTER (WHERE isfinite(${value})) AS s${index}_max`, `avg(${value}) FILTER (WHERE isfinite(${value})) AS s${index}_mean`, `stddev_samp(${value}) FILTER (WHERE isfinite(${value})) AS s${index}_stddev`];
    });
    const data = await queryColumns(`SELECT ${aggregates.join(', ')} FROM ${TABLE} WHERE ${where}`);
    numeric.forEach((col, index) => {
      columns.column.push(col.name); columns.count.push(totalRows);
      for (const stat of ['valid','min','max','mean','stddev']) columns[stat].push(data[`s${index}_${stat}`][0]);
    });
  }
  return {columns, rowCount: numeric.length, totalRows, sourceRows: sourceMeta.rowCount, meta: {kind:'summary'}};
}

async function distribution(spec) {
  const columns = sourceMeta.columns;
  const y = selectedColumns(spec, columns);
  if (!y.length) throw new Error('Choose one or more numeric signals.');
  const {where} = baseSQL(spec, columns, internalRow);
  const selections = y.map(name => `SELECT ${numericField(name, columns)} AS v FROM ${TABLE} WHERE ${where}`);
  const pooled = selections.join(' UNION ALL ');
  const stats = await queryColumns(`SELECT min(v) AS low, max(v) AS high, count(*) AS n, quantile_cont(v, 0.25) AS q1, quantile_cont(v, 0.75) AS q3, stddev_samp(v) AS stddev FROM (${pooled}) WHERE ${finite('v')}`);
  const low = stats.low[0], high = stats.high[0], n = stats.n[0];
  if (!n) throw new Error('No finite values remain after filtering.');
  let settings = typeof spec.bins === 'number' ? {mode: 'count', count: spec.bins} : {...spec.bins};
  if (!settings.mode || settings.mode === 'auto') {
    let width = 2 * (stats.q3[0] - stats.q1[0]) * n ** (-1 / 3);
    if (!(width > 0)) width = 3.5 * stats.stddev[0] * n ** (-1 / 3);
    if (!(width > 0)) width = (high - low) / Math.max(1, Math.ceil(Math.log2(n) + 1)) || 1;
    settings = {mode: 'width', width};
  }
  const edges = makeBinEdges(low, high, {...settings, maxBins: 10000});
  const binCount = edges.length - 1;
  const output = {bin_left: edges.slice(0, -1), bin_right: edges.slice(1), bin_center: edges.slice(1).map((right, i) => edges[i] + (right - edges[i]) / 2)};
  const observationCounts = {};
  for (const name of y) {
    const value = numericField(name, columns);
    const branches = edges.slice(1).map((edge, i) => `WHEN v ${i === binCount - 1 ? '<=' : '<'} ${quoteValue(edge)} THEN ${i}`).join(' ');
    const data = await queryColumns(`SELECT CASE ${branches} END AS bin, count(*) AS n FROM (SELECT ${value} AS v FROM ${TABLE} WHERE ${where}) WHERE ${finite('v')} GROUP BY bin ORDER BY bin`);
    const counts = Array(binCount).fill(0);
    for (let i = 0; i < data.bin.length; i++) if (data.bin[i] != null) counts[data.bin[i]] = Number(data.n[i]);
    output[name] = counts;
    observationCounts[name] = counts.reduce((sum, number) => sum + number, 0);
  }
  const totalRows = Number(await queryScalar(`SELECT count(*) FROM ${TABLE} WHERE ${where}`));
  return {columns: output, rowCount: binCount, totalRows, sourceRows: sourceMeta.rowCount, meta: {kind: 'distribution', width: edges[1] - edges[0], binCount, range: [low, high], observationCounts}};
}
async function query(spec) {
  if (!sourceMeta) throw new Error('Import a dataset first.');
  if (spec.timeAxis?.timezone || spec.timeAxis?.outputTimezone || spec.preparedAxis?.timeAxis?.timezone || spec.filters?.some(condition => condition.timeAxis?.timezone)) await loadExtension('icu');

  if (!['preview','export'].includes(spec.mode)) {
    const chartNames = new Set([spec.x, ...selectedColumns(spec, sourceMeta.columns), spec.color, spec.marker].filter(Boolean));
    const unsafeTypes = sourceMeta.columns.filter(col => chartNames.has(col.name) && /(?:U?BIGINT|U?HUGEINT)/i.test(col.type));
    if (unsafeTypes.length) {
      const {where} = baseSQL(spec, sourceMeta.columns, internalRow);
      for (const col of unsafeTypes) {
        if (col.name === spec.x && spec.timeAxis?.profile?.toLowerCase().includes('epoch')) continue;
        const name = field(col.name, sourceMeta.columns);
        const unsafe = Number(await queryScalar(`SELECT count(*) FROM ${TABLE} WHERE ${where} AND (${name} > 9007199254740991 OR ${name} < -9007199254740991)`));
        if (unsafe) throw new Error(`Column “${col.name}” contains integers too large to plot accurately in JavaScript. Scale that column before plotting; full values remain available in preview and export.`);
      }
    }
  }
  if (spec.mode === 'distribution') return distribution(spec);
  const sql = buildQuerySQL(spec, sourceMeta.columns);
  const columns = await queryColumns(sql);
  const xName = spec.x || ROW_COLUMN;
  if (spec.timeAxis?.outputTimezone && columns[xName]) columns[xName] = columns[xName].map(value => inDisplayTimezone(value, spec.timeAxis.outputTimezone));
  const {where, filterWhere, preparedExpr} = baseSQL(spec, sourceMeta.columns, internalRow);
  const totalRows = Number(await queryScalar(`SELECT count(*) FROM ${TABLE} WHERE ${where}`));
  const invalidXRows = preparedExpr ? Number(await queryScalar(`SELECT count(*) FROM ${TABLE} WHERE ${filterWhere} AND ${preparedExpr} IS NULL`)) : 0;
  const result = {columns, rowCount: Object.values(columns)[0]?.length || 0, totalRows, sourceRows: sourceMeta.rowCount, meta: {kind: spec.mode || 'trends', x: spec.x || ROW_COLUMN, sampling: false, invalidXRows, groups: {}}};
  for (const encoding of ['color', 'marker']) {
    const source = spec[encoding];
    const settings = spec[`${encoding}Binning`];
    if (!source || !settings || ['Individual values', 'individual', 'none'].includes(settings.mode)) continue;
    if (!columns[source]) throw new Error(`The ${encoding} grouping source is not in the selected chart.`);
    const binned = assignBins(columns[source], settings);
    let name = `__${encoding}_group`;
    while (Object.hasOwn(columns, name)) name += '_';
    columns[name] = binned.groups;
    result.meta.groups[encoding] = {column: name, source, ...binned, groups: undefined};
  }
  return result;
}

export async function handleEngineMessage(message) {
  const {method, args = []} = message;
  switch (method) {
    case 'init': await initialize(...args); return {ready: true};
    case 'importFile': return importFile(...args);
    case 'importRows': return importRows(...args);
    case 'summary': return summary(...args);
    case 'query': return query(...args);
    case 'close': await connection?.close(); await database?.terminate(); databaseWorker?.terminate(); connection = database = sourceMeta = undefined; loadedExtensions.clear(); return {closed: true};
    default: throw new Error(`Unknown local-engine operation: ${method}`);
  }
}
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  // One operation at a time; termination is the cancellation boundary.
  let queue = Promise.resolve();
  self.onmessage = event => {
    const message = event.data;
    queue = queue.then(async () => {
      try { const result = await handleEngineMessage(message); self.postMessage({id: message.id, result}); }
      catch (error) { self.postMessage({id: message.id, error: {message: String(error?.message || error), name: error?.name || 'Error'}}); }
    });
  };
}
