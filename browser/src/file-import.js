/** Local-only spreadsheet and MATLAB readers. No dataset bytes leave the browser. */
export const EXCEL_EXTENSIONS = ['.xlsx', '.xls', '.xlsb', '.ods'];
export const MAT_EXTENSION = '.mat';
const HDF5_SIGNATURE = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

function nonempty(value) { return value !== null && value !== undefined && value !== ''; }
/** Keep every column even with repeated/blank spreadsheet headers. */
export function uniqueColumnNames(headers) {
  const used = new Set();
  return headers.map((header, i) => {
    const base = nonempty(header) ? String(header) : `Column ${i + 1}`;
    let name = base, suffix = 2;
    while (used.has(name)) name = `${base} (${suffix++})`;
    used.add(name); return name;
  });
}
function cellValue(value) {
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value.toISOString();
  return value === undefined ? null : value;
}
/** Convert a rectangular worksheet without dropping internal blank rows or any values. */
export function worksheetRows(matrix) {
  if (!Array.isArray(matrix) || !matrix.length) throw new Error('The selected worksheet is empty.');
  const width = matrix.reduce((max, row) => Math.max(max, row?.length || 0), 0);
  if (!width) throw new Error('The selected worksheet is empty.');
  const columns = uniqueColumnNames(Array.from({length: width}, (_, i) => matrix[0]?.[i]));
  const data = matrix.slice(1);
  // Spreadsheet dimensions can include formatted but empty cells at the end.
  while (data.length && !data[data.length - 1]?.some(nonempty)) data.pop();
  const rows = data.map(row => Object.fromEntries(columns.map((name, i) => [name, cellValue(row?.[i])])));
  return {columns, rows};
}
export async function readExcel(file, {sheetName, sheetIndex = 0, xlsx} = {}) {
  const XLSX = xlsx || await import('xlsx');
  const bytes = file instanceof ArrayBuffer ? file : await file.arrayBuffer();
  const workbook = XLSX.read(bytes, {type: 'array', cellDates: true, cellFormula: false, cellHTML: false});
  const sheets = workbook.SheetNames;
  if (!sheets.length) throw new Error('This workbook has no worksheets.');
  const selected = sheetName ?? sheets[sheetIndex];
  if (!Object.hasOwn(workbook.Sheets, selected)) throw new Error(`Worksheet '${selected}' does not exist.`);
  const matrix = XLSX.utils.sheet_to_json(workbook.Sheets[selected], {header: 1, raw: true, defval: null, blankrows: true});
  const result = worksheetRows(matrix);
  if (!result.rows.length) throw new Error('The selected worksheet has no data rows below its headers.');
  return {name: file.name || 'Workbook', ...result, sheets: [...sheets], sheetName: selected, warnings: []};
}
export function isHdf5(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return [0, 512].some(offset => HDF5_SIGNATURE.every((byte, i) => data[offset + i] === byte));
}
export function isMatlabHeader(bytes) {
  return new TextDecoder('ascii').decode(bytes.subarray(0, 116)).startsWith('MATLAB 7.3 MAT-file');
}
function numericValue(value, path) {
  if (typeof value === 'bigint') {
    const number = Number(value);
    if (!Number.isSafeInteger(number)) throw new Error(`Dataset '${path}' has 64-bit integers too large to represent exactly in JavaScript.`);
    return number;
  }
  if (typeof value !== 'number') throw new Error(`Dataset '${path}' is not a plain numeric array.`);
  return value;
}
/** MATLAB v7.3 stores matrix dimensions reversed relative to MATLAB's row/column view. */
export function numericDatasetColumns({name, shape, values}, {matlab = true} = {}) {
  if (!Array.isArray(shape) || shape.length < 1 || shape.length > 2 || shape.some(size => !Number.isSafeInteger(size) || size < 0)) throw new Error(`Dataset '${name}' must be a numeric vector or 2-D matrix.`);
  const count = shape.reduce((n, size) => n * size, 1);
  if (values.length !== count) throw new Error(`Dataset '${name}' has an inconsistent shape.`);
  if (!count) return [];
  if (shape.length === 1 || shape.includes(1)) {
    return [{name, values: Array.from(values, value => numericValue(value, name))}];
  }
  const rowCount = matlab ? shape[1] : shape[0];
  const columnCount = matlab ? shape[0] : shape[1];
  return Array.from({length: columnCount}, (_, column) => ({
    name: `${name} [${column + 1}]`,
    values: Array.from({length: rowCount}, (_, row) => numericValue(values[matlab ? column * rowCount + row : row * columnCount + column], name)),
  }));
}
export function numericColumnsToRows(columns) {
  if (!columns.length) throw new Error('No plain numeric vectors or matrices were found. MATLAB objects, cells and timetables with object references are not supported in this browser importer. Export the desired table to Parquet or use the existing Streamlit version.');
  const expected = columns[0].values.length;
  if (columns.some(column => column.values.length !== expected)) throw new Error('The numeric datasets have different row counts. Select datasets from one table, or export the desired MATLAB table to Parquet. No rows have been discarded.');
  const names = uniqueColumnNames(columns.map(column => column.name));
  const rows = Array.from({length: expected}, (_, row) => Object.fromEntries(names.map((name, i) => [name, columns[i].values[row]])));
  return {columns: names, rows};
}
function attributeText(value) {
  if (typeof value === 'string') return value;
  if (ArrayBuffer.isView(value) || Array.isArray(value)) return Array.from(value, item => typeof item === 'number' ? String.fromCharCode(item) : String(item)).join('');
  return '';
}
/** Collect metadata before reading values, so unsupported reference/object arrays are never mistaken for numeric data. */
export function collectNumericDatasets(root, {datasetPaths, maxDepth = 8} = {}) {
  const selected = datasetPaths && new Set(datasetPaths.map(path => path.replace(/^\//, '')));
  const datasets = [], excluded = [];
  const seen = new Set();
  function scan(group, prefix = '', depth = 0) {
    if (depth > maxDepth) {excluded.push(`${prefix} (nested more than ${maxDepth} levels)`); return;}
    for (const key of group.keys()) {
      const path = prefix ? `${prefix}/${key}` : key;
      if (seen.has(path)) continue;
      seen.add(path);
      if (key.startsWith('#')) {excluded.push(`${path} (MATLAB internal references)`); continue;}
      const node = group.get(key); if (!node) continue;
      if (node.type === 'Group' || typeof node.keys === 'function') {
        const cls = attributeText(node.attrs?.MATLAB_class?.value);
        if (cls && !['struct'].includes(cls)) {excluded.push(`${path} (MATLAB ${cls} object)`); continue;}
        scan(node, path, depth + 1); continue;
      }
      if (selected && !selected.has(path)) continue;
      const cls = attributeText(node.attrs?.MATLAB_class?.value);
      const metadata = node.metadata;
      const shape = node.shape;
      if (!metadata || ![0, 1].includes(metadata.type) || !Array.isArray(shape) || shape.length < 1 || shape.length > 2 || cls === 'char' || node.attrs?.MATLAB_complex) {
        excluded.push(`${path} (not a plain real numeric vector or matrix)`); continue;
      }
      datasets.push({name: path, node, shape: [...shape]});
    }
  }
  scan(root);
  if (selected) {
    const found = new Set(datasets.map(dataset => dataset.name));
    const missing = [...selected].filter(path => !found.has(path));
    if (missing.length) throw new Error(`The requested numeric datasets are unavailable: ${missing.join(', ')}.`);
  }
  return {datasets, excluded};
}
/**
 * Read plain real numeric vectors/matrices from HDF5 / MATLAB v7.3 locally.
 * This intentionally does not claim compatibility with MATLAB timetable object references.
 * file.arrayBuffer is needed on the main thread; moving this helper to a worker permits WORKERFS.
 */
export async function readMatV73(file, {datasetPaths, hdf5} = {}) {
  const bytes = new Uint8Array(file instanceof ArrayBuffer ? file : await file.arrayBuffer());
  if (!isHdf5(bytes)) throw new Error('Only MATLAB v7.3 (HDF5) .mat files are supported. Save with -v7.3 or export to Parquet.');
  const module = hdf5 || (await import('h5wasm')).default;
  const {FS} = await module.ready;
  const path = `/plotter-${globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`}.mat`;
  let handle;
  try {
    FS.writeFile(path, bytes);
    handle = new module.File(path, 'r');
    const {datasets, excluded} = collectNumericDatasets(handle, {datasetPaths});
    const columns = datasets.flatMap(dataset => numericDatasetColumns({name: dataset.name, shape: dataset.shape, values: dataset.node.value}, {matlab: isMatlabHeader(bytes)}));
    const result = numericColumnsToRows(columns);
    return {name: file.name || 'MATLAB file', ...result, datasets: datasets.map(dataset => dataset.name), warnings: excluded.length ? [`Excluded unsupported/internal MATLAB objects: ${excluded.join('; ')}. Only the listed numeric datasets were imported.`] : []};
  } finally {
    handle?.close();
    try {FS.unlink(path);} catch { /* Opening an invalid file may fail before the virtual file exists. */ }
  }
}
