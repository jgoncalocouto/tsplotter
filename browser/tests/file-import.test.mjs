import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as XLSX from 'xlsx';
import {readExcel, worksheetRows, uniqueColumnNames, isHdf5, numericDatasetColumns, numericColumnsToRows, collectNumericDatasets} from '../src/file-import.js';

test('spreadsheet headers preserve duplicate, blank, special and unicode columns', () => {
  const names = uniqueColumnNames(['x', 'x', null, '__proto__', '日', 'x (2)', 'x']);
  assert.deepEqual(names, ['x', 'x (2)', 'Column 3', '__proto__', '日', 'x (2) (2)', 'x (3)']);
});
test('spreadsheet rows preserve interior blank rows, all columns and valid falsy values', () => {
  const result = worksheetRows([['x', 'y', 'flag'], [0, 2, false], [null, null, null], [3, 4, true], [null, null, null]]);
  assert.equal(result.rows.length, 3); assert.deepEqual(result.rows[0], {x: 0, y: 2, flag: false}); assert.deepEqual(result.rows[1], {x: null, y: null, flag: null});
});
test('spreadsheet rows preserve values beyond original header width and serialize dates', () => {
  const result = worksheetRows([['time'], [new Date('2025-01-31T12:30:00Z'), 12]]);
  assert.deepEqual(result.columns, ['time', 'Column 2']); assert.equal(result.rows[0].time, '2025-01-31T12:30:00.000Z'); assert.equal(result.rows[0]['Column 2'], 12);
});
test('spreadsheet special names do not modify row prototypes', () => {
  const {rows} = worksheetRows([['__proto__', 'constructor'], ['private', 'safe']]);
  assert.equal(rows[0].__proto__, 'private'); assert.equal(Object.getPrototypeOf(rows[0]), Object.prototype); assert.equal({}.private, undefined);
});
for (const bookType of ['xlsx', 'xls']) test(`actual ${bookType} workbook imports requested sheet locally`, async () => {
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([['ignore'], [100]]), 'First');
  XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([['signal', 'other'], [1, 2], [3, 4]]), 'Data');
  const array = XLSX.write(workbook, {bookType, type: 'array'});
  const result = await readExcel({name: `test.${bookType}`, arrayBuffer: async () => array}, {sheetName: 'Data'});
  assert.equal(result.sheetName, 'Data'); assert.deepEqual(result.sheets, ['First', 'Data']); assert.deepEqual(result.rows, [{signal: 1, other: 2}, {signal: 3, other: 4}]);
});
test('invalid spreadsheet selection and empty sheet fail explicitly', async () => {
  const workbook = XLSX.utils.book_new(); XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([]), 'Empty');
  const data = XLSX.write(workbook, {bookType: 'xlsx', type: 'array'});
  await assert.rejects(() => readExcel(data, {sheetName: 'Missing'}), /does not exist/);
  await assert.rejects(() => readExcel(data), /empty/);
});
test('HDF5 signature accepted at offset zero or MATLAB offset512, old MAT rejected', () => {
  const signature = new Uint8Array([0x89, 0x48, 0x44, 0x46, 0xd, 0xa, 0x1a, 0xa]);
  assert.equal(isHdf5(signature), true);
  const bytes = new Uint8Array(520); bytes.set(signature, 512); assert.equal(isHdf5(bytes), true);
  assert.equal(isHdf5(new TextEncoder().encode('MATLAB 5.0 MAT-file')), false);
});
test('numeric vector import preserves every point and both row/column vector orientations', () => {
  for (const shape of [[4], [1, 4], [4, 1]]) assert.deepEqual(numericDatasetColumns({name: 'signal', shape, values: new Float64Array([1, 2, 3, 4])}), [{name: 'signal', values: [1, 2, 3, 4]}]);
});
test('MATLAB matrix HDF dimensions are reversed into MATLAB rows and columns', () => {
  const columns = numericDatasetColumns({name: 'matrix', shape: [2, 3], values: new Float64Array([1, 2, 3, 10, 20, 30])});
  assert.deepEqual(columns, [{name: 'matrix [1]', values: [1, 2, 3]}, {name: 'matrix [2]', values: [10, 20, 30]}]);
  assert.deepEqual(numericColumnsToRows(columns).rows, [{'matrix [1]': 1, 'matrix [2]': 10}, {'matrix [1]': 2, 'matrix [2]': 20}, {'matrix [1]': 3, 'matrix [2]': 30}]);
});
test('plain HDF5 matrix import uses HDF rows and columns directly', () => {
  const result = numericDatasetColumns({name: 'm', shape: [2, 3], values: [1, 2, 3, 4, 5, 6]}, {matlab: false});
  assert.deepEqual(result.map(column => column.values), [[1, 4], [2, 5], [3, 6]]);
});
test('mismatching datasets fail without silently dropping rows', () => {
  assert.throws(() => numericColumnsToRows([{name: 'a', values: [1, 2]}, {name: 'b', values: [3]}]), /different row counts/);
  assert.throws(() => numericColumnsToRows([]), /object references are not supported/);
});
test('big integers beyond JS exact precision are rejected rather than silently rounded', () => {
  assert.throws(() => numericDatasetColumns({name: 'n', shape: [1], values: [9007199254740993n]}), /too large/);
  assert.deepEqual(numericDatasetColumns({name: 'n', shape: [1], values: [123n]})[0].values, [123]);
});
test('metadata scan excludes MATLAB chars, object references and internals without reading values', () => {
  const node = (name, metadata, shape, attrs = {}) => ({name, metadata, shape, attrs, get value() {throw new Error('should not read yet');}});
  const nodes = {signal: node('signal', {type: 1}, [1, 3]), text: node('text', {type: 0}, [3, 1], {MATLAB_class: {value: 'char'}}), refs: node('refs', {type: 7}, [1, 3]), '#refs#': {keys: () => {throw new Error('never explore refs');}}};
  const root = {keys: () => Object.keys(nodes), get: key => nodes[key]};
  const {datasets, excluded} = collectNumericDatasets(root);
  assert.deepEqual(datasets.map(dataset => dataset.name), ['signal']); assert.equal(excluded.length, 3);
  assert.throws(() => collectNumericDatasets(root, {datasetPaths: ['missing']}), /unavailable/);
});
test('actual h5wasm numeric dataset import metadata and values', async () => {
  const hdf5 = await import('h5wasm/node'); await hdf5.ready;
  const directory = await mkdtemp(join(tmpdir(), 'plotter-hdf-test-'));
  const path = join(directory, 'vectors.mat');
  let file;
  try {
    file = new hdf5.File(path, 'w');
    file.create_dataset({name: 'signal', data: new Float64Array([1, 2, 3]), shape: [1, 3]});
    file.create_dataset({name: 'labels', data: new Uint16Array([65, 66, 67]), shape: [1, 3]}).create_attribute('MATLAB_class', 'char');
    const {datasets, excluded} = collectNumericDatasets(file);
    assert.deepEqual(datasets.map(dataset => dataset.name), ['signal']); assert.equal(excluded.length, 1);
    const columns = datasets.flatMap(dataset => numericDatasetColumns({name: dataset.name, shape: dataset.shape, values: dataset.node.value}));
    assert.deepEqual(numericColumnsToRows(columns).rows, [{signal: 1}, {signal: 2}, {signal: 3}]);
  } finally {file?.close(); await rm(directory, {recursive: true, force: true});}
});
