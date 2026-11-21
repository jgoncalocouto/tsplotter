/** Synthetic Parquet fixture, generated locally without a Python dependency. */
import duckdb from '@duckdb/duckdb-wasm/dist/duckdb-node-blocking.cjs';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(here, '../node_modules/@duckdb/duckdb-wasm/dist');
const rows = Number(process.env.TEST_ROWS || 35001);
if (!Number.isSafeInteger(rows) || rows < 3 || rows > 5_000_000) throw new Error('TEST_ROWS must be a whole number from 3 to 5,000,000');
const output = process.env.PARQUET_FIXTURE || '/tmp/plotter-full-points.parquet';
const db = await duckdb.createDuckDB({
  mvp: { mainModule: path.join(dist, 'duckdb-mvp.wasm') },
  eh: { mainModule: path.join(dist, 'duckdb-eh.wasm') },
}, new duckdb.VoidLogger(), duckdb.NODE_RUNTIME);
await db.instantiate();
const connection = db.connect();
await mkdir('/tmp/plotter-qa-extensions', { recursive: true });
connection.query("SET extension_directory='/tmp/plotter-qa-extensions'");
if (process.env.EXTENSION_REPOSITORY) connection.query(`SET custom_extension_repository='${process.env.EXTENSION_REPOSITORY.replaceAll("'", "''")}'`);
connection.query('LOAD parquet');
connection.query(`COPY (SELECT i::DOUBLE AS x, (2 * i + 1)::DOUBLE AS signal, (i % 101)::DOUBLE AS aux, CASE WHEN i % 2 = 1 THEN 'odd' ELSE 'even' END AS category FROM range(${rows}) t(i)) TO 'fixture.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)`);
await writeFile(output, db.copyFileToBuffer('fixture.parquet'));
connection.close();
console.log(JSON.stringify({ path: output, rows, compressed: true }));
