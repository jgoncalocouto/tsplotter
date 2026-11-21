/** Reproduce the deployed local DuckDB assets from pinned packages and verified official extensions. */
import {createHash, randomBytes} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {copyFile, mkdir, open, readFile, rename, stat, unlink} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const browserRoot = fileURLToPath(new URL('../', import.meta.url));
const expectedRuntimeFiles = ['duckdb-mvp.wasm', 'duckdb-eh.wasm', 'duckdb-browser-mvp.worker.js', 'duckdb-browser-eh.worker.js'];
const extensionNames = ['parquet', 'json', 'icu'];
const platforms = ['wasm_mvp', 'wasm_eh'];
const sha256Pattern = /^[a-f0-9]{64}$/;
const wasmMagic = Buffer.from([0, 0x61, 0x73, 0x6d]);

export function validateManifest(manifest) {
  if (manifest?.schemaVersion !== 1 || !/^\d+\.\d+\.\d+$/.test(manifest.duckdbWasmVersion || '') || !/^v\d+\.\d+\.\d+$/.test(manifest.duckdbEngineVersion || '')) throw new Error('Invalid DuckDB asset manifest header.');
  if (!Array.isArray(manifest.runtimeFiles) || manifest.runtimeFiles.length !== expectedRuntimeFiles.length || new Set(manifest.runtimeFiles.map(file => file.name)).size !== expectedRuntimeFiles.length) throw new Error('The asset manifest must contain each expected runtime file once.');
  for (const file of manifest.runtimeFiles) {
    if (!expectedRuntimeFiles.includes(file.name) || !sha256Pattern.test(file.sha256) || !Number.isSafeInteger(file.bytes) || file.bytes < 1 || file.bytes > 64 * 1024 * 1024) throw new Error(`Invalid runtime asset: ${file.name}.`);
  }
  if (!Array.isArray(manifest.extensions) || manifest.extensions.length !== extensionNames.length * platforms.length) throw new Error('The asset manifest must contain six extensions.');
  const seen = new Set();
  for (const entry of manifest.extensions) {
    const expectedPath = `extensions/duckdb-wasm/${manifest.duckdbEngineVersion}/${entry.platform}/${entry.name}.duckdb_extension.wasm`;
    const expectedURL = `https://extensions.duckdb.org/${manifest.duckdbEngineVersion}/${entry.platform}/${entry.name}.duckdb_extension.wasm`;
    const key = `${entry.platform}/${entry.name}`;
    if (!extensionNames.includes(entry.name) || !platforms.includes(entry.platform) || entry.path !== expectedPath || entry.url !== expectedURL || !sha256Pattern.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 8 || entry.bytes > 32 * 1024 * 1024 || seen.has(key)) throw new Error(`Invalid extension manifest entry: ${key}.`);
    seen.add(key);
  }
  return manifest;
}
export async function sha256File(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}
export async function matchesAsset(filename, expected) {
  try {
    const info = await stat(filename);
    return info.isFile() && info.size === expected.bytes && await sha256File(filename) === expected.sha256;
  } catch (error) {if (error.code === 'ENOENT') return false; throw error;}
}
async function temporaryPath(destination) {
  await mkdir(path.dirname(destination), {recursive: true});
  return path.join(path.dirname(destination), `.${path.basename(destination)}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`);
}
async function atomicWrite(destination, bytes) {
  const temporary = await temporaryPath(destination);
  let handle;
  try {
    handle = await open(temporary, 'wx', 0o644);
    await handle.writeFile(bytes); await handle.sync(); await handle.close(); handle = undefined;
    await rename(temporary, destination);
  } finally {
    await handle?.close();
    await unlink(temporary).catch(error => {if (error.code !== 'ENOENT') throw error;});
  }
}
async function atomicCopy(source, destination) {
  const temporary = await temporaryPath(destination);
  try {await copyFile(source, temporary); await rename(temporary, destination);}
  finally {await unlink(temporary).catch(error => {if (error.code !== 'ENOENT') throw error;});}
}
export async function fetchExtension(entry, {fetchImpl = globalThis.fetch} = {}) {
  // Node fetch transparently decodes HTTP Content-Encoding, including Brotli.
  // Hash the decoded .wasm bytes, not the compressed transfer representation.
  const response = await fetchImpl(entry.url, {redirect: 'error', signal: AbortSignal.timeout(60_000)});
  if (!response.ok) throw new Error(`Could not download ${entry.platform}/${entry.name}: HTTP ${response.status}.`);
  if (!response.body) throw new Error(`Empty extension response: ${entry.name}.`);
  const chunks = [];
  let length = 0;
  const hash = createHash('sha256');
  for await (const chunk of response.body) {
    const bytes = Buffer.from(chunk); length += bytes.length;
    if (length > entry.bytes) throw new Error(`Unexpected size for ${entry.platform}/${entry.name}; expected ${entry.bytes} decoded bytes.`);
    hash.update(bytes); chunks.push(bytes);
  }
  if (length !== entry.bytes || hash.digest('hex') !== entry.sha256) throw new Error(`SHA256/size mismatch for ${entry.platform}/${entry.name}. Existing assets were kept; do not update the manifest without verifying the new files.`);
  const bytes = Buffer.concat(chunks, length);
  if (!bytes.subarray(0, 4).equals(wasmMagic)) throw new Error(`Downloaded asset is not WebAssembly: ${entry.platform}/${entry.name}.`);
  return bytes;
}
export async function ensureExtension(entry, assetRoot, options = {}) {
  const destination = path.join(assetRoot, ...entry.path.split('/'));
  if (await matchesAsset(destination, entry)) return {name: `${entry.platform}/${entry.name}`, status: 'cached'};
  const bytes = await fetchExtension(entry, options);
  await atomicWrite(destination, bytes);
  return {name: `${entry.platform}/${entry.name}`, status: 'downloaded'};
}
export async function prepareAssets({root = browserRoot, fetchImpl = globalThis.fetch, log = console.log} = {}) {
  const manifest = validateManifest(JSON.parse(await readFile(path.join(root, 'scripts/duckdb-extensions.json'), 'utf8')));
  const packageRoot = path.join(root, 'node_modules/@duckdb/duckdb-wasm');
  const packageInfo = JSON.parse(await readFile(path.join(packageRoot, 'package.json'), 'utf8'));
  if (packageInfo.version !== manifest.duckdbWasmVersion) throw new Error(`DuckDB-Wasm ${manifest.duckdbWasmVersion} is required; installed ${packageInfo.version}. Run npm ci.`);
  const output = path.join(root, 'public/duckdb');
  const results = [];
  for (const entry of manifest.runtimeFiles) {
    const source = path.join(packageRoot, 'dist', entry.name);
    if (!await matchesAsset(source, entry)) throw new Error(`Pinned runtime file failed verification: ${entry.name}. Run npm ci to restore the package.`);
    const destination = path.join(output, entry.name);
    const cached = await matchesAsset(destination, entry);
    if (!cached) await atomicCopy(source, destination);
    results.push({name: entry.name, status: cached ? 'cached' : 'copied'});
  }
  for (const entry of manifest.extensions) results.push(await ensureExtension(entry, output, {fetchImpl}));
  for (const result of results) log(`${result.status}: ${result.name}`);
  return results;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  prepareAssets().catch(error => {console.error(`Asset preparation failed: ${error.message}`); process.exitCode = 1;});
}
