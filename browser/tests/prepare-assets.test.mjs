import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, readdir, rm, writeFile, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {validateManifest, ensureExtension, fetchExtension} from '../scripts/prepare-assets.mjs';
const manifest = JSON.parse(await readFile(new URL('../scripts/duckdb-extensions.json', import.meta.url), 'utf8'));
const wasm = Buffer.from([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]);
const entry = {...manifest.extensions[0], bytes: wasm.length, sha256: createHash('sha256').update(wasm).digest('hex')};
async function temporary(run) {const root = await mkdtemp(path.join(tmpdir(), 'plotter-assets-')); try {await run(root);} finally {await rm(root, {recursive: true, force: true});}}
const destination = root => path.join(root, entry.path);

test('manifest pins the deployed runtime and six distinct official extension URLs', () => {
  assert.equal(validateManifest(manifest), manifest);
  assert.equal(manifest.duckdbWasmVersion, '1.30.0'); assert.equal(manifest.duckdbEngineVersion, 'v1.3.2');
  assert.equal(new Set(manifest.extensions.map(asset => asset.path)).size, 6);
});
test('manifest rejects alternate origins, traversal, duplicates and unexpected hashes', () => {
  for (const change of [asset => asset.url = 'https://example.org/extension.wasm', asset => asset.path = '../../private', asset => asset.sha256 = 'wrong']) {
    const changed = structuredClone(manifest); change(changed.extensions[0]); assert.throws(() => validateManifest(changed), /Invalid extension/);
  }
  const duplicate = structuredClone(manifest); duplicate.extensions[1] = {...duplicate.extensions[0]}; assert.throws(() => validateManifest(duplicate), /Invalid extension/);
});
test('verified cached bytes require no network', () => temporary(async root => {
  await mkdir(path.dirname(destination(root)), {recursive: true}); await writeFile(destination(root), wasm);
  const result = await ensureExtension(entry, root, {fetchImpl: () => {throw new Error('Network must not be used for valid cached assets');}});
  assert.equal(result.status, 'cached');
}));
test('missing asset downloads once and the verified result is then usable offline', () => temporary(async root => {
  let calls = 0;
  const fetchImpl = async (url, options) => {calls++; assert.equal(url, entry.url); assert.equal(options.redirect, 'error'); return new Response(wasm, {headers: {'Content-Encoding': 'br'}});};
  assert.equal((await ensureExtension(entry, root, {fetchImpl})).status, 'downloaded');
  assert.deepEqual(await readFile(destination(root)), wasm);
  assert.equal((await ensureExtension(entry, root, {fetchImpl})).status, 'cached'); assert.equal(calls, 1);
  assert.deepEqual(await readdir(path.dirname(destination(root))), [path.basename(destination(root))]);
}));
test('same-size cache tampering is repaired from verified official bytes', () => temporary(async root => {
  await mkdir(path.dirname(destination(root)), {recursive: true}); await writeFile(destination(root), Buffer.alloc(wasm.length));
  const result = await ensureExtension(entry, root, {fetchImpl: async () => new Response(wasm)});
  assert.equal(result.status, 'downloaded'); assert.deepEqual(await readFile(destination(root)), wasm);
}));
test('unexpected download hash never replaces existing bytes or leaves partial files', () => temporary(async root => {
  const original = Buffer.alloc(wasm.length, 2); await mkdir(path.dirname(destination(root)), {recursive: true}); await writeFile(destination(root), original);
  await assert.rejects(() => ensureExtension(entry, root, {fetchImpl: async () => new Response(Buffer.alloc(wasm.length, 3))}), /mismatch/);
  assert.deepEqual(await readFile(destination(root)), original);
  assert.deepEqual(await readdir(path.dirname(destination(root))), [path.basename(destination(root))]);
}));
test('HTTP failures, truncated/oversized responses and non-Wasm bytes fail explicitly', async () => {
  await assert.rejects(() => fetchExtension(entry, {fetchImpl: async () => new Response('missing', {status: 404})}), /HTTP 404/);
  await assert.rejects(() => fetchExtension(entry, {fetchImpl: async () => new Response(wasm.subarray(0, 7))}), /mismatch/);
  await assert.rejects(() => fetchExtension(entry, {fetchImpl: async () => new Response(Buffer.alloc(wasm.length + 1))}), /Unexpected size/);
  const bytes = Buffer.alloc(wasm.length); const wrongType = {...entry, sha256: createHash('sha256').update(bytes).digest('hex')};
  await assert.rejects(() => fetchExtension(wrongType, {fetchImpl: async () => new Response(bytes)}), /not WebAssembly/);
});
