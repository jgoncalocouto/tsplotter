import test from 'node:test';
import assert from 'node:assert/strict';
import {TrendDataCache,projectTrendResult} from '../src/trend-cache.js';
const source=()=>({columns:{t:[0,1,2,3],a:[1,null,null,4],b:[null,2,null,5],c:[1,2,3,4]},rowCount:4,totalRows:4,sourceRows:7,
 meta:{kind:'trends',rows:{source:7,matched:5,filteredOut:2,invalidTime:1,usable:4,excludedValues:0},seriesCounts:{a:{usable:2,excludedValues:2},b:{usable:2,excludedValues:2},c:{usable:4,excludedValues:0}}}});
const request=(y,filters=[])=>({mode:'trends',x:'t',y,filters});
test('multiple instances load one full union and share arrays with exact per-instance accounting',async()=>{
 const cache=new TrendDataCache(),raw=source(),calls=[];
 const results=await cache.query([request(['a','a']),request(['a','b']),request(['c'])],async spec=>{calls.push(spec);return raw;});
 assert.deepEqual(calls.map(spec=>spec.y),[['a','b','c']]);
 for(const result of results)assert.equal(result.columns.t,raw.columns.t);
 assert.equal(results[0].columns.a,raw.columns.a);assert.equal(results[1].columns.a,raw.columns.a);
 assert.deepEqual(results.map(result=>result.meta.rows.usable),[2,3,4]);
 assert.deepEqual(results.map(result=>result.meta.rows.excludedValues),[2,1,0]);
 assert.deepEqual(Object.keys(results[0].columns),['t','a']);assert.deepEqual(Object.keys(results[1].meta.seriesCounts),['a','b']);
 assert.equal(raw.meta.rows.usable,4);
});
test('layout/style redraws and subsets reuse data; removed signals are not retained indefinitely',async()=>{
 const cache=new TrendDataCache();let calls=0;
 const load=async()=>{calls++;return source();};
 await cache.query([request(['a','b'])],load);
 assert.equal(cache.canReuse([request(['b']),request(['a'])]),true);
 await cache.query([request(['b']),request(['a'])],load);assert.equal(calls,1);
 await cache.query([request(['a'])],load);assert.equal(calls,1);
 assert.deepEqual(Object.keys(cache.entry.result.columns),['t','a']);
 assert.equal(cache.canReuse([request(['b'])]),false);
 await cache.query([request(['b'])],load);assert.equal(calls,2);
 await cache.query([request(['b'],[{column:'a',low:1}])],load);assert.equal(calls,3);
 cache.clear();await cache.query([request(['b'])],load);assert.equal(calls,4);
});
test('changing the dataset during a load cannot repopulate the cache with stale data',async()=>{
 const cache=new TrendDataCache();let finish;
 const pending=cache.query([request(['a'])],()=>new Promise(resolve=>{finish=resolve;}));
 cache.clear();finish(source());await assert.rejects(pending,{name:'AbortError'});assert.equal(cache.entry,null);
});
test('choosing the X field as a new Y signal loads its missing validity counts',async()=>{
 const cache=new TrendDataCache();await cache.query([request(['a'])],async()=>source());
 assert.equal(cache.canReuse([request(['t'])]),false);
 const results=await cache.query([request(['t'])],async spec=>{
  assert.deepEqual(spec.y,['t']);const raw=source();raw.meta.seriesCounts.t={usable:4,excludedValues:0};return raw;
 });
 assert.deepEqual(Object.keys(results[0].columns),['t']);assert.equal(results[0].meta.rows.usable,4);
});
test('projection respects invalid numeric X, all-empty signals and independent scopes',async()=>{
 const raw=source();raw.columns.t[3]=null;raw.meta.seriesCounts.a.usable=1;raw.meta.seriesCounts.b.usable=1;
 assert.equal(projectTrendResult(raw,request(['a','b'])).meta.rows.usable,2);
 const cache=new TrendDataCache();await assert.rejects(cache.query([request(['a']),{...request(['b']),x:'other'}],async()=>raw),/same dataset/);
});
