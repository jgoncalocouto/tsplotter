import test from 'node:test';
import assert from 'node:assert/strict';
import {quoteIdentifier, quoteValue, columnKind, buildFilterSQL, buildTimeExpression, buildQuerySQL, makeBinEdges, assignBins, binLabel, inDisplayTimezone} from '../src/data-worker.js';
const schema = [{name:'time',type:'TIMESTAMP',kind:'datetime'}, {name:'sensor "A"',type:'DOUBLE',kind:'numeric'}, {name:'value',type:'DOUBLE',kind:'numeric'}, {name:'group',type:'VARCHAR',kind:'categorical'}];

test('identifiers and values are separately escaped', () => {
  assert.equal(quoteIdentifier('sensor "A"'), '"sensor ""A"""');
  assert.equal(quoteValue("a' OR true --"), "'a'' OR true --'");
  assert.equal(quoteValue(12345678901234567890n), '12345678901234567890');
  assert.throws(() => quoteIdentifier('a\0b'));
  assert.throws(() => quoteValue(Infinity));
});
test('type classification preserves integer, date and categorical semantics', () => {
  assert.equal(columnKind('BIGINT'), 'numeric'); assert.equal(columnKind('DECIMAL(38, 12)'), 'numeric');
  assert.equal(columnKind('TIMESTAMP WITH TIME ZONE'), 'datetime'); assert.equal(columnKind('BOOLEAN'), 'categorical');
});
test('numeric ranges are inclusive and combine left-to-right', () => {
  const sql = buildFilterSQL([{kind:'numeric',column:'value',low:1,up:2}, {kind:'categorical',column:'group',op:'is one of',values:['A'],logic:'OR'}, {kind:'numeric',column:'value',up:8,logic:'AND'}], schema);
  assert.match(sql,/"value" AS DOUBLE\) >= 1/); assert.match(sql,/"value" AS DOUBLE\) <= 2/);
  assert.ok(sql.lastIndexOf('AND') > sql.indexOf('OR')); assert.ok(sql.startsWith('((('));
});
test('empty filters are noops and unknown columns cannot become SQL', () => {
  assert.equal(buildFilterSQL([{kind:'numeric',column:'value'},{kind:'categorical',column:'group',values:[]}],schema),'TRUE');
  assert.throws(()=>buildFilterSQL([{kind:'numeric',column:'missing',low:1}],schema),/not in this dataset/);
  assert.throws(()=>buildFilterSQL([{kind:'numeric',column:'value',low:1,logic:'AND 1=1'}],schema));
});
test('literal contains does not treat wildcard characters as regex', () => {
  const sql=buildFilterSQL([{kind:'categorical',column:'group',op:'contains',values:['A_%'],caseSensitive:false}],schema);
  assert.match(sql,/contains\(lower\(CAST\("group"/); assert.match(sql,/'a_%'/); assert.doesNotMatch(sql,/LIKE|regexp/);
});
test('membership remains case-sensitive like Python filter',()=>{
  assert.doesNotMatch(buildFilterSQL([{kind:'categorical',column:'group',op:'is one of',values:['A'],caseSensitive:false}],schema),/lower/);
});
test('null-match includes missing values even for negated text filters',()=>{
  assert.match(buildFilterSQL([{kind:'categorical',column:'group',op:'not contains',values:['a'],naMatch:true}],schema),/OR "group" IS NULL/);
});
test('datetime filters use typed timestamps and nullable bounds',()=>{
  const sql=buildFilterSQL([{kind:'datetime',column:'time',low:'2024-01-01',up:'2024-12-31T23:59:59'}],schema);
  assert.match(sql,/TRY_CAST\("time" AS TIMESTAMP\)/); assert.match(sql,/'2024-01-01' AS TIMESTAMP/);
});
test('row index filters address generated row number',()=>{
  assert.match(buildFilterSQL([{kind:'numeric',column:'__row__',low:10}],schema,'internal'),/"internal" AS DOUBLE/);
});
test('epoch and custom time parsing are explicit',()=>{
  assert.match(buildTimeExpression('value',schema,{profile:'epoch',unit:'seconds'}), /\* 1000/);
  assert.match(buildTimeExpression('group',schema,{format:'%d/%m/%Y'}),/try_strptime/);
  assert.equal(buildTimeExpression('time',schema,{}),'"time"');
});
for (const mode of ['trends','scatter','grouped-scatter','rolling','derivatives']) test(`${mode} queries never sample or limit chart rows`,()=>{
  const sql=buildQuerySQL({mode,x:'time',y:['value'],color:'group'},schema);
  assert.doesNotMatch(sql,/\bLIMIT\b|TABLESAMPLE|USING SAMPLE|random\(/i);
});
test('only previews have a bounded row count; export remains complete',()=>{
  assert.match(buildQuerySQL({mode:'preview',limit:200},schema),/LIMIT 200$/);
  assert.doesNotMatch(buildQuerySQL({mode:'export'},schema),/LIMIT/);
});
test('all source identifiers are validated and deduplicated',()=>{
  assert.throws(()=>buildQuerySQL({mode:'trends',y:['DROP TABLE foo']},schema));
  const sql=buildQuerySQL({mode:'scatter',x:'value',y:['value'],color:'group'},schema);
  assert.equal((sql.match(/AS "value"/g)||[]).length,1);
});
test('scatter excludes selected incomplete/nonfinite rows and keeps source order',()=>{
  const sql=buildQuerySQL({mode:'scatter',x:'value',y:['sensor "A"'],color:'group'},schema);
  assert.match(sql,/isfinite/); assert.match(sql,/"group" IS NOT NULL/); assert.match(sql,/ORDER BY "__plotter_internal_row__"$/);
});
test('rolling window includes current by default and requires valid counts when requested',()=>{
  const sql=buildQuerySQL({mode:'rolling',x:'time',y:['value'],window:3,requireFull:true},schema);
  assert.match(sql,/ROWS BETWEEN 2 PRECEDING AND CURRENT ROW/); assert.match(sql,/>= 3 THEN/);
  for(const stat of ['min','mean','max']) assert.ok(sql.includes(`value__roll_${stat}`));
});
test('rolling excluding current uses exactly previous n rows',()=>{
  const sql=buildQuerySQL({mode:'rolling',y:['value'],window:5,includeCurrent:false},schema);
  assert.match(sql,/ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING/);
  assert.throws(()=>buildQuerySQL({mode:'rolling',y:['value'],window:0},schema));
});
test('derivatives use backward differences with zero-spacing guard and unsampled outputs',()=>{
  const sql=buildQuerySQL({mode:'derivatives',x:'time',y:['value'],derivativeUnit:'hours',smoothWindow:3},schema);
  assert.match(sql,/epoch\("time"\) \/ 3600/);
  assert.match(sql,/\(v - lag\(v\) OVER w\) \/ nullif\(dx - lag\(dx\) OVER w, 0\)/);
  assert.match(sql,/\(d1 - lag\(d1\) OVER w\) \/ nullif\(dx - lag\(dx\) OVER w, 0\)/);
  assert.match(sql,/__order > 1 THEN avg\(d1\)/); assert.match(sql,/__order > 2 THEN avg\(d2\)/);
});
test('decimal bin edges do not shift common decimal boundaries',()=>{
  assert.deepEqual(makeBinEdges(0,0.3,{mode:'width',width:0.1}),[0,0.1,0.2,0.3]);
  assert.deepEqual(makeBinEdges(0,1,{mode:'count',count:4}),[0,0.25,0.5,0.75,1]);
});
test('bin groups put interior edges in next bin and include final maximum',()=>{
  const result=assignBins([0,0.1,0.2,0.3],{mode:'width',width:0.1});
  assert.deepEqual(result.counts,[1,1,2]);
  assert.deepEqual(result.groups,['[0, 0.1)','[0.1, 0.2)','[0.2, 0.3]','[0.2, 0.3]']);
});
test('empty bins stay in membership counts and widths can extend beyond maximum',()=>{
  const result=assignBins([1,4],{mode:'width',width:2});
  assert.deepEqual(result.edges,[1,3,5]); assert.deepEqual(result.counts,[1,1]);
  const many=assignBins([0,10],{mode:'count',count:5}); assert.deepEqual(many.counts,[1,0,0,0,1]);
});
test('constant numeric grouping has a single exact label',()=>{
  const result=assignBins([1.1,1.1,null,Infinity]); assert.deepEqual(result.labels,['1.1']); assert.deepEqual(result.counts,[2]);
  assert.equal(binLabel([1.1,1.1],0),'1.1');
});
test('binning rejects unsafe integer precision, excessive bins and invalid width',()=>{
  assert.throws(()=>assignBins([9007199254740992,9007199254740994]),/too large/);
  assert.throws(()=>makeBinEdges(0,100,{mode:'width',width:0.01}),/more than 100/);
  assert.throws(()=>makeBinEdges(0,100,{mode:'width',width:0}));
  assert.throws(()=>assignBins([null,Infinity]),/no finite/);
});

test('original Unix/DMY profiles and keep timezone are understood',()=>{
 assert.match(buildTimeExpression('value',schema,{profile:'Unix epoch (milliseconds)'}),/\* 1 /);
 assert.match(buildTimeExpression('group',schema,{profile:'DMY (dd/mm/yyyy)'}),/%d\/%m\/%Y/);
 assert.doesNotMatch(buildTimeExpression('time',schema,{profile:'Auto-detect',outputTimezone:'(keep)'}),/timezone/);
});
test('display timezone retains instant and uses actual DST offset',()=>{
 const summer=inDisplayTimezone('2026-07-01T12:00:00.000Z','Europe/Lisbon');
 assert.equal(summer,'2026-07-01T13:00:00.000+01:00');
 assert.equal(new Date(summer).toISOString(),'2026-07-01T12:00:00.000Z');
 assert.equal(inDisplayTimezone('2026-01-01T12:00:00.000Z','Europe/Lisbon'),'2026-01-01T12:00:00.000+00:00');
});
