import test from 'node:test';
import assert from 'node:assert/strict';
import {axisRangeNumbers,comparisonAxes,linkedRangePatch} from '../src/comparisons.js';
const layout={xaxis:{type:'linear',range:[0,100]},yaxis:{type:'linear',range:[0,20]}};
test('shared date ranges preserve displayed wall times without applying the browser local timezone',()=>{
 assert.deepEqual(axisRangeNumbers(['2026-07-01 12:00:00','2026-07-01T13:00:00'],'date'),[Date.UTC(2026,6,1,12),Date.UTC(2026,6,1,13)]);
 assert.deepEqual(axisRangeNumbers(['2026-07-01T12:00:00+01:00',Date.UTC(2026,6,1,13)],'date'),[Date.UTC(2026,6,1,11),Date.UTC(2026,6,1,13)]);
});
test('linking propagates matching variables and leaves unrelated axes independent',()=>{
 const source=comparisonAxes('Scatter',{x:'time',y:'temperature'},null,layout);
 const target=comparisonAxes('Scatter',{x:'time',y:'power'},null,layout);
 assert.deepEqual(linkedRangePatch({'xaxis.range[0]':10,'xaxis.range[1]':30,'yaxis.range':[2,4]},source,layout,target,{x:true,y:true}),{'xaxis.range':[10,30],'xaxis.autorange':false});
 assert.deepEqual(linkedRangePatch({'xaxis.autorange':true},source,layout,target,{x:true}),{'xaxis.autorange':true});
 assert.deepEqual(linkedRangePatch({'xaxis.range':[10,30]},source,layout,target,{x:false}),{});
 target.xaxis.scale='log';assert.deepEqual(linkedRangePatch({'xaxis.range':[10,30]},source,layout,target,{x:true}),{});
});
test('distribution ranges match selected signals unless shared bins explicitly establish a common axis',()=>{
 const a={columns:['a']},b={columns:['b']};
 const source=comparisonAxes('Distribution',a,null,layout),target=comparisonAxes('Distribution',b,null,layout);
 assert.deepEqual(linkedRangePatch({'xaxis.range':[0,10]},source,layout,target,{x:true}),{});
 assert.deepEqual(linkedRangePatch({'xaxis.range':[0,10]},comparisonAxes('Distribution',a,null,layout,true),layout,comparisonAxes('Distribution',b,null,layout,true),{x:true}),{'xaxis.range':[0,10],'xaxis.autorange':false});
});
test('derivative orders and units never link to the wrong vertical axis',()=>{
 const three={...layout,xaxis2:{},xaxis3:{},yaxis2:{},yaxis3:{}};
 const panel={y:'v',timeUnit:'seconds',xMode:'index step (1)'};
 const a=comparisonAxes('Derivatives',panel,'time',three),b=comparisonAxes('Derivatives',{...panel,timeUnit:'minutes'},'time',three);
 assert.deepEqual(linkedRangePatch({'yaxis2.range':[0,1]},a,three,b,{y:true}),{});
 assert.deepEqual(linkedRangePatch({'yaxis.range':[0,1]},a,three,b,{y:true}),{'yaxis.range':[0,1],'yaxis.autorange':false});
});

test('nested Trend axes use each row signal set while every horizontal axis keeps the shared variable',()=>{
 const three={...layout,xaxis2:{},xaxis3:{},yaxis2:{type:'linear'},yaxis3:{type:'log'}};
 const panel={subplots:[{columns:['temperature']},{columns:['humidity','temperature','humidity']},{columns:['pressure']}]};
 const axes=comparisonAxes('Trends',panel,'time',three);
 assert.deepEqual([axes.xaxis.key,axes.xaxis2.key,axes.xaxis3.key],['field:time','field:time','field:time']);
 assert.equal(axes.yaxis.key,'signals:["temperature"]');
 assert.equal(axes.yaxis2.key,'signals:["humidity","temperature"]');
 assert.equal(axes.yaxis3.key,'signals:["pressure"]');
 assert.equal(axes.yaxis3.scale,'log');
});

test('external Trend Y links follow equal signal sets across row order and reject unrelated variables or scales',()=>{
 const three={...layout,xaxis2:{},xaxis3:{},yaxis2:{},yaxis3:{type:'log'}};
 const source=comparisonAxes('Trends',{subplots:[{columns:['temperature']},{columns:['humidity','temperature']},{columns:['pressure']}]},'time',three);
 const target=comparisonAxes('Trends',{subplots:[{columns:['temperature','humidity']},{columns:['temperature']},{columns:['pressure']}]},'time',three);
 assert.deepEqual(linkedRangePatch({'yaxis.range':[10,20]},source,three,target,{y:true}),{'yaxis2.range':[10,20],'yaxis2.autorange':false});
 assert.deepEqual(linkedRangePatch({'yaxis2.range':[2,8]},source,three,target,{y:true}),{'yaxis.range':[2,8],'yaxis.autorange':false});
 assert.deepEqual(linkedRangePatch({'yaxis3.range':[0,2]},source,three,target,{y:true}),{'yaxis3.range':[0,2],'yaxis3.autorange':false});
 target.yaxis3.scale='linear';
 assert.deepEqual(linkedRangePatch({'yaxis3.range':[0,2]},source,three,target,{y:true}),{});
 assert.deepEqual(linkedRangePatch({'yaxis.range':[10,20]},source,three,target,{y:false}),{});
});

test('nested Trend X linking reaches every matching row and reset preserves unrelated X variables',()=>{
 const two={...layout,xaxis2:{},yaxis2:{}};
 const panel={subplots:[{columns:['temperature']},{columns:['humidity']}]};
 const source=comparisonAxes('Trends',panel,'time',two),target=comparisonAxes('Trends',panel,'time',two);
 assert.deepEqual(linkedRangePatch({'xaxis2.range':[1,5]},source,two,target,{x:true}),{'xaxis.range':[1,5],'xaxis.autorange':false,'xaxis2.range':[1,5],'xaxis2.autorange':false});
 assert.deepEqual(linkedRangePatch({'xaxis2.autorange':true},source,two,target,{x:true}),{'xaxis.autorange':true,'xaxis2.autorange':true});
 assert.deepEqual(linkedRangePatch({'xaxis.range':[1,5]},source,two,comparisonAxes('Trends',panel,'samples',two),{x:true}),{});
});

test('blank Trend rows have no linkable vertical variable and legacy flat panels retain their signal identity',()=>{
 const two={...layout,xaxis2:{},yaxis2:{}};
 const blank=comparisonAxes('Trends',{subplots:[{columns:['temperature']},{columns:[]}]},'time',two);
 assert.equal(blank.yaxis2,undefined);assert.equal(blank.xaxis2.key,'field:time');
 assert.deepEqual(linkedRangePatch({'yaxis2.range':[0,1]},blank,two,blank,{y:true}),{});
 const flat=comparisonAxes('Trends',{columns:['temperature']},'time',layout);
 assert.equal(flat.yaxis.key,blank.yaxis.key);
 assert.deepEqual(linkedRangePatch({'yaxis.range':[5,9]},flat,layout,blank,{y:true}),{'yaxis.range':[5,9],'yaxis.autorange':false});
});
