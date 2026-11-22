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
