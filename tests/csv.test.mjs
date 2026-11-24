import test from 'node:test';
import assert from 'node:assert/strict';
import {csvBlob} from '../src/csv.js';

test('chart CSV preserves literal columns, quotes, Unicode, missing values and valid falsy values',async()=>{
 const blob=csvBlob({rowCount:3,columns:{'signal,raw':[0,null,-2.5],label:['a"b','line\nend','温度'],flag:[false,true,null]}});
 assert.equal(await blob.text(),'"signal,raw",label,flag\r\n0,"a""b",false\r\n,"line\nend",true\r\n-2.5,温度,');
 assert.match(blob.type,/text\/csv/);
});

test('large chart CSV retains every row and exact endpoints across output chunks',async()=>{
 const count=315034,x=Array.from({length:count},(_,index)=>index),signal=x.map(value=>2*value+1);
 const text=await csvBlob({rowCount:count,columns:{x,signal}}).text();
 assert.equal(text.split('\r\n').length,count+1);
 assert.ok(text.startsWith('x,signal\r\n0,1\r\n'));
 assert.ok(text.endsWith(`${count-1},${2*count-1}`));
});
