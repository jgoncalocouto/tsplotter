import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
const baseURL=process.env.BASE_URL||'http://127.0.0.1:5174/';
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox','--disable-dev-shm-usage']});
const page=await browser.newPage();const writes=[],external=[];
page.on('request',r=>{if(r.method()!=='GET')writes.push(r.url());if(new URL(r.url()).origin!==new URL(baseURL).origin&&!r.url().startsWith('blob:'))external.push(r.url());});
try {
 await page.goto(baseURL,{waitUntil:'networkidle'});
 const r=await page.evaluate(async()=>{
  const {DataEngine}=await import('/src/engine-client.js');const e=new DataEngine({assetBase:new URL('duckdb/',document.baseURI).href});
  const meta=await e.importRows([{x:0,v:0},{x:1,v:1},{x:2,v:4},{x:3,v:9}]);
  const d=await e.query({mode:'derivatives',x:'x',y:['v'],derivativeUnit:'numeric'});
  const rolling=await e.query({mode:'rolling',x:'x',y:['v'],window:2,includeCurrent:false,requireFull:true});
  const h=await e.query({mode:'distribution',y:['v'],bins:{mode:'count',count:3}});
  const g=await e.query({mode:'grouped-scatter',x:'x',y:['v'],color:'v',colorBinning:{mode:'count',count:3}});
  const s=await e.summary([]);
  await e.importRows([{date:'01/02/2026 12:30:00',v:1},{date:'02/02/2026 12:30:00',v:2},{date:'broken',v:3}]);
  const axis={x:'date',timeAxis:{profile:'DMY (dd/mm/yyyy)'}};
  const dmy=await e.query({mode:'trends',x:'date',y:['v'],timeAxis:axis.timeAxis});
  const f=await e.query({mode:'scatter',x:'v',y:['v'],preparedAxis:axis,filters:[{kind:'datetime',column:'date',low:'2026-02-02',timeAxis:axis.timeAxis}]});
  const preview=await e.query({mode:'preview',preparedAxis:axis});const summary=await e.summary([],{preparedAxis:axis});
  await e.importRows([{date:0,v:1},{date:1000,v:2}]);const epoch=await e.query({mode:'trends',x:'date',y:['v'],timeAxis:{profile:'Unix epoch (milliseconds)'}});
  await e.importRows([{date:'2026-03-29 00:30:00',v:0},{date:'2026-03-29 02:30:00',v:3600},{date:'2026-03-29 03:30:00',v:7200}]);
  const tz=await e.query({mode:'derivatives',x:'date',y:['v'],derivativeUnit:'seconds',timeAxis:{profile:'Auto-detect',timezone:'Europe/Lisbon',outputTimezone:'Europe/Lisbon'}});
  await e.close();return {meta,d,rolling,h,g,s,dmy,f,preview,summary,epoch,tz};
 });
 assert.equal(r.meta.rowCount,4);assert.deepEqual(r.d.columns.v__d1,[null,1,3,5]);assert.deepEqual(r.d.columns.v__d2,[null,null,2,2]);
 assert.deepEqual(r.rolling.columns.v__roll_mean,[null,null,.5,2.5]);assert.deepEqual(r.rolling.columns.v__roll_min,[null,null,0,1]);assert.deepEqual(r.rolling.columns.v__roll_max,[null,null,1,4]);
 assert.deepEqual(r.h.columns.v,[2,1,1]);assert.deepEqual(r.g.columns.v,[0,1,4,9]);assert.deepEqual(r.g.meta.groups.color.counts,[2,1,1]);assert.deepEqual(r.s.columns.mean,[1.5,3.5]);
 assert.deepEqual(r.dmy.columns.date,['2026-02-01T12:30:00.000Z','2026-02-02T12:30:00.000Z']);assert.equal(r.dmy.meta.invalidXRows,1);assert.deepEqual(r.f.columns.v,[2]);assert.equal(r.preview.rowCount,2);assert.equal(r.summary.totalRows,2);
 assert.deepEqual(r.epoch.columns.date,['1970-01-01T00:00:00.000Z','1970-01-01T00:00:01.000Z']);assert.deepEqual(r.tz.columns.v__d1,[null,1,1]);assert.deepEqual(r.tz.columns.date,['2026-03-29T00:30:00.000+00:00','2026-03-29T02:30:00.000+01:00','2026-03-29T03:30:00.000+01:00']);
 assert.deepEqual(writes,[]);assert.deepEqual(external,[]);console.log('PASS: JSON, derivatives, rolling, histograms, bins, summaries, DMY, prepared axes, epoch, DST, no uploads/external requests');
}finally{await browser.close();}
