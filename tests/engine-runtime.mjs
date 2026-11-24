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
  await e.importRows([{x:0,v:null},{x:1,v:5},{x:2,v:null}]);
  const emptyFiltered=await e.query({mode:'distribution',y:['v'],filters:[{kind:'numeric',column:'x',low:4}]});
  const emptyFinite=await e.query({mode:'distribution',y:['v'],filters:[{kind:'numeric',column:'x',up:0}]});
  const validAfterEmpty=await e.query({mode:'distribution',y:['v'],bins:{mode:'count',count:3}});
  await e.importRows([
   {x:0,date:'2026-01-01',v:0,w:10},{x:1,date:'2026-01-02',v:1,w:null},
   {x:2,date:'bad',v:2,w:20},{x:3,date:'2026-01-04',v:null,w:30},
   {x:4,date:'bad',v:null,w:null},{x:5,date:'2026-01-06',v:null,w:null},
   {x:6,date:'2026-01-07',v:6,w:60},{x:7,date:'2026-01-08',v:7,w:70},
  ]);
  const ledgerOptions={x:'x',preparedAxis:{x:'date',timeAxis:{profile:'Auto-detect'}},filters:[{kind:'numeric',column:'x',low:1,up:6}]};
  const ledgerScatter=await e.query({...ledgerOptions,mode:'scatter',y:['v']});
  const ledgerTrends=await e.query({...ledgerOptions,mode:'trends',y:['v','w']});
  const ledgerRolling=await e.query({...ledgerOptions,mode:'rolling',y:['v'],window:2});
  const ledgerDerivative=await e.query({...ledgerOptions,mode:'derivatives',y:['v'],derivativeUnit:'numeric'});
  const ledgerDistribution=await e.query({...ledgerOptions,mode:'distribution',y:['v','w'],bins:{mode:'count',count:2}});
  const ledgerSummary=await e.summary(ledgerOptions.filters,{preparedAxis:ledgerOptions.preparedAxis});
  const ledgerPreview=await e.query({...ledgerOptions,mode:'preview'});
  const ledgerCSVResult=await e.exportCSV(ledgerOptions.filters,{preparedAxis:ledgerOptions.preparedAxis});
  const ledgerCSV={rowCount:ledgerCSVResult.rowCount,meta:ledgerCSVResult.meta,isBlob:ledgerCSVResult.blob instanceof Blob,text:await ledgerCSVResult.blob.text(),hasColumns:Object.hasOwn(ledgerCSVResult,'columns')};
  const pooledV=await e.query({...ledgerOptions,mode:'distribution',y:['v'],bins:{mode:'count',count:2},binFitColumns:['v','w']});
  const pooledW=await e.query({...ledgerOptions,mode:'distribution',y:['w'],bins:{mode:'count',count:2},binFitColumns:['v','w']});
  const pooledDataset=await e.query({...ledgerOptions,mode:'distribution',y:['v'],bins:{mode:'count',count:2,scope:'dataset'},binFitColumns:['v','w']});
  const explicitBins=await e.query({...ledgerOptions,mode:'distribution',y:['w'],bins:{edges:pooledDataset.meta.binEdges}});
  let outsideEdgesError='';try{await e.query({...ledgerOptions,mode:'distribution',y:['v'],bins:{edges:[2,4]}});}catch(error){outsideEdgesError=error.message;}
  await e.importRows(Array.from({length:10},(_,x)=>({x,v:2*x})));
  const binOptions={mode:'scatter',x:'x',y:['v'],color:'x',marker:'x',filters:[{kind:'numeric',column:'x',low:3,up:6}]};
  const filteredBins=await e.query({...binOptions,colorBinning:{mode:'count',count:2},markerBinning:{mode:'count',count:2}});
  const datasetBins=await e.query({...binOptions,colorBinning:{mode:'count',count:2,scope:'dataset'},markerBinning:{mode:'count',count:2,scope:'dataset'}});
  const datasetBinsLater=await e.query({...binOptions,filters:[{kind:'numeric',column:'x',low:5,up:7}],colorBinning:{mode:'count',count:2,scope:'dataset'}});
  const emptyGrouped=await e.query({...binOptions,filters:[{kind:'numeric',column:'x',low:100}],colorBinning:{mode:'count',count:2}});
  const emptyDatasetGrouped=await e.query({...binOptions,filters:[{kind:'numeric',column:'x',low:100}],colorBinning:{mode:'count',count:2,scope:'dataset'}});
  await e.importRows(['D','A','C','B'].map((station,x)=>({x,v:x+1,station,level:[40,10,30,20][x]})));
  const domainSpec={mode:'scatter',x:'x',y:['v'],color:'station',marker:'level'};
  const domainFull=await e.query(domainSpec),domainFiltered=await e.query({...domainSpec,filters:[{kind:'numeric',column:'x',low:1}]});
  const continuousDomain=await e.query({mode:'scatter',x:'x',y:['v'],color:'level'});
  const numericDiscreteDomain=await e.query({mode:'scatter',x:'x',y:['v'],color:'level',colorMode:'discrete'});
  const {buildFigure}=await import('/src/charts.js');
  const domainSettings={plot_type:'Scatter',scatter_panels:[{x:'x',y:'v',color:'station',marker:'level'}]};
  const domainFigure=buildFigure(domainFull,domainSettings,'x'),filteredDomainFigure=buildFigure(domainFiltered,domainSettings,'x');
  const encodings={fullColor:domainFigure.data[0].marker.color,fullSymbol:domainFigure.data[0].marker.symbol,filteredColor:filteredDomainFigure.data[0].marker.color,filteredSymbol:filteredDomainFigure.data[0].marker.symbol};
  await e.importRows([{x:0,v:1,station:'Z'}]);
  const resetDomain=await e.query({mode:'scatter',x:'x',y:['v'],color:'station'});
  const wideRows=Array.from({length:257},(_,row)=>Object.fromEntries(Array.from({length:233},(_,column)=>[`signal_${column}`,row!==256&&row%233===column?row:null])));
  wideRows[0].signal_1=1;
  // Sparse JSON objects wider than200 fields are inferred as one MAP by DuckDB.
  // CSV gives this regression a stable233-column numeric schema.
  const wideNames=Object.keys(wideRows[0]),wideText=wideNames.join(',')+'\n'+wideRows.map(row=>wideNames.map(name=>row[name]??'').join(',')).join('\n');
  const wideMetadata=await e.importFile(new File([wideText],'sparse-summary.csv',{type:'text/csv'}));
  const wideSummary=await e.summary([]),wideCSVResult=await e.exportCSV();
  const wideCSV={rowCount:wideCSVResult.rowCount,isBlob:wideCSVResult.blob instanceof Blob,text:await wideCSVResult.blob.text(),meta:wideCSVResult.meta};
  await e.importRows([{order:3,text:'comma,"quote"\nline',empty:null},{order:1,text:'é😀',empty:null},{order:2,text:'last',empty:null}]);
  const streamCSVResult=await e.exportCSV(),filteredCSVResult=await e.exportCSV([{kind:'numeric',column:'order',up:2}],{columns:['text','order']}),emptyCSVResult=await e.exportCSV([{kind:'numeric',column:'order',low:10}]);
  const csvExports={all:await streamCSVResult.blob.text(),filtered:await filteredCSVResult.blob.text(),empty:await emptyCSVResult.blob.text(),rows:streamCSVResult.rowCount,filteredRows:filteredCSVResult.rowCount,emptyRows:emptyCSVResult.rowCount};
  await e.importFile(new File(['large,nullable\n9223372036854775807,\n-9223372036854775808,1\n'],'exact-integers.csv',{type:'text/csv'}));
  const integerCSV=await (await e.exportCSV()).blob.text();
  await e.importRows([{date:'01/02/2026 12:30:00',v:1},{date:'02/02/2026 12:30:00',v:2},{date:'broken',v:3}]);
  const axis={x:'date',timeAxis:{profile:'DMY (dd/mm/yyyy)'}};
  const dmy=await e.query({mode:'trends',x:'date',y:['v'],timeAxis:axis.timeAxis});
  const f=await e.query({mode:'scatter',x:'v',y:['v'],preparedAxis:axis,filters:[{kind:'datetime',column:'date',low:'2026-02-02',timeAxis:axis.timeAxis}]});
  const preview=await e.query({mode:'preview',preparedAxis:axis});const summary=await e.summary([],{preparedAxis:axis});
  await e.importRows([{date:0,v:1},{date:1000,v:2}]);const epoch=await e.query({mode:'trends',x:'date',y:['v'],timeAxis:{profile:'Unix epoch (milliseconds)'}});
  await e.importRows([{date:'2026-03-29 00:30:00',v:0},{date:'2026-03-29 02:30:00',v:3600},{date:'2026-03-29 03:30:00',v:7200}]);
  const tz=await e.query({mode:'derivatives',x:'date',y:['v'],derivativeUnit:'seconds',timeAxis:{profile:'Auto-detect',timezone:'Europe/Lisbon',outputTimezone:'Europe/Lisbon'}});
  const timezoneDates=['2026-03-29T00:30:00.000Z','2026-03-29T01:30:00.000Z','2026-10-25T00:30:00.000Z','2026-10-25T01:30:00.000Z'];
  await e.importRows([...timezoneDates.map((date,index)=>({v:index+1,date,other:date})),{v:5,date:null,other:timezoneDates[0]}]);
  const timezoneOptions={x:'date',timeAxis:{profile:'Auto-detect',outputTimezone:'Europe/Lisbon'},columns:['v','date','other']};
  const timezonePreview=await e.query({...timezoneOptions,mode:'preview'}),timezoneExport=await e.exportCSV([],timezoneOptions);
  const timezoneCSV={text:await timezoneExport.blob.text(),rowCount:timezoneExport.rowCount,rows:timezoneExport.meta.rows,previewDates:timezonePreview.columns.date,otherDates:timezonePreview.columns.other};
  const origin=Date.parse('2026-01-01T00:00:00.357Z');
  const shortCSV='date,v\n'+Array.from({length:32},(_,index)=>`${new Date(origin+index*64).toISOString().slice(0,-1)},${index*index}`).join('\n');
  const shortMetadata=await e.importFile(new File([shortCSV],'millisecond-derivatives.csv',{type:'text/csv'}));
  const shortDerivatives=await e.query({mode:'derivatives',x:'date',y:['v'],derivativeUnit:'minutes'});
  const dayMetadata=await e.importFile(new File(['date,v\n2026-01-01,0\n2026-01-02,1\n2026-01-03,4\n'],'date-derivatives.csv',{type:'text/csv'}));
  const dayDerivatives=await e.query({mode:'derivatives',x:'date',y:['v'],derivativeUnit:'days'});
  await e.close();return {meta,d,rolling,h,g,s,emptyFiltered,emptyFinite,validAfterEmpty,ledgerScatter,ledgerTrends,ledgerRolling,ledgerDerivative,ledgerDistribution,ledgerSummary,ledgerPreview,ledgerCSV,pooledV,pooledW,pooledDataset,explicitBins,outsideEdgesError,filteredBins,datasetBins,datasetBinsLater,emptyGrouped,emptyDatasetGrouped,domainFull,domainFiltered,continuousDomain,numericDiscreteDomain,encodings,resetDomain,wideMetadata,wideSummary,wideCSV,csvExports,integerCSV,dmy,f,preview,summary,epoch,tz,timezoneCSV,shortMetadata,shortDerivatives,dayMetadata,dayDerivatives};
 });
 assert.equal(r.meta.rowCount,4);assert.deepEqual(r.d.columns.v__d1,[null,1,3,5]);assert.deepEqual(r.d.columns.v__d2,[null,null,2,2]);
 assert.deepEqual(r.rolling.columns.v__roll_mean,[null,null,.5,2.5]);assert.deepEqual(r.rolling.columns.v__roll_min,[null,null,0,1]);assert.deepEqual(r.rolling.columns.v__roll_max,[null,null,1,4]);
 assert.deepEqual(r.h.columns.v,[2,1,1]);assert.deepEqual(r.g.columns.v,[0,1,4,9]);assert.deepEqual(r.g.meta.groups.color.counts,[2,1,1]);assert.deepEqual(r.s.columns.mean,[1.5,3.5]);
 for(const [result,totalRows]of[[r.emptyFiltered,0],[r.emptyFinite,1]]){
  assert.deepEqual(result.columns,{bin_left:[],bin_right:[],bin_center:[],v:[]});assert.equal(result.rowCount,0);assert.equal(result.totalRows,totalRows);assert.equal(result.sourceRows,3);
  assert.equal(result.meta.kind,'distribution');assert.equal(result.meta.binCount,0);assert.deepEqual(result.meta.observationCounts,{v:0});
 }
 assert.equal(r.validAfterEmpty.totalRows,3);assert.equal(r.validAfterEmpty.columns.v.reduce((sum,count)=>sum+count,0),1);
 const singleLedger={source:8,matched:6,filteredOut:2,invalidTime:2,usable:2,excludedValues:2};
 for(const result of [r.ledgerScatter,r.ledgerRolling,r.ledgerDerivative])assert.deepEqual(result.meta.rows,singleLedger);
 for(const result of [r.ledgerTrends,r.ledgerDistribution]){
  assert.deepEqual(result.meta.rows,{...singleLedger,usable:3,excludedValues:1});
  assert.deepEqual(result.meta.seriesCounts,{v:{usable:2,excludedValues:2},w:{usable:2,excludedValues:2}});
 }
 assert.equal(r.ledgerTrends.rowCount,4);assert.deepEqual(r.ledgerTrends.columns.v,[1,null,null,6]);assert.deepEqual(r.ledgerRolling.columns.v,[1,null,null,6]);
 assert.equal(r.ledgerDerivative.rowCount,2);assert.deepEqual(r.ledgerDerivative.columns.v__d1,[null,1]);assert.deepEqual(r.ledgerDerivative.columns.v__d2,[null,null]);
 assert.deepEqual(r.ledgerDistribution.meta.observationCounts,{v:2,w:2});
 for(const result of [r.ledgerSummary,r.ledgerPreview])assert.deepEqual(result.meta.rows,{...singleLedger,usable:4,excludedValues:0});
 assert.deepEqual(r.ledgerSummary.meta.seriesCounts.v,{usable:2,excludedValues:2});
 assert.equal(r.ledgerCSV.isBlob,true);assert.equal(r.ledgerCSV.hasColumns,false);assert.equal(r.ledgerCSV.rowCount,4);assert.deepEqual(r.ledgerCSV.meta.rows,{...singleLedger,usable:4,excludedValues:0});
 assert.deepEqual(r.ledgerCSV.text.split('\r\n').slice(1).map(row=>Number(row.split(',')[0])),[1,3,5,6]);
 for(const result of [r.ledgerScatter,r.ledgerTrends,r.ledgerDistribution,r.ledgerSummary,r.ledgerPreview]){
  const rows=result.meta.rows;assert.equal(rows.source,rows.filteredOut+rows.invalidTime+rows.excludedValues+rows.usable);
 }
 assert.deepEqual(r.pooledV.meta.binEdges,[1,30.5,60]);assert.deepEqual(r.pooledW.meta.binEdges,r.pooledV.meta.binEdges);
 assert.deepEqual(r.pooledDataset.meta.binEdges,[0,35,70]);assert.deepEqual(r.explicitBins.meta.binEdges,r.pooledDataset.meta.binEdges);
 assert.equal(r.pooledV.meta.observationCounts.v,2);assert.equal(r.pooledW.meta.observationCounts.w,2);assert.equal(r.explicitBins.meta.observationCounts.w,2);assert.match(r.outsideEdgesError,/cover every finite selected value/);
 assert.deepEqual(r.filteredBins.meta.groups.color.edges,[3,4.5,6]);assert.deepEqual(r.datasetBins.meta.groups.color.edges,[0,4.5,9]);
 assert.deepEqual(r.datasetBins.meta.groups.marker.edges,r.datasetBins.meta.groups.color.edges);assert.deepEqual(r.datasetBinsLater.meta.groups.color.edges,r.datasetBins.meta.groups.color.edges);
 assert.deepEqual(r.datasetBins.columns.x,[3,4,5,6]);assert.deepEqual(r.datasetBins.columns.v,[6,8,10,12]);assert.equal(r.datasetBins.meta.groups.color.counts.reduce((sum,n)=>sum+n,0),4);
 assert.equal(r.emptyGrouped.rowCount,0);assert.equal(r.emptyGrouped.meta.rows.matched,0);assert.equal(r.emptyDatasetGrouped.rowCount,0);assert.deepEqual(r.emptyDatasetGrouped.meta.groups.color.counts,[0,0]);
 assert.deepEqual(r.domainFull.meta.categoryDomains,{color:['A','B','C','D'],marker:[10,20,30,40]});
 assert.deepEqual(r.domainFiltered.meta.categoryDomains,r.domainFull.meta.categoryDomains);
 assert.deepEqual(r.continuousDomain.meta.categoryDomains,{});assert.deepEqual(r.numericDiscreteDomain.meta.categoryDomains.color,[10,20,30,40]);
 assert.equal(new Set(r.encodings.fullColor).size,4);assert.equal(new Set(r.encodings.fullSymbol).size,4);
 assert.deepEqual(r.encodings.filteredColor,r.encodings.fullColor.slice(1));assert.deepEqual(r.encodings.filteredSymbol,r.encodings.fullSymbol.slice(1));
 assert.deepEqual(r.resetDomain.meta.categoryDomains.color,['Z']);
 assert.equal(r.wideMetadata.columns.filter(column=>column.kind==='numeric').length,233,'Sparse fixture must retain233 numeric source fields');
 assert.equal(r.wideSummary.rowCount,233);assert.equal(r.wideSummary.columns.column.length,233);assert.deepEqual(r.wideSummary.meta.rows,{source:257,matched:257,filteredOut:0,invalidTime:0,usable:256,excludedValues:1});
 for(let column=0;column<233;column++){
  const values=Array.from({length:256},(_,row)=>row%233===column?row:column===1&&row===0?1:null).filter(value=>value!==null);
  const mean=values.reduce((sum,value)=>sum+value,0)/values.length;
  const stddev=values.length>1?Math.sqrt(values.reduce((sum,value)=>sum+(value-mean)**2,0)/(values.length-1)):null;
  assert.equal(r.wideSummary.columns.valid[column],values.length);assert.equal(r.wideSummary.columns.min[column],Math.min(...values));assert.equal(r.wideSummary.columns.max[column],Math.max(...values));assert.equal(r.wideSummary.columns.mean[column],mean);
  if(stddev===null)assert.equal(r.wideSummary.columns.stddev[column],null);else assert.ok(Math.abs(r.wideSummary.columns.stddev[column]-stddev)<1e-10);
 }
 assert.equal(r.wideCSV.isBlob,true);assert.equal(r.wideCSV.rowCount,257);assert.equal(r.wideCSV.meta.rows.usable,257);const wideLines=r.wideCSV.text.split('\r\n');assert.equal(wideLines.length,258);assert.equal(wideLines[0].split(',').length,233);assert.ok(wideLines.slice(1).every(line=>line.split(',').length===233));
 assert.equal(r.csvExports.all,'order,text,empty\r\n3,"comma,""quote""\nline",\r\n1,é😀,\r\n2,last,');assert.equal(r.csvExports.filtered,'text,order\r\né😀,1\r\nlast,2');assert.equal(r.csvExports.empty,'order,text,empty');
 assert.equal(r.csvExports.rows,3);assert.equal(r.csvExports.filteredRows,2);assert.equal(r.csvExports.emptyRows,0);
 assert.equal(r.integerCSV,'large,nullable\r\n9223372036854775807,\r\n-9223372036854775808,1');
 assert.deepEqual(r.dmy.columns.date,['2026-02-01T12:30:00.000Z','2026-02-02T12:30:00.000Z']);assert.equal(r.dmy.meta.invalidXRows,1);assert.deepEqual(r.f.columns.v,[2]);assert.equal(r.preview.rowCount,2);assert.equal(r.summary.totalRows,2);
 assert.deepEqual(r.epoch.columns.date,['1970-01-01T00:00:00.000Z','1970-01-01T00:00:01.000Z']);assert.deepEqual(r.tz.columns.v__d1,[null,1,1]);assert.deepEqual(r.tz.columns.date,['2026-03-29T00:30:00.000+00:00','2026-03-29T02:30:00.000+01:00','2026-03-29T03:30:00.000+01:00']);
 const displayDates=['2026-03-29T00:30:00.000+00:00','2026-03-29T02:30:00.000+01:00','2026-10-25T01:30:00.000+01:00','2026-10-25T01:30:00.000+00:00'];
 const utcDates=['2026-03-29T00:30:00.000Z','2026-03-29T01:30:00.000Z','2026-10-25T00:30:00.000Z','2026-10-25T01:30:00.000Z'];
 assert.deepEqual(r.timezoneCSV.previewDates,displayDates);assert.deepEqual(r.timezoneCSV.otherDates,utcDates);
 assert.equal(r.timezoneCSV.text,'v,date,other'+displayDates.map((date,index)=>`\r\n${index+1},${date},${utcDates[index]}`).join(''));assert.equal(r.timezoneCSV.rowCount,4);
 assert.deepEqual(r.timezoneCSV.rows,{source:5,matched:5,filteredOut:0,invalidTime:1,usable:4,excludedValues:0});
 assert.equal(r.shortMetadata.columns.find(column=>column.name==='date').type,'TIMESTAMP');
 assert.equal(r.shortDerivatives.rowCount,32);assert.equal(r.shortDerivatives.columns.v__d1[0],null);assert.deepEqual(r.shortDerivatives.columns.v__d2.slice(0,2),[null,null]);
 for(let index=1;index<32;index++)assert.ok(Math.abs(r.shortDerivatives.columns.v__d1[index]-(2*index-1)*937.5)<1e-8);
 for(let index=2;index<32;index++)assert.ok(Math.abs(r.shortDerivatives.columns.v__d2[index]-1757812.5)<1e-8);
 assert.equal(r.dayMetadata.columns.find(column=>column.name==='date').type,'DATE');assert.deepEqual(r.dayDerivatives.columns.v__d1,[null,1,3]);assert.deepEqual(r.dayDerivatives.columns.v__d2,[null,null,2]);
 assert.deepEqual(writes,[]);assert.deepEqual(external,[]);console.log('PASS: row ledgers, scoped/shared/explicit bins, JSON, derivatives, rolling, histograms, summaries, DMY, prepared axes, epoch, DST, no uploads/external requests');
}finally{await browser.close();}
