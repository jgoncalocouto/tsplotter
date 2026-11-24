/** Full-source Trend interaction regression. No sampling; reports contain timing/count metadata only. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp,readFile,writeFile,stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createInterface} from 'node:readline';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {chooseSignals,waitForChart} from './browser-controls.mjs';

const baseURL=process.env.BASE_URL||'http://127.0.0.1:5173/';
const fixture=path.resolve(process.env.REFERENCE_PARQUET||'tests/MasterLogD4_2_decoded_registers.parquet');
const artifacts=await mkdtemp(path.join(tmpdir(),'tsplotter-trend-performance-'));
const expectedRows=315034;
const signals=['R2045_Inlet_Water_Temp','R2046_Outlet_Water_Temp','R2048_Ambient_Temp_AT','R2012_Unit_mode'];
const hoverSelector=process.env.TREND_HOVER_SELECTOR||':is(.hoverlayer .hovertext,.hoverlayer .legend)';
const spikeSelector=process.env.TREND_SPIKE_SELECTOR||'.spikeline';
const interactionBudget=Number(process.env.TREND_INTERACTION_BUDGET_MS||2000);
const hoverBudget=Number(process.env.TREND_HOVER_BUDGET_MS||500);
const longTaskBudget=Number(process.env.TREND_LONG_TASK_BUDGET_MS||500);
const report={baseURL,sourceRows:expectedRows,sourceColumns:234,fixtureBytes:(await stat(fixture)).size,stages:[],checks:[],errors:[],workerErrors:[],graphicsWarnings:[],writes:[],external:[],budgets:{interactionMs:interactionBudget,hoverMs:hoverBudget,longTaskMs:longTaskBudget},artifacts};
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1050},acceptDownloads:true});page.setDefaultTimeout(120000);
page.on('pageerror',e=>report.errors.push(String(e).slice(0,1000)));
page.on('console',m=>{if(/(?:webgl.*(?:lost|too many|limit)|too many.*(?:webgl|context)|context.*lost)/i.test(m.text()))report.graphicsWarnings.push(m.text().slice(0,1000));});
page.on('request',r=>{if(['POST','PUT','PATCH','DELETE'].includes(r.method()))report.writes.push(r.url());if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==new URL(baseURL).origin)report.external.push(r.url());});
await page.addInitScript(()=>{
 window.__trendPerf={tasks:[],events:[],queries:[],contexts:new WeakMap(),workerErrors:[]};let next=1;
 new PerformanceObserver(list=>{for(const e of list.getEntries())window.__trendPerf.tasks.push({start:e.startTime,duration:e.duration});}).observe({type:'longtask',buffered:true});
 const post=Worker.prototype.postMessage,observed=new WeakSet();
 Worker.prototype.postMessage=function(m,...args){if(!observed.has(this)){observed.add(this);this.addEventListener('error',e=>window.__trendPerf.workerErrors.push(String(e.message)));}if(m?.method==='query')window.__trendPerf.queries.push({mode:m.args?.[0]?.mode});return post.call(this,m,...args);};
 const getContext=HTMLCanvasElement.prototype.getContext;
 HTMLCanvasElement.prototype.getContext=function(type,...args){const c=getContext.call(this,type,...args);if(c&&String(type).includes('webgl')&&!window.__trendPerf.contexts.has(this))window.__trendPerf.contexts.set(this,{context:c,id:next++});return c;};
});
const cdp=await page.context().newCDPSession(page);await cdp.send('Performance.enable');
const chart=index=>index?`#trend-chart-${index}`:'#chart';
const pass=text=>{report.checks.push(text);console.log(text);};
async function persist(){await writeFile(path.join(artifacts,'report.json'),JSON.stringify(report,null,2));}
async function field(key,value){const e=page.locator(`[data-panel-key="${key}"]`);if(!await e.isVisible())await e.evaluate(n=>n.closest('details').open=true);if(typeof value==='boolean')await e.setChecked(value);else if(await e.evaluate(n=>n.tagName==='SELECT'))await e.selectOption(value);else{await e.fill(String(value));await e.press('Tab');}}
async function update(panels=1){await page.locator('#render-button').click();await waitForChart(page);assert.equal(await page.locator('[data-trend-card]').count(),panels);}
async function queries(){return await page.evaluate(()=>window.__trendPerf.queries.length);}
async function hook(index=0){await page.locator(chart(index)).evaluate(g=>{if(typeof g.on!=='function')return;for(const [e,fn]of g.__performanceHooks||[])g.removeListener?.(e,fn);g.__performanceHooks=[];for(const e of ['plotly_afterplot','plotly_relayout','plotly_relayouting']){const fn=()=>window.__trendPerf.events.push({event:e,time:performance.now()});g.on(e,fn);g.__performanceHooks.push([e,fn]);}});}
async function graphState(index=0){return await page.locator(chart(index)).evaluate(g=>({
 traces:g.data.filter(t=>t.meta?.trendSeries).map(t=>({row:t.meta.subplotIndex,x:t.x.length,y:t.y.length})),
 ranges:Object.fromEntries(Object.entries(g._fullLayout).filter(([k])=>/^[xy]axis\d*$/.test(k)).map(([k,v])=>[k,v.range])),
 gl:[...g.querySelectorAll('canvas')].map(c=>window.__trendPerf.contexts.get(c)).filter(Boolean).map(({id,context})=>({id,lost:context.isContextLost(),width:context.drawingBufferWidth,height:context.drawingBufferHeight})),
 hovermode:g._fullLayout.hovermode,
}));}
async function assertPoints(panels=1){for(let i=0;i<panels;i++){const s=await graphState(i);assert.ok(s.traces.length);assert.ok(s.traces.every(t=>t.x===expectedRows&&t.y===expectedRows),'Every full-source point must remain in every subplot trace');assert.ok(s.gl.length&&s.gl.every(c=>!c.lost&&c.width&&c.height));}assert.equal(Number((await page.locator('#row-count').textContent()).replace(/[^0-9]/g,'')),expectedRows);}
async function measure(label,action,{panels=1,interactive=false,budget=interactionBudget,allowQuery=false}={}){
 await hook();const start=await page.evaluate(()=>({time:performance.now(),queries:window.__trendPerf.queries.length}));const wall=performance.now();let failure;
 try{await action();await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));}catch(e){failure=e;}
 const elapsed=performance.now()-wall;
 const result=await page.evaluate(start=>{const tasks=window.__trendPerf.tasks.filter(t=>t.start>=start.time),events=window.__trendPerf.events.filter(e=>e.time>=start.time);return {longTasks:tasks.length,mainThreadMs:tasks.reduce((n,t)=>n+t.duration,0),longestTaskMs:Math.max(0,...tasks.map(t=>t.duration)),lastGraphEventMs:events.length?events.at(-1).time-start.time:null,graphEvents:events.reduce((n,e)=>(n[e.event]=(n[e.event]||0)+1,n),{}),workerQueries:window.__trendPerf.queries.length-start.queries};},start);
 const row={label,elapsedMs:elapsed,...result,mainHeapBytes:(await cdp.send('Runtime.getHeapUsage')).usedSize,ok:!failure};report.stages.push(row);await persist();console.log(JSON.stringify(row));if(failure)throw failure;
 if(!allowQuery)assert.equal(result.workerQueries,0,'Interactions and cached Updates must not query data');
 await assertPoints(panels);
 if(interactive){assert.ok(elapsed<=budget,`${label} exceeded${budget}ms interaction budget (${Math.round(elapsed)}ms)`);assert.ok(result.longestTaskMs<=longTaskBudget,`${label} blocked the main thread for${result.longestTaskMs}ms`);}
 return row;
}
async function dragArea(row=0,index=0){const drag=page.locator(`${chart(index)} .draglayer .nsewdrag`).nth(row);await drag.scrollIntoViewIfNeeded();const box=await drag.boundingBox();assert.ok(box?.width>30&&box.height>30);const viewport=page.viewportSize();assert.ok(box.y+box.height*.5>=0&&box.y+box.height*.5<viewport.height,'The actual drag target must be within the viewport');return box;}
async function modebar(title,index=0){await page.locator(`${chart(index)} .modebar-btn[data-title="${title}"]`).click({force:true});}
async function hoverPoint(row=0,index=0){await dragArea(row,index);const box=await page.locator(chart(index)).boundingBox();return await page.locator(chart(index)).evaluate((g,{row,left,top})=>{
 const t=g.data.find(t=>t.meta?.trendSeries&&t.meta.subplotIndex===row);let i=Math.floor(t.x.length*.55);while(i<t.x.length&&!Number.isFinite(t.y[i]))i++;if(i===t.x.length)throw new Error('No finite full-source hover point');
 const axis=(family,ref)=>g._fullLayout[`${family}axis${ref.slice(1)}`],x=axis('x',t.xaxis||'x'),y=axis('y',t.yaxis||'y');
 return {x:left+x._offset+x.d2p(t.x[i]),y:top+y._offset+y.d2p(t.y[i]),name:t.name,row};
 },{row,left:box.x,top:box.y});}
async function spikeCount(index=0){return await page.locator(chart(index)).evaluate((g,selector)=>[...g.querySelectorAll(selector)].filter(e=>{const style=getComputedStyle(e);return style.display!=='none'&&style.visibility!=='hidden'&&Number(e.getAttribute('stroke-width')||style.strokeWidth)>0&&(e.getAttribute('x1')!==e.getAttribute('x2')||e.getAttribute('y1')!==e.getAttribute('y2')||Number(e.getAttribute('r'))>0);}).length,spikeSelector);}
async function hover(tag,{row=0,index=0,spikes=true,panels=1}={}){const p=await hoverPoint(row,index);await measure(`${tag}: actual hover`,async()=>{await page.mouse.move(p.x,p.y);await page.locator(`${chart(index)} ${hoverSelector}`).waitFor({state:'visible'});assert.ok(await page.locator(chart(index)).evaluate((g,row)=>(g._hoverdata||[]).some(point=>{const curve=point.curveNumber??point.fullData?.index,at=point.pointNumber??point.index,trace=g.data[curve];return trace?.meta?.subplotIndex===row&&Number.isInteger(at)&&at>=0&&at<trace.y.length&&point.y===trace.y[at];}),row),'Native hover must reference an exact source observation in the selected subplot');if(spikes)assert.ok(await spikeCount(index));},{panels,interactive:true,budget:hoverBudget});}
async function interactions(tag,{panels=1,row=0}={}){
 await hover(tag,{panels,row});
 for(const title of ['Zoom in','Zoom out']){const before=JSON.stringify((await graphState()).ranges.xaxis);await measure(`${tag}: ${title}`,async()=>{await modebar(title);await page.waitForFunction(before=>JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range)!==before,before);},{panels,interactive:true});}
 await measure(`${tag}: box zoom`,async()=>{await modebar('Zoom');const b=await dragArea(row);const before=JSON.stringify((await graphState()).ranges.xaxis);await page.mouse.move(b.x+b.width*.2,b.y+b.height*.25);await page.mouse.down();await page.mouse.move(b.x+b.width*.8,b.y+b.height*.75,{steps:4});await page.mouse.up();await page.waitForFunction(before=>JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range)!==before,before);},{panels,interactive:true});
 await measure(`${tag}: pan`,async()=>{await modebar('Pan');const b=await dragArea(row);const before=JSON.stringify((await graphState()).ranges.xaxis);await page.mouse.move(b.x+b.width*.5,b.y+b.height*.5);await page.mouse.down();await page.mouse.move(b.x+b.width*.65,b.y+b.height*.5,{steps:4});await page.mouse.up();await page.waitForFunction(before=>JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range)!==before,before);},{panels,interactive:true});
 const beforeReset=JSON.stringify((await graphState()).ranges.xaxis);
 await measure(`${tag}: reset axes`,async()=>{await modebar('Reset axes');await page.waitForFunction(before=>JSON.stringify(document.querySelector('#chart')._fullLayout.xaxis.range)!==before,beforeReset);},{panels,interactive:true});
 await hover(`${tag} after reset`,{panels,row});
 await page.mouse.move(2,2);await page.waitForFunction(selector=>[...document.querySelectorAll(selector)].every(e=>!e.getClientRects().length||getComputedStyle(e).visibility==='hidden'||getComputedStyle(e).display==='none'),hoverSelector);
}
async function csvAllRows(){const event=page.waitForEvent('download');await page.locator('[data-trend-export="0"]').click();const d=await event;let count=0,header;
 for await(const line of createInterface({input:createReadStream(await d.path()),crlfDelay:Infinity})){if(header===undefined)header=line.replace(/^\uFEFF/,'').split(',');else count++;}
 assert.equal(count,expectedRows);assert.deepEqual([...header].sort(),['timestamp',...signals].sort());assert.equal(new Set(header).size,5);await d.delete();return {rows:count,columns:header.length};}

try{
 await page.goto(baseURL,{waitUntil:'networkidle'});await page.setInputFiles('#file-input',fixture);await waitForChart(page);
 await chooseSignals(page,[signals[0]]);await field('shape','linear');await field('markers',false);await field('rangeslider',true);await field('unified',true);await field('spikes',true);
 await measure('One full signal: initial Update',()=>update(),{allowQuery:true});await interactions('One signal, unified hover, slider on');
 pass('One full315034-point signal supports visible tooltips, spikes and real zoom/box/pan interactions');
 await chooseSignals(page,signals);await measure('Four full signals: initial Update',()=>update(),{allowQuery:true});
 await interactions('Four signals, unified hover, slider on');
 const guide=page.locator('#chart .legend').getByText(signals[0],{exact:true}).first().locator('..').locator('.legendtoggle');
 await measure('Four signals: legend hide',async()=>{await guide.click();await page.waitForFunction(name=>document.querySelector('#chart').data.find(trace=>trace.name===name)?.visible==='legendonly',signals[0]);});
 await page.waitForTimeout(350);
 await measure('Four signals: legend restore',async()=>{await guide.click();await page.waitForFunction(name=>document.querySelector('#chart').data.find(trace=>trace.name===name)?.visible!=='legendonly',signals[0]);});
 await hover('Four signals after legend restore');
 const gl=(await graphState()).gl.map(c=>c.id);await measure('Four signals: unchanged Update',()=>update(),{interactive:true});assert.deepEqual((await graphState()).gl.map(c=>c.id),gl);
 report.chartCSV=await csvAllRows();pass('Four315034-point traces stay intact, cached Update preserves graphics contexts and full CSV preserves every row');
 await field('rangeslider',false);await measure('Four signals: slider off style Update',()=>update());await interactions('Four signals, unified hover, slider off');
 await field('unified',false);await field('ytype','log');await measure('Four signals: closest hover/log style Update',()=>update());await interactions('Four signals, closest hover, log axis');
 await field('spikes',false);await measure('Four signals: spikes off style Update',()=>update());await hover('Spikes disabled',{spikes:false});assert.equal(await spikeCount(),0);
 pass('Slider, unified/closest hover, log axes and spike controls preserve responsive interaction without data queries');
 await field('ytype','linear');await field('spikes',true);await field('unified',true);await chooseSignals(page,signals.slice(0,2));await page.locator('#add-trend-subplot').click();await chooseSignals(page,signals.slice(2),'select[data-trend-subplot="1"]');
 await measure('Four signals in two subplots: cached Update',()=>update());await interactions('Two full-source subplots',{row:1});
 const nested=await graphState();assert.deepEqual(nested.traces.map(t=>t.row),[0,0,1,1]);assert.deepEqual(nested.ranges.xaxis,nested.ranges.xaxis2);
 await page.locator('#duplicate-trend').click();await measure('Two instances with two subplots: cached Update',()=>update(2),{panels:2});await hover('Second instance lower subplot',{row:1,index:1,panels:2});
 pass('Multiple instances and nested subplots retain all full-source arrays with responsive actual hover');
 const event=page.waitForEvent('download');await page.locator('[data-trend-image="0"]').click();const png=await event;const bytes=await readFile(await png.path());assert.deepEqual([...bytes.subarray(0,8)],[137,80,78,71,13,10,26,10]);assert.ok(bytes.length>1000);await png.delete();
 await page.screenshot({path:path.join(artifacts,'full-source-trend-subplots.png'),fullPage:true});
 try{report.baseline=JSON.parse(await readFile('artifacts/trend-interaction-baseline.json','utf8')).stages.filter(s=>/zoom in|zoom out|box zoom|: pan|mousemove/.test(s.label)).map(({label,wallMs,mainThreadMs,longestTaskMs})=>({label,wallMs,mainThreadMs,longestTaskMs}));}catch(e){if(e.code!=='ENOENT')throw e;}
 report.workerErrors=await page.evaluate(()=>window.__trendPerf.workerErrors);assert.deepEqual(report.workerErrors,[]);assert.deepEqual(report.errors,[]);assert.deepEqual(report.graphicsWarnings,[]);assert.deepEqual(report.writes,[]);assert.deepEqual(report.external,[]);report.result='PASS';
}catch(error){report.result='FAIL';report.error=String(error);await page.screenshot({path:path.join(artifacts,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{report.heapScope='Main renderer JS only; excludes Worker WASM and GPU allocations';await persist();console.log(JSON.stringify(report,null,2));await browser.close();}
