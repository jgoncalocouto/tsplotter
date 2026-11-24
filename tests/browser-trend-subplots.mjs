/** Exercise nested Trend subplots, data reuse and portable configurations in the real UI. */
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {mkdtemp,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import * as XLSX from 'xlsx';
import {chooseSignals,waitForChart} from './browser-controls.mjs';

const baseURL=process.env.BASE_URL||'http://127.0.0.1:5173/';
const artifacts=await mkdtemp(path.join(tmpdir(),'tsplotter-trend-subplots-'));
const rows=1000,filename='private-trend-subplots.csv';
const file={name:filename,mimeType:'text/csv',buffer:Buffer.from('x,a,b,c,d,category\n'+Array.from({length:rows},(_,i)=>`${i},${i+1},${100+2*i},${5+i%17},${20+i%31},dataset-secret-${i%2}`).join('\n'))};
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1050},acceptDownloads:true});
const checks=[],errors=[],writes=[],external=[],graphicsWarnings=[];
page.setDefaultTimeout(60000);
page.on('pageerror',e=>errors.push(String(e)));
page.on('console',m=>{if(/(?:webgl.*(?:lost|too many|limit)|too many.*(?:webgl|context)|context.*lost)/i.test(m.text()))graphicsWarnings.push(m.text());});
page.on('request',r=>{if(['POST','PUT','PATCH','DELETE'].includes(r.method()))writes.push(r.url());if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==new URL(baseURL).origin)external.push(r.url());});
await page.addInitScript(()=>{
 window.__trendQueries=[];window.__trendGL=new WeakMap();window.__trendCanvases=new WeakMap();let nextGL=1;
 const post=Worker.prototype.postMessage;
 Worker.prototype.postMessage=function(message,...args){if(message?.method==='query'&&message.args?.[0]?.mode==='trends'){const s=message.args[0];window.__trendQueries.push({x:s.x,y:[...s.y]});}return post.call(this,message,...args);};
 const getContext=HTMLCanvasElement.prototype.getContext;
 HTMLCanvasElement.prototype.getContext=function(type,...args){const c=getContext.call(this,type,...args);if(c&&String(type).includes('webgl')){if(!window.__trendGL.has(c))window.__trendGL.set(c,nextGL++);window.__trendCanvases.set(this,c);}return c;};
});
const pass=text=>{checks.push(text);console.log(text);};
const chart=index=>index?`#trend-chart-${index}`:'#chart';
async function ready(count=1){await waitForChart(page);assert.equal(await page.locator('[data-trend-card]').count(),count);}
async function update(count=1){await page.locator('#render-button').click();await ready(count);}
async function reveal(field){if(!await field.isVisible())await field.evaluate(e=>e.closest('details').open=true);}
async function field(key,value){const el=page.locator(`[data-panel-key="${key}"]`);await reveal(el);if(typeof value==='boolean')await el.setChecked(value);else if(await el.evaluate(e=>e.tagName==='SELECT'))await el.selectOption(value);else{await el.fill(String(value));await el.press('Tab');}}
async function subplot(index,columns){await chooseSignals(page,columns,`select[data-trend-subplot="${index}"]`);}
async function edit(index){await page.locator(`[data-trend-panel="${index}"]`).click();assert.equal(await page.locator(`[data-trend-panel="${index}"]`).getAttribute('aria-pressed'),'true');}
async function queries(){return await page.evaluate(()=>window.__trendQueries.length);}
async function state(index=0){return await page.locator(chart(index)).evaluate(g=>({
 traces:(g.data||[]).map(t=>({name:t.name,row:t.meta?.subplotIndex,xaxis:t.xaxis,yaxis:t.yaxis,x:t.x?.length,y:t.y?.length,first:t.y?.[0],last:t.y?.at(-1)})),
 annotations:(g.layout?.annotations||[]).map(a=>a.text),
 ranges:Object.fromEntries(Object.entries(g._fullLayout||{}).filter(([k])=>/^[xy]axis\d*$/.test(k)).map(([k,v])=>[k,v.range])),
 gl:[...g.querySelectorAll('canvas')].map(c=>{const gl=window.__trendCanvases.get(c);return gl?{id:window.__trendGL.get(gl),lost:gl.isContextLost(),width:gl.drawingBufferWidth,height:gl.drawingBufferHeight}:null;}).filter(Boolean),
}));}
async function retained(index=0,count=rows){const s=await state(index);assert.ok(s.traces.length);for(const t of s.traces){assert.equal(t.x,count);assert.equal(t.y,count);}assert.ok(s.gl.length&&s.gl.every(c=>!c.lost&&c.width&&c.height));return s;}
async function saved(){const event=page.waitForEvent('download');await page.locator('#save-config').click();const d=await event;const text=await readFile(await d.path(),'utf8');await d.delete();const result=JSON.parse(text);assert.equal(result.version,4);assert.ok(!text.includes(filename)&&!text.includes('dataset-secret-'));for(const p of result.settings.trend_panels){assert.ok(p.subplots.length);assert.ok(!Object.hasOwn(p,'columns'));}return result;}
async function load(config,count=1){await page.setInputFiles('#config-input',{name:'trend-settings.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(config))});await ready(count);}
async function exported(index,expectedColumns,count=rows){const event=page.waitForEvent('download');await page.locator(`[data-trend-export="${index}"]`).click();const d=await event;const text=await readFile(await d.path(),'utf8');await d.delete();const wb=XLSX.read(text,{type:'string',raw:true});const table=XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]],{header:1,raw:true});assert.equal(table.length,count+1);assert.deepEqual([...table[0]].sort(),[...expectedColumns].sort());assert.equal(new Set(table[0]).size,expectedColumns.length);return table;}
async function checkbox(selector,value){const el=page.locator(selector);await reveal(el);await el.setChecked(value);}

try{
 await page.goto(baseURL,{waitUntil:'networkidle'});await page.setInputFiles('#file-input',file);await ready();
 await page.selectOption('[data-key="setup_time_mode"]','Relative time');await chooseSignals(page, ['x'], '#x-column');
 await subplot(0,['a','b']);await page.locator('#add-trend-subplot').click();await subplot(1,['c']);await update();
 let s=await retained();assert.deepEqual(s.traces.map(t=>[t.name,t.row,t.xaxis,t.yaxis]),[['a',0,'x','y'],['b',0,'x','y'],['c',1,'x2','y2']]);
 assert.ok(s.ranges.yaxis[1]>s.ranges.yaxis2[1]);
 pass('One Trend instance renders independently selected subplot rows with separate Y axes and every source point');

 const start=await queries(),gl=s.gl.map(c=>c.id);
 await update();assert.equal(await queries(),start);assert.deepEqual((await state()).gl.map(c=>c.id),gl);
 await field('width',4);await field('shape','hv');await field('rangeslider',false);await field('unified',false);await update();
 assert.equal(await queries(),start);assert.deepEqual((await state()).gl.map(c=>c.id),gl);
 await subplot(0,['a']);await subplot(1,['b']);await update();
 assert.equal(await queries(),start);assert.deepEqual((await state()).gl.map(c=>c.id),gl);
 pass('Unchanged, style and subset Updates reuse loaded data without Worker queries or replacing WebGL contexts');

 await page.locator('[data-duplicate-trend-subplot="0"]').click();assert.equal(await page.locator('select[data-trend-subplot]').count(),3);
 await subplot(2,['b']);assert.deepEqual(await page.locator('select[data-trend-subplot="0"]').evaluate(e=>[...e.selectedOptions].map(o=>o.value)),['a']);
 assert.ok(await page.locator('[data-trend-export="0"]').isDisabled());await update();assert.equal(await queries(),start);
 await page.locator('[data-remove-trend-subplot="1"]').click();await update();
 assert.deepEqual((await state()).traces.map(t=>[t.name,t.row]),[['a',0],['b',1]]);
 pass('Duplicate and remove preserve independent row selections, renumber rows and guard stale exports');

 await subplot(0,['a','c']);await subplot(1,['c']);await update();const csv=await exported(0,['x','a','c']);
 assert.deepEqual(csv[1].map(Number),[0,1,5]);assert.equal(Number(csv.at(-1)[0]),rows-1);
 pass('Raw Trend CSV contains the unique union of selected columns once and retains all rows');

 await page.locator('#add-trend').click();await subplot(0,['b']);await page.locator('#add-trend-subplot').click();await subplot(1,['a','c']);
 const beforeNew=await queries();await subplot(0,['b','d']);await update(2);
 assert.equal(await queries(),beforeNew+1,'All instances must acquire their combined new signal union in one Worker query');
 const lastRequest=await page.evaluate(()=>window.__trendQueries.at(-1));assert.deepEqual([...lastRequest.y].sort(),['a','b','c','d']);
 await retained(0);await retained(1);await exported(1,['x','a','b','c','d']);
 pass('Independent Trend instances share one query for the unique combined signal union while exporting their own selections');

 await edit(0);await subplot(0,['a']);await subplot(1,['b']);
 await edit(1);await subplot(0,['b']);await subplot(1,['a']);
 await checkbox('#comparison-link-x',true);await checkbox('#comparison-link-y',true);await update(2);
 const beforeLinks=await queries();
 const unrelatedY=(await state(1)).ranges.yaxis;
 await page.locator('#chart .modebar-btn[data-title="Zoom"]').click({force:true});
 const target=page.locator('#chart .draglayer .nsewdrag').first();await target.scrollIntoViewIfNeeded();const box=await target.boundingBox();
 await page.mouse.move(box.x+box.width*.2,box.y+box.height*.2);await page.mouse.down();
 await page.mouse.move(box.x+box.width*.8,box.y+box.height*.8,{steps:4});await page.mouse.up();
 await page.waitForFunction(()=>{const a=document.querySelector('#chart')._fullLayout,b=document.querySelector('#trend-chart-1')._fullLayout;return a.xaxis.autorange===false&&['xaxis','xaxis2'].every(k=>JSON.stringify(a[k].range)===JSON.stringify(a.xaxis.range)&&JSON.stringify(b[k].range)===JSON.stringify(a.xaxis.range))&&JSON.stringify(b.yaxis2.range)===JSON.stringify(a.yaxis.range);});
 assert.deepEqual((await state(1)).ranges.yaxis,unrelatedY);assert.equal(await queries(),beforeLinks);
 pass('Shared X ranges span every subplot and instance while linked Y ranges follow matching signal rows');

 await edit(0);await subplot(1,[]);await update(2);s=await state();assert.equal(s.traces.length,1);assert.ok(s.annotations.some(text=>/subplot 2/i.test(text)));
 const blankRow=await page.locator('#chart').evaluate(g=>{
  const row=g.querySelector('.draglayer .nsewdrag').getBoundingClientRect();
  const prompt=[...g.querySelectorAll('.annotation-text')].find(e=>e.textContent==='Choose signals for subplot 2').getBoundingClientRect();
  return {plotBottom:row.bottom,promptCenter:prompt.top+prompt.height/2};
 });assert.ok(blankRow.promptCenter>blankRow.plotBottom+12,'An empty subplot prompt must sit below the preceding plot');
 const savedWorkspace=await saved();assert.deepEqual(savedWorkspace.settings.trend_panels[0].subplots[1].columns,[]);
 await load(savedWorkspace,2);assert.deepEqual(await page.locator('select[data-trend-subplot="1"]').evaluate(e=>[...e.selectedOptions].map(o=>o.value)),[]);
 await page.reload({waitUntil:'networkidle'});assert.ok(await page.locator('#workspace').evaluate(e=>e.classList.contains('hidden')));
 const recovery=await page.evaluate(()=>localStorage.getItem('tsplotter.workspace.v3'));assert.ok(recovery&&!recovery.includes(filename)&&!recovery.includes('dataset-secret-'));
 await page.setInputFiles('#file-input',file);await ready(2);await edit(0);assert.deepEqual(await page.locator('select[data-trend-subplot="1"]').evaluate(e=>[...e.selectedOptions].map(o=>o.value)),[]);
 pass('Version4 files and device recovery preserve nested instances and intentionally cleared rows without storing dataset bytes');

 const beforeBlank=await queries();await subplot(0,[]);await page.locator('#render-button').click();
 await page.waitForFunction(()=>!document.querySelector('#render-button').disabled&&document.querySelector('#status').classList.contains('error'));
 assert.match(await page.locator('#status').textContent(),/choose.*signal/i);assert.equal(await queries(),beforeBlank);
 assert.ok(await page.locator('[data-trend-export="0"]').isDisabled());
 const blank=await saved();assert.ok(blank.settings.trend_panels[0].subplots.every(row=>row.columns.length===0));
 await subplot(0,['a']);await update(2);await retained();
 pass('A completely cleared instance remains portable but rejects plotting recoverably before any query');

 for(const version of [1,2]){
  await load({app:'tsplotter',version,settings:{plot_type:'Trends',setup_time_mode:'Relative time',setup_index:'x',mlc_n_subplots:3,mlc_cols_0:['a'],mlc_cols_1:['b','c'],mlc_cols_2:[],mlc_shape:'hv'}},1);
  assert.equal(await page.locator('select[data-trend-subplot]').count(),3);assert.deepEqual((await state()).traces.map(t=>[t.name,t.row]),[['a',0],['b',1],['c',1]]);
 }
 await load({app:'tsplotter',version:3,settings:{plot_type:'Trends',setup_time_mode:'Relative time',setup_index:'x',trend_panels:[{title:'Older first',columns:['a','b']},{title:'Older second',columns:['c']}]}},2);
 assert.equal(await page.locator('select[data-trend-subplot]').count(),1);assert.deepEqual((await state()).traces.map(t=>t.name),['a','b']);assert.deepEqual((await state(1)).traces.map(t=>t.name),['c']);
 pass('Legacy v1/v2 subplot counts restore one nested Trend; v3 flat panels preserve independent instances');

 await edit(0);await field('shape','spline');await field('ytype','log');await field('unified',false);await update(2);
 assert.ok(await page.locator('#chart').evaluate(g=>g.data.every(trace=>trace.type==='scatter')));
 const svgArea=page.locator('#chart .draglayer .nsewdrag').first();await svgArea.scrollIntoViewIfNeeded();
 const chartBox=await page.locator('#chart').boundingBox();
 const point=await page.locator('#chart').evaluate((g,box)=>{const trace=g.data[0],at=500;return {x:box.x+g._fullLayout.xaxis._offset+g._fullLayout.xaxis.d2p(trace.x[at]),y:box.y+g._fullLayout.yaxis._offset+g._fullLayout.yaxis.d2p(trace.y[at])};},chartBox);
 await page.mouse.move(point.x,point.y);
 await page.waitForFunction(()=>{const g=document.querySelector('#chart');return (g._hoverdata||[]).some(point=>{const curve=point.curveNumber??point.fullData?.index,at=point.pointNumber??point.index;return Number.isInteger(at)&&point.y===g.data[curve]?.y[at];});});
 assert.ok(await page.locator('#chart .hoverlayer .hovertext').isVisible());
 await page.mouse.move(2,2);await field('shape','linear');await field('ytype','linear');await update(2);
 pass('Spline interpolation and logarithmic Y axes retain native tooltips for exact source observations');

 const config=await saved();config.settings.comparison_link_x=false;config.settings.comparison_link_y=false;
 config.settings.trend_panels=Array.from({length:6},(_,i)=>({...config.settings.trend_panels[0],title:`Maximum ${i+1}`,shape:'linear',rangeslider:false,subplots:Array.from({length:6},(_,row)=>({columns:[['a','b','c'][row%3]]}))}));
 await load(config,6);assert.ok(await page.locator('#add-trend').isDisabled());assert.ok(await page.locator('#duplicate-trend').isDisabled());
 assert.ok(await page.locator('#add-trend-subplot').isDisabled());for(const button of await page.locator('[data-duplicate-trend-subplot]').all())assert.ok(await button.isDisabled());
 for(let i=0;i<6;i++){const current=await retained(i);assert.equal(current.traces.length,6);}
 while(await page.locator('select[data-trend-subplot]').count()>1)await page.locator('[data-remove-trend-subplot="0"]').click();
 assert.ok(await page.locator('[data-remove-trend-subplot="0"]').isDisabled());
 while(await page.locator('[data-trend-panel]').count()>1)await page.locator('#remove-trend').click();assert.ok(await page.locator('#remove-trend').isDisabled());
 await update();await retained();
 pass('Six independent Trend instances each support six full-data subplots and enforce both minimum and maximum guards');

 await load(savedWorkspace,2);await page.setViewportSize({width:390,height:844});await page.waitForTimeout(200);
 const overflow=await page.evaluate(()=>Math.max(document.documentElement.scrollWidth,document.body.scrollWidth)-innerWidth);assert.ok(overflow<=1);
 await page.screenshot({path:path.join(artifacts,'mobile-trend-subplots.png'),fullPage:true});
 pass('Nested subplot controls and multiple Trend cards fit a390px viewport without horizontal overflow');
 assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);assert.deepEqual(external,[]);assert.deepEqual(graphicsWarnings,[]);
 console.log(JSON.stringify({result:'PASS',checks,artifacts},null,2));
}catch(error){await page.screenshot({path:path.join(artifacts,'failure.png'),fullPage:true}).catch(()=>{});console.error(JSON.stringify({result:'FAIL',message:String(error),status:await page.locator('#status').textContent().catch(()=>''),checks,errors,writes,external,graphicsWarnings,artifacts},null,2));process.exitCode=1;}
finally{await browser.close();}
