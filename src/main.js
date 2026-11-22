import './style.css';
import Plotly from 'plotly.js-dist-min';
import {DataEngine} from './engine-client.js';
import {defaultSettings,decodeConfig,encodeConfig,normalizeConfiguration,filtersFromSettings,PLOT_TYPES,PALETTES,TIMEZONES,BIN_MODES,BIN_SCOPES,defaultScatterPanel,scatterPanels,MAX_SCATTER_PANELS,defaultDistributionPanel,distributionPanels,MAX_DISTRIBUTION_PANELS,MAX_ANALYSIS_PANELS,defaultTrendPanel,trendPanels,defaultRollingPanel,rollingPanels,defaultDerivativePanel,derivativePanels,COLORMAPS,MARKER_PRESETS} from './config.js';
import {buildFigure,chartRequest} from './charts.js';
import {enhancePickers,closePickers} from './pickers.js';
import {configurationSignature,saveWorkspace,loadWorkspace,clearWorkspace} from './workspace-recovery.js';
import {axisRangeNumbers,comparisonAxes,linkedRangePatch} from './comparisons.js';

const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s),fmt=n=>Number(n??0).toLocaleString();
let schema=null,state=defaultSettings(),lastResult=null,currentName='',busy=false,pendingSettings=null,dirty=false,lastFile=null;
let activePanels={},lastPanelResults=[],savedConfig=null,recoveryAvailable=false,recoverySuppressed=false,recoveryProblem='',linkingAxes=false;
const panelViews={
 Trends:{slug:'trend',label:'Trend',key:'trend_panels',max:MAX_ANALYSIS_PANELS,read:trendPanels,create:defaultTrendPanel},
 Scatter:{slug:'scatter',label:'Scatter',key:'scatter_panels',max:MAX_SCATTER_PANELS,read:scatterPanels,create:defaultScatterPanel},
 Distribution:{slug:'distribution',label:'Distribution',key:'distribution_panels',max:MAX_DISTRIBUTION_PANELS,read:distributionPanels,create:defaultDistributionPanel},
 'Rolling window':{slug:'rolling',label:'Rolling analysis',key:'rolling_panels',max:MAX_ANALYSIS_PANELS,read:rollingPanels,create:defaultRollingPanel},
 Derivatives:{slug:'derivative',label:'Derivative analysis',key:'derivative_panels',max:MAX_ANALYSIS_PANELS,read:derivativePanels,create:defaultDerivativePanel},
};
const panelView=()=>panelViews[state.plot_type];
const activePanelIndex=()=>activePanels[state.plot_type]??0;
const engine=new DataEngine({assetBase:new URL('duckdb/',document.baseURI).href});
engine.onProgress=event=>status(event.message??'Preparing local analysis…','busy');
const icons={
 file:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9L14 3Z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
 upload:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 16V4m-4 4 4-4 4 4M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></svg>',
 sliders:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7h7m4 0h5M4 17h3m4 0h9"/><circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></svg>',
 save:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m5 3 14 0 2 2v16H3V3h2ZM7 3v6h10V3M7 21v-8h10v8"/></svg>',
 download:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M4 16v4h16v-4"/></svg>',
 refresh:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 3M4 16l2 3a7 7 0 0 0 12-2"/></svg>',
 image:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 16 5-5 4 4 4-6 5 7"/></svg>',
 filter:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 5h18l-7 8v6l-4 2v-8L3 5Z"/></svg>',
 arrow:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5"/></svg>',
 plus:'<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>'
};
const modeMeta={
 'Trends':['Follow changes over time','Compare signals on a shared time axis, with independent panels when you need them.','<path d="m3 17 5-6 4 3 5-9 4 4"/>'],
 'Distribution':['Understand the distribution','Create independent distribution panels, each with its own signals and histogram bins.','<path d="M4 20V13h4v7m1 0V5h5v15m1 0V9h5v11"/>'],
 'Scatter':['Compare relationships','Create independent scatter panels, with optional color, marker shapes and numeric bins.','<circle cx="5" cy="17" r="1.2"/><circle cx="8" cy="11" r="1.2"/><circle cx="13" cy="14" r="1.2"/><circle cx="16" cy="7" r="1.2"/><circle cx="20" cy="4" r="1.2"/>'],
 'Rolling window':['See the underlying trend','Explore the trailing average and range while keeping the original signal in view.','<path d="m3 14 4-5 4 7 4-10 6 5M3 17c5-7 12-9 18-9"/>'],
 'Derivatives':['Inspect rates of change','Compare the original signal with its first and second derivatives.','<path d="M3 17c3 0 4-11 8-11s5 11 10 11M3 12h18"/>']
};
$('#app').innerHTML=`<div id="progress" class="progress hidden"></div>
<div id="sidebar-backdrop" class="sidebar-backdrop" hidden></div>
<div class="shell"><aside id="sidebar" aria-label="Dataset and workspace settings">
 <div class="sidebar-brand"><div class="brand"><span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m3 16 4-5 4 3 5-9 5 4"/></svg></span>TSPlotter</div><button id="close-sidebar" class="icon-button mobile-only" aria-label="Close settings">×</button></div>
 <p class="brand-note">Make sense of your signals.</p>
 <section class="sidebar-section"><div class="section-heading"><span class="section-number">01</span><h2>Dataset</h2></div>
 <label id="file-drop" class="file-drop" tabindex="0" role="button" aria-label="Choose or drop a local dataset">${icons.upload}<strong>Open a dataset</strong><span>Drop a file here or <em>browse files</em></span><input class="native-control" type="file" tabindex="-1" id="file-input" accept=".parquet,.csv,.tsv,.json,.jsonl,.xlsx,.xls,.mat"></label>
 <p class="format-hint">Parquet · CSV · Excel · JSON · MAT</p>
 <button id="example-button" class="text-button example-link">Explore example data <span>↗</span></button>
 <details class="import-options" id="file-options"><summary>Import options</summary><div><label class="field">Excel worksheet<input id="sheet-name" placeholder="First worksheet"></label><button id="reload-sheet" class="subtle full-width">Apply worksheet</button></div></details>
 <div id="source-info" class="source-info hidden"></div>
 </section>
 <section class="sidebar-section"><div class="section-heading"><span class="section-number">02</span><h2>Time axis</h2></div><div id="axis-controls"></div></section>
 <section class="sidebar-section"><div class="section-heading"><span class="section-number">03</span><h2>Workspace</h2></div>
 <div class="workspace-buttons"><label class="button-label" tabindex="0">${icons.upload}Load settings<input class="native-control" id="config-input" tabindex="-1" type="file" accept=".json"></label><button id="save-config" class="secondary">${icons.save}Save settings</button></div>
 <p class="field-help">Keep your chart selections, filters and styling.</p><p id="settings-indicator" class="settings-indicator" role="status" aria-live="polite" data-unsaved="false">Settings recovery stays on this device.</p><button id="clear-recovery" class="text-button subtle">Clear recovery</button></section>
 <div class="sidebar-footer"><div class="privacy-note"><span class="privacy-dot"></span>Files stay on your device</div></div>
 </aside>
 <main><header class="topbar"><div><div class="breadcrumb">Workspace <span>/</span> Analysis</div><h1>Time Series Studio</h1></div><div class="tools"><button id="settings-button" class="secondary mobile-settings">${icons.sliders}Dataset & settings</button><button id="cancel-button" class="secondary hidden">Cancel analysis</button><span class="workspace-tag"><span></span>Private workspace</span></div></header>
 <div id="status" role="status" aria-live="polite" class="status"></div>
 <section id="welcome" class="welcome"><div class="welcome-copy"><span class="eyebrow">A workspace for curiosity</span><h2>Find the story<br>in your <span>signals.</span></h2><p>Start with a dataset. Explore every point, compare patterns and turn observations into a clearer picture.</p><div class="welcome-actions"><button id="welcome-open" class="primary">${icons.upload}Open a dataset</button><button id="welcome-example" class="secondary">Try example data ${icons.arrow}</button></div><div class="welcome-note"><span class="privacy-dot"></span>Private by design. Your files stay with you.</div></div><div class="welcome-visual" aria-hidden="true"><div class="mini-window"><div class="mini-toolbar"><span class="mini-dot"></span><span class="mini-dot"></span><span class="mini-dot"></span><span class="mini-title">Signal overview</span><span class="mini-live">LIVE VIEW</span></div><div class="mini-label">Temperature <strong>24.6 <small>°C</small></strong><span>+3.2 from baseline</span></div><svg viewBox="0 0 480 230" class="mini-chart"><defs><linearGradient id="chart-fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#7560d9" stop-opacity=".18"/><stop offset="100%" stop-color="#7560d9" stop-opacity="0"/></linearGradient></defs><path class="mini-grid" d="M0 35h480M0 85h480M0 135h480M0 185h480M40 0v230M140 0v230M240 0v230M340 0v230M440 0v230"/><path d="M0 175C20 180 30 100 60 118S95 182 125 136 155 48 185 76 210 150 240 100 265 28 295 58 320 120 350 75 380 12 405 36 440 85 480 28V230H0Z" fill="url(#chart-fade)"/><path class="mini-line muted-line" d="M0 150C40 145 40 172 80 158S140 118 170 134 210 157 250 130 290 115 320 125 380 105 420 95 450 108 480 88"/><path class="mini-line" d="M0 175C20 180 30 100 60 118S95 182 125 136 155 48 185 76 210 150 240 100 265 28 295 58 320 120 350 75 380 12 405 36 440 85 480 28"/></svg><div class="mini-legend"><span><i></i>Temperature</span><span><i></i>Baseline</span><span class="mini-all">All points included</span></div></div><div class="visual-caption">One dataset. More ways to understand it.</div></div><div class="welcome-features"><div><span>01</span><h3>Bring your data</h3><p>Open your local files. No uploads, no waiting for a server.</p></div><div><span>02</span><h3>Explore with purpose</h3><p>Five focused views, flexible filters and familiar Plotly tools.</p></div><div><span>03</span><h3>Keep every detail</h3><p>Every selected point, with portable settings and full exports.</p></div></div></section>
 <div id="workspace" class="hidden">
 <section class="dataset-header"><div class="dataset-identity"><span class="dataset-icon">${icons.file}</span><div><h2 id="source-label"></h2><div class="dataset-subtitle"><span id="file-format">Dataset</span><span>·</span><span>Local file</span></div></div></div><button id="replace-file" class="text-button">Change dataset ${icons.refresh}</button></section>
 <section class="stats" aria-label="Dataset summary"><div class="stat"><small>Matched rows</small><strong id="row-count">—</strong></div><div class="stat"><small>Numeric signals</small><strong id="numeric-count">—</strong></div><div class="stat"><small>Columns</small><strong id="column-count">—</strong></div><div class="stat"><small>Filtered out</small><strong id="filtered-count">—</strong></div><div class="stat"><small>Invalid time</small><strong id="invalid-time-count">0</strong></div></section>
 <nav class="chart-tabs" aria-label="Chart views" role="tablist">${PLOT_TYPES.map(t=>`<button type="button" class="chart-tab" role="tab" aria-selected="${t==='Trends'}" data-chart-mode="${esc(t)}"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true">${modeMeta[t][2]}</svg><span>${esc(t)}</span></button>`).join('')}<select id="chart-type" tabindex="-1" class="native-control" aria-label="Chart type">${PLOT_TYPES.map(t=>`<option>${esc(t)}</option>`).join('')}</select></nav>
 <section class="chart-workspace card"><header class="chart-heading"><div><h2 id="chart-title">${modeMeta.Trends[0]}</h2><p id="chart-description">${modeMeta.Trends[1]}</p></div><span id="dirty-indicator" class="dirty-indicator hidden">Unapplied changes</span></header>
 ${Object.values(panelViews).map(view=>`<div id="${view.slug}-toolbar" class="instance-toolbar hidden" aria-label="${view.label} panels"></div>`).join('')}
 <div class="chart-setup"><div id="chart-options" class="options"></div><button id="render-button" class="primary">${icons.refresh}<span>Update chart</span></button></div>
 <div class="control-panels"><details id="filter-panel" class="controls-panel"><summary>${icons.filter}Filter data<span id="filter-count" class="count-badge hidden">0</span></summary><div class="details-body"><div class="panel-description">Narrow the dataset. Filters apply to charts, statistics and full exports.</div><div id="filters"></div><div class="button-row"><button id="add-filter" class="secondary subtle">${icons.plus}Add filter</button><button id="clear-filters" class="text-button subtle">Clear all</button></div></div></details><details id="appearance-panel" class="controls-panel"><summary>${icons.sliders}Chart appearance</summary><div id="appearance-options" class="options details-body"></div></details></div>
 <details id="comparison-panel" class="comparison-panel"><summary>Compare panels</summary><div id="comparison-options"></div></details>
 <div id="plot-grid"><article id="primary-plot" class="plot-panel">
 <header id="primary-panel-heading" class="plot-heading hidden"><div><h3></h3><p></p></div><button class="text-button subtle">Edit settings</button></header>
 <div id="chart" class="plot-canvas"><div id="chart-empty">Choose signals to begin.</div></div><div class="chart-footer"><div class="chart-readout"><span class="readout-dot"></span><span id="plotted-count">—</span></div><div class="chart-export-tools"><button id="chart-export" class="text-button subtle">${icons.download}Chart data</button><button id="image-export" class="text-button subtle">${icons.image}Save image</button></div></div><details id="bin-panel" class="bin-panel hidden"><summary>Group intervals & counts</summary><div id="bin-details" class="bin-details"></div></details>
 </article></div>
 </section>
 <section class="data-panels"><details class="panel" id="preview-panel"><summary>${icons.file}Data preview & export<span>Inspect the rows behind your analysis</span></summary><div class="details-body"><div class="panel-toolbar"><p class="field-help">First 100 rows · up to 30 columns. Full exports contain every filtered row.</p><div class="button-row"><button id="preview-button" class="secondary subtle">Show preview</button><button id="data-export" class="secondary subtle">${icons.download}Export full dataset</button></div></div><div id="preview-table"></div></div></details><details class="panel" id="summary-panel"><summary>${icons.sliders}Summary statistics<span>A closer look at your numeric signals</span></summary><div class="details-body"><button id="summary-button" class="secondary subtle">Calculate summary</button><div id="summary-table"></div></div></details></section>
 </div><footer class="small-footer"><span>TSPlotter</span><span>Every signal has a story.</span></footer></main></div>`;

function status(text,kind=''){const el=$('#status');el.textContent=text;el.className=`status ${kind}`;}
function syncWorkspaceSettings(){
 const indicator=$('#settings-indicator');
 if(!schema&&!pendingSettings){indicator.dataset.unsaved='false';indicator.textContent=recoveryProblem||'Settings recovery stays on this device.';return;}
 try{
  const settings=!schema&&pendingSettings?pendingSettings:state,config=encodeConfig(settings);
  const unsaved=!savedConfig||configurationSignature(config)!==configurationSignature(savedConfig);
  indicator.dataset.unsaved=String(unsaved);
  if(!recoverySuppressed){
   try{saveWorkspace(window.localStorage,settings,savedConfig);recoveryAvailable=true;recoveryProblem='';}
   catch{recoveryProblem='Device recovery is unavailable. Save a settings file to keep your work.';}
  }
  indicator.textContent=recoveryProblem||(unsaved?`Unsaved settings${recoveryAvailable?' · recovery saved on this device':''}`:'Settings file up to date');
 }catch{
  indicator.dataset.unsaved='true';indicator.textContent=`Unsaved settings · finish invalid fields${recoveryAvailable?'; recovery keeps the last valid settings':''}.`;
 }
 $('#clear-recovery').disabled=busy||!recoveryAvailable;
}
function initializeRecovery(){
 try{
  const recovered=loadWorkspace(window.localStorage);
  if(recovered){pendingSettings=recovered.settings;savedConfig=recovered.savedConfig;recoveryAvailable=true;status('Recovered settings. Open your dataset to continue.');}
 }catch{recoveryProblem='Saved recovery could not be read. You can load a settings file.';}
 syncWorkspaceSettings();
}
function syncActions(){
 const exportable=result=>result?.rowCount&&(!result.meta?.rows||result.meta.rows.usable>0);
 for(const id of ['chart-export','image-export'])$(`#${id}`).disabled=busy||dirty||!exportable(lastResult);
 const view=panelView();
 if(view)for(const button of document.querySelectorAll(`[data-${view.slug}-export],[data-${view.slug}-image]`)){
  const index=Number(button.getAttribute(`data-${view.slug}-export`)??button.getAttribute(`data-${view.slug}-image`));
  button.disabled=busy||dirty||!exportable(lastPanelResults[index]?.result);
 }
 for(const id of ['render-button','preview-button','data-export','summary-button'])$(`#${id}`).disabled=busy||!schema;
 $('#reload-sheet').disabled=busy||!lastFile||!/\.xlsx?$/i.test(lastFile.name);
 $('#clear-recovery').disabled=busy||!recoveryAvailable;
 for(const view of Object.values(panelViews)){
  const count=state[view.key]?.length??1;
  for(const action of ['add','duplicate'])if($(`#${action}-${view.slug}`))$(`#${action}-${view.slug}`).disabled=busy||!schema||count>=view.max;
  if($(`#remove-${view.slug}`))$(`#remove-${view.slug}`).disabled=busy||!schema||count<=1;
  for(const button of document.querySelectorAll(`[data-${view.slug}-panel],[data-edit-${view.slug}]`))button.disabled=busy;
 }
}
function setDirty(value=true){dirty=Boolean(value&&schema);if(value)recoverySuppressed=false;$('#dirty-indicator').classList.toggle('hidden',!dirty);$('#render-button').classList.toggle('has-changes',dirty);syncWorkspaceSettings();syncActions();}
function setBusy(value){
 busy=value;if(value)closePickers();$('#progress').classList.toggle('hidden',!value);$('#cancel-button').classList.toggle('hidden',!value);
 for(const el of document.querySelectorAll('button,input,select'))if(!['cancel-button','close-sidebar','settings-button'].includes(el.id))el.disabled=value;
 $('#render-button').innerHTML=value?'<span class="button-spinner"></span><span>Updating…</span>':`${icons.refresh}<span>${panelView()&&(state[panelView().key]?.length??1)>1?'Update charts':'Update chart'}</span>`;
 syncActions();
}
async function task(fn){if(busy)return false;setBusy(true);try{await fn();return true;}catch(error){console.error(error);if(schema)setDirty(true);status(error.message??String(error),'error');return false;}finally{setBusy(false);}}
function refreshView(){
 const mode=state.plot_type??'Trends';$('#chart-type').value=mode;
 for(const tab of document.querySelectorAll('[data-chart-mode]')){const selected=tab.dataset.chartMode===mode;tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;}
 $('#chart-title').textContent=modeMeta[mode]?.[0]??mode;$('#chart-description').textContent=modeMeta[mode]?.[1]??'';
 $('#chart').setAttribute('aria-label',`${mode} chart`);
 const n=Number(state.flt2_n)||0;$('#filter-count').textContent=n;$('#filter-count').classList.toggle('hidden',!n);
}
const narrowScreen=window.matchMedia('(max-width: 780px)');
function setSidebar(open,{focus=true}={}){
 const active=Boolean(open&&narrowScreen.matches);document.body.classList.toggle('sidebar-open',active);$('#sidebar-backdrop').hidden=!active;
 $('#sidebar').inert=narrowScreen.matches&&!active;document.querySelector('main').inert=active;$('#settings-button').setAttribute('aria-expanded',String(active));
 if(focus&&narrowScreen.matches)(active?$('#close-sidebar'):$('#settings-button')).focus();
}

const names=()=>schema?.columns.map(c=>c.name)??[];
const nums=()=>schema?.columns.filter(c=>c.kind==='numeric').map(c=>c.name)??[];
function options(items,value,multiple=false){return items.map(item=>{const [v,label]=Array.isArray(item)?item:[item,item];return `<option value="${esc(v)}" ${(multiple?Array.isArray(value)&&value.includes(v):String(v)===String(value))?'selected':''}>${esc(label)}</option>`;}).join('');}
function select(label,key,items,{multiple=false,id='',className=''}={}){return `<label class="field ${className}">${esc(label)}<select ${id?`id="${id}"`:''} data-key="${key}" ${multiple?'multiple':''}>${options(items,state[key],multiple)}</select></label>`;}
function input(label,key,{type='number',min='',max='',step='any',id=''}={}){return `<label class="field">${esc(label)}<input ${id?`id="${id}"`:''} data-key="${key}" type="${type}" value="${esc(state[key]??'')}" ${min!==''?`min="${min}"`:''} ${max!==''?`max="${max}"`:''} ${type==='number'?`step="${step}"`:''}></label>`;}
function check(label,key,{id=''}={}){return `<label class="check"><input ${id?`id="${id}"`:''} data-key="${key}" type="checkbox" ${state[key]?'checked':''}>${esc(label)}</label>`;}
function currentPanels(){
 const view=panelView();
 if(!Array.isArray(state[view.key])||!state[view.key].length)state[view.key]=view.read(state,schema??{});
 activePanels[state.plot_type]=Math.max(0,Math.min(activePanelIndex(),state[view.key].length-1));
 return state[view.key];
}
function panelSelect(label,key,items,{id='',multiple=false,className=''}={}){
 const panel=currentPanels()[activePanelIndex()];
 return `<label class="field ${className}">${esc(label)}<select ${id?`id="${id}"`:''} data-panel-key="${key}" ${multiple?'multiple':''}>${options(items,panel[key],multiple)}</select></label>`;
}
function panelInput(label,key,{type='number',min='',max='',step='any',maxLength=''}={}){
 const panel=currentPanels()[activePanelIndex()];
 return `<label class="field">${esc(label)}<input data-panel-key="${key}" type="${type}" value="${esc(panel[key]??'')}" ${min!==''?`min="${min}"`:''} ${max!==''?`max="${max}"`:''} ${maxLength!==''?`maxlength="${maxLength}"`:''} ${type==='number'?`step="${step}"`:''}></label>`;
}
function panelCheck(label,key){return `<label class="check"><input data-panel-key="${key}" type="checkbox" ${currentPanels()[activePanelIndex()][key]?'checked':''}>${esc(label)}</label>`;}
function panelBinOptions(encoding,label){
 const panel=currentPanels()[activePanelIndex()];
 if(!nums().includes(panel[encoding]))return '';
 const mode=panel[`${encoding}BinMode`];
 return panelSelect(`${label} groups`,`${encoding}BinMode`,BIN_MODES)+(mode==='Bin width'
  ?panelInput(`${label} bin width`,`${encoding}BinWidth`,{min:0})
  :mode==='Number of bins'?panelInput(`${label} bin count`,`${encoding}BinCount`,{min:1,max:100,step:1}):'')
  +(mode!=='Individual values'?panelSelect(`Fit ${label.toLowerCase()} bins on`,`${encoding}BinScope`,BIN_SCOPES):'');
}
function renderPanelToolbar(){
 const view=panelView();
 for(const entry of Object.values(panelViews))$(`#${entry.slug}-toolbar`).classList.toggle('hidden',entry!==view);
 if(!view)return;
 const panels=currentPanels(),toolbar=$(`#${view.slug}-toolbar`);
 toolbar.innerHTML=`<div class="instance-choices" role="group" aria-label="Choose ${view.slug} to edit">${panels.map((panel,index)=>`<button type="button" class="instance-choice" data-${view.slug}-panel="${index}" aria-pressed="${index===activePanelIndex()}" title="${esc(panel.title||`${view.label} ${index+1}`)}">${esc(panel.title||`${view.label} ${index+1}`)}</button>`).join('')}</div><div class="instance-actions"><button id="add-${view.slug}" class="secondary subtle" ${panels.length>=view.max?'disabled':''}>${icons.plus}Add ${view.slug}</button><button id="duplicate-${view.slug}" class="text-button subtle" ${panels.length>=view.max?'disabled':''}>Duplicate</button><button id="remove-${view.slug}" class="text-button subtle" ${panels.length===1?'disabled':''}>Remove</button><span class="field-help">${panels.length} of ${view.max}</span></div>`;
 for(const card of document.querySelectorAll(`[data-${view.slug}-card]`))card.classList.toggle('editing',Number(card.getAttribute(`data-${view.slug}-card`))===activePanelIndex());
 $('#comparison-panel').classList.toggle('hidden',panels.length<2);
 $('#comparison-options').innerHTML=check('Link matching X axes','comparison_link_x',{id:'comparison-link-x'})+check('Link matching Y axes','comparison_link_y',{id:'comparison-link-y'})
  +(state.plot_type==='Distribution'?check('Use shared bins from the first panel','distribution_shared_bins',{id:'distribution-shared-bins'}):'')
  +'<p class="field-help">Linked axes share ranges for matching signals and scales. Filters apply to every panel.</p>';
}
function capture() {
  for (const el of document.querySelectorAll('[data-key]')) {
    const key = el.dataset.key;
    if (el.type === 'checkbox') state[key] = el.checked;
    else if (el.multiple) state[key] = [...el.selectedOptions].map(option => option.value);
    else if (el.type === 'number') state[key] = el.value === '' ? undefined : Number(el.value);
    else if (el.dataset.membershipFilter !== undefined) {
      const arrayKey = `flt2_vals_${el.dataset.membershipFilter}`;
      // Keep imported literal values (which may contain commas) until this field is edited.
      if (el.value !== el.dataset.originalValue) delete state[arrayKey];
      if (!Array.isArray(state[arrayKey])) state[key] = el.value;
    } else state[key] = el.value;
  }
  for (const el of document.querySelectorAll('[data-date-bound]')) {
    const i = el.dataset.filterIndex, side = el.dataset.dateBound;
    const value = el.value, suffix = el.dataset.dateSuffix || 'col';
    state[`flt2_use_${side}dt_${suffix}_${i}`] = Boolean(value);
    if (value) {
      const [date, time] = value.split('T');
      state[`flt2_${side}_date_${suffix}_${i}`] = date;
      state[`flt2_${side}_time_${suffix}_${i}`] = time || '00:00:00';
    }
  }
  for(const el of document.querySelectorAll('[data-panel-key]')){
    const key=el.dataset.panelKey;if(!key)continue;
    currentPanels()[activePanelIndex()][key]=el.type==='checkbox'?el.checked:el.multiple?[...el.selectedOptions].map(option=>option.value):el.type==='number'?(el.value===''?undefined:Number(el.value)):el.value;
  }
}
function renderAxis(){
 const all=names(),absolute=state.setup_time_mode==='Absolute time',wasOpen=$('#time-options')?.open??false;
 const profile=state.setup_profile,tzMode=state.setup_tz_mode;
 let advanced='';
 if(absolute){advanced=select('Input layout','setup_profile',['Auto-detect','DMY (dd/mm/yyyy)','MDY (mm/dd/yyyy)','YMD (yyyy-mm-dd)','Unix epoch (seconds)','Unix epoch (milliseconds)','Custom strptime']);
 if(profile==='Custom strptime')advanced+=input('Datetime format','setup_format',{type:'text'});
 advanced+=select('Timezone handling','setup_tz_mode',['Treat as naive (no tz)','Assume timezone and localize','Input already has timezone']);
 if(tzMode==='Assume timezone and localize'){advanced+=select('Source timezone','setup_assume_tz',[...TIMEZONES,'Custom...']);if(state.setup_assume_tz==='Custom...')advanced+=input('Source timezone name','setup_custom_tz',{type:'text'});}
 advanced+=select('Display timezone','setup_out_tz',['(keep)',...TIMEZONES,'Custom...']);
 if(state.setup_out_tz==='Custom...')advanced+=input('Display timezone name','setup_target_tz',{type:'text'});
 }
 $('#axis-controls').innerHTML=select('Time mode','setup_time_mode',['Absolute time','Relative time'])+select(absolute?'Time column':'X-axis column',absolute?'setup_dt_col':'setup_index',absolute?all:[['(row number 0..N-1)','Row number (0…N−1)'],...all],{id:'x-column'})+(advanced?`<details id="time-options" class="time-options" ${wasOpen?'open':''}><summary>Time parsing & timezone</summary><div class="details-body">${advanced}</div></details>`:'');
}

function renderOptions(){const focusedPanelKey=document.activeElement?.dataset.panelKey;const mode=state.plot_type,numeric=nums(),all=names(),none=['(none)',...all];let main='',appearance='';
 if(mode==='Trends'){
  main=panelInput('Trend title','title',{type:'text',maxLength:80})+panelSelect('Signals','columns',numeric,{multiple:true,id:'y-columns',className:'wide'});
  appearance=panelSelect('Line shape','shape',['linear','spline','hv','vh','hvh','vhv'])+panelInput('Line width','width',{min:1,max:8,step:1})+panelSelect('Y-axis scale','ytype',['linear','log'])
   +panelCheck('Show markers','markers')+panelCheck('Range slider','rangeslider')+panelCheck('Unified hover','unified')+panelCheck('Spike lines','spikes');
 }
 if(mode==='Distribution'){
  const panel=currentPanels()[activePanelIndex()];
  main=panelInput('Distribution title','title',{type:'text',maxLength:80})+panelSelect('Signals','columns',numeric,{multiple:true,id:'y-columns',className:'wide'});
  if(state.distribution_shared_bins&&activePanelIndex()>0){main+=`<p class="field-help wide">Shared bins use the settings from ${esc(currentPanels()[0].title||'Distribution 1')}. Select that panel to edit them.</p>`;}
  else{
   main+=panelSelect('Binning','binMode',['Auto (width)','Fixed width','Target #bins']);
   if(panel.binMode==='Fixed width')main+=panelInput('Bin width','binWidth',{min:1e-12});
   if(panel.binMode==='Target #bins')main+=panelInput('Target bins','binCount',{min:10,max:200,step:1});
   main+=panelSelect('Fit bins on','binScope',BIN_SCOPES);
  }
 }
 if(mode==='Scatter'){
  const panel=currentPanels()[activePanelIndex()];
  main=panelInput('Scatter title','title',{type:'text',maxLength:80})+panelSelect('X variable','x',all,{id:'scatter-x'})+panelSelect('Y variable','y',all,{id:'y-columns'})
   +panelSelect('Color by','color',none)+panelBinOptions('color','Color')+panelSelect('Marker by','marker',none)+panelBinOptions('marker','Marker');
  if(nums().includes(panel.color)&&panel.colorBinMode==='Individual values')main+=panelSelect('Color display','colorMode',[['auto','Automatic (numeric scale)'],['continuous','Continuous scale'],['discrete','Distinct groups']]);
  appearance=panelInput('Marker size','size',{min:3,max:20,step:1})+panelInput('Opacity','opacity',{min:.2,max:1,step:.05});
  if(panel.color!=='(none)')appearance+=numeric.includes(panel.color)&&panel.colorBinMode==='Individual values'&&panel.colorMode!=='discrete'
   ?panelSelect('Continuous colormap','colormap',COLORMAPS):panelSelect('Color palette','palette',PALETTES);
  if(panel.marker!=='(none)')appearance+=panelSelect('Marker preset','markerPreset',MARKER_PRESETS)+panelInput('Custom symbols (comma separated)','customSymbols',{type:'text'});
 }

 if(mode==='Rolling window'){
  main=panelInput('Analysis title','title',{type:'text',maxLength:80})+panelSelect('Signal','y',numeric,{id:'y-columns'})+panelInput('Window size (rows)','window',{min:2,max:200000,step:1})
   +panelCheck('Include current row','includeCurrent')+panelCheck('Require full window','requireFull');
  appearance=panelCheck('Shade min/max band','band')+panelCheck('Unified hover','unified')+panelCheck('Range slider','rangeslider')+panelCheck('Spike lines','spikes')+panelInput('Marker size','size',{min:3,max:16,step:1})+panelInput('Marker opacity','opacity',{min:.2,max:1,step:.05});
 }
 if(mode==='Derivatives'){
  main=panelInput('Analysis title','title',{type:'text',maxLength:80})+panelSelect('Signal','y',numeric,{id:'y-columns'})+panelSelect('Time unit','timeUnit',['seconds','minutes','hours','days','none (sample index)'])
   +panelSelect('Relative X spacing','xMode',['index step (1)','use numeric index values'])+panelInput('Derivative smoothing window','smoothWindow',{min:1,max:9999,step:1});
  appearance=panelCheck('Show markers','markers')+panelInput('Line width','width',{min:1,max:8,step:1})+panelCheck('Unified hover','unified')+panelCheck('Range slider','rangeslider')+panelCheck('Spike lines','spikes');
 }
 $('#chart-options').innerHTML=main;$('#chart-options').className='options';$('#appearance-options').innerHTML=appearance;$('#appearance-panel').classList.toggle('hidden',!appearance);renderPanelToolbar();refreshView();enhancePickers($('#chart-options'));if(focusedPanelKey){const field=$(`[data-panel-key="${focusedPanelKey}"]`);if(field&&!field.multiple)field.focus({preventScroll:true});}}
function renderFilters() {
  const timeColumn = schema?.timeIndex || (state.setup_time_mode === 'Absolute time' && names().includes(state.setup_dt_col) ? state.setup_dt_col : undefined);
  const filterOptions = timeColumn ? [['(time index)', `Time index · ${timeColumn}`], ...names()] : names();
  $('#filters').innerHTML = Array.from({length: Number(state.flt2_n) || 0}, (_, i) => {
    const column = state[`flt2_var_${i}`] ?? names()[0];
    const isTimeIndex = column === '(time index)';
    const kind = isTimeIndex ? 'datetime' : schema?.columns.find(item => item.name === column)?.kind;
    state[`flt2_var_${i}`] = column;
    let fields = '';
    if (kind === 'numeric') {
      fields = input('Minimum', `flt2_low_num_${i}`, {type: 'text'}) + input('Maximum', `flt2_up_num_${i}`, {type: 'text'});
    } else if (kind === 'datetime') {
      const suffix = isTimeIndex ? 'idx' : 'col';
      fields = ['low', 'up'].map(side => {
        const date = state[`flt2_${side}_date_${suffix}_${i}`];
        const time = state[`flt2_${side}_time_${suffix}_${i}`] || (side === 'low' ? '00:00:00' : '23:59:59.999999');
        const enabled = state[`flt2_use_${side}dt_${suffix}_${i}`];
        return `<label class="field">${side === 'low' ? 'From' : 'To'}<input type="datetime-local" step=".000001" data-date-bound="${side}" data-date-suffix="${suffix}" data-filter-index="${i}" value="${enabled && date ? esc(date + 'T' + time) : ''}"></label>`;
      }).join('');
    } else {
      const membership = ['is one of', 'is not one of', undefined].includes(state[`flt2_op_${i}`]);
      fields = select('Match', `flt2_op_${i}`, ['is one of', 'is not one of', 'contains', 'not contains', 'starts with', 'ends with', 'regex']);
      if (membership) {
        const value = Array.isArray(state[`flt2_vals_${i}`]) ? state[`flt2_vals_${i}`].join(', ') : state[`flt2_vals_csv_${i}`] || '';
        fields += `<label class="field">Value(s), comma separated<input data-key="flt2_vals_csv_${i}" data-membership-filter="${i}" data-original-value="${esc(value)}" type="text" value="${esc(value)}"></label>`;
      } else fields += input('Pattern', `flt2_patt_${i}`, {type: 'text'});
    }
    return `<div class="filter-row" data-filter="${i}">${select('Combine', `flt2_logic_${i}`, ['AND', 'OR'])}${select('Column', `flt2_var_${i}`, filterOptions)}${fields}<button data-remove-filter="${i}" title="Remove filter" aria-label="Remove filter ${i + 1}">×</button>${kind !== 'numeric' && kind !== 'datetime' ? check('Case sensitive', `flt2_case_${i}`) + check('Include missing', `flt2_na_${i}`) : ''}</div>`;
  }).join('');
}
function xColumn(){return state.setup_time_mode==='Absolute time'?state.setup_dt_col:state.setup_index==='(row number 0..N-1)'?'__row__':state.setup_index;}
function analysisOptions(){
 if(state.setup_time_mode!=='Absolute time')return {};
 const timeAxis={profile:state.setup_profile,format:state.setup_profile==='Custom strptime'?state.setup_format:undefined,
  timezone:state.setup_tz_mode==='Assume timezone and localize'?(state.setup_assume_tz==='Custom...'?state.setup_custom_tz:state.setup_assume_tz):undefined,
  outputTimezone:state.setup_out_tz==='Custom...'?state.setup_target_tz:(state.setup_out_tz==='(keep)'?undefined:state.setup_out_tz)};
 return {preparedAxis:{x:xColumn(),timeAxis},x:xColumn(),timeAxis};
}
function filters(){const f=filtersFromSettings(state,schema);const options=analysisOptions();for(const condition of f){if(condition.column==='(time index)')condition.column=xColumn();if(condition.kind==='datetime'&&condition.column===xColumn()&&options.timeAxis)condition.timeAxis=options.timeAxis;}return f;}
async function importFile(file){await task(async()=>{status('Reading your file on this device…','busy');capture();const isFirst=!schema;schema=null;lastResult=null;activePanels={};lastPanelResults=[];$('#preview-table').replaceChildren();$('#summary-table').replaceChildren();resetPlotOutputs();$('#workspace').classList.add('hidden');$('#welcome').classList.remove('hidden');let imported,importWarnings=[];
 if(/\.xlsx?$/i.test(file.name)){const {readExcel}=await import('./file-import.js');const result=await readExcel(file,{sheetName:$('#sheet-name').value||undefined});imported=await engine.importRows(result.rows,{sourceName:file.name});importWarnings=result.warnings??[];}
 else if(/\.mat$/i.test(file.name)){const {readMatV73}=await import('./file-import.js');const result=await readMatV73(file);imported=await engine.importRows(result.rows,{sourceName:file.name});importWarnings=result.warnings??[];}
 else imported=await engine.importFile(file);
 schema=imported;currentName=file.name;lastFile=file;$('#file-format').textContent=file.name.split('.').pop().toUpperCase();$('#file-options').open=/\.xlsx?$/i.test(file.name);state={...defaultSettings(schema),...(pendingSettings??(!isFirst?state:{}))};pendingSettings=null;const normalized=normalizeConfiguration(state,schema);state=normalized.settings;renderAxis();renderOptions();renderFilters();$('#welcome').classList.add('hidden');$('#workspace').classList.remove('hidden');$('#source-label').textContent=currentName;$('#source-info').classList.remove('hidden');$('#source-info').textContent=`${fmt(schema.rowCount)} rows · ${fmt(schema.columns.length)} columns · local file`;$('#numeric-count').textContent=fmt(nums().length);$('#column-count').textContent=fmt(schema.columns.length);setBusy(true);await updateChart();setSidebar(false,{focus:false});if(normalized.warnings.length||importWarnings.length)status(`Ready. ${[...normalized.warnings,...importWarnings].join(' ')}`);});}
async function example(){const rows=Array.from({length:720},(_,i)=>({timestamp:new Date(Date.UTC(2025,0,1)+i*600000).toISOString(),temperature:Math.round((21+3*Math.sin(i*Math.PI/72)) *100)/100,humidity:Math.round((55-8*Math.sin(i*Math.PI/72))*100)/100,power_kw:Math.round((2.5+.7*Math.sin(i*Math.PI/72-.7))*100)/100,shift:i%144<72?'Day':'Night'}));const file=new File([JSON.stringify(rows)],'Example · sensor readings.json',{type:'application/json'});await importFile(file);}
const panelChartId=index=>index===0?'chart':`${panelView().slug}-chart-${index}`;
function unbindScatterLegend(graph){
 if(graph?._scatterLegendHandler){graph.removeListener?.('plotly_legendclick',graph._scatterLegendHandler);delete graph._scatterLegendHandler;}
 if(graph?._scatterDoubleHandler){graph.removeListener?.('plotly_legenddoubleclick',graph._scatterDoubleHandler);delete graph._scatterDoubleHandler;}
}
function purgePlot(graph){
 unbindScatterLegend(graph);
 unbindComparison(graph);
 if(graph)Plotly.purge(graph);
}
function removeExtraPlots(){
 for(const card of $('#plot-grid').querySelectorAll('.plot-panel'))if(card.id!=='primary-plot'){
  purgePlot(card.querySelector('.plot-canvas'));card.remove();
 }
}
function resetPlotOutputs(){
 removeExtraPlots();purgePlot($('#chart'));lastResult=null;lastPanelResults=[];
 $('#chart').innerHTML='<div class="plot-empty">Choose signals and update the chart.</div>';
 $('#bin-panel').classList.add('hidden');$('#bin-details').textContent='';$('#plotted-count').textContent='—';
 $('#primary-plot').querySelector('.plot-warnings')?.replaceChildren();$('#primary-plot').querySelector('[data-row-ledger]')?.remove();
}
function setPanelLayout(panels){
 const view=panelView(),shown=Boolean(view),primary=$('#primary-plot'),heading=$('#primary-panel-heading');
 $('#plot-grid').classList.toggle('panel-grid',shown);$('#plot-grid').classList.toggle('multiple',shown&&panels.length>1);
 heading.classList.toggle('hidden',!shown);
 for(const entry of Object.values(panelViews)){
  primary.removeAttribute(`data-${entry.slug}-card`);
  $('#chart-export').removeAttribute(`data-${entry.slug}-export`);$('#image-export').removeAttribute(`data-${entry.slug}-image`);
  heading.querySelector('button').removeAttribute(`data-edit-${entry.slug}`);
 }
 if(!shown){
  removeExtraPlots();delete primary.dataset.panelKind;
  primary.classList.remove('editing');primary.querySelector('.plot-warnings')?.replaceChildren();return;
 }
 primary.dataset.panelKind=view.slug;primary.setAttribute(`data-${view.slug}-card`,'0');
 $('#chart-export').setAttribute(`data-${view.slug}-export`,'0');$('#image-export').setAttribute(`data-${view.slug}-image`,'0');
 heading.querySelector('button').setAttribute(`data-edit-${view.slug}`,'0');
 for(const card of $('#plot-grid').querySelectorAll('.plot-panel'))if(card!==primary&&(card.dataset.panelKind!==view.slug||Number(card.getAttribute(`data-${view.slug}-card`))>=panels.length)){purgePlot(card.querySelector('.plot-canvas'));card.remove();}
 panels.forEach((panel,index)=>{
  let card=$(`[data-${view.slug}-card="${index}"]`);
  if(!card){
   card=document.createElement('article');card.className='plot-panel';card.dataset.panelKind=view.slug;card.setAttribute(`data-${view.slug}-card`,String(index));
   card.innerHTML=`<header class="plot-heading"><div><h3></h3><p></p></div><button class="text-button subtle" data-edit-${view.slug}="${index}">Edit settings</button></header><div id="${panelChartId(index)}" class="plot-canvas"><div class="plot-empty">Update charts to plot this ${view.slug}.</div></div><div class="chart-footer"><div class="chart-readout"><span class="readout-dot"></span><span class="panel-count">—</span></div><div class="chart-export-tools"><button class="text-button subtle" data-${view.slug}-export="${index}" disabled>${icons.download}Chart data</button><button class="text-button subtle" data-${view.slug}-image="${index}" disabled>${icons.image}Save image</button></div></div><details class="bin-panel hidden"><summary>Group intervals & counts</summary><div class="bin-details"></div></details><div class="plot-warnings" role="status"></div>`;
   $('#plot-grid').append(card);
  }
  card.querySelector('h3').textContent=panel.title||`${view.label} ${index+1}`;
  card.querySelector('.plot-heading p').textContent=view.slug==='scatter'?`${panel.y} vs ${panel.x}`:panel.columns?panel.columns.join(', '):view.slug==='rolling'?`${panel.y} · ${panel.window} rows`: `${panel.y} · original, first and second derivative`;
  card.querySelector('.plot-canvas').setAttribute('aria-label',`${panel.title||`${view.label} ${index+1}`} chart`);
  card.classList.toggle('editing',index===activePanelIndex());
 });
}
function binDescription(result){
 return Object.values(result.meta?.groups??{}).filter(group=>group?.labels).map(group=>`${group.source??group.column}: ${group.labels.map((label,index)=>`${label} (${fmt(group.counts?.[index])})`).join(' · ')}`).join('\n');
}
function rowsFor(result){
 return result.meta?.rows??{source:result.sourceRows??schema?.rowCount??0,matched:result.totalRows??result.rowCount,filteredOut:Math.max(0,(result.sourceRows??schema?.rowCount??0)-(result.totalRows??result.rowCount)),invalidTime:result.meta?.invalidXRows??0,usable:result.rowCount,excludedValues:0};
}
function renderRowSummary(result){
 const rows=rowsFor(result);$('#row-count').textContent=fmt(rows.matched);$('#filtered-count').textContent=fmt(rows.filteredOut);$('#invalid-time-count').textContent=fmt(rows.invalidTime);
}
function renderRowLedger(card,result){
 const rows=rowsFor(result);let details=card.querySelector('[data-row-ledger]');
 if(!details){details=document.createElement('details');details.className='row-ledger';details.setAttribute('data-row-ledger','');details.innerHTML='<summary></summary><div></div>';card.append(details);}
 details.dataset.ledger=JSON.stringify(rows);
 details.querySelector('summary').textContent=`${fmt(rows.usable)} usable rows · ${fmt(rows.invalidTime)} invalid time · ${fmt(rows.excludedValues)} missing or nonfinite`;
 const lines=[`${fmt(rows.source)} source rows → ${fmt(rows.matched)} matched filters → ${fmt(rows.usable)} usable rows.`,
  `${fmt(rows.filteredOut)} filtered out; ${fmt(rows.invalidTime)} invalid time values; ${fmt(rows.excludedValues)} excluded for missing or nonfinite selected values.`];
 for(const [name,counts]of Object.entries(result.meta?.seriesCounts??{}))lines.push(`${name}: ${fmt(counts.usable)} usable values; ${fmt(counts.excludedValues)} missing or nonfinite.`);
 details.querySelector('div').textContent=lines.join('\n');
}
function unbindComparison(graph){
 if(graph?._comparisonHandler)graph.removeListener?.('plotly_relayout',graph._comparisonHandler);
 if(graph){delete graph._comparisonHandler;delete graph._comparisonAxes;}
}
function bindComparison(graph,panel,figure){
 unbindComparison(graph);
 graph._comparisonAxes=comparisonAxes(state.plot_type,panel,xColumn(),figure.layout,state.distribution_shared_bins);
 for(const [name,axis]of Object.entries(graph._comparisonAxes))axis.scale=graph._fullLayout?.[name]?.type||axis.scale;
 const options={x:state.comparison_link_x,y:state.comparison_link_y};
 graph._comparisonHandler=async event=>{
  if(busy||linkingAxes||(!options.x&&!options.y))return;
  linkingAxes=true;
  try{
   const changes=[];
   for(const other of $('#plot-grid').querySelectorAll('.plot-canvas')){
    if(other===graph||!other._comparisonAxes)continue;
    const patch=linkedRangePatch(event,graph._comparisonAxes,graph.layout,other._comparisonAxes,options);
    if(Object.keys(patch).length)changes.push(Plotly.relayout(other,patch));
   }
   await Promise.all(changes);
   if(Object.keys(event).some(key=>/^[xy]axis\d*\.autorange$/.test(key)&&event[key]===true))await alignLinkedAxes(options);
  }catch(error){status(`Axis ranges could not be linked: ${error.message}`,'error');}
  finally{linkingAxes=false;}
 };
 graph.on('plotly_relayout',graph._comparisonHandler);
}
async function alignLinkedAxes(options={x:state.comparison_link_x,y:state.comparison_link_y}){
 if(!options.x&&!options.y)return;
 const groups=new Map();
 for(const graph of $('#plot-grid').querySelectorAll('.plot-canvas'))for(const [name,axis]of Object.entries(graph._comparisonAxes??{})){
  if(!(axis.family==='x'?options.x:options.y)||['category','multicategory'].includes(axis.scale))continue;
  const range=graph._fullLayout?.[name]?.range;if(!range?.length)continue;
  const values=axisRangeNumbers(range,axis.scale);if(!values.every(Number.isFinite))continue;
  const key=JSON.stringify(axis),group=groups.get(key)??[];group.push({graph,name,values,date:axis.scale==='date'});groups.set(key,group);
 }
 linkingAxes=true;
 try{
  const patches=new Map();
  for(const group of groups.values()){
   if(new Set(group.map(item=>item.graph)).size<2)continue;
   const range=[Math.min(...group.flatMap(item=>item.values)),Math.max(...group.flatMap(item=>item.values))];
   for(const item of group){const patch=patches.get(item.graph)??{};patch[`${item.name}.range`]=item.date?range.map(value=>new Date(value).toISOString()):range;patch[`${item.name}.autorange`]=false;patches.set(item.graph,patch);}
  }
  await Promise.all([...patches].map(([graph,patch])=>Plotly.relayout(graph,patch)));
 }finally{linkingAxes=false;}
}
function bindScatterLegend(graph,figure){
 if(graph._scatterLegendHandler)graph.removeListener('plotly_legendclick',graph._scatterLegendHandler);
 if(graph._scatterDoubleHandler)graph.removeListener('plotly_legenddoubleclick',graph._scatterDoubleHandler);
 const original={...figure.data[0],marker:{...figure.data[0].marker}},hidden={color:new Set(),marker:new Set()};
 const groups=figure.scatterGroups??{};
 graph._scatterLegendHandler=event=>{
  if(busy)return false;
  const guide=graph.data[event.curveNumber]?.meta;if(!guide?.scatterGuide)return;
  const set=hidden[guide.encoding];if(set.has(guide.category))set.delete(guide.category);else set.add(guide.category);
  const indices=original.x.map((_,index)=>index).filter(index=>!hidden.color.has(groups.color?.[index])&&!hidden.marker.has(groups.marker?.[index]));
  const subset=value=>Array.isArray(value)||ArrayBuffer.isView(value)?indices.map(index=>value[index]):value;
  const changes={x:[subset(original.x)],y:[subset(original.y)],'marker.color':[subset(original.marker.color)],'marker.symbol':[subset(original.marker.symbol)]};
  if(original.customdata)changes.customdata=[subset(original.customdata)];
  Plotly.restyle(graph,changes,[0]);
  Plotly.restyle(graph,{opacity:set.has(guide.category) ? .3 : 1},[event.curveNumber]);
  return false;
 };
 graph._scatterDoubleHandler=()=>false;
 graph.on('plotly_legendclick',graph._scatterLegendHandler);graph.on('plotly_legenddoubleclick',graph._scatterDoubleHandler);
}
async function updatePanelCharts(sharedFilters,preparedAxis){
 const view=panelView(),panels=structuredClone(currentPanels());
 const sharedBins=view.slug==='distribution'&&state.distribution_shared_bins;
 const effectivePanels=sharedBins?panels.map(panel=>({...panel,...Object.fromEntries(['binMode','binCount','binWidth','binScope'].map(key=>[key,panels[0][key]]))})):panels;
 encodeConfig({[view.key]:effectivePanels});
 const requests=panels.map((panel,index)=>{
  const request={...chartRequest(state,xColumn(),sharedFilters,index),...(preparedAxis?{preparedAxis}:{})};
  if(view.slug==='scatter'&&(!request.x||!request.y?.[0]))throw new Error(`${panel.title||`Scatter ${index+1}`}: choose both X and Y variables.`);
  if(view.slug!=='scatter'&&(!request.y?.length||request.y.some(name=>!name)))throw new Error(`${panel.title||`${view.label} ${index+1}`}: choose at least one numeric signal.`);
  return request;
 });
 if(sharedBins){
  const firstBins={...requests[0].bins},fitColumns=[...new Set(panels.flatMap(panel=>panel.columns))];
  for(const request of requests){request.bins={...firstBins};request.binFitColumns=fitColumns;}
 }
 const results=[];
 for(let index=0;index<requests.length;index++){
  status(`Calculating ${view.slug} ${index+1} of ${panels.length} · every selected row…`,'busy');
  if(sharedBins&&index>0&&results[0].meta?.binEdges?.length>=2)requests[index].bins.edges=[...results[0].meta.binEdges];
  const result=await engine.query(requests[index]);result.meta??={};
  if(view.slug==='scatter')result.meta.colorKind=schema.columns.find(column=>column.name===panels[index].color)?.kind;
  results.push(result);
 }
 setPanelLayout(panels);lastPanelResults=results.map((result,index)=>({result,panel:panels[index]}));lastResult=results[0];
 let plotted=0;
 for(let index=0;index<panels.length;index++){
  const result=results[index],graph=$(`#${panelChartId(index)}`),card=$(`[data-${view.slug}-card="${index}"]`);
  const rows=rowsFor(result);
  if(result.rowCount&&rows.usable>0){
   const figure=buildFigure(result,state,requests[index].x,index);
   if(panels.length>1&&view.slug!=='derivative'){figure.layout.height=440+Math.max(0,figure.layout.height-510);figure.layout.legend.y=1-36/figure.layout.height;}
   graph.querySelectorAll('.plot-empty,#chart-empty').forEach(element=>element.remove());
   unbindScatterLegend(graph);unbindComparison(graph);await Plotly.react(graph,figure.data,figure.layout,figure.config);
   if(view.slug==='scatter')bindScatterLegend(graph,figure);
   bindComparison(graph,panels[index],figure);
   let warnings=card.querySelector('.plot-warnings');
   if(!warnings){warnings=document.createElement('div');warnings.className='plot-warnings';warnings.setAttribute('role','status');card.append(warnings);}
   warnings.textContent=(figure.warnings??[]).join(' ');
  }else{purgePlot(graph);graph.innerHTML=`<div class="plot-empty">No usable values for this ${view.slug}. Check its signals and the shared filters.</div>`;card.querySelector('.plot-warnings')?.replaceChildren();}
  const count=index===0?$('#plotted-count'):card.querySelector('.panel-count');
  count.textContent=view.slug==='distribution'?`${fmt(rows.usable)} usable rows · ${fmt(result.rowCount)} histogram bins · no sampling`:`${fmt(rows.usable)} usable rows · no sampling`;
  const binPanel=card.querySelector('.bin-panel');
  binPanel.classList.toggle('hidden',!Object.keys(result.meta?.groups??{}).length);binPanel.querySelector('.bin-details').textContent=binDescription(result);
  renderRowLedger(card,result);plotted+=rows.usable;
 }
 await alignLinkedAxes();renderRowSummary(results[0]);
 return {plotted,count:panels.length};
}
async function updateChart(){
 capture();if(!schema)return;$('#preview-table').replaceChildren();$('#summary-table').replaceChildren();
 const started=performance.now(),updated=await updatePanelCharts(filters(),analysisOptions().preparedAxis);setDirty(false);
 status(`Ready · ${updated.count} ${panelView().label.toLowerCase()}${updated.count===1?'':'s'} updated in ${((performance.now()-started)/1000).toFixed(2)} s${updated.plotted?'':' · no usable values; check signals or filters.'}`);
}
function download(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function csv(result){const keys=Object.keys(result.columns);const quote=x=>{const s=x===null||x===undefined?'':typeof x==='object'?JSON.stringify(x):String(x);return /[",\r\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s;};const lines=[keys.map(quote).join(',')];for(let i=0;i<result.rowCount;i++)lines.push(keys.map(k=>quote(result.columns[k][i])).join(','));return new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'});}
function table(result){const keys=Object.keys(result.columns).slice(0,30);return `<div class="table-wrap"><table><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>${Array.from({length:Math.min(result.rowCount,100)},(_,i)=>`<tr>${keys.map(k=>`<td>${esc(result.columns[k][i])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function exportChartData(index){
 if(busy||dirty)return;
 const view=panelView(),result=view?lastPanelResults[index]?.result:lastResult;
 if(result?.rowCount)download(csv(result),view?`${view.slug}_${index+1}_data.csv`:'chart_data.csv');
}
function exportChartImage(index){
 if(busy||dirty)return;
 const view=panelView(),result=view?lastPanelResults[index]?.result:lastResult;
 if(result?.rowCount)Plotly.downloadImage(panelChartId(view?index:0),{format:'png',filename:view?`${view.slug}_${index+1}`:'tsplotter',scale:2});
}
function editPanel(index){
 if(busy||index<0||index>=currentPanels().length)return;
 const view=panelView(),focusedChoice=document.activeElement?.hasAttribute(`data-${view.slug}-panel`);
 capture();activePanels[state.plot_type]=index;renderOptions();syncActions();
 if(focusedChoice)$(`[data-${view.slug}-panel="${index}"]`).focus();
}
function changePanels(action){
 if(busy||!schema)return;capture();const view=panelView(),panels=currentPanels(),active=activePanelIndex();
 if(action==='remove'){
  if(panels.length===1)return;
  panels.splice(active,1);activePanels[state.plot_type]=Math.min(active,panels.length-1);
 }else{
  if(panels.length>=view.max)return;
  const panel=action==='duplicate'?{...structuredClone(panels[active]),title:`${(panels[active].title||view.label).slice(0,74)} copy`}:view.create(schema,panels.length);
  panels.push(panel);activePanels[state.plot_type]=panels.length-1;
 }
 resetPlotOutputs();setPanelLayout(panels);renderOptions();setDirty(true);
 $('#render-button').innerHTML=`${icons.refresh}<span>${panels.length>1?'Update charts':'Update chart'}</span>`;
 status(`${view.label} layout changed. Update charts to apply it.`);
 if(action==='remove')$(`[data-${view.slug}-panel="${activePanelIndex()}"]`).focus();
 else $('[data-panel-key="title"]').focus();
}
for(const view of Object.values(panelViews))$(`#${view.slug}-toolbar`).addEventListener('click',event=>{
 const choice=event.target.closest(`[data-${view.slug}-panel]`);if(choice){editPanel(Number(choice.getAttribute(`data-${view.slug}-panel`)));return;}
 const id=event.target.closest('button')?.id;
 for(const action of ['add','duplicate','remove'])if(id===`${action}-${view.slug}`)changePanels(action);
});
$('#plot-grid').addEventListener('click',event=>{
 const view=panelView();if(!view||busy)return;
 const edit=event.target.closest(`[data-edit-${view.slug}]`);if(edit){editPanel(Number(edit.getAttribute(`data-edit-${view.slug}`)));$('[data-panel-key="title"]').focus({preventScroll:true});$(`#${view.slug}-toolbar`).scrollIntoView({block:'nearest',behavior:'smooth'});return;}
 const exportButton=event.target.closest(`[data-${view.slug}-export]`);if(exportButton&&exportButton.id!=='chart-export'){exportChartData(Number(exportButton.getAttribute(`data-${view.slug}-export`)));return;}
 const imageButton=event.target.closest(`[data-${view.slug}-image]`);if(imageButton&&imageButton.id!=='image-export')exportChartImage(Number(imageButton.getAttribute(`data-${view.slug}-image`)));
});
function changeMode(mode){if(busy)return;capture();state.plot_type=mode;renderOptions();setDirty(true);task(updateChart);}
$('#file-input').addEventListener('change',event=>{if(event.target.files[0])importFile(event.target.files[0]);});
for(const id of ['welcome-open','replace-file'])$(`#${id}`).onclick=()=>$('#file-input').click();
$('#file-drop').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('#file-input').click();}});
for(const eventName of ['dragenter','dragover'])$('#file-drop').addEventListener(eventName,event=>{event.preventDefault();if(!busy)$('#file-drop').classList.add('drag-over');});
for(const eventName of ['dragleave','drop'])$('#file-drop').addEventListener(eventName,event=>{event.preventDefault();$('#file-drop').classList.remove('drag-over');if(eventName==='drop'&&!busy&&event.dataTransfer?.files[0])importFile(event.dataTransfer.files[0]);});
$('#reload-sheet').onclick=()=>{if(lastFile)importFile(lastFile);};
$('#example-button').onclick=example;$('#welcome-example').onclick=example;
$('#render-button').onclick=()=>task(updateChart);
$('#chart-type').onchange=()=>changeMode($('#chart-type').value);
for(const tab of document.querySelectorAll('[data-chart-mode]')){
 tab.onclick=()=>changeMode(tab.dataset.chartMode);
 tab.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=[...document.querySelectorAll('[data-chart-mode]')];const at=tabs.indexOf(tab);tabs[event.key==='Home'?0:event.key==='End'?tabs.length-1:(at+(event.key==='ArrowLeft'?-1:1)+tabs.length)%tabs.length].focus();});
}
$('#app').addEventListener('change',event=>{
 const panelKey=event.target.dataset.panelKey;
 if(panelKey){
  if(busy)return;capture();const panel=currentPanels()[activePanelIndex()],view=panelView();
  if(['color','marker'].includes(panelKey)&&!nums().includes(panel[panelKey]))panel[`${panelKey}BinMode`]='Individual values';
  if(['color','marker','colorMode','colorBinMode','markerBinMode','binMode'].includes(panelKey))renderOptions();
  if(panelKey==='title'){
   const choice=$(`[data-${view.slug}-panel="${activePanelIndex()}"]`);if(choice){choice.textContent=panel.title||`${view.label} ${activePanelIndex()+1}`;choice.title=choice.textContent;}
  }
  setDirty(true);return;
 }
 if(event.target.dataset.dateBound){if(!busy){capture();setDirty(true);}return;}
 const key=event.target.dataset.key;if(!key||busy)return;capture();
 if(['setup_time_mode','setup_profile','setup_tz_mode','setup_assume_tz','setup_out_tz'].includes(key))renderAxis();
 if(key==='distribution_shared_bins'){
  renderOptions();
 }
 if(/^flt2_(var|op)_/.test(key))renderFilters();refreshView();setDirty(true);
});
$('#app').addEventListener('input',event=>{
 if(busy||!schema||event.target.tagName!=='INPUT')return;
 if(event.target.dataset.key||event.target.dataset.panelKey||event.target.dataset.dateBound){capture();setDirty(true);}
});
$('#settings-button').onclick=()=>setSidebar(!document.body.classList.contains('sidebar-open'));
$('#close-sidebar').onclick=()=>setSidebar(false);$('#sidebar-backdrop').onclick=()=>setSidebar(false);
narrowScreen.addEventListener('change',()=>setSidebar(false,{focus:false}));
document.addEventListener('keydown',event=>{
 if(event.key==='Escape'&&document.body.classList.contains('sidebar-open'))setSidebar(false);
 if(event.key==='Tab'&&document.body.classList.contains('sidebar-open')){const focusable=[...$('#sidebar').querySelectorAll('button,input,select,[tabindex="0"],summary,a')].filter(el=>!el.disabled&&el.offsetParent!==null&&!el.classList.contains('native-control'));const first=focusable[0],last=focusable.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}
});
$('.button-label').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('#config-input').click();}});
$('#add-filter').onclick=()=>{capture();if(state.flt2_n>=10){status('You can combine up to 10 filters.','error');return;}const i=state.flt2_n++;state[`flt2_logic_${i}`]='AND';state[`flt2_var_${i}`]=names()[0];renderFilters();refreshView();setDirty(true);};
$('#clear-filters').onclick=()=>{for(const k of Object.keys(state))if(k.startsWith('flt2_'))delete state[k];state.flt2_n=0;renderFilters();refreshView();setDirty(true);task(updateChart);};
$('#filters').addEventListener('click',e=>{const button=e.target.closest('[data-remove-filter]');if(!button)return;capture();const at=Number(button.dataset.removeFilter),next={};for(const [key,value]of Object.entries(state)){if(!key.startsWith('flt2_')||key==='flt2_n'){next[key]=value;continue;}const match=key.match(/_(\d+)$/);if(match){const i=Number(match[1]);if(i===at)continue;next[key.replace(/_\d+$/,`_${i>at?i-1:i}`)]=value;}}state=next;state.flt2_n=Math.max(0,state.flt2_n-1);renderFilters();refreshView();setDirty(true);});
$('#save-config').onclick=()=>{try{capture();const config=encodeConfig(!schema&&pendingSettings?pendingSettings:state);download(new Blob([config],{type:'application/json'}),'tsplotter-settings.json');savedConfig=config;syncWorkspaceSettings();status('Plot settings saved. Your dataset is not included.');}catch(e){status(e.message,'error');}};
$('#config-input').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const loaded=decodeConfig(await file.text());const sampling=Object.keys(loaded.settings).some(k=>k.endsWith('_maxpts'));if(schema){capture();const next=normalizeConfiguration(loaded.settings,schema);state={...defaultSettings(schema),...next.settings};savedConfig=encodeConfig(state);activePanels={};renderAxis();renderOptions();renderFilters();const applied=await task(updateChart);if(!applied)return;if(loaded.warnings.length||next.warnings.length||sampling)status(`Settings loaded. ${[...loaded.warnings,...next.warnings,sampling?'Legacy point limits were ignored; every selected point is retained.':''].filter(Boolean).join(' ')}`);}else{pendingSettings=loaded.settings;savedConfig=encodeConfig(loaded.settings);syncWorkspaceSettings();status('Settings loaded. Open your dataset to apply them.');}}catch(error){status(error.message,'error');}};
$('#clear-recovery').onclick=()=>{try{clearWorkspace(window.localStorage);recoveryAvailable=false;recoverySuppressed=true;recoveryProblem='Recovery cleared. A new edit will start recovery again.';if(!schema){pendingSettings=null;savedConfig=null;state=defaultSettings();renderAxis();renderOptions();}syncWorkspaceSettings();syncActions();}catch{status('Device recovery could not be cleared.','error');}};
$('#chart-export').onclick=()=>exportChartData(0);
$('#data-export').onclick=()=>task(async()=>{capture();syncWorkspaceSettings();status('Preparing the full filtered CSV on this device…','busy');const result=await engine.query({...analysisOptions(),mode:'export',x:xColumn(),filters:filters()});renderRowSummary(result);download(csv(result),'filtered_dataset.csv');status(`Ready · ${fmt(result.rowCount)} rows exported locally.${dirty?' Chart changes remain unapplied.':''}`);});
$('#preview-button').onclick=()=>task(async()=>{capture();syncWorkspaceSettings();const result=await engine.query({...analysisOptions(),mode:'preview',x:xColumn(),filters:filters(),limit:100});renderRowSummary(result);$('#preview-table').innerHTML=table(result);status(`Ready · preview loaded; all rows remain available for analysis.${dirty?' Chart changes remain unapplied.':''}`);});
$('#summary-button').onclick=()=>task(async()=>{capture();syncWorkspaceSettings();const result=await engine.summary(filters(),analysisOptions());renderRowSummary(result);$('#summary-table').innerHTML=table(result);status(`Ready · statistics calculated from all selected rows.${dirty?' Chart changes remain unapplied.':''}`);});
$('#image-export').onclick=()=>exportChartImage(0);
$('#cancel-button').onclick=()=>{engine.cancel();schema=null;resetPlotOutputs();setBusy(false);$('#workspace').classList.add('hidden');$('#welcome').classList.remove('hidden');status('Analysis cancelled. Reopen your local file to continue.');};
window.addEventListener('pagehide',()=>{capture();syncWorkspaceSettings();});
initializeRecovery();renderAxis();renderOptions();setSidebar(false,{focus:false});syncActions();
