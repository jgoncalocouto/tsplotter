import './style.css';
import Plotly from 'plotly.js-dist-min';
import {DataEngine} from './engine-client.js';
import {defaultSettings,decodeConfig,encodeConfig,normalizeConfiguration,filtersFromSettings,PLOT_TYPES,PALETTES,TIMEZONES,BIN_MODES} from './config.js';
import {buildFigure,chartRequest} from './charts.js';
import {enhancePickers,closePickers} from './pickers.js';

const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s),fmt=n=>Number(n??0).toLocaleString();
let schema=null,state=defaultSettings(),lastResult=null,currentName='',busy=false,pendingSettings=null,dirty=false,lastFile=null;
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
 'Distribution':['Understand the distribution','Compare frequencies across your signals using shared, adjustable histogram bins.','<path d="M4 20V13h4v7m1 0V5h5v15m1 0V9h5v11"/>'],
 'Scatter':['Explore relationships','Plot one signal against another and use colour to reveal patterns.','<circle cx="5" cy="17" r="1.2"/><circle cx="8" cy="11" r="1.2"/><circle cx="13" cy="14" r="1.2"/><circle cx="16" cy="7" r="1.2"/><circle cx="20" cy="4" r="1.2"/>'],
 'Grouped scatter':['Compare groups','Separate relationships with independent colours, marker shapes and numeric bins.','<circle cx="6" cy="7" r="2"/><circle cx="10" cy="15" r="2"/><path d="m16 4 3 5h-6l3-5ZM18 14l3 5h-6l3-5Z"/>'],
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
 <p class="field-help">Keep your chart selections, filters and styling.</p></section>
 <div class="sidebar-footer"><div class="privacy-note"><span class="privacy-dot"></span>Files stay on your device</div><a href="/plotter/?legacy=1">Open legacy Streamlit version ↗</a></div>
 </aside>
 <main><header class="topbar"><div><div class="breadcrumb">Workspace <span>/</span> Analysis</div><h1>Time Series Studio</h1></div><div class="tools"><button id="settings-button" class="secondary mobile-settings">${icons.sliders}Dataset & settings</button><button id="cancel-button" class="secondary hidden">Cancel analysis</button><span class="workspace-tag"><span></span>Private workspace</span></div></header>
 <div id="status" role="status" aria-live="polite" class="status"></div>
 <section id="welcome" class="welcome"><div class="welcome-copy"><span class="eyebrow">A workspace for curiosity</span><h2>Find the story<br>in your <span>signals.</span></h2><p>Start with a dataset. Explore every point, compare patterns and turn observations into a clearer picture.</p><div class="welcome-actions"><button id="welcome-open" class="primary">${icons.upload}Open a dataset</button><button id="welcome-example" class="secondary">Try example data ${icons.arrow}</button></div><div class="welcome-note"><span class="privacy-dot"></span>Private by design. Your files stay with you.</div></div><div class="welcome-visual" aria-hidden="true"><div class="mini-window"><div class="mini-toolbar"><span class="mini-dot"></span><span class="mini-dot"></span><span class="mini-dot"></span><span class="mini-title">Signal overview</span><span class="mini-live">LIVE VIEW</span></div><div class="mini-label">Temperature <strong>24.6 <small>°C</small></strong><span>+3.2 from baseline</span></div><svg viewBox="0 0 480 230" class="mini-chart"><defs><linearGradient id="chart-fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#7560d9" stop-opacity=".18"/><stop offset="100%" stop-color="#7560d9" stop-opacity="0"/></linearGradient></defs><path class="mini-grid" d="M0 35h480M0 85h480M0 135h480M0 185h480M40 0v230M140 0v230M240 0v230M340 0v230M440 0v230"/><path d="M0 175C20 180 30 100 60 118S95 182 125 136 155 48 185 76 210 150 240 100 265 28 295 58 320 120 350 75 380 12 405 36 440 85 480 28V230H0Z" fill="url(#chart-fade)"/><path class="mini-line muted-line" d="M0 150C40 145 40 172 80 158S140 118 170 134 210 157 250 130 290 115 320 125 380 105 420 95 450 108 480 88"/><path class="mini-line" d="M0 175C20 180 30 100 60 118S95 182 125 136 155 48 185 76 210 150 240 100 265 28 295 58 320 120 350 75 380 12 405 36 440 85 480 28"/></svg><div class="mini-legend"><span><i></i>Temperature</span><span><i></i>Baseline</span><span class="mini-all">All points included</span></div></div><div class="visual-caption">One dataset. More ways to understand it.</div></div><div class="welcome-features"><div><span>01</span><h3>Bring your data</h3><p>Open your local files. No uploads, no waiting for a server.</p></div><div><span>02</span><h3>Explore with purpose</h3><p>Six focused views, flexible filters and familiar Plotly tools.</p></div><div><span>03</span><h3>Keep every detail</h3><p>Every selected point, with portable settings and full exports.</p></div></div></section>
 <div id="workspace" class="hidden">
 <section class="dataset-header"><div class="dataset-identity"><span class="dataset-icon">${icons.file}</span><div><h2 id="source-label"></h2><div class="dataset-subtitle"><span id="file-format">Dataset</span><span>·</span><span>Local file</span></div></div></div><button id="replace-file" class="text-button">Change dataset ${icons.refresh}</button></section>
 <section class="stats" aria-label="Dataset summary"><div class="stat"><small>Rows in view</small><strong id="row-count">—</strong></div><div class="stat"><small>Numeric signals</small><strong id="numeric-count">—</strong></div><div class="stat"><small>Columns</small><strong id="column-count">—</strong></div><div class="stat"><small>Filtered out</small><strong id="filtered-count">—</strong></div><div class="stats-note"><span class="privacy-dot"></span>All points retained</div></section>
 <nav class="chart-tabs" aria-label="Chart views" role="tablist">${PLOT_TYPES.map(t=>`<button type="button" class="chart-tab" role="tab" aria-selected="${t==='Trends'}" data-chart-mode="${esc(t)}"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true">${modeMeta[t][2]}</svg><span>${esc(t)}</span></button>`).join('')}<select id="chart-type" tabindex="-1" class="native-control" aria-label="Chart type">${PLOT_TYPES.map(t=>`<option>${esc(t)}</option>`).join('')}</select></nav>
 <section class="chart-workspace card"><header class="chart-heading"><div><h2 id="chart-title">${modeMeta.Trends[0]}</h2><p id="chart-description">${modeMeta.Trends[1]}</p></div><span id="dirty-indicator" class="dirty-indicator hidden">Unapplied changes</span></header>
 <div class="chart-setup"><div id="chart-options" class="options"></div><button id="render-button" class="primary">${icons.refresh}<span>Update chart</span></button></div>
 <div class="control-panels"><details id="filter-panel" class="controls-panel"><summary>${icons.filter}Filter data<span id="filter-count" class="count-badge hidden">0</span></summary><div class="details-body"><div class="panel-description">Narrow the dataset. Filters apply to charts, statistics and full exports.</div><div id="filters"></div><div class="button-row"><button id="add-filter" class="secondary subtle">${icons.plus}Add filter</button><button id="clear-filters" class="text-button subtle">Clear all</button></div></div></details><details id="appearance-panel" class="controls-panel"><summary>${icons.sliders}Chart appearance</summary><div id="appearance-options" class="options details-body"></div></details></div>
 <div id="chart"><div id="chart-empty">Choose signals to begin.</div></div><div class="chart-footer"><div class="chart-readout"><span class="readout-dot"></span><span id="plotted-count">—</span></div><div class="chart-export-tools"><button id="chart-export" class="text-button subtle">${icons.download}Chart data</button><button id="image-export" class="text-button subtle">${icons.image}Save image</button></div></div><details id="bin-panel" class="bin-panel hidden"><summary>Group intervals & counts</summary><div id="bin-details" class="bin-details"></div></details>
 </section>
 <section class="data-panels"><details class="panel" id="preview-panel"><summary>${icons.file}Data preview & export<span>Inspect the rows behind your analysis</span></summary><div class="details-body"><div class="panel-toolbar"><p class="field-help">First 100 rows · up to 30 columns. Full exports contain every filtered row.</p><div class="button-row"><button id="preview-button" class="secondary subtle">Show preview</button><button id="data-export" class="secondary subtle">${icons.download}Export full dataset</button></div></div><div id="preview-table"></div></div></details><details class="panel" id="summary-panel"><summary>${icons.sliders}Summary statistics<span>A closer look at your numeric signals</span></summary><div class="details-body"><button id="summary-button" class="secondary subtle">Calculate summary</button><div id="summary-table"></div></div></details></section>
 </div><footer class="small-footer"><span>TSPlotter</span><span>Every signal has a story.</span></footer></main></div>`;

function status(text,kind=''){const el=$('#status');el.textContent=text;el.className=`status ${kind}`;}
function syncActions(){
 for(const id of ['chart-export','image-export'])$(`#${id}`).disabled=busy||dirty||!lastResult;
 for(const id of ['render-button','preview-button','data-export','summary-button'])$(`#${id}`).disabled=busy||!schema;
 $('#reload-sheet').disabled=busy||!lastFile||!/\.xlsx?$/i.test(lastFile.name);
}
function setDirty(value=true){dirty=Boolean(value&&schema);$('#dirty-indicator').classList.toggle('hidden',!dirty);$('#render-button').classList.toggle('has-changes',dirty);syncActions();}
function setBusy(value){
 busy=value;if(value)closePickers();$('#progress').classList.toggle('hidden',!value);$('#cancel-button').classList.toggle('hidden',!value);
 for(const el of document.querySelectorAll('button,input,select'))if(!['cancel-button','close-sidebar','settings-button'].includes(el.id))el.disabled=value;
 $('#render-button').innerHTML=value?'<span class="button-spinner"></span><span>Updating…</span>':`${icons.refresh}<span>Update chart</span>`;
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
function check(label,key){return `<label class="check"><input data-key="${key}" type="checkbox" ${state[key]?'checked':''}>${esc(label)}</label>`;}
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

function renderOptions(){const mode=state.plot_type,numeric=nums(),all=names(),none=['(none)',...all];let main='',appearance='';
 if(mode==='Trends'){main+=input('Panels','mlc_n_subplots',{min:1,max:6,step:1});for(let i=0;i<Math.min(6,state.mlc_n_subplots||1);i++)main+=select(state.mlc_n_subplots===1?'Signals':`Signals · panel ${i+1}`,`mlc_cols_${i}`,numeric,{multiple:true,id:i===0?'y-columns':'',className:'wide'});appearance=select('Line shape','mlc_shape',['linear','spline','hv','vh','hvh','vhv'])+input('Line width','mlc_width',{min:1,max:8})+select('Y-axis scale','mlc_ytype',['linear','log'])+check('Show markers','mlc_markers')+check('Range slider','mlc_rangeslider')+check('Unified hover','mlc_unified')+check('Spike lines','mlc_spikes');}
 if(mode==='Distribution'){main=select('Signals','hist_cols_main',numeric,{multiple:true,id:'y-columns',className:'wide'})+select('Binning','hist_binmode',['Auto (width)','Fixed width','Target #bins'])+(state.hist_binmode==='Fixed width'?input('Bin width','hist_width_input',{min:.000000000001}):input('Target bins','hist_bins_main',{min:10,max:200,step:1}));}
 if(mode==='Scatter'||mode==='Grouped scatter'){const p=mode==='Scatter'?'sc':'sc2';main=select('X variable',`${p}_x`,all,{id:'scatter-x'})+select('Y variable',`${p}_y`,all,{id:'y-columns'})+select('Color by',p==='sc'?'sc_group':'sc2_g1',none)+binOptions(p==='sc'?'sc_color':'sc2_color','Color',p==='sc'?state.sc_group:state.sc2_g1);if(p==='sc2')main+=select('Marker by','sc2_g2',none)+binOptions('sc2_marker','Marker',state.sc2_g2);appearance=input('Marker size',`${p}_msize`,{min:3,max:20})+input('Opacity',`${p}_mop`,{min:.2,max:1,step:.05})+select('Color palette',p==='sc'?'sc_cmap_disc':'sc2_palette',PALETTES);if(p==='sc')appearance+=select('Continuous colormap','sc_cmap_cont',['Viridis','Plasma','Inferno','Magma','Cividis','Turbo','Blues','Greens','Greys','Oranges','Purples','Reds']);else appearance+=select('Marker preset','sc2_marker_preset',['Core Shapes','Crosshair Set','Directional','Starburst','Minimal Dots'])+input('Custom symbols (comma separated)','sc2_marker_custom',{type:'text'});}
 if(mode==='Rolling window')main=select('Signal','rw_y',numeric,{id:'y-columns'})+input('Window size (rows)','rw_window',{min:2,max:200000,step:1})+check('Include current row','rw_include_current')+check('Require full window','rw_full');
 if(mode==='Rolling window')appearance=check('Shade min/max band','rw_band')+check('Unified hover','rw_unified')+check('Range slider','rw_rangeslider')+check('Spike lines','rw_spikes')+input('Marker size','rw_msize',{min:3,max:16})+input('Marker opacity','rw_mop',{min:.2,max:1,step:.05});
 if(mode==='Derivatives'){main=select('Signal','der_y',numeric,{id:'y-columns'})+select('Time unit','der_tunit',['seconds','minutes','hours','days','none (sample index)'])+select('Relative X spacing','der_xmode',['index step (1)','use numeric index values'])+input('Derivative smoothing window','der_smooth',{min:1,max:9999,step:1});appearance=check('Show markers','der_markers')+input('Line width','der_lw',{min:1,max:8})+check('Unified hover','der_unified')+check('Range slider','der_rangeslider')+check('Spike lines','der_spikes');}
 $('#chart-options').innerHTML=main;$('#chart-options').className=`options ${mode==='Trends'?'trend-options':''}`;$('#appearance-options').innerHTML=appearance;refreshView();enhancePickers($('#chart-options'));}
function binOptions(prefix,label,source){if(!nums().includes(source))return '';return select(`${label} groups`,`${prefix}_bin_mode`,BIN_MODES)+(state[`${prefix}_bin_mode`]==='Bin width'?input(`${label} bin width`,`${prefix}_bin_width`,{min:.000000000001}):state[`${prefix}_bin_mode`]==='Number of bins'?input(`${label} bin count`,`${prefix}_bin_count`,{min:1,max:100,step:1}):'');}
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
 const request=chartRequest({...state,plot_type:'Trends'},xColumn(),[]);
 return state.setup_time_mode==='Absolute time'?{preparedAxis:{x:xColumn(),timeAxis:request.timeAxis},x:xColumn(),timeAxis:request.timeAxis}:{};
}
function filters(){const f=filtersFromSettings(state,schema);const options=analysisOptions();for(const condition of f){if(condition.column==='(time index)')condition.column=xColumn();if(condition.kind==='datetime'&&condition.column===xColumn()&&options.timeAxis)condition.timeAxis=options.timeAxis;}return f;}
async function importFile(file){await task(async()=>{status('Reading your file on this device…','busy');capture();const isFirst=!schema;schema=null;lastResult=null;$('#preview-table').replaceChildren();$('#summary-table').replaceChildren();Plotly.purge('chart');$('#workspace').classList.add('hidden');$('#welcome').classList.remove('hidden');let imported,importWarnings=[];
 if(/\.xlsx?$/i.test(file.name)){const {readExcel}=await import('./file-import.js');const result=await readExcel(file,{sheetName:$('#sheet-name').value||undefined});imported=await engine.importRows(result.rows,{sourceName:file.name});importWarnings=result.warnings??[];}
 else if(/\.mat$/i.test(file.name)){const {readMatV73}=await import('./file-import.js');const result=await readMatV73(file);imported=await engine.importRows(result.rows,{sourceName:file.name});importWarnings=result.warnings??[];}
 else imported=await engine.importFile(file);
 schema=imported;currentName=file.name;lastFile=file;$('#file-format').textContent=file.name.split('.').pop().toUpperCase();$('#file-options').open=/\.xlsx?$/i.test(file.name);state={...defaultSettings(schema),...(pendingSettings??(!isFirst?state:{}))};pendingSettings=null;const normalized=normalizeConfiguration(state,schema);state=normalized.settings;renderAxis();renderOptions();renderFilters();$('#welcome').classList.add('hidden');$('#workspace').classList.remove('hidden');$('#source-label').textContent=currentName;$('#source-info').classList.remove('hidden');$('#source-info').textContent=`${fmt(schema.rowCount)} rows · ${fmt(schema.columns.length)} columns · local file`;$('#numeric-count').textContent=fmt(nums().length);$('#column-count').textContent=fmt(schema.columns.length);setBusy(true);await updateChart();setSidebar(false,{focus:false});if(normalized.warnings.length||importWarnings.length)status(`Ready. ${[...normalized.warnings,...importWarnings].join(' ')}`);});}
async function example(){const rows=Array.from({length:720},(_,i)=>({timestamp:new Date(Date.UTC(2025,0,1)+i*600000).toISOString(),temperature:Math.round((21+3*Math.sin(i*Math.PI/72)) *100)/100,humidity:Math.round((55-8*Math.sin(i*Math.PI/72))*100)/100,power_kw:Math.round((2.5+.7*Math.sin(i*Math.PI/72-.7))*100)/100,shift:i%144<72?'Day':'Night'}));const file=new File([JSON.stringify(rows)],'Example · sensor readings.json',{type:'application/json'});await importFile(file);}
async function updateChart(){capture();if(!schema)return;$('#preview-table').replaceChildren();$('#summary-table').replaceChildren();const request={...chartRequest(state,xColumn(),filters()),...(analysisOptions().preparedAxis?{preparedAxis:analysisOptions().preparedAxis}:{})};if(!request.y?.length||request.y.some(c=>!c))throw new Error('Choose at least one signal to plot.');status('Calculating every selected row locally…','busy');const started=performance.now();lastResult=await engine.query(request);if(!lastResult.rowCount){Plotly.purge('chart');$('#chart').innerHTML='<div id="chart-empty">No usable rows match this view. Clear or widen your filters.</div>';$('#row-count').textContent='0';$('#filtered-count').textContent=fmt(schema.rowCount);$('#plotted-count').textContent='0 rows plotted';setDirty(false);status('Ready · no rows match this view. Clear or widen your filters.');return;}lastResult.meta??={};const color=state.plot_type==='Scatter'?state.sc_group:state.sc2_g1;lastResult.meta.colorKind=schema.columns.find(c=>c.name===color)?.kind;const figure=buildFigure(lastResult,state,request.x);$('#chart-empty')?.remove();await Plotly.react('chart',figure.data,figure.layout,figure.config);const selected=lastResult.totalRows??lastResult.rowCount;$('#row-count').textContent=fmt(selected);$('#filtered-count').textContent=fmt(Math.max(0,schema.rowCount-selected));$('#plotted-count').textContent=state.plot_type==='Distribution'?`${fmt(selected)} selected rows · ${fmt(lastResult.rowCount)} histogram bins · no sampling`:`${fmt(lastResult.rowCount)} rows plotted · no sampling`;$('#bin-panel').classList.toggle('hidden',!Object.keys(lastResult.meta?.groups??{}).length);$('#bin-details').textContent=lastResult.meta?.groups?Object.values(lastResult.meta.groups).map(g=>g?.labels?`${g.source??g.column}: ${g.labels.map((label,i)=>`${label} (${fmt(g.counts?.[i])})`).join(' · ')}`:'').filter(Boolean).join('\n'):'';setDirty(false);status(`Ready · updated in ${((performance.now()-started)/1000).toFixed(2)} s${lastResult.meta?.invalidXRows?' · '+fmt(lastResult.meta.invalidXRows)+' rows with invalid timestamps excluded.':''}`);}
function download(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function csv(result){const keys=Object.keys(result.columns);const quote=x=>{const s=x===null||x===undefined?'':typeof x==='object'?JSON.stringify(x):String(x);return /[",\r\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s;};const lines=[keys.map(quote).join(',')];for(let i=0;i<result.rowCount;i++)lines.push(keys.map(k=>quote(result.columns[k][i])).join(','));return new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'});}
function table(result){const keys=Object.keys(result.columns).slice(0,30);return `<div class="table-wrap"><table><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>${Array.from({length:Math.min(result.rowCount,100)},(_,i)=>`<tr>${keys.map(k=>`<td>${esc(result.columns[k][i])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
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
 const key=event.target.dataset.key;if(!key||busy)return;capture();
 if(['setup_time_mode','setup_profile','setup_tz_mode','setup_assume_tz','setup_out_tz'].includes(key))renderAxis();
 if(key==='mlc_n_subplots'||key==='hist_binmode'||key.endsWith('_bin_mode')||['sc_group','sc2_g1','sc2_g2'].includes(key)){
  for(const [group,prefix]of [['sc_group','sc_color'],['sc2_g1','sc2_color'],['sc2_g2','sc2_marker']])if(key===group&&!nums().includes(state[group]))state[`${prefix}_bin_mode`]='Individual values';
  renderOptions();
 }
 if(/^flt2_(var|op)_/.test(key))renderFilters();refreshView();setDirty(true);
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
$('#save-config').onclick=()=>{try{capture();download(new Blob([encodeConfig(!schema&&pendingSettings?pendingSettings:state)],{type:'application/json'}),'tsplotter-settings.json');status('Plot settings saved. Your dataset is not included.');}catch(e){status(e.message,'error');}};
$('#config-input').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const loaded=decodeConfig(await file.text());const sampling=Object.keys(loaded.settings).some(k=>k.endsWith('_maxpts'));if(schema){capture();const next=normalizeConfiguration({...defaultSettings(schema),...loaded.settings},schema);state=next.settings;renderAxis();renderOptions();renderFilters();const applied=await task(updateChart);if(!applied)return;if(loaded.warnings.length||next.warnings.length||sampling)status(`Settings loaded. ${[...loaded.warnings,...next.warnings,sampling?'Legacy point limits were ignored; every selected point is retained.':''].filter(Boolean).join(' ')}`);}else{pendingSettings=loaded.settings;status('Settings loaded. Open your dataset to apply them.');}}catch(error){status(error.message,'error');}};
$('#chart-export').onclick=()=>{if(lastResult)download(csv(lastResult),'chart_data.csv');};
$('#data-export').onclick=()=>task(async()=>{if(dirty)await updateChart();capture();status('Preparing the full filtered CSV on this device…','busy');const result=await engine.query({...analysisOptions(),mode:'export',x:xColumn(),filters:filters()});download(csv(result),'filtered_dataset.csv');status(`Ready · ${fmt(result.rowCount)} rows exported locally.`);});
$('#preview-button').onclick=()=>task(async()=>{if(dirty)await updateChart();capture();const result=await engine.query({...analysisOptions(),mode:'preview',x:xColumn(),filters:filters(),limit:100});$('#preview-table').innerHTML=table(result);status('Ready · preview loaded; all rows remain available for analysis.');});
$('#summary-button').onclick=()=>task(async()=>{if(dirty)await updateChart();capture();const result=await engine.summary(filters(),analysisOptions());$('#summary-table').innerHTML=table(result);status('Ready · statistics calculated from all selected rows.');});
$('#image-export').onclick=()=>{if(lastResult)Plotly.downloadImage('chart',{format:'png',filename:'tsplotter',scale:2});};
$('#cancel-button').onclick=()=>{engine.cancel();schema=null;lastResult=null;setBusy(false);$('#workspace').classList.add('hidden');$('#welcome').classList.remove('hidden');status('Analysis cancelled. Reopen your local file to continue.');};
renderAxis();renderOptions();setSidebar(false,{focus:false});syncActions();
