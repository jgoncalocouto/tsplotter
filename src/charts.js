import {distributionPanels,scatterPanels} from './config.js';

export const COLORS=['#6251c6','#13a79b','#e89b40','#df617b','#438cd8','#9c6ec6','#717f98','#9bbf58','#b77662','#d071bb'];
const PALETTE_COLORS={
 Plotly:COLORS,D3:['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'],
 Set1:['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
 Set2:['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'],
 Set3:['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'],
 Pastel:['#66c5cc','#f6cf71','#f89c74','#dcb0f2','#87c55f','#9eb9f3','#fe88b1','#c9db74','#8be0a4','#b497e7'],
 Bold:['#7f3c8d','#11a579','#3969ac','#f2b701','#e73f74','#80ba5a','#e68310','#008695','#cf1c90','#f97b72'],
 Dark24:['#2e91e5','#e15f99','#1ca71c','#fb0d0d','#da16ff','#222a2a','#b68100','#750d86','#eb663b','#511cfb','#00a08b','#fb00d1','#fc0080','#b2828d','#6c7c32','#778aae','#862a16','#a777f1','#620042','#1616a7','#da60ca','#6c4516','#0d2a63','#af0038'],
 Light24:['#fd3216','#00fe35','#6a76fc','#fed4c4','#fe00ce','#0df9ff','#f6f926','#ff9616','#479b55','#eea6fb','#dc587d','#d626ff','#6e899c','#00b5f7','#b68e00','#c9fbe5','#ff0092','#22ffa7','#e3ee9e','#86ce00','#bc7196','#7e7dcd','#fc6955','#e48f72']
};
const MARKER_PRESETS={
 'Core Shapes':['circle','square','diamond','cross','x','triangle-up','triangle-down','star'],
 'Crosshair Set':['cross','x','cross-thin','x-thin','asterisk'],
 'Directional':['triangle-up','triangle-right','triangle-down','triangle-left','diamond'],
 'Starburst':['star','hexagram','star-triangle-up','star-square','star-diamond'],
 'Minimal Dots':['circle','circle-open','square','square-open','diamond','diamond-open'],
};
const escapeLabel=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const isGroup=name=>Boolean(name)&&name!=='(none)';
const finiteNumber=value=>typeof value==='number'&&Number.isFinite(value);
const values=(columns,name)=>columns[name]??[];
const axis=(i)=>i===0?'':String(i+1);
const num=(s,key,fallback)=>Number.isFinite(Number(s[key]))?Number(s[key]):fallback;
const check=(s,key,fallback)=>s[key]===undefined?fallback:Boolean(s[key]);
export function buildFigure(result,settings,xName,panelIndex=0){
 const c=result.columns, mode=settings.plot_type??'Trends';
 const traces=[],warnings=[]; let panels=1,scatterGroups;
 const layout={
  autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#ffffff',
  font:{family:'Inter, "Segoe UI", system-ui, sans-serif',size:13,color:'#48556b'},
  margin:{l:70,r:30,t:88,b:64,pad:0},height:510,showlegend:true,
  legend:{orientation:'h',yref:'container',xref:'container',y:1,x:.02,xanchor:'left',yanchor:'top',font:{size:13,color:'#48556b'},itemsizing:'constant',itemwidth:30,tracegroupgap:12,maxheight:42,bgcolor:'rgba(255,255,255,0)'},
  hoverlabel:{bgcolor:'#ffffff',bordercolor:'#dce2ed',font:{family:'Inter, "Segoe UI", system-ui, sans-serif',size:14,color:'#28364d'},align:'left',namelength:-1},
  hovermode:'x unified',hoverdistance:32,spikedistance:1000,uirevision:'plotter-browser',colorway:COLORS,
  modebar:{bgcolor:'rgba(255,255,255,0.92)',color:'#8290a6',activecolor:'#6251c6'},
 };

 const x=values(c,xName);
 const line=(name,y,i=0,extra={})=>{const suffix=axis(i);traces.push({type:extra.fill||(mode==='Trends'&&settings.mlc_shape&&settings.mlc_shape!=='linear')?'scatter':'scattergl',mode:'lines',x,y,name,xaxis:`x${suffix}`,yaxis:`y${suffix}`,line:{color:COLORS[traces.length%COLORS.length],width:2,simplify:false},connectgaps:false,...extra});};
 const axes=(n)=>{
  panels=n;layout.height=510+(n-1)*175;
  const gap=n>1?.065:0,panelHeight=(1-gap*(n-1))/n;
  for(let i=0;i<n;i++){
   const s=axis(i),bottom=1-(i+1)*panelHeight-i*gap;
   const shared={automargin:true,showgrid:true,gridcolor:'#eaf0f7',gridwidth:1,zeroline:false,showline:true,linecolor:'#dce3ee',linewidth:1,ticks:'outside',ticklen:4,tickcolor:'#dce3ee',tickfont:{size:13,color:'#738096'},showspikes:true,spikemode:'across',spikesnap:'cursor',spikedash:'dot',spikecolor:'#a8b4c8',spikethickness:1};
   layout[`xaxis${s}`]={...shared,title:i===n-1?{text:xName==='__row__'?'Row number':xName,font:{size:13,color:'#48556b'},standoff:18}:undefined,anchor:`y${s}`,matches:i?'x':undefined,showticklabels:i===n-1,showspikes:check(settings,mode==='Rolling window'?'rw_spikes':mode==='Derivatives'?'der_spikes':'mlc_spikes',true)};
   layout[`yaxis${s}`]={...shared,domain:[Math.max(0,bottom),Math.min(1,bottom+panelHeight)],anchor:`x${s}`,title:{standoff:18,font:{size:13,color:'#48556b'}},nticks:n>1?5:7,showspikes:false};
  }
 };
 const rangeSlider=(name,visible)=>{
  layout[name].rangeslider={visible,thickness:panels===1?.11:.075,bgcolor:'#f7f8fc',bordercolor:'#e1e6f0',borderwidth:1};
  if(visible){layout.height+=50;layout.margin.b=76;}
 };

 if(mode==='Trends'){
  const count=Math.max(1,Math.min(6,num(settings,'mlc_n_subplots',1)));axes(count);
  for(let i=0;i<count;i++){let cols=settings[`mlc_cols_${i}`]??[];for(const name of cols){line(name,values(c,name),i,{mode:check(settings,'mlc_markers',false)?'lines+markers':'lines',marker:{size:4},line:{width:num(settings,'mlc_width',2),shape:settings.mlc_shape??'linear',simplify:false}});}layout[`yaxis${axis(i)}`].type=settings.mlc_ytype??'linear';}
  rangeSlider(`xaxis${axis(count-1)}`,check(settings,'mlc_rangeslider',true));layout.hovermode=check(settings,'mlc_unified',true)?'x unified':'closest';
 } else if(mode==='Distribution'){
  const panel=distributionPanels(settings)[panelIndex];
  if(!panel)throw new Error(`Distribution panel ${panelIndex+1} does not exist.`);
  axes(1);layout.hovermode='closest';layout.barmode='overlay';layout.xaxis.title={text:'Value'};layout.yaxis.title={text:'Count'};
  layout.uirevision=`distribution:${panelIndex}:${JSON.stringify([panel.columns,panel.binMode,panel.binCount,panel.binWidth])}`;
  for(const name of panel.columns){traces.push({type:'bar',name,x:values(c,'bin_center'),y:values(c,name),width:values(c,'bin_right').map((v,i)=>v-c.bin_left[i]),opacity:.65,customdata:values(c,'bin_left').map((v,i)=>[v,c.bin_right[i]]),hovertemplate:'[%{customdata[0]}, %{customdata[1]})<br>Count: %{y}<extra>%{fullData.name}</extra>'});}
 } else if(mode==='Scatter'||mode==='Grouped scatter'){
  const panel=scatterPanels(settings)[panelIndex];
  if(!panel)throw new Error(`Scatter panel ${panelIndex+1} does not exist.`);
  axes(1);layout.hovermode='closest';layout.uirevision=`scatter:${panelIndex}:${panel.x}:${panel.y}`;
  layout.xaxis.title={text:panel.x};layout.yaxis.title={text:panel.y};
  layout.xaxis.showspikes=false;layout.yaxis.showspikes=false;
  const xx=values(c,panel.x),yy=values(c,panel.y);
  const colorOn=isGroup(panel.color),markerOn=isGroup(panel.marker);
  const colorMeta=result.meta?.groups?.color,markerMeta=result.meta?.groups?.marker;
  const colorRaw=values(c,panel.color),markerRaw=values(c,panel.marker);
  const numericColor=result.meta?.colorKind==='numeric'||colorRaw.length>0&&colorRaw.every(finiteNumber);
  const continuous=colorOn&&!colorMeta&&panel.colorMode!=='discrete'&&numericColor;
  if(colorOn&&!colorMeta&&panel.colorMode==='continuous'&&!numericColor)warnings.push('Continuous color needs a numeric column. Categorical colors are being used instead.');
  const colorRows=colorOn&&!continuous?Array.from(values(c,colorMeta?.column??panel.color),String):[];
  const markerRows=markerOn?Array.from(values(c,markerMeta?.column??panel.marker),String):[];
  const colorOrder=colorMeta?.labels?.map(String)??[...new Set(colorRows)];
  const markerOrder=markerMeta?.labels?.map(String)??[...new Set(markerRows)];
  const occupiedColors=new Set(colorRows),occupiedMarkers=new Set(markerRows);
  const palette=PALETTE_COLORS[panel.palette]??COLORS;
  const custom=panel.customSymbols?.split(',').map(symbol=>symbol.trim()).filter(Boolean);
  const symbols=custom?.length?custom:MARKER_PRESETS[panel.markerPreset]??MARKER_PRESETS['Core Shapes'];
  const colorMap=new Map(colorOrder.map((category,index)=>[category,palette[index%palette.length]]));
  const symbolMap=new Map(markerOrder.map((category,index)=>[category,symbols[index%symbols.length]]));
  if(new Set([...occupiedColors].map(category=>colorMap.get(category))).size<occupiedColors.size)warnings.push('Some color groups share a palette color. Use fewer bins or a larger palette to distinguish every group.');
  if(new Set([...occupiedMarkers].map(category=>symbolMap.get(category))).size<occupiedMarkers.size)warnings.push('Some marker groups share a symbol. Use fewer bins or more symbols to distinguish every group.');
  const marker={size:panel.size??7,opacity:panel.opacity??.7,color:COLORS[0],symbol:'circle'};
  if(continuous){
   marker.color=colorRaw;marker.colorscale=panel.colormap??'Viridis';marker.showscale=true;
   let low=Infinity,high=-Infinity;
   for(const value of colorRaw){low=Math.min(low,value);high=Math.max(high,value);}
   if(Number.isFinite(low)&&Number.isFinite(high)){marker.cmin=low;marker.cmax=high;}
   marker.colorbar={title:{text:panel.color},thickness:14};layout.margin.r=84;
  }else if(colorOn)marker.color=colorRows.map(category=>colorMap.get(category));
  if(markerOn)marker.symbol=markerRows.map(category=>symbolMap.get(category));
  const hoverColumns=[...new Set([colorOn?panel.color:undefined,markerOn?panel.marker:undefined].filter(Boolean))];
  const hoverFields=hoverColumns.map(name=>({name,values:values(c,name)}));
  if(colorMeta)hoverFields.push({name:'Color interval',values:values(c,colorMeta.column)});
  if(markerMeta)hoverFields.push({name:'Marker interval',values:values(c,markerMeta.column)});
  let hovertemplate=`${escapeLabel(panel.x)}: %{x}<br>${escapeLabel(panel.y)}: %{y}`;
  hoverFields.forEach((field,index)=>{hovertemplate+=`<br>${escapeLabel(field.name)}: %{customdata[${index}]}`;});
  traces.push({type:'scattergl',mode:'markers',name:panel.title||`Scatter ${panelIndex+1}`,x:xx,y:yy,marker,showlegend:false,
   customdata:hoverFields.length?Array.from(xx,(_,index)=>hoverFields.map(field=>field.values[index])):undefined,
   hovertemplate:hovertemplate+'<extra></extra>',meta:{scatterPoints:true,panelIndex}});
  for(const [encoding,name,order,occupied,mapping]of[
   ['color',panel.color,colorOrder,occupiedColors,colorMap],['marker',panel.marker,markerOrder,occupiedMarkers,symbolMap],
  ]){
   if(encoding==='color'&&!colorOn||encoding==='marker'&&!markerOn||encoding==='color'&&continuous)continue;
   let first=true;
   for(const category of order){
    if(!occupied.has(category))continue;
    const guideMarker={size:9,color:encoding==='color'?mapping.get(category):'#717f98',symbol:encoding==='marker'?mapping.get(category):'circle'};
    traces.push({type:'scatter',mode:'markers',x:[null],y:[null],name:category,marker:guideMarker,showlegend:true,hoverinfo:'skip',
     legendgroup:`scatter-${panelIndex}-${encoding}-${category}`,legendgrouptitle:first?{text:`${encoding==='color'?'Color':'Marker'}: ${name}`}:undefined,
     meta:{scatterGuide:true,encoding,category,panelIndex}});
    first=false;
   }
  }
  scatterGroups={color:colorOn&&!continuous?colorRows:undefined,marker:markerOn?markerRows:undefined};
 }else if(mode==='Rolling window'){
  axes(1);const y=settings.rw_y;line(y,values(c,y),0,{mode:'markers',marker:{size:num(settings,'rw_msize',7),opacity:num(settings,'rw_mop',.7),color:'#9b93c7'}});
  line('Rolling minimum',values(c,`${y}__roll_min`),0,{line:{color:'#13a79b',width:1,simplify:false}});
  line('Rolling maximum',values(c,`${y}__roll_max`),0,{...(check(settings,'rw_band',true)?{fill:'tonexty',fillcolor:'rgba(19,167,155,.12)'}:{}),line:{color:'#13a79b',width:1,simplify:false}});
  line('Rolling mean',values(c,`${y}__roll_mean`),0,{line:{color:'#6251c6',width:2,simplify:false}});rangeSlider('xaxis',check(settings,'rw_rangeslider',true));layout.hovermode=check(settings,'rw_unified',true)?'x unified':'closest';
 }else if(mode==='Derivatives'){
  axes(3);const y=settings.der_y;const lw=num(settings,'der_lw',2);for(const [i,[label,col]]of [[y,y],['First derivative',`${y}__d1`],['Second derivative',`${y}__d2`]].entries()){line(label,values(c,col),i,{mode:check(settings,'der_markers',true)?'lines+markers':'lines',marker:{size:4},line:{width:lw,simplify:false}});layout[`yaxis${axis(i)}`].title={text:label};}rangeSlider('xaxis3',check(settings,'der_rangeslider',true));layout.hovermode=check(settings,'der_unified',true)?'x unified':'closest';
 }
 // The modebar occupies the first row. A container-anchored legend begins
 // 36 px below it, independent of subplot count, slider height, or screen width.
 layout.legend.y=1-36/layout.height;
 return {data:traces,layout,warnings,scatterGroups,config:{responsive:true,scrollZoom:true,displayModeBar:true,displaylogo:false,showTips:false,toImageButtonOptions:{format:'png',filename:'tsplotter',scale:2}}};
}
export function chartRequest(settings,x,filters,panelIndex=0){
 const map={'Trends':'trends','Distribution':'distribution','Scatter':'scatter','Grouped scatter':'grouped-scatter','Rolling window':'rolling','Derivatives':'derivatives'};let request={mode:map[settings.plot_type]??'trends',x,filters,timeAxis:{profile:settings.setup_profile,format:settings.setup_profile==='Custom strptime'?settings.setup_format:undefined,timezone:settings.setup_tz_mode==='Assume timezone and localize'?(settings.setup_assume_tz==='Custom...'?settings.setup_custom_tz:settings.setup_assume_tz):undefined,outputTimezone:settings.setup_out_tz==='Custom...'?settings.setup_target_tz:(settings.setup_out_tz==='(keep)'?undefined:settings.setup_out_tz)}};
 if(settings.setup_time_mode!=='Absolute time')delete request.timeAxis;
 if(request.mode==='trends')request.y=[...new Set(Array.from({length:Number(settings.mlc_n_subplots)||1},(_,i)=>settings[`mlc_cols_${i}`]??[]).flat())];
 if(request.mode==='distribution'){
  const panel=distributionPanels(settings)[panelIndex];
  if(!panel)throw new Error(`Distribution panel ${panelIndex+1} does not exist.`);
  request.y=[...panel.columns];
  request.bins={mode:panel.binMode==='Fixed width'?'width':panel.binMode==='Target #bins'?'count':'auto',count:Number(panel.binCount??50),width:Number(panel.binWidth??1)};
 }
 if(request.mode==='scatter'||request.mode==='grouped-scatter'){
  const panel=scatterPanels(settings)[panelIndex];
  if(!panel)throw new Error(`Scatter panel ${panelIndex+1} does not exist.`);
  request.mode='scatter';delete request.timeAxis;request.x=panel.x;request.y=[panel.y];
  if(isGroup(panel.color))request.color=panel.color;
  if(isGroup(panel.marker))request.marker=panel.marker;
  for(const encoding of ['color','marker']){
   const mode=panel[`${encoding}BinMode`];
   if(isGroup(panel[encoding])&&mode&&mode!=='Individual values')request[`${encoding}Binning`]={mode:mode==='Bin width'?'width':'count',count:Number(panel[`${encoding}BinCount`]??5),width:Number(panel[`${encoding}BinWidth`]??1)};
  }
 }
 if(request.mode==='rolling'){request.y=[settings.rw_y];request.window=Number(settings.rw_window)||50;request.includeCurrent=settings.rw_include_current??true;request.requireFull=settings.rw_full??false;}
 if(request.mode==='derivatives'){request.y=[settings.der_y];request.derivativeUnit=(settings.setup_time_mode!=='Absolute time'&&settings.der_xmode!=='use numeric index values')?'sample':(settings.der_tunit??'seconds');request.xMode=settings.der_xmode??'index step (1)';request.smoothWindow=Number(settings.der_smooth)||1;}
 return request;
}
