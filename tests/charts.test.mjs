import test from 'node:test';
import assert from 'node:assert/strict';
import {buildFigure,chartRequest} from '../src/charts.js';
import {defaultSettings,defaultTrendPanel,defaultRollingPanel,defaultDerivativePanel} from '../src/config.js';
const schema={columns:[{name:'x',kind:'numeric'},{name:'y',kind:'numeric'}]};
test('trends retain every point beyond former sampling caps',()=>{const n=200001,s={...defaultSettings(schema),trend_panels:[{...defaultTrendPanel(schema),columns:['y']}]};const cols={x:Array.from({length:n},(_,i)=>i),y:Array.from({length:n},(_,i)=>i*2)};const fig=buildFigure({columns:cols,rowCount:n},s,'x');assert.equal(fig.data[0].x.length,n);assert.equal(fig.data[0].y.at(-1),400000);assert.equal(fig.data[0].line.simplify,false);});
test('scatter does not treat none labels as source columns or parse numeric x as timestamps',()=>{const s={...defaultSettings(schema),plot_type:'Scatter',setup_time_mode:'Absolute time'};const request=chartRequest(s,'x',[]);assert.equal(request.color,undefined);assert.equal(request.marker,undefined);assert.equal(request.timeAxis,undefined);});
test('relative axes preserve numeric values without implicit datetime parsing',()=>{const s=defaultSettings(schema);assert.equal(chartRequest(s,'x',[]).timeAxis,undefined);});
test('legacy derivative settings map smoothing and explicit numeric spacing to engine',()=>{const s={plot_type:'Derivatives',der_y:'y',der_smooth:7,der_xmode:'use numeric index values',der_tunit:'seconds'};const request=chartRequest(s,'x',[]);assert.equal(request.smoothWindow,7);assert.equal(request.derivativeUnit,'seconds');});
test('legacy grouped scatter settings retain raw rows with separate encoding guides',()=>{
 const settings={plot_type:'Grouped scatter',sc2_x:'x',sc2_y:'y',sc2_g1:'label',sc2_g2:'shape'};
 const result={columns:{x:[1,2,3],y:[4,5,6],label:['a','b','a'],shape:['one','two','one']},meta:{}};
 const figure=buildFigure(result,settings,'x');
 assert.equal(figure.data.filter(trace=>trace.meta?.scatterPoints).reduce((count,trace)=>count+trace.x.length,0),3);
 assert.deepEqual(figure.data[0].x,result.columns.x);assert.deepEqual(figure.data[0].y,result.columns.y);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>trace.name),['a','b']);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='marker').map(trace=>trace.name),['one','two']);
});
test('legacy Trends panels become independent full-area figures',()=>{
 const columns={x:[0,1,2],y:[1,3,2]};
 for(let count=1;count<=6;count++){
  const settings={plot_type:'Trends',mlc_n_subplots:count,mlc_rangeslider:true};
  for(let i=0;i<count;i++)settings[`mlc_cols_${i}`]=['y'];
  for(let i=0;i<count;i++){
   const figure=buildFigure({columns,rowCount:3},settings,'x',i);
   assert.deepEqual(figure.layout.yaxis.domain,[0,1]);assert.equal(figure.layout.xaxis.anchor,'y');assert.equal(figure.layout.xaxis.showticklabels,true);
   assert.equal(figure.layout.xaxis.rangeslider.visible,true);assert.equal(figure.layout.yaxis2,undefined);assert.equal(figure.data.length,1);
   assert.deepEqual(figure.data[0].x,columns.x);assert.deepEqual(figure.data[0].y,columns.y);assert.equal(figure.data[0].line.simplify,false);
  }
 }
});
test('window and derivative spike controls change presentation without changing plotted series',()=>{
 const columns={x:[0,1,2],y:[1,3,2],y__roll_min:[1,1,2],y__roll_mean:[1,2,2.5],y__roll_max:[1,3,3],y__d1:[null,2,-1],y__d2:[null,null,-3]};
 for(const [mode,prefix]of[['Rolling window','rw'],['Derivatives','der']]){
  const settings={plot_type:mode,[`${prefix}_y`]:'y',[`${prefix}_spikes`]:false};
  const noSpikes=buildFigure({columns,rowCount:3},settings,'x');
  const withSpikes=buildFigure({columns,rowCount:3},{...settings,[`${prefix}_spikes`]:true},'x');
  assert.equal(noSpikes.layout.xaxis.showspikes,false);assert.equal(withSpikes.layout.xaxis.showspikes,true);
  assert.deepEqual(noSpikes.data,withSpikes.data);
 }
});

const scatterPanel=overrides=>({
 title:'Scatter 1',x:'x',y:'y',color:'(none)',marker:'(none)',colorMode:'auto',
 colorBinMode:'Individual values',colorBinCount:5,colorBinWidth:1,
 markerBinMode:'Individual values',markerBinCount:5,markerBinWidth:1,
 palette:'Plotly',colormap:'Viridis',markerPreset:'Core Shapes',customSymbols:'',size:7,opacity:.7,
 ...overrides,
});
const scatterSettings=panels=>({plot_type:'Scatter',setup_time_mode:'Absolute time',setup_dt_col:'time',scatter_panels:panels});

test('each scatter index builds its own query with independent axes, groups and bin parameters',()=>{
 const panels=[
  scatterPanel({title:'Temperature',x:'temperature',y:'response',color:'humidity',colorBinMode:'Number of bins',colorBinCount:3}),
  scatterPanel({title:'Power',x:'power',y:'response',marker:'humidity',markerBinMode:'Bin width',markerBinWidth:2.5}),
 ];
 const settings=scatterSettings(panels),before=structuredClone(settings),filters=[{kind:'numeric',column:'response',low:4}];
 const first=chartRequest(settings,'time',filters,0),second=chartRequest(settings,'time',filters,1);
 assert.equal(first.mode,'scatter');assert.equal(second.mode,'scatter');
 assert.equal(first.x,'temperature');assert.deepEqual(first.y,['response']);assert.equal(first.color,'humidity');assert.equal(first.marker,undefined);
 assert.deepEqual(first.colorBinning,{mode:'count',count:3,width:1,scope:'filtered'});assert.equal(first.markerBinning,undefined);
 assert.equal(second.x,'power');assert.deepEqual(second.y,['response']);assert.equal(second.color,undefined);assert.equal(second.marker,'humidity');
 assert.deepEqual(second.markerBinning,{mode:'width',count:5,width:2.5,scope:'filtered'});assert.equal(second.colorBinning,undefined);
 assert.equal(first.timeAxis,undefined);assert.equal(second.timeAxis,undefined);
 assert.equal(first.filters,filters);assert.equal(second.filters,filters);
 assert.deepEqual(settings,before);
 assert.throws(()=>chartRequest(settings,'time',filters,2),/Scatter panel 3/);
});

test('continuous numeric color and independent marker grouping use one full point trace',()=>{
 const settings=scatterSettings([scatterPanel({color:'temperature',marker:'station',colormap:'Plasma'})]);
 const result={columns:{x:[3,1,2],y:[9,1,4],temperature:[30,10,20],station:['B','A','B']},rowCount:3,meta:{colorKind:'numeric',groups:{}}};
 const figure=buildFigure(result,settings,'x'),points=figure.data[0];
 assert.equal(points.type,'scattergl');assert.equal(points.x.length,3);
 assert.deepEqual(points.marker.color,[30,10,20]);assert.equal(points.marker.symbol[0],points.marker.symbol[2]);assert.notEqual(points.marker.symbol[0],points.marker.symbol[1]);
 assert.equal(points.marker.showscale,true);assert.equal(points.marker.colorscale,'Plasma');assert.equal(points.marker.cmin,10);assert.equal(points.marker.cmax,30);
 assert.equal(figure.data.filter(trace=>trace.marker?.showscale).length,1);
 assert.equal(figure.data.filter(trace=>trace.meta?.encoding==='color').length,0);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='marker').map(trace=>trace.name),['B','A']);
 assert.equal(figure.scatterGroups.color,undefined);assert.deepEqual(figure.scatterGroups.marker,['B','A','B']);
 assert.deepEqual(points.customdata,[[30,'B'],[10,'A'],[20,'B']]);
 assert.equal(figure.warnings.length,0);
});

test('explicit discrete numeric color remains categorical without changing numeric axes',()=>{
 const settings=scatterSettings([scatterPanel({color:'x',colorMode:'discrete'})]);
 const result={columns:{x:[20,10,20],y:[4,5,6]},meta:{colorKind:'numeric',groups:{}}};
 const figure=buildFigure(result,settings,'x'),points=figure.data[0];
 assert.deepEqual(points.x,[20,10,20]);assert.deepEqual(points.y,[4,5,6]);
 assert.equal(points.marker.color[0],points.marker.color[2]);assert.notEqual(points.marker.color[0],points.marker.color[1]);
 assert.equal(points.marker.showscale,undefined);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>trace.name),['20','10']);
});

test('continuous color requested for a categorical source falls back with feedback',()=>{
 const settings=scatterSettings([scatterPanel({color:'station',colorMode:'continuous'})]);
 const figure=buildFigure({columns:{x:[1,2],y:[3,4],station:['North','South']},meta:{colorKind:'categorical'}},settings,'x');
 assert.ok(figure.data[0].marker.color.every(color=>/^#[a-f0-9]{6}$/i.test(color)));
 assert.equal(figure.data[0].marker.showscale,undefined);
 assert.match(figure.warnings.join(' '),/numeric column/);
});

test('independent bin metadata controls ordered guides, including collision-safe derived columns',()=>{
 const colorLabels=['[0, 10)','[10, 20)','[20, 30]'],markerLabels=['[0, 15)','[15, 30]'];
 const x=[22,1,23,24],y=[2,4,6,8];
 const result={columns:{x,y,__color_group:['raw','raw','raw','raw'],__color_group_:[colorLabels[2],colorLabels[0],colorLabels[2],colorLabels[2]],__marker_group:[markerLabels[1],markerLabels[0],markerLabels[1],markerLabels[1]]},meta:{colorKind:'numeric',groups:{
  color:{source:'x',column:'__color_group_',labels:colorLabels,counts:[1,0,3]},
  marker:{source:'x',column:'__marker_group',labels:markerLabels,counts:[1,3]},
 }}};
 const panel=scatterPanel({color:'x',marker:'x',colorMode:'continuous',colorBinMode:'Number of bins',markerBinMode:'Bin width'});
 const figure=buildFigure(result,scatterSettings([panel]),'x'),points=figure.data[0];
 assert.equal(points.x,x);assert.equal(points.y,y);
 assert.equal(points.marker.showscale,undefined);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>trace.name),[colorLabels[0],colorLabels[2]]);
 assert.deepEqual(figure.data.filter(trace=>trace.meta?.encoding==='marker').map(trace=>trace.name),markerLabels);
 const colorGuides=new Map(figure.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>[trace.name,trace.marker.color]));
 const markerGuides=new Map(figure.data.filter(trace=>trace.meta?.encoding==='marker').map(trace=>[trace.name,trace.marker.symbol]));
 assert.deepEqual(points.marker.color,[colorGuides.get(colorLabels[2]),colorGuides.get(colorLabels[0]),colorGuides.get(colorLabels[2]),colorGuides.get(colorLabels[2])]);
 assert.deepEqual(points.marker.symbol,[markerGuides.get(markerLabels[1]),markerGuides.get(markerLabels[0]),markerGuides.get(markerLabels[1]),markerGuides.get(markerLabels[1])]);
 assert.deepEqual(points.customdata[0],[22,colorLabels[2],markerLabels[1]]);
 assert.deepEqual(figure.scatterGroups.color,result.columns.__color_group_);
 assert.deepEqual(figure.scatterGroups.marker,result.columns.__marker_group);
 const guide=figure.data.find(trace=>trace.meta?.encoding==='color');
 assert.deepEqual(guide.meta,{scatterGuide:true,encoding:'color',category:colorLabels[0],panelIndex:0});
});

test('many color and marker combinations retain every point without a trace per combination',()=>{
 const count=35001,columns={x:[],y:[],color:[],shape:[]};
 for(let index=0;index<count;index++){columns.x.push(index);columns.y.push(2*index+1);columns.color.push(`color-${index%100}`);columns.shape.push(`shape-${Math.floor(index/100)%100}`);}
 const panel=scatterPanel({color:'color',marker:'shape'});
 const figure=buildFigure({columns,rowCount:count,meta:{groups:{}}},scatterSettings([panel]),'x');
 const pointTraces=figure.data.filter(trace=>trace.meta?.scatterPoints);
 assert.equal(pointTraces.length,1);assert.equal(pointTraces[0].x.length,count);assert.equal(pointTraces[0].y.at(-1),70001);
 assert.equal(pointTraces[0].marker.color.length,count);assert.equal(pointTraces[0].marker.symbol.length,count);
 assert.equal(figure.data.length,201);
 assert.match(figure.warnings.join(' '),/palette color/);assert.match(figure.warnings.join(' '),/symbol/);
});

test('scatter figures have independent axis labels, appearance and zoom identity',()=>{
 const panels=[scatterPanel({x:'x',y:'y',size:4,opacity:.4}),scatterPanel({title:'Reverse',x:'y',y:'x',size:12,opacity:.9})];
 const settings=scatterSettings(panels),result={columns:{x:[1,2],y:[3,4]},meta:{}};
 const first=buildFigure(result,settings,'time',0),second=buildFigure(result,settings,'time',1);
 assert.equal(first.layout.xaxis.title.text,'x');assert.equal(second.layout.xaxis.title.text,'y');
 assert.equal(first.layout.yaxis.title.text,'y');assert.equal(second.layout.yaxis.title.text,'x');
 assert.equal(first.data[0].marker.size,4);assert.equal(second.data[0].marker.size,12);
 assert.equal(first.data[0].marker.opacity,.4);assert.equal(second.data[0].marker.opacity,.9);
 assert.notEqual(first.layout.uirevision,second.layout.uirevision);
 assert.deepEqual(second.data[0].x,[3,4]);assert.deepEqual(second.data[0].y,[1,2]);
 assert.throws(()=>buildFigure(result,settings,'time',2),/Scatter panel 3/);
});

test('legacy continuous and grouped scatter requests map to the unified engine mode',()=>{
 const ordinary={plot_type:'Scatter',sc_x:'a',sc_y:'b',sc_group:'temperature',sc_cmap_cont:'Plasma',sc_color_bin_mode:'Individual values'};
 const grouped={plot_type:'Grouped scatter',sc2_x:'c',sc2_y:'d',sc2_g1:'temperature',sc2_g2:'station',sc2_color_bin_mode:'Number of bins',sc2_color_bin_count:4};
 const first=chartRequest(ordinary,'time',[]),second=chartRequest(grouped,'time',[]);
 assert.equal(first.mode,'scatter');assert.equal(first.x,'a');assert.deepEqual(first.y,['b']);assert.equal(first.color,'temperature');assert.equal(first.marker,undefined);
 assert.equal(second.mode,'scatter');assert.equal(second.x,'c');assert.deepEqual(second.y,['d']);assert.equal(second.color,'temperature');assert.equal(second.marker,'station');assert.equal(second.colorBinning.count,4);
});

test('invalid active bin widths are passed to validation instead of silently becoming one',()=>{
 const settings=scatterSettings([scatterPanel({color:'x',colorBinMode:'Bin width',colorBinWidth:0})]);
 assert.equal(chartRequest(settings,'x',[]).colorBinning.width,0);
});

const distributionPanel=overrides=>({title:'Distribution 1',columns:['y'],binMode:'Auto (width)',binCount:50,binWidth:1,...overrides});
const distributionSettings=panels=>({plot_type:'Distribution',setup_time_mode:'Relative time',distribution_panels:panels});

test('distribution indices select independent signals and bin queries with shared filtering',()=>{
 const settings=distributionSettings([
  distributionPanel({title:'Temperature',columns:['temperature','humidity'],binMode:'Target #bins',binCount:20}),
  distributionPanel({title:'Power',columns:['power'],binMode:'Fixed width',binWidth:.25}),
  distributionPanel({title:'Automatic',columns:['pressure']}),
 ]);
 const before=structuredClone(settings),filters=[{kind:'numeric',column:'quality',low:4}];
 const first=chartRequest(settings,'time',filters,0),second=chartRequest(settings,'time',filters,1),third=chartRequest(settings,'time',filters,2);
 assert.equal(first.mode,'distribution');assert.equal(second.mode,'distribution');
 assert.deepEqual(first.y,['temperature','humidity']);assert.deepEqual(second.y,['power']);assert.deepEqual(third.y,['pressure']);
 assert.deepEqual(first.bins,{mode:'count',count:20,width:1,scope:'filtered'});
 assert.deepEqual(second.bins,{mode:'width',count:50,width:.25,scope:'filtered'});
 assert.deepEqual(third.bins,{mode:'auto',count:50,width:1,scope:'filtered'});
 assert.equal(first.x,'time');assert.equal(second.x,'time');assert.equal(first.filters,filters);assert.equal(second.filters,filters);
 assert.equal(first.timeAxis,undefined);assert.deepEqual(settings,before);
 assert.throws(()=>chartRequest(settings,'time',filters,3),/Distribution panel 4/);
});

test('distribution queries own their selected columns and bin settings',()=>{
 const settings=distributionSettings([distributionPanel({columns:['temperature','humidity']})]),before=structuredClone(settings);
 const first=chartRequest(settings,'time',[]),second=chartRequest(settings,'time',[]);
 first.y.pop();first.bins.width=99;
 assert.deepEqual(settings,before);assert.deepEqual(second.y,['temperature','humidity']);assert.equal(second.bins.width,1);
});

test('distribution figures use each panel result without losing counts or interval widths',()=>{
 const settings=distributionSettings([
  distributionPanel({columns:['temperature','humidity'],binMode:'Target #bins',binCount:20}),
  distributionPanel({columns:['power'],binMode:'Fixed width',binWidth:.25}),
 ]);
 const results=[
  {columns:{bin_center:[5,15],bin_left:[0,10],bin_right:[10,20],temperature:[200001,150000],humidity:[100000,250001],power:[9,9]},rowCount:2},
  {columns:{bin_center:[.125,.375,.625],bin_left:[0,.25,.5],bin_right:[.25,.5,.75],power:[150001,100000,100000],temperature:[8,8,8]},rowCount:3},
 ];
 const before=structuredClone({settings,results}),first=buildFigure(results[0],settings,'time',0),second=buildFigure(results[1],settings,'time',1);
 assert.deepEqual(first.data.map(trace=>trace.name),['temperature','humidity']);assert.deepEqual(second.data.map(trace=>trace.name),['power']);
 assert.deepEqual(first.data[0].x,[5,15]);assert.deepEqual(first.data[0].width,[10,10]);assert.deepEqual(first.data[0].customdata,[[0,10],[10,20]]);
 assert.deepEqual(second.data[0].x,[.125,.375,.625]);assert.deepEqual(second.data[0].width,[.25,.25,.25]);assert.deepEqual(second.data[0].customdata,[[0,.25],[.25,.5],[.5,.75]]);
 for(const trace of [...first.data,...second.data]){assert.equal(trace.type,'bar');assert.equal(trace.y.reduce((sum,count)=>sum+count,0),350001);}
 assert.equal(first.layout.xaxis.title.text,'Value');assert.equal(first.layout.yaxis.title.text,'Count');assert.equal(first.layout.barmode,'overlay');
 assert.notEqual(first.layout.uirevision,second.layout.uirevision);assert.deepEqual({settings,results},before);
 assert.throws(()=>buildFigure(results[0],settings,'time',2),/Distribution panel 3/);
});

test('distribution zoom identity stays local and resets for different selected signals or bins',()=>{
 const result={columns:{bin_center:[.5],bin_left:[0],bin_right:[1],y:[10],x:[12]},rowCount:1};
 const settings=distributionSettings([distributionPanel(),distributionPanel()]);
 const identity=buildFigure(result,settings,'time',0).layout.uirevision;
 assert.equal(buildFigure(result,settings,'time',0).layout.uirevision,identity);
 assert.notEqual(buildFigure(result,settings,'time',1).layout.uirevision,identity);
 const renamed=structuredClone(settings);renamed.distribution_panels[0].title='Renamed';
 assert.equal(buildFigure(result,renamed,'time',0).layout.uirevision,identity);
 for(const change of [{columns:['x']},{binMode:'Fixed width',binWidth:.5},{binMode:'Target #bins',binCount:25}]){
  const next=structuredClone(settings);Object.assign(next.distribution_panels[0],change);
  assert.notEqual(buildFigure(result,next,'time',0).layout.uirevision,identity);
 }
});

test('legacy distribution settings retain their signals, bins and absolute time context',()=>{
 const settings={plot_type:'Distribution',hist_cols_main:['temperature','humidity'],hist_binmode:'Fixed width',hist_width_input:2.5,hist_bins_main:70,setup_time_mode:'Absolute time',setup_profile:'Unix epoch (seconds)',setup_out_tz:'(keep)'};
 const request=chartRequest(settings,'timestamp',[]);
 assert.deepEqual(request.y,['temperature','humidity']);assert.deepEqual(request.bins,{mode:'width',count:70,width:2.5,scope:'filtered'});assert.equal(request.timeAxis.profile,'Unix epoch (seconds)');
 const figure=buildFigure({columns:{bin_center:[1.25],bin_left:[0],bin_right:[2.5],temperature:[10],humidity:[12]}},settings,'timestamp');
 assert.deepEqual(figure.data.map(trace=>trace.name),['temperature','humidity']);assert.deepEqual(figure.data.map(trace=>trace.y),[[10],[12]]);
 const canonical={...settings,distribution_panels:[distributionPanel({columns:['power']})]};
 assert.deepEqual(chartRequest(canonical,'timestamp',[]).y,['power']);
});

test('invalid active distribution width stays invalid instead of silently becoming one',()=>{
 const canonical=distributionSettings([distributionPanel({binMode:'Fixed width',binWidth:0})]);
 const legacy={plot_type:'Distribution',hist_cols_main:['y'],hist_binmode:'Fixed width',hist_width_input:0};
 assert.equal(chartRequest(canonical,'x',[]).bins.width,0);assert.equal(chartRequest(legacy,'x',[]).bins.width,0);
});

test('Trends instances own signals and presentation while source colors survive reordering',()=>{
 const first={...defaultTrendPanel(schema),columns:['y','x'],shape:'spline',width:4,ytype:'log',markers:true,rangeslider:false,unified:false,spikes:false};
 const second={...defaultTrendPanel(schema,1),columns:['x','y'],shape:'linear',width:1,ytype:'linear',markers:false,rangeslider:true,unified:true,spikes:true};
 const settings={plot_type:'Trends',trend_panels:[first,second]},before=structuredClone(settings),columns={x:[0,1,2],y:[1,null,3]};
 const a=chartRequest(settings,'x',[],0),b=chartRequest(settings,'x',[],1);
 assert.deepEqual(a.y,['y','x']);assert.deepEqual(b.y,['x','y']);a.y.pop();assert.deepEqual(settings,before);
 const figures=[0,1].map(index=>buildFigure({columns,rowCount:3},settings,'x',index));
 assert.deepEqual(figures[0].data[0].y,[1,null,3]);assert.equal(figures[0].data[0].connectgaps,false);
 assert.equal(figures[0].data[0].type,'scatter');assert.equal(figures[0].data[0].line.shape,'spline');assert.equal(figures[0].data[0].line.width,4);
 assert.equal(figures[0].data[0].mode,'lines+markers');assert.equal(figures[1].data[0].mode,'lines');
 assert.equal(figures[0].layout.yaxis.type,'log');assert.equal(figures[1].layout.yaxis.type,'linear');
 assert.equal(figures[0].layout.xaxis.rangeslider.visible,false);assert.equal(figures[1].layout.xaxis.rangeslider.visible,true);
 assert.equal(figures[0].layout.xaxis.showspikes,false);assert.equal(figures[1].layout.xaxis.showspikes,true);
 assert.equal(figures[0].layout.hovermode,'closest');assert.equal(figures[1].layout.hovermode,'x unified');
 const colors=figure=>new Map(figure.data.map(trace=>[trace.name,trace.line.color]));
 assert.deepEqual(colors(figures[0]),colors(figures[1]));assert.notEqual(figures[0].layout.uirevision,figures[1].layout.uirevision);
 assert.throws(()=>chartRequest(settings,'x',[],2),/Trend panel 3/);assert.throws(()=>buildFigure({columns},settings,'x',2),/Trend panel 3/);
});

test('Rolling instances independently query windows and render their own signal and band',()=>{
 const settings={plot_type:'Rolling window',rolling_panels:[
  {...defaultRollingPanel(schema),y:'y',window:3,includeCurrent:false,requireFull:true,band:false,size:4,opacity:.4,spikes:false,rangeslider:false},
  {...defaultRollingPanel(schema,1),y:'x',window:7,includeCurrent:true,requireFull:false,band:true,size:11,opacity:.9,spikes:true,rangeslider:true},
 ]};
 const before=structuredClone(settings),a=chartRequest(settings,'time',[],0),b=chartRequest(settings,'time',[],1);
 assert.deepEqual(a.y,['y']);assert.equal(a.window,3);assert.equal(a.includeCurrent,false);assert.equal(a.requireFull,true);
 assert.deepEqual(b.y,['x']);assert.equal(b.window,7);assert.equal(b.includeCurrent,true);assert.equal(b.requireFull,false);
 const columns={time:[0,1,2],x:[1,3,2],y:[4,null,6],x__roll_min:[1,1,1],x__roll_mean:[1,2,2],x__roll_max:[1,3,3],y__roll_min:[null,null,4],y__roll_mean:[null,null,4],y__roll_max:[null,null,4]};
 const figures=[0,1].map(index=>buildFigure({columns},settings,'time',index));
 assert.deepEqual(figures[0].data[0].y,[4,null,6]);assert.deepEqual(figures[1].data[0].y,[1,3,2]);
 assert.equal(figures[0].data[0].marker.size,4);assert.equal(figures[1].data[0].marker.size,11);
 assert.equal(figures[0].data[2].fill,undefined);assert.equal(figures[1].data[2].fill,'tonexty');
 assert.equal(figures[0].layout.xaxis.showspikes,false);assert.equal(figures[1].layout.xaxis.showspikes,true);
 assert.notEqual(figures[0].layout.uirevision,figures[1].layout.uirevision);assert.deepEqual(settings,before);
 assert.throws(()=>chartRequest(settings,'time',[],2),/Rolling panel 3/);assert.throws(()=>buildFigure({columns},settings,'time',2),/Rolling panel 3/);
});

test('Derivative instances retain three aligned axes and independent differentiation settings',()=>{
 const settings={plot_type:'Derivatives',setup_time_mode:'Absolute time',derivative_panels:[
  {...defaultDerivativePanel(schema),y:'y',timeUnit:'hours',xMode:'use numeric index values',smoothWindow:5,markers:false,width:4,rangeslider:false,spikes:false},
  {...defaultDerivativePanel(schema,1),y:'x',timeUnit:'seconds',smoothWindow:1,markers:true,width:1,rangeslider:true,spikes:true},
 ]};
 const a=chartRequest(settings,'time',[],0),b=chartRequest(settings,'time',[],1);
 assert.deepEqual(a.y,['y']);assert.equal(a.derivativeUnit,'hours');assert.equal(a.smoothWindow,5);assert.equal(a.xMode,'use numeric index values');
 assert.deepEqual(b.y,['x']);assert.equal(b.derivativeUnit,'seconds');assert.equal(b.smoothWindow,1);
 const columns={time:[0,1,2],x:[1,2,3],y:[2,4,8],x__d1:[null,1,1],x__d2:[null,null,0],y__d1:[null,2,4],y__d2:[null,null,2]};
 for(let index=0;index<2;index++){
  const figure=buildFigure({columns},settings,'time',index),domains=[figure.layout.yaxis.domain,figure.layout.yaxis2.domain,figure.layout.yaxis3.domain];
  assert.equal(figure.data.length,3);assert.deepEqual(figure.data.map(trace=>trace.y),index?[columns.x,columns.x__d1,columns.x__d2]:[columns.y,columns.y__d1,columns.y__d2]);
  assert.equal(domains[0][1],1);assert.ok(Math.abs(domains[2][0])<1e-12);assert.ok(domains[0][0]>domains[1][1]);assert.ok(domains[1][0]>domains[2][1]);
  assert.equal(figure.layout.xaxis.showticklabels,false);assert.equal(figure.layout.xaxis3.showticklabels,true);assert.equal(figure.layout.xaxis3.matches,'x');
  assert.equal(figure.layout.xaxis3.rangeslider.visible,Boolean(index));assert.equal(figure.data[0].mode,index?'lines+markers':'lines');
 }
 assert.throws(()=>chartRequest(settings,'time',[],2),/Derivative panel 3/);assert.throws(()=>buildFigure({columns},settings,'time',2),/Derivative panel 3/);
});

test('categorical Scatter colors and symbols survive filtering and independent panel ordering',()=>{
 const panel=scatterPanel({color:'station',marker:'station'}),settings=scatterSettings([panel,{...panel,title:'Other panel'}]);
 const full=buildFigure({columns:{x:[1,2,3],y:[3,4,5],station:['A','B','C']}},settings,'x',0);
 const filtered=buildFigure({columns:{x:[3,2],y:[5,4],station:['C','B']}},settings,'x',1);
 assert.deepEqual(filtered.data[0].marker.color,[full.data[0].marker.color[2],full.data[0].marker.color[1]]);
 assert.deepEqual(filtered.data[0].marker.symbol,[full.data[0].marker.symbol[2],full.data[0].marker.symbol[1]]);
});

test('Distribution signal colors and bin scopes are independent of selected order and panel index',()=>{
 const result={columns:{bin_center:[.5],bin_left:[0],bin_right:[1],x:[2],y:[3]}},settings=distributionSettings([
  distributionPanel({columns:['x','y'],binScope:'Full dataset'}),distributionPanel({columns:['y','x'],binScope:'Filtered data'}),
 ]);
 const a=buildFigure(result,settings,'time',0),b=buildFigure(result,settings,'time',1);
 assert.equal(a.data[0].marker.color,b.data[1].marker.color);assert.equal(a.data[1].marker.color,b.data[0].marker.color);
 assert.equal(chartRequest(settings,'time',[],0).bins.scope,'dataset');assert.equal(chartRequest(settings,'time',[],1).bins.scope,'filtered');
 const scatter=scatterSettings([scatterPanel({color:'x',marker:'y',colorBinMode:'Number of bins',markerBinMode:'Number of bins',colorBinScope:'Full dataset',markerBinScope:'Filtered data'})]);
 const request=chartRequest(scatter,'time',[]);assert.equal(request.colorBinning.scope,'dataset');assert.equal(request.markerBinning.scope,'filtered');
});

test('example signals remain distinguishable with stable source colors',()=>{
 const names=['temperature','humidity','power_kw'];
 const settings={plot_type:'Trends',trend_panels:[{...defaultTrendPanel(),columns:names}]};
 const figure=buildFigure({columns:{x:[0,1],...Object.fromEntries(names.map(name=>[name,[1,2]]))}},settings,'x');
 assert.equal(new Set(figure.data.map(trace=>trace.line.color)).size,3);
});

test('full source category domains avoid collisions and retain encodings in filtered cards',()=>{
 const domain={color:['A','B','C','D'],marker:[10,20,30,40]};
 const full={columns:{x:[0,1,2,3],y:[4,5,6,7],station:['D','A','C','B'],level:[40,10,30,20]},meta:{categoryDomains:domain}};
 const subset={columns:{x:[3,0],y:[7,4],station:['B','D'],level:[20,40]},meta:{categoryDomains:domain}};
 const panels=[scatterPanel({color:'station',marker:'level'}),scatterPanel({color:'station',marker:'level'})];
 const before=structuredClone({full,subset}),a=buildFigure(full,scatterSettings(panels),'x',0),b=buildFigure(subset,scatterSettings(panels),'x',1);
 assert.equal(new Set(a.data[0].marker.color).size,4);assert.equal(new Set(a.data[0].marker.symbol).size,4);assert.deepEqual(a.warnings,[]);
 assert.deepEqual(b.data[0].marker.color,[a.data[0].marker.color[3],a.data[0].marker.color[0]]);
 assert.deepEqual(b.data[0].marker.symbol,[a.data[0].marker.symbol[3],a.data[0].marker.symbol[0]]);
 assert.deepEqual(a.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>trace.name),['D','A','C','B']);
 assert.deepEqual({full,subset},before);
});

test('bin ordinals reserve empty intervals and distinguish every bin within palette capacity',()=>{
 const labels=['[0, 1)','[1, 2)','[2, 3)','[3, 4]'],groups={color:{source:'x',column:'color_bin',labels},marker:{source:'x',column:'marker_bin',labels}};
 const full={columns:{x:[.5,1.5,2.5,3.5],y:[1,2,3,4],color_bin:labels,marker_bin:labels},meta:{groups}};
 const settings=scatterSettings([scatterPanel({color:'x',marker:'x'})]),a=buildFigure(full,settings,'x');
 assert.equal(new Set(a.data[0].marker.color).size,4);assert.equal(new Set(a.data[0].marker.symbol).size,4);
 const sparse=buildFigure({columns:{x:[2.5,3.5],y:[3,4],color_bin:labels.slice(2),marker_bin:labels.slice(2)},meta:{groups}},settings,'x');
 assert.deepEqual(sparse.data[0].marker.color,a.data[0].marker.color.slice(2));assert.deepEqual(sparse.data[0].marker.symbol,a.data[0].marker.symbol.slice(2));
 assert.deepEqual(sparse.data.filter(trace=>trace.meta?.encoding==='color').map(trace=>trace.name),labels.slice(2));
});
