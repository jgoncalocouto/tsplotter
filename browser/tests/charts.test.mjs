import test from 'node:test';
import assert from 'node:assert/strict';
import {buildFigure,chartRequest} from '../src/charts.js';
import {defaultSettings} from '../src/config.js';
const schema={columns:[{name:'x',kind:'numeric'},{name:'y',kind:'numeric'}]};
test('trends retain every point beyond former sampling caps',()=>{const n=200001,s={...defaultSettings(schema),mlc_cols_0:['y']};const cols={x:Array.from({length:n},(_,i)=>i),y:Array.from({length:n},(_,i)=>i*2)};const fig=buildFigure({columns:cols,rowCount:n},s,'x');assert.equal(fig.data[0].x.length,n);assert.equal(fig.data[0].y.at(-1),400000);assert.equal(fig.data[0].line.simplify,false);});
test('scatter does not treat none labels as source columns or parse numeric x as timestamps',()=>{const s={...defaultSettings(schema),plot_type:'Scatter',setup_time_mode:'Absolute time'};const request=chartRequest(s,'x',[]);assert.equal(request.color,undefined);assert.equal(request.marker,undefined);assert.equal(request.timeAxis,undefined);});
test('relative axes preserve numeric values without implicit datetime parsing',()=>{const s=defaultSettings(schema);assert.equal(chartRequest(s,'x',[]).timeAxis,undefined);});
test('derivative settings map smoothing and explicit numeric spacing to engine',()=>{const s={...defaultSettings(schema),plot_type:'Derivatives',der_smooth:7,der_xmode:'use numeric index values',der_tunit:'seconds'};const request=chartRequest(s,'x',[]);assert.equal(request.smoothWindow,7);assert.equal(request.derivativeUnit,'seconds');});
test('scatter render groups retain all source rows',()=>{const s={...defaultSettings(schema),plot_type:'Grouped scatter',sc2_x:'x',sc2_y:'y',sc2_g1:'label',sc2_g2:'shape'};const result={columns:{x:[1,2,3],y:[4,5,6],label:['a','b','a'],shape:['one','two','one']},meta:{}};const fig=buildFigure(result,s,'x');assert.equal(fig.data.reduce((n,t)=>n+t.x.length,0),3);});
test('shared multi-panel charts use the available plotting area without overlapping panels',()=>{
 const columns={x:[0,1,2],y:[1,3,2]};
 for(let count=1;count<=6;count++){
  const settings={...defaultSettings(schema),plot_type:'Trends',mlc_n_subplots:count,mlc_rangeslider:true};
  for(let i=0;i<count;i++)settings[`mlc_cols_${i}`]=['y'];
  const figure=buildFigure({columns,rowCount:3},settings,'x');
  const domains=Array.from({length:count},(_,i)=>figure.layout[`yaxis${i?i+1:''}`].domain);
  assert.equal(domains[0][1],1);assert.ok(Math.abs(domains.at(-1)[0])<1e-12);
  for(let i=0;i<count;i++){
   const suffix=i?String(i+1):'';const xaxis=figure.layout[`xaxis${suffix}`];
   assert.ok(domains[i][0]>=0&&domains[i][1]<=1&&domains[i][0]<domains[i][1]);
   if(i)assert.ok(domains[i-1][0]>domains[i][1]);
   assert.equal(xaxis.anchor,`y${suffix}`);assert.equal(xaxis.showticklabels,i===count-1);
   if(i<count-1)assert.equal(xaxis.rangeslider,undefined);
  }
  assert.equal(figure.layout[`xaxis${count===1?'':count}`].rangeslider.visible,true);
  assert.equal(figure.data.length,count);
  for(const trace of figure.data){assert.deepEqual(trace.x,columns.x);assert.deepEqual(trace.y,columns.y);assert.equal(trace.line.simplify,false);}
 }
});
test('window and derivative spike controls change presentation without changing plotted series',()=>{
 const columns={x:[0,1,2],y:[1,3,2],y__roll_min:[1,1,2],y__roll_mean:[1,2,2.5],y__roll_max:[1,3,3],y__d1:[null,2,-1],y__d2:[null,null,-3]};
 for(const [mode,prefix]of[['Rolling window','rw'],['Derivatives','der']]){
  const settings={...defaultSettings(schema),plot_type:mode,[`${prefix}_y`]:'y',[`${prefix}_spikes`]:false};
  const noSpikes=buildFigure({columns,rowCount:3},settings,'x');
  const withSpikes=buildFigure({columns,rowCount:3},{...settings,[`${prefix}_spikes`]:true},'x');
  assert.equal(noSpikes.layout.xaxis.showspikes,false);assert.equal(withSpikes.layout.xaxis.showspikes,true);
  assert.deepEqual(noSpikes.data,withSpikes.data);
 }
});
