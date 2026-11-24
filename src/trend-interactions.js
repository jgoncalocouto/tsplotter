// Plotly3.7's scattergl point-cluster queries can dominate interaction time.
// Index every finite observation by Plotly's own calculated X coordinate, then
// let its original hover code format the exact selected observation. Geometry,
// data arrays, native labels, spikes, events and exports remain Plotly-owned.
const bindings=new WeakMap(),wrappedModules=new WeakSet();
const coordinateCache=new WeakMap();
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const axisName=id=>id[0]+'axis'+id.slice(1);

export function createTrendHoverIndex(x,y,{logX=false,logY=false}={}){
 const rows=[];
 let last=-Infinity,ordered=true;
 for(let index=0;index<Math.min(x.length,y.length);index++){
  if(!finite(x[index])||!finite(y[index])||logX&&x[index]<=0||logY&&y[index]<=0)continue;
  if(x[index]<last)ordered=false;last=x[index];rows.push(index);
 }
 const indices=Uint32Array.from(rows);
 if(!ordered)indices.sort((a,b)=>x[a]-x[b]||a-b);
 return {x,y,indices};
}

function lowerBound(index,target){
 const {x,indices}=index;let low=0,high=indices.length;
 while(low<high){const middle=(low+high)>>>1;if(x[indices[middle]]<target)low=middle+1;else high=middle;}
 return low;
}

/** Exact nearest point; branch bounds use X distance, never observation caps. */
export function nearestTrendPoint(index,xval,yval,xa,ya,{mode='x',distance=32,svg=false,markers=false}={}){
 const {x,y,indices}=index,xpx=xa.c2p(xval),ypx=ya.c2p(yval);
 if(!indices.length||!finite(xpx)||!finite(ypx)||!(distance>=0))return undefined;
 let right=lowerBound(index,xval),left=right-1,best=distance,point;
 const slack=svg?3:0,minRadius=markers?3:.5;
 while(left>=0||right<indices.length){
  const leftDistance=left>=0?Math.abs(xa.c2p(x[indices[left]])-xpx):Infinity;
  const rightDistance=right<indices.length?Math.abs(xa.c2p(x[indices[right]])-xpx):Infinity;
  const dx=Math.min(leftDistance,rightDistance);
  if(dx>best+slack||!Number.isFinite(dx))break;
  const row=leftDistance<=rightDistance?indices[left--]:indices[right++];
  const dy=ya.c2p(y[row])-ypx;
  if(!finite(dy))continue;
  let d=mode==='x'?dx:Math.sqrt(dx*dx+dy*dy);
  if(svg)d=mode==='x'?(dx<3?2*dx/9:dx-3+2/3):Math.max(d-minRadius,0);
  if(d<best||(d===best&&(svg||point!==undefined)&&row<(point?.index??Infinity))){best=d;point={index:row,distance:d};}
  if(mode==='x'&&best===0)break;
 }
 return point;
}

function installNativeDelegate(module){
 if(!module?.hoverPoints||wrappedModules.has(module))return;
 const original=module.hoverPoints;
 module.hoverPoints=function(pointData,xval,yval,mode,...rest){
  const binding=bindings.get(pointData.cd);
  if(!binding||!['x','closest'].includes(mode))return original.call(this,pointData,xval,yval,mode,...rest);
  const manual=Number.isInteger(pointData.index)&&pointData.index>=0;
  const selected=manual?{index:pointData.index,distance:0}:nearestTrendPoint(binding.index,xval,yval,pointData.xa,pointData.ya,
   {mode,distance:pointData.distance,svg:!binding.gl,markers:pointData.trace.mode?.includes('markers')});
  if(binding.gl){
   const stash=pointData.cd[0].t,tree=stash.tree;
   binding.candidates.length=0;if(selected)binding.candidates.push(selected.index);
   stash.tree=binding.tree;
   try{
    return original.call(this,pointData,manual?binding.index.x[selected.index]:xval,manual?binding.index.y[selected.index]:yval,mode,...rest);
   }finally{stash.tree=tree;}
  }
  if(!selected){pointData.index=false;return [pointData];}
  pointData.index=selected.index;
  const result=original.call(this,pointData,xval,yval,mode,...rest);
  // SVG's explicit-index path deliberately sets distance=0. Restore the actual
  // search metric so closest-hover competition between traces stays correct.
  if(!manual)for(const point of result??[])point.distance=selected.distance;
  return result;
 };
 wrappedModules.add(module);
}

function coordinates(raw,trace,xa,cd){
 let cached=coordinateCache.get(raw);
 if(!cached){cached=new Map();coordinateCache.set(raw,cached);}
 const key=JSON.stringify([xa.type,xa.calendar,trace.xcalendar]);
 if(!cached.has(key)){
  // Reuse WebGL's exact coordinates, including its date/timezone semantics.
  // SVG may mask X at a Y gap, so calculate X independently of its Y values.
  cached.set(key,cd[0]?.t?.x??xa.makeCalcdata(trace,'x'));
 }
 return cached.get(key);
}

export function unbindTrendInteractions(graph){
 const current=graph?._trendInteractions;
 if(!current)return;
 for(const event of ['plotly_afterplot','plotly_relayout','plotly_restyle'])graph.removeListener?.(event,current.refresh);
 for(const cd of current.entries)bindings.delete(cd);
 delete graph._trendInteractions;delete graph.dataset.trendFastHover;
}

export function bindTrendInteractions(graph,{allowUnmarked=false}={}){
 unbindTrendInteractions(graph);
 const current={entries:[],signatures:[],refresh:null};
 current.refresh=()=>{
  const fullData=graph._fullData??[],calcdata=graph.calcdata??[],layout=graph._fullLayout;
  if(!layout)return;
  // Layout-only updates may replace module objects without recalculating data
  // or emitting afterplot. Keep delegation installed without rebuilding indexes.
  for(let index=0;index<fullData.length;index++){
   const trace=fullData[index],cd=calcdata[index];
   if(allowUnmarked||trace.meta?.trendSeries){installNativeDelegate(trace._module);installNativeDelegate(cd?.[0]?.trace?._module);}
  }
  const signatures=fullData.map((trace,index)=>[calcdata[index],graph.data?.[index]?.x,graph.data?.[index]?.y,trace.visible,trace.type,trace.xaxis,trace.yaxis,layout[axisName(trace.xaxis??'x')]?.type,layout[axisName(trace.yaxis??'y')]?.type]);
  if(signatures.length===current.signatures.length&&signatures.every((row,index)=>row.every((value,key)=>value===current.signatures[index]?.[key])))return;
  for(const cd of current.entries)bindings.delete(cd);current.entries=[];current.signatures=signatures;
  const reused=new Map();
  for(let index=0;index<fullData.length;index++){
   const trace=fullData[index],cd=calcdata[index],raw=graph.data?.[index];
   if(!raw||trace.visible!==true||(!allowUnmarked&&!trace.meta?.trendSeries)||!['scattergl','scatter'].includes(trace.type)||!cd?.length||trace.xperiod||trace.yperiod||trace.fill&&trace.fill!=='none')continue;
   const xa=layout[axisName(trace.xaxis??'x')],ya=layout[axisName(trace.yaxis??'y')];
   if(!xa||!ya||!raw.x||!raw.y)continue;
   let byY=reused.get(raw.x);if(!byY){byY=new Map();reused.set(raw.x,byY);}
   let indexByScale=byY.get(raw.y);if(!indexByScale){indexByScale=new Map();byY.set(raw.y,indexByScale);}
   const key=xa.type+':'+ya.type;
   if(!indexByScale.has(key)){
    const numericY=cd[0]?.t?.y??ya.makeCalcdata(trace,'y');
    indexByScale.set(key,createTrendHoverIndex(coordinates(raw.x,trace,xa,cd),numericY,{logX:xa.type==='log',logY:ya.type==='log'}));
   }
   const binding={index:indexByScale.get(key),gl:trace.type==='scattergl',candidates:[]};binding.tree={range:()=>binding.candidates};
   bindings.set(cd,binding);current.entries.push(cd);installNativeDelegate(trace._module);installNativeDelegate(cd[0]?.trace?._module);
  }
  graph.dataset.trendFastHover=String(current.entries.length);
 };
 graph._trendInteractions=current;
 for(const event of ['plotly_afterplot','plotly_relayout','plotly_restyle'])graph.on(event,current.refresh);
 current.refresh();
 return ()=>unbindTrendInteractions(graph);
}
