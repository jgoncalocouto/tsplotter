import test from 'node:test';
import assert from 'node:assert/strict';
import {createTrendHoverIndex,nearestTrendPoint,bindTrendInteractions,unbindTrendInteractions} from '../src/trend-interactions.js';
const axis=(scale=1,offset=0)=>({type:'linear',c2p:value=>scale*value+offset});

test('sorted hover indexes retain finite source rows, gaps, duplicate X and original row identity',()=>{
 const x=[30,10,20,10,NaN,40,50],y=[3,1,null,2,4,Infinity,5],before=[...x];
 const index=createTrendHoverIndex(x,y);
 assert.deepEqual([...index.indices],[1,3,0,6]);assert.deepEqual(x,before);assert.equal(index.x,x);assert.equal(index.y,y);
 assert.equal(nearestTrendPoint(index,10,2,axis(),axis(),{mode:'closest'}).index,3);
 assert.equal(nearestTrendPoint(index,30,3,axis(),axis(),{mode:'x'}).index,0);
 assert.equal(nearestTrendPoint(index,100,0,axis(),axis(),{distance:1}),undefined);
});

test('nearest source observations agree with exhaustive pixel-distance search on reversed and logarithmic axes',()=>{
 const x=Array.from({length:2000},(_,i)=>1+(i*317)%2000),y=x.map((v,i)=>i%23===0?NaN:1+(i*13)%100);
 for(const log of [false,true])for(const reverse of [false,true]){
  const xa=axis(reverse?-.2:.2),ya=log?{type:'log',c2p:value=>Math.log10(value)*50}:axis(2);
  const index=createTrendHoverIndex(x,y,{logY:log});
  for(let trial=0;trial<30;trial++)for(const mode of ['x','closest']){
   const xx=35+trial*61,yy=2+trial*3,distance=32;
   let best=distance;
   for(let i=0;i<x.length;i++)if(Number.isFinite(y[i])){
    const dx=Math.abs(xa.c2p(x[i])-xa.c2p(xx)),dy=ya.c2p(y[i])-ya.c2p(yy);
    best=Math.min(best,mode==='x'?dx:Math.hypot(dx,dy));
   }
   const selected=nearestTrendPoint(index,xx,yy,xa,ya,{mode,distance});
   if(best===distance)assert.equal(selected,undefined);
   else{assert.ok(selected);assert.ok(Math.abs(selected.distance-best)<1e-10);assert.ok(Number.isFinite(y[selected.index]));}
  }
 }
});

test('a full 315034-observation time axis needs logarithmic lookup rather than a full hover scan',()=>{
 const count=315034,x=Float64Array.from({length:count},(_,i)=>i),y=new Float64Array(count).fill(1);
 const index=createTrendHoverIndex(x,y);assert.equal(index.indices.length,count);
 let calls=0;const xa={c2p:value=>{calls++;return value/300;}};
 const hit=nearestTrendPoint(index,173170,1,xa,axis(),{mode:'x',distance:32});
 assert.equal(hit.index,173170);assert.ok(calls<20,`Unexpected full-data hover scan: ${calls} coordinate conversions`);
});

function fixture(){
 const x=[1,2,3,4],y=[10,20,30,40],tree={range:()=>{tree.calls++;return [0,1,2,3];},calls:0};
 const module={hoverPoints(point){const rows=point.cd[0].t.tree.range();point.index=rows[0];return [point];}};
 const trace={type:'scattergl',visible:true,meta:{trendSeries:true},mode:'lines',xaxis:'x',yaxis:'y',_module:module};
 const cd=[{t:{x,y,tree},trace}],listeners=new Map();
 const graph={data:[{x,y}],_fullData:[trace],calcdata:[cd],dataset:{},_fullLayout:{xaxis:axis(),yaxis:axis()},
  on(event,fn){if(!listeners.has(event))listeners.set(event,new Set());listeners.get(event).add(fn);},
  removeListener(event,fn){listeners.get(event)?.delete(fn);},emit(event){for(const fn of listeners.get(event)??[])fn();}};
 return {graph,trace,cd,tree,module,listeners};
}

test('native hover delegation preserves original point indexes and restores Plotly search state',()=>{
 const f=fixture();bindTrendInteractions(f.graph);
 const data={cd:f.cd,trace:f.trace,xa:axis(),ya:axis(),distance:32};
 const result=f.module.hoverPoints(data,3,30,'x');
 assert.equal(result[0].index,2);assert.equal(f.tree.calls,0);assert.equal(f.cd[0].t.tree,f.tree);
 assert.equal(f.graph.dataset.trendFastHover,'1');
 unbindTrendInteractions(f.graph);
 assert.ok([...f.listeners.values()].every(set=>set.size===0));
 f.module.hoverPoints({...data},3,30,'x');assert.equal(f.tree.calls,1);
});

test('module replacements on layout-only updates stay accelerated without mutating raw source arrays',()=>{
 const f=fixture();bindTrendInteractions(f.graph);
 const replacement={hoverPoints:f.module.hoverPoints};
 f.trace._module=replacement;f.graph.emit('plotly_relayout');
 const point={cd:f.cd,trace:f.trace,xa:axis(),ya:axis(),distance:32};
 assert.equal(replacement.hoverPoints(point,4,40,'closest')[0].index,3);
 assert.deepEqual(f.graph.data[0].x,[1,2,3,4]);assert.deepEqual(f.graph.data[0].y,[10,20,30,40]);
 unbindTrendInteractions(f.graph);
});
