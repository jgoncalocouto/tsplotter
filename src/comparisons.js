import {trendSubplots} from './config.js';

const signalKey=columns=>JSON.stringify([...new Set(columns)].sort());

export function axisRangeNumbers(range,scale){
 return range.map(value=>{
  if(scale!=='date'||typeof value==='number')return Number(value);
  let date=String(value).replace(' ','T');
  if(!/(?:Z|[+-]\d{2}:?\d{2})$/i.test(date))date+='Z';
  return Date.parse(date);
 });
}

/** Axes link only when their variable, units/order, and numeric/time scale agree. */
export function comparisonAxes(mode,panel,xName,layout,sharedBins=false){
 const axes={},trendRows=mode==='Trends'?trendSubplots(panel):[];
 for(const name of Object.keys(layout).filter(key=>/^[xy]axis\d*$/.test(key))){
  const family=name[0],order=Number(name.replace(/^[xy]axis/,'' )||1)-1;
  let key;
  if(family==='x')key=mode==='Scatter'?`field:${panel.x}`:mode==='Distribution'?(sharedBins?'distribution:shared':`distribution:${signalKey(panel.columns)}`):`field:${xName}`;
  else if(mode==='Scatter')key=`field:${panel.y}`;
  else if(mode==='Distribution')key='distribution:count';
  else if(mode==='Trends'){
   const columns=trendRows[order]?.columns??[];
   if(!columns.length)continue; // A blank subplot has no vertical variable to compare.
   key=`signals:${signalKey(columns)}`;
  }
  else if(mode==='Rolling window')key=`field:${panel.y}`;
  else key=order===0?`field:${panel.y}`:`derivative:${panel.y}:${order}:${panel.timeUnit}:${panel.xMode}`;
  axes[name]={family,key,scale:layout[name].type||'linear'};
 }
 return axes;
}

export function linkedRangePatch(event,sourceAxes,sourceLayout,targetAxes,{x=false,y=false}={}){
 const patch={};
 const changed=new Set(Object.keys(event).map(key=>/^([xy]axis\d*)\.(?:range(?:\[[01]\])?|autorange)$/.exec(key)?.[1]).filter(Boolean));
 for(const name of changed){
  const source=sourceAxes[name];if(!source||!(source.family==='x'?x:y)||source.scale==='category'||source.scale==='multicategory')continue;
  const reset=event[`${name}.autorange`]===true;
  const range=event[`${name}.range`]??[event[`${name}.range[0]`]??sourceLayout[name]?.range?.[0],event[`${name}.range[1]`]??sourceLayout[name]?.range?.[1]];
  for(const [destination,target]of Object.entries(targetAxes)){
   if(source.family!==target.family||source.key!==target.key||source.scale!==target.scale)continue;
   if(reset)patch[`${destination}.autorange`]=true;
   else if(range?.length===2&&range.every(value=>value!==undefined)){patch[`${destination}.range`]=[...range];patch[`${destination}.autorange`]=false;}
  }
 }
 return patch;
}
