const finite=value=>typeof value==='number'&&Number.isFinite(value);
const validX=value=>value!=null&&(typeof value!=='number'||Number.isFinite(value));
const unique=values=>[...new Set(values)];
const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'
 ?Object.fromEntries(Object.keys(value).sort().filter(key=>value[key]!==undefined).map(key=>[key,stable(value[key])])):value;
const scope=request=>JSON.stringify(stable({...request,y:[]}));

/** Share full source arrays between figures, with each figure's own exact ledger. */
export function projectTrendResult(source,request){
 const x=request.x||'__row__',signals=unique(request.y),columns={[x]:source.columns[x]};
 for(const name of signals){
  if(!Object.hasOwn(source.columns,name))throw new Error(`Missing Trend signal: ${name}`);
  columns[name]=source.columns[name];
 }
 const eligible=source.totalRows;
 const seriesCounts=Object.fromEntries(signals.map(name=>[name,source.meta.seriesCounts[name]]));
 let usable=0;
 const sourceSignals=Object.keys(source.meta.seriesCounts);
 if(sourceSignals.length===signals.length&&signals.every(name=>Object.hasOwn(source.meta.seriesCounts,name)))usable=source.meta.rows.usable;
 else if(signals.some(name=>seriesCounts[name].usable===eligible))usable=eligible;
 else if(signals.length===1)usable=seriesCounts[signals[0]].usable;
 else if(signals.some(name=>seriesCounts[name].usable)){
  const arrays=signals.map(name=>columns[name]),xx=columns[x];
  for(let row=0;row<source.rowCount;row++)if(validX(xx[row])&&arrays.some(values=>finite(values[row])))usable++;
 }
 return {...source,columns,meta:{...source.meta,seriesCounts,rows:{...source.meta.rows,usable,excludedValues:eligible-usable}}};
}

/** One bounded cache for the current dataset/filter/time scope, never a history. */
export class TrendDataCache{
 constructor(){this.generation=0;this.entry=null;}
 clear(){this.generation++;this.entry=null;}
 canReuse(requests){
  return Boolean(this.entry&&requests.length&&requests.every(request=>scope(request)===this.entry.key&&request.y.every(name=>Object.hasOwn(this.entry.result.columns,name)&&Object.hasOwn(this.entry.result.meta.seriesCounts,name))));
 }
 async query(requests,load){
  if(!requests.length)throw new Error('Choose a Trend instance.');
  const key=scope(requests[0]);
  if(requests.some(request=>request.mode!=='trends'||scope(request)!==key))throw new Error('Trend instances must use the same dataset, filters and X axis.');
  const union={...requests[0],y:unique(requests.flatMap(request=>request.y))};
  if(!this.canReuse(requests)){
   this.entry=null;
   const generation=this.generation,result=await load(union);
   if(generation!==this.generation)throw new DOMException('Analysis cancelled.','AbortError');
   this.entry={key,result};
  }
  // Trim removed selections instead of retaining every signal ever explored.
  this.entry.result=projectTrendResult(this.entry.result,union);
  return requests.map(request=>projectTrendResult(this.entry.result,request));
 }
}
