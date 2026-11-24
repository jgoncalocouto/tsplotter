/** Build chart CSV without retaining both an all-row string array and its joined copy. */
export function csvBlob(result){
 const keys=Object.keys(result.columns),parts=[];
 const quote=value=>{
  const text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);
  return /[",\r\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text;
 };
 let chunk=keys.map(quote).join(',');
 for(let row=0;row<result.rowCount;row++){
  chunk+='\r\n'+keys.map(key=>quote(result.columns[key][row])).join(',');
  if(chunk.length>=256*1024){parts.push(chunk);chunk='';}
 }
 if(chunk)parts.push(chunk);
 return new Blob(parts,{type:'text/csv;charset=utf-8'});
}
