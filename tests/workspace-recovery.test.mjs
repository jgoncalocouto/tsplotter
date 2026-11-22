import test from 'node:test';
import assert from 'node:assert/strict';
import {saveWorkspace,loadWorkspace,clearWorkspace,configurationSignature,WORKSPACE_KEY} from '../src/workspace-recovery.js';
import {defaultSettings,encodeConfig} from '../src/config.js';
const storage=()=>{const entries=new Map();return {getItem:key=>entries.get(key)??null,setItem:(key,value)=>entries.set(key,value),removeItem:key=>entries.delete(key)};};

test('recovery stores only portable preferences and restores the exported baseline independently',()=>{
 const store=storage(),settings=defaultSettings({columns:[{name:'signal',kind:'numeric'}]});
 const saved=encodeConfig(settings);settings.distribution_panels[0].binMode='Fixed width';settings.distribution_panels[0].binWidth=.25;
 saveWorkspace(store,{...settings,dataset:{private:'secret'},fileName:'private.csv',fileHandle:{path:'C:/private'}},saved);
 const record=loadWorkspace(store);
 assert.equal(record.settings.distribution_panels[0].binWidth,.25);
 assert.equal(record.savedConfig,saved);
 assert.ok(!store.getItem(WORKSPACE_KEY).includes('private'));
 assert.notEqual(configurationSignature(record.config),configurationSignature(record.savedConfig));
 clearWorkspace(store);assert.equal(loadWorkspace(store),null);
});

test('invalid unfinished settings never overwrite the last recoverable configuration',()=>{
 const store=storage(),settings=defaultSettings();saveWorkspace(store,settings);
 const before=store.getItem(WORKSPACE_KEY);
 settings.distribution_panels[0].binMode='Fixed width';settings.distribution_panels[0].binWidth=0;
 assert.throws(()=>saveWorkspace(store,settings),/binWidth/);
 assert.equal(store.getItem(WORKSPACE_KEY),before);
});

test('recovery also sanitizes the saved baseline and externally modified local records',()=>{
 const store=storage(),foreign=JSON.stringify({app:'tsplotter',version:3,settings:{plot_type:'Trends',dataset:{secret:'private rows'},fileName:'private.csv'}});
 saveWorkspace(store,defaultSettings(),foreign);
 assert.ok(!store.getItem(WORKSPACE_KEY).includes('private'));
 store.setItem(WORKSPACE_KEY,JSON.stringify({config:foreign,savedConfig:foreign}));
 const recovered=loadWorkspace(store);
 assert.ok(!recovered.config.includes('private'));assert.ok(!recovered.savedConfig.includes('private'));
});

test('signatures ignore object field order and storage failures remain catchable by the UI',()=>{
 assert.equal(configurationSignature({plot_type:'Scatter',setup_index:'x'}),configurationSignature({setup_index:'x',plot_type:'Scatter'}));
 assert.throws(()=>saveWorkspace({setItem(){throw new Error('Quota exceeded');}},defaultSettings()),/Quota/);
 const store=storage();store.setItem(WORKSPACE_KEY,'{invalid');assert.throws(()=>loadWorkspace(store));
 store.setItem(WORKSPACE_KEY,JSON.stringify({config:'{}',savedConfig:null}));assert.throws(()=>loadWorkspace(store),/TSPlotter/);
});
