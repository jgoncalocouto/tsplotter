import {decodeConfig,encodeConfig,MAX_CONFIG_BYTES} from './config.js';

export const WORKSPACE_KEY='tsplotter.workspace.v3';
const sorted=value=>Array.isArray(value)?value.map(sorted):value&&typeof value==='object'
 ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,sorted(value[key])])):value;

export function configurationSignature(config){
 const settings=typeof config==='string'?decodeConfig(config).settings:decodeConfig(encodeConfig(config)).settings;
 return JSON.stringify(sorted(settings));
}

/** Store only the same allowlisted settings accepted by portable configuration files. */
export function saveWorkspace(storage,settings,savedConfig=null){
 const config=encodeConfig(settings);
 const baseline=savedConfig===null?null:encodeConfig(decodeConfig(savedConfig).settings);
 storage.setItem(WORKSPACE_KEY,JSON.stringify({config,savedConfig:baseline}));
 return config;
}

export function loadWorkspace(storage){
 const text=storage.getItem(WORKSPACE_KEY);
 if(text===null)return null;
 if(text.length>MAX_CONFIG_BYTES*5)throw new Error('Stored workspace is too large.');
 const record=JSON.parse(text);
 if(!record||typeof record.config!=='string'||!(record.savedConfig===null||typeof record.savedConfig==='string'))throw new Error('Stored workspace is invalid.');
 const decoded=decodeConfig(record.config);
 const baseline=record.savedConfig===null?null:encodeConfig(decodeConfig(record.savedConfig).settings);
 return {...decoded,config:encodeConfig(decoded.settings),savedConfig:baseline};
}

export function clearWorkspace(storage){storage.removeItem(WORKSPACE_KEY);}
