import {defineConfig} from 'vite';
export default defineConfig({worker:{format:'es'},build:{chunkSizeWarningLimit:6000},server:{host:'127.0.0.1'}});
