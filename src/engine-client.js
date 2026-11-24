/** UI facade. File/Blob data is cloned directly into a browser Worker, never uploaded. */
export class DataEngine {
  constructor(options = {}) {
    this.options = {...options};
    delete this.options.onProgress;
    if (this.options.assetBase && typeof document !== 'undefined') this.options.assetBase = new URL(this.options.assetBase, document.baseURI).href;
    this.nextId = 1;
    this.pending = new Map();
    this.onProgress = options.onProgress || (() => {});
    this.worker = undefined;
  }
  _start() {
    if (this.worker) return;
    this.worker = new Worker(new URL('./data-worker.js', import.meta.url), {type: 'module'});
    this.worker.onmessage = ({data}) => {
      if (data.type === 'progress') { this.onProgress(data); return; }
      const pending = this.pending.get(data.id);
      if (!pending) return;
      this.pending.delete(data.id);
      if (data.error) { const error = new Error(data.error.message); error.name = data.error.name; pending.reject(error); }
      else pending.resolve(data.result);
    };
    this.worker.onerror = event => {
      this._rejectAll(new Error(event.message || 'The local analysis worker stopped. Try reimporting the file.'));
      this.worker?.terminate(); this.worker = undefined;
    };
  }
  _rejectAll(error) { for (const request of this.pending.values()) request.reject(error); this.pending.clear(); }
  _call(method, args) {
    this._start();
    const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, {resolve, reject}); this.worker.postMessage({id, method, args}); });
  }
  init() { return this._call('init', [this.options]); }
  importFile(file, options = {}) { return this._call('importFile', [file, {...this.options, ...options}]); }
  importRows(rows, options = {}) { return this._call('importRows', [rows, {...this.options, ...options}]); }
  query(spec) { return this._call('query', [spec]); }
  summary(filters = [], options = {}) { return this._call('summary', [filters, options]); }
  exportCSV(filters = [], options = {}) { return this._call('exportCSV', [filters, options]); }
  cancel() {
    // Terminating the parent worker also terminates its dedicated DuckDB child worker.
    this.worker?.terminate(); this.worker = undefined;
    this._rejectAll(new DOMException('Local analysis cancelled. Reimport your dataset to continue.', 'AbortError'));
  }
  async close() {
    if (!this.worker) return;
    try { await this._call('close', []); } finally { this.cancel(); }
  }
}
export default DataEngine;
