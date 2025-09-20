// src/engine/worker.js
export class WorkerClient {
  constructor() {
    // Resolve worker URL relative to this file so it works on static servers
    const url = new URL('../worker/elm-worker.js', import.meta.url);
    // Classic worker (uses importScripts inside elm-worker.js)
    this.w = new Worker(url, { type: 'classic' });

    this.seq = 0;
    this.pending = new Map();

    this.w.onmessage = (e) => {
      const { id, ok, result, error } = e.data || {};
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      ok ? p.resolve(result) : p.reject(new Error(error || 'Worker error'));
    };
  }

  /**
   * Post a message to the worker. If the worker complains about an unknown action,
   * we retry with common aliases (helps when the worker is an older/newer build).
   */
  call(action, payload) {
    const normalized = String(action || '').trim();

    // Fallback maps: if the worker returns "Unknown action", try these in order.
    const FALLBACKS = {
      train: ['fit', 'update', 'trainFromData'],
      predictLogits: ['predictlogits', 'predict', 'predictProbaFromVector'],
      initELM: ['init'],
      initOnlineELM: ['init'],
      init: [] // base
    };


    const tryOnce = (act) => new Promise((resolve, reject) => {
      const id = ++this.seq;
      this.pending.set(id, { resolve, reject, _act: act });
      this.w.postMessage({ id, action: act, payload });
    });

    const tryWithFallbacks = async (primary, alts) => {
      try {
        return await tryOnce(primary);
      } catch (err) {
        const msg = (err && err.message) ? err.message : String(err);
        // Only fall back if the worker explicitly says it doesn't know this action
        const isUnknown = /Unknown action/i.test(msg);
        if (!isUnknown || !alts || alts.length === 0) throw err;
        for (const alt of alts) {
          try {
            return await tryOnce(alt);
          } catch (e2) {
            const m2 = (e2 && e2.message) ? e2.message : String(e2);
            if (!/Unknown action/i.test(m2)) throw e2; // real error, bubble up
          }
        }
        // If we exhaust all fallbacks and still get "Unknown action", bubble the last one
        throw new Error(`Worker does not support action "${primary}" or its aliases [${alts.join(', ')}]`);
      }
    };

    const alts = FALLBACKS[normalized] || [];
    return tryWithFallbacks(normalized, alts);
  }

  terminate() {
    try { this.w.terminate(); } catch { }
  }
}
