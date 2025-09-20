/* global self */
// Classic worker (not module). Requires /astermind.umd.js to be served at site root.

self.onmessage = async (e) => {
  const { id, action, payload } = e.data || {};
  const respond = (ok, result, error) => self.postMessage({ id, ok, result, error });

  // Normalize/alias the action to prevent subtle mismatches
  const act = String(action || '').trim().toLowerCase();
  // accepted aliases that should behave like "train"
  const isTrain =
    act === 'train' ||
    act === 'fit' ||
    act === 'update' ||
    act === 'trainfromdata';

  try {
    if (act === 'init') {
      // Load the UMD bundle into the worker global
      importScripts('/astermind.umd.js');

      const am = self.astermind || {};
      const { engine, config } = payload || {};

      if (engine === 'elm') {
        if (!am.ELM) throw new Error('ELM class not found on UMD.');
        self.model = new am.ELM(config);
      } else if (engine === 'kernel') {
        if (!am.KernelELM) throw new Error('KernelELM class not found on UMD.');
        self.model = new am.KernelELM(config);
      } else if (engine === 'online') {
        if (!am.OnlineELM) throw new Error('OnlineELM class not found on UMD.');
        self.model = new am.OnlineELM(config);
      } else {
        throw new Error('Unknown engine: ' + engine);
      }
      respond(true, true);

    } else if (isTrain) {
      const { engine, X, y, task } = payload || {};
      if (!self.model) throw new Error('Model not initialized');

      if (engine === 'online') {
        if (!self._initialized) { self.model.init(X, y); self._initialized = true; }
        else self.model.update(X, y);
      } else if (engine === 'kernel') {
        // KernelELM API uses fit(X, Y)
        self.model.fit(X, y);
      } else {
        // Plain ELM: try trainFromData then fallback to train
        if (typeof self.model.trainFromData === 'function') {
          self.model.trainFromData(X, y, { task });
        } else if (typeof self.model.train === 'function') {
          self.model.train(X, y);
        } else {
          throw new Error('No train method on model');
        }
      }
      respond(true, true);

    } else if (act === 'predictlogits') {
      const { x } = payload || {};
      if (!self.model) throw new Error('Model not initialized');

      let logits = null;
      if (typeof self.model.predictLogitsFromVector === 'function') {
        logits = self.model.predictLogitsFromVector(x);
      } else if (typeof self.model.predictLogits === 'function') {
        logits = self.model.predictLogits(x);
      } else if (typeof self.model.predictProbaFromVector === 'function') {
        const p = self.model.predictProbaFromVector(x);
        logits = p.map(v => Math.log((v || 1e-12)));
      } else if (typeof self.model.predictFromVector === 'function') {
        const r = self.model.predictFromVector([x])[0];
        if (Array.isArray(r)) logits = r;
        else if (r && Array.isArray(r.probs)) logits = r.probs.map(v => Math.log((v || 1e-12)));
      } else {
        throw new Error('No vector-safe predict on model');
      }
      respond(true, logits);

    } else {
      // Helpful diagnostics if something else slips through
      respond(false, null, `Unknown action: ${action}. Expected one of: init, train, fit, update, trainFromData, predictLogits`);
    }
  } catch (err) {
    respond(false, null, err && err.message ? err.message : String(err));
  }
};
