/* Classic (non-module) Web Worker for AsterMind ELM demos.
 * Place this file next to index.html and ensure /astermind.umd.js is served from your site root.
 */
self.onmessage = async (e) => {
  const { id, action, payload } = e.data || {};
  const respond = (ok, result, error) => self.postMessage({ id, ok, result, error });

  const act = String(action || "").trim().toLowerCase();
  const isTrain = (act === "train" || act === "fit" || act === "update" || act === "trainfromdata");

  try {
    if (act === "init") {
      // Load the UMD bundle into the worker global
      importScripts("/astermind.umd.js");
      const am = self.astermind || self.Astermind || self.AstermindELM || {};
      const { engine, config } = payload || {};

      if (engine === "elm") {
        if (!am.ELM) throw new Error("ELM class not found on UMD.");
        self.model = new am.ELM(config);
      } else if (engine === "kernel") {
        if (!am.KernelELM) throw new Error("KernelELM class not found on UMD.");
        self.model = new am.KernelELM(config);
      } else if (engine === "online") {
        if (!am.OnlineELM) throw new Error("OnlineELM class not found on UMD.");
        self.model = new am.OnlineELM(config);
      } else {
        throw new Error("Unknown engine: " + engine);
      }
      respond(true, true);
      return;
    }

    if (isTrain) {
      if (!self.model) throw new Error("Model not initialized");
      const { engine, X, y, task } = payload || {};

      if (engine === "online") {
        if (typeof self._initialized === "undefined" || !self._initialized) {
          self.model.init(X, y);
          self._initialized = true;
        } else {
          // Support both 'update' and repeated 'train' calls
          if (typeof self.model.update === "function") self.model.update(X, y);
          else if (typeof self.model.train === "function") self.model.train(X, y);
          else throw new Error("No update/train on OnlineELM");
        }
      } else if (engine === "kernel") {
        if (typeof self.model.fit === "function") self.model.fit(X, y);
        else if (typeof self.model.train === "function") self.model.train(X, y);
        else throw new Error("No fit/train on KernelELM");
      } else {
        // Plain ELM: try labels, then one-hot inferred from y rows
        if (typeof self.model.trainFromData === "function") {
          try { self.model.trainFromData(X, y, { task: task || "classification" }); }
          catch { self.model.trainFromData(X, y, { task: task || "classification" }); }
        } else if (typeof self.model.train === "function") {
          self.model.train(X, y);
        } else {
          throw new Error("No train method on model");
        }
      }
      respond(true, true);
      return;
    }

    if (act === "predictlogits") {
      if (!self.model) throw new Error("Model not initialized");
      const { x } = payload || {};
      let logits = null;
      if (typeof self.model.predictLogitsFromVector === "function") logits = self.model.predictLogitsFromVector(x);
      else if (typeof self.model.predictLogits === "function") logits = self.model.predictLogits(x);
      else if (typeof self.model.predictProbaFromVector === "function") {
        const p = self.model.predictProbaFromVector(x);
        logits = p.map(v => Math.log((v || 1e-12)));
      } else if (typeof self.model.predictFromVector === "function") {
        const r = self.model.predictFromVector([x])[0];
        if (Array.isArray(r)) logits = r;
        else if (r && Array.isArray(r.probs)) logits = r.probs.map(v => Math.log((v || 1e-12)));
      } else {
        throw new Error("No vector-safe predict on model");
      }
      respond(true, logits);
      return;
    }

    if (act === "tojson") {
      if (!self.model) throw new Error("Model not initialized");
      if (typeof self.model.toJSON === "function") {
        respond(true, self.model.toJSON());
      } else {
        respond(false, null, "Model has no toJSON()");
      }
      return;
    }

    if (act === "dispose") {
      self.model = null; self._initialized = false;
      respond(true, true); return;
    }

    respond(false, null, "Unknown action: " + action);
  } catch (err) {
    respond(false, null, err && err.message ? err.message : String(err));
  }
};
