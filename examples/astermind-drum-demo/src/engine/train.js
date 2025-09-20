// src/engine/train.js
import { ID2STR } from '../drums/tokens.js';

/* ---------- helpers (plain arrays, fixed width) ---------- */
const clampInt = (x, lo, hi) => {
  const xi = x | 0;
  return xi < lo ? lo : (xi > hi ? hi : xi);
};

// Return Array<number> of EXACT width C (pad/truncate, coerce to finite)
function toFixedRow(rowLike, C) {
  const out = new Array(C);
  for (let i = 0; i < C; i++) {
    const v = rowLike && Number.isFinite(rowLike[i]) ? Number(rowLike[i]) : 0;
    out[i] = v;
  }
  return out;
}

// Return Array<Array<number>> with EACH row EXACT width C
function toFixedRect(M, C, name = 'matrix') {
  if (!Array.isArray(M) || M.length === 0) throw new Error(`${name} is empty`);
  const out = new Array(M.length);
  for (let r = 0; r < M.length; r++) out[r] = toFixedRow(M[r], C);
  // validate
  for (let r = 0; r < out.length; r++) {
    const row = out[r];
    if (!row || row.length !== C) throw new Error(`${name} row ${r} has length ${row && row.length}, expected ${C}`);
    for (let c = 0; c < C; c++) {
      const v = row[c];
      if (!Number.isFinite(v)) throw new Error(`${name} row ${r}, col ${c} not finite: ${v}`);
    }
  }
  return out;
}

// Build one-hot Y (Array<Array<number>>) of EXACT width V
function toOneHot(labels, V) {
  const n = labels.length | 0;
  const Y = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = clampInt(labels[i], 0, V - 1);
    const row = new Array(V).fill(0);
    row[j] = 1;
    Y[i] = row;
  }
  return Y;
}

function looksLikeOneHot(Y) {
  return Array.isArray(Y) && Array.isArray(Y[0]) && Number.isFinite(Y[0][0]);
}

/* ---------- main API ---------- */
export async function trainOnMain(engine, model, X, yOrY) {
  const V = ID2STR.length;                 // number of classes
  const D = (X && X[0]) ? (X[0].length | 0) : 0;
  if (D <= 0) throw new Error('trainOnMain: inputs X are empty or have zero width');

  // Rectangularize X
  const Xrect = toFixedRect(X, D, 'inputs X');

  // Accept labels or one-hot for targets; rectangularize to width V
  const Yrect = looksLikeOneHot(yOrY)
    ? toFixedRect(yOrY, V, 'targets Y')
    : toFixedRect(toOneHot(yOrY, V), V, 'targets Y');

  // Debug (remove if noisy)
  console.log('[train.js] -> X:', Xrect.length, 'x', Xrect[0].length, '| Y:', Yrect.length, 'x', Yrect[0].length);

  // ---- Engine-specific paths ----
  if (engine === 'kernel' && typeof model.fit === 'function') {
    model.fit(Xrect, Yrect);
    return;
  }

  if (engine === 'online') {
    // Prefer init once, then update if available
    if (!model._initialized && typeof model.init === 'function') {
      model.init(Xrect, Yrect);
      model._initialized = true;
      return;
    }
    if (typeof model.update === 'function') {
      model.update(Xrect, Yrect);
      return;
    }
    // Fallback to fit if that's the API
    if (typeof model.fit === 'function') {
      model.fit(Xrect, Yrect);
      return;
    }
    // Otherwise fall through to plain ELM handling
  }

  // Plain ELM (numeric). Prefer trainFromData; fallback to train.
  const tryAwait = async (r) => { if (r && typeof r.then === 'function') await r; };

  if (typeof model.trainFromData === 'function') {
    // Many builds accept labels OR one-hot; we pass one-hot to be explicit.
    await tryAwait(model.trainFromData(Xrect, Yrect, { task: 'classification' }));
    return;
  }

  if (typeof model.train === 'function') {
    // Some builds support numeric train(X, Y).
    await tryAwait(model.train(Xrect, Yrect));
    return;
  }

  throw new Error('Model has no train/trainFromData/fit/update API');
}
