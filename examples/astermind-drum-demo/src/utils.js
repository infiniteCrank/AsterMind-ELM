import { TOK } from './drums/tokens.js';

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export function softmax(logits, temp = 1) {
  let m = -Infinity; for (const v of logits) if (v > m) m = v;
  const ex = logits.map(v => Math.exp((v - m) / temp)); const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map(v => v / s);
}
export function sampleTopK(probs, k) {
  const arr = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
  const sum = arr.reduce((s, [p]) => s + p, 0) || 1;
  let r = Math.random() * sum;
  for (const [p, i] of arr) { r -= p; if (r <= 0) return i; }
  return arr[arr.length - 1][1];
}

// --- src/utils.js ---
// (keep your other exports; replace/add the ones below)

export function oneHot(idx, size) {
  const C = size | 0;
  const j = Math.max(0, Math.min(C - 1, idx | 0));
  const row = new Array(C).fill(0);
  row[j] = 1;
  return row; // plain Array<number>, fixed width
}

export function toOneHot(y, numClasses) {
  const C = numClasses | 0;
  const Y = new Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const j = Math.max(0, Math.min(C - 1, y[i] | 0));
    const row = new Array(C).fill(0);
    row[j] = 1;
    Y[i] = row; // plain Array<number>, fixed width
  }
  return Y;
}

/** Coerce any row-like (Float32Array, TypedArray, etc.) to plain Array<number> of length C */
export function toRowOfWidth(rowLike, C) {
  const out = new Array(C);
  for (let i = 0; i < C; i++) {
    const v = rowLike && Number.isFinite(rowLike[i]) ? Number(rowLike[i]) : 0;
    out[i] = v;
  }
  return out;
}

/** Ensure a rectangular matrix: returns Array<Array<number>> with each row length = C */
export function ensureRectMatrix(M, C) {
  const out = new Array(M.length);
  for (let r = 0; r < M.length; r++) {
    const row = M[r];
    // If it's already a plain array of exactly C, shallow-copy & coerce numbers
    if (Array.isArray(row) && row.length === C) {
      const rr = new Array(C);
      for (let i = 0; i < C; i++) rr[i] = Number.isFinite(row[i]) ? Number(row[i]) : 0;
      out[r] = rr;
    } else {
      // Coerce anything else (typed array, shorter/longer array) into width C
      out[r] = toRowOfWidth(row, C);
    }
  }
  return out;
}

/** Validate matrix shape; throws readable errors before hitting ELM internals */
export function validateRectMatrix(M, C, name = 'matrix') {
  for (let r = 0; r < M.length; r++) {
    const row = M[r];
    if (!row || row.length !== C) {
      throw new Error(`${name} row ${r} has length ${row && row.length}, expected ${C}`);
    }
    for (let c = 0; c < C; c++) {
      const v = row[c];
      if (!Number.isFinite(v)) throw new Error(`${name} row ${r}, col ${c} not finite: ${v}`);
    }
  }
}

export function buildInputWindow(tokens, pos, N, V) {
  const out = new Float32Array(N * V); let o = 0;
  for (let i = pos - N; i < pos; i++) {
    const t = i >= 0 ? tokens[i] : TOK.WAIT_1;
    out.set(oneHot(t, V), o); o += V;
  }
  return out;
}
export function makeXY(seqs, N, V) {
  const X = [], y = [];
  for (const seq of seqs) {
    for (let t = 1; t < seq.length; t++) {
      X.push(Array.from(buildInputWindow(seq, t, N, V)));
      y.push(seq[t]);
    }
  }
  return { X, y };
}
