// elm-worker.js — classic worker (not type: 'module')
function resolve(url) { try { return new URL(url, self.location.href).toString(); } catch { return url; } }
try { importScripts(resolve('astermind.umd.js')); } catch { }
const lib = self.astermind || {};
const { ELM, EncoderELM, TFIDFVectorizer } = lib;

postMessage({ type: 'status', payload: `UMD: ${ELM ? 'ELM available' : 'no ELM'}; TFIDF: ${TFIDFVectorizer ? 'yes' : 'no'}` });

// ---------- Activations ----------
const Acts = {
    relu: x => Math.max(0, x),
    leakyRelu: (x, a = 0.01) => (x >= 0 ? x : a * x),
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    tanh: x => Math.tanh(x),
};
postMessage({ type: 'actCurve', payload: { fnName: 'relu', fnList: Object.keys(Acts) } });

// ---------- Tokenization / Vectorization ----------
function simpleTokens(text) {
    return (text || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/).filter(Boolean);
}

let basisFrozen = false;
let tfidf = null;
let vocab = null;        // Map token -> index
let idf = null;          // Map token -> weight
let featureNames = null; // Array indexed by feature id

function buildFrozenFallback(texts) {
    const df = new Map();
    const tokensPerDoc = texts.map(simpleTokens);
    const N = tokensPerDoc.length;
    const v = new Map(); let next = 0;
    for (const toks of tokensPerDoc) {
        const seen = new Set();
        for (const t of toks) {
            if (!v.has(t)) v.set(t, next++);
            if (!seen.has(t)) { seen.add(t); df.set(t, (df.get(t) || 0) + 1); }
        }
    }
    const id = new Map();
    for (const [t, d] of df.entries()) id.set(t, Math.log((N + 1) / (d + 1)) + 1);
    const names = [];
    for (const [t, j] of v.entries()) names[j] = t;
    vocab = v; idf = id; featureNames = names;
}

function vecFrozenFallback(text) {
    const toks = simpleTokens(text);
    const dim = vocab ? vocab.size : 0;
    const v = new Float32Array(dim);
    for (const t of toks) {
        const j = vocab.get(t);
        if (j != null) v[j] += (idf.get(t) || 1);
    }
    return { tokens: toks, vector: Array.from(v), featureNames };
}

function isolatedPreviewVector(text) {
    const toks = simpleTokens(text);
    const idx = new Map(); const names = [];
    for (const t of toks) if (!idx.has(t)) { idx.set(t, idx.size); names.push(t); }
    const v = new Float32Array(names.length);
    for (const t of toks) v[idx.get(t)] += 1;
    return { tokens: toks, vector: Array.from(v), featureNames: names, usedTFIDF: false };
}

// ---------- Hidden-layer preview for demo ----------
let lastHidden = { W: null, b: null }; // W: hidden x inputDim
function initHidden({ inputDim = 512, hidden = 32 }) {
    const W = Array.from({ length: hidden }, () =>
        Array.from({ length: inputDim }, () => (Math.random() * 2 - 1) * 0.5)
    );
    const b = Array.from({ length: hidden }, () => (Math.random() * 2 - 1) * 0.1);
    lastHidden.W = W; lastHidden.b = b;
    postMessage({ type: 'hidden_init', payload: { W, b } });
}
function projectHidden({ x }) {
    if (!lastHidden.W) initHidden({ inputDim: x.length, hidden: 32 });
    const { W, b } = lastHidden;
    const H = new Array(W.length);
    const Z = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
        const row = W[i];
        let s = b[i];
        for (let j = 0; j < row.length; j++) s += row[j] * x[j];
        Z[i] = s; H[i] = Acts.relu(s);
    }
    postMessage({ type: 'hidden_project', payload: { Hx: H, Z, activation: 'relu' } });
}

// ---------- Minimal math helpers (fallback) ----------
const transpose = A => {
    const R = A.length, C = A[0].length;
    const T = Array.from({ length: C }, () => Array(R));
    for (let j = 0; j < C; j++) for (let i = 0; i < R; i++) T[j][i] = A[i][j];
    return T;
};
const matmul = (A, B) => {
    const R = A.length, C = A[0].length, C2 = B[0].length;
    const out = Array.from({ length: R }, () => Array(C2).fill(0));
    for (let i = 0; i < R; i++) {
        for (let k = 0; k < C; k++) {
            const aik = A[i][k], Bk = B[k];
            for (let j = 0; j < C2; j++) out[i][j] += aik * Bk[j];
        }
    }
    return out;
};
function addRidgeI(A, lambda) {
    const n = A.length; const out = A.map(r => r.slice());
    for (let i = 0; i < n; i++) out[i][i] += lambda;
    return out;
}
// Gauss–Jordan for small systems
function solveLinear(A, B) {
    const n = A.length, m = B[0].length;
    const M = Array.from({ length: n }, (_, i) => A[i].concat(B[i]));
    for (let col = 0; col < n; col++) {
        let piv = col, max = Math.abs(M[col][col]);
        for (let r = col + 1; r < n; r++) { const v = Math.abs(M[r][col]); if (v > max) { max = v; piv = r; } }
        if (max < 1e-12) continue;
        if (piv !== col) { const tmp = M[piv]; M[piv] = M[col]; M[col] = tmp; }
        const p = M[col][col];
        for (let j = col; j < n + m; j++) M[col][j] /= p;
        for (let r = 0; r < n; r++) {
            if (r === col) continue;
            const f = M[r][col]; if (Math.abs(f) < 1e-12) continue;
            for (let j = col; j < n + m; j++) M[r][j] -= f * M[col][j];
        }
    }
    return Array.from({ length: n }, (_, i) => M[i].slice(n));
}
function ridgeSolveBeta(H, Y, lambda = 1e-2) {
    const Ht = transpose(H);
    const A = addRidgeI(matmul(Ht, H), lambda);
    const B = matmul(Ht, Y);
    return solveLinear(A, B); // returns h x k
}

// ---------- β visual downsampler ----------
const MAXB = 64;
function downsampleBeta(B, maxR = MAXB, maxC = MAXB) {
    if (!B || !B.length || !B[0].length) return null;
    const R = B.length, C = B[0].length;
    const dr = Math.max(1, Math.ceil(R / maxR));
    const dc = Math.max(1, Math.ceil(C / maxC));
    const out = [];
    for (let i = 0; i < R; i += dr) {
        const row = [];
        for (let j = 0; j < C; j += dc) row.push(B[i][j]);
        out.push(row);
    }
    return out;
}

// ---------- Model state ----------
let model = null;     // UMD ELM instance or fallback
let usingFallback = false;
let labelSpace = [];  // e.g., [1,2,3,4]

// ---------- Messages ----------
self.onmessage = async (e) => {
    const { type, payload } = e.data || {};

    if (type === 'hello') postMessage({ type: 'status', payload: 'ready' });
    if (type === 'list_activations') postMessage({ type: 'actCurve', payload: { fnName: 'relu', fnList: Object.keys(Acts) } });

    // Encode preview / frozen
    if (type === 'encode') {
        const { text, method, corpus } = payload;

        // quick isolated one-hot view (per-doc)
        if (method === 'isolated') {
            const result = isolatedPreviewVector(text);
            postMessage({ type: 'encoded', payload: { ...result, methodUsed: 'isolated' } });
            return;
        }

        // If we've already frozen a basis (after training), reuse it.
        if (basisFrozen) {
            if (method === 'bow') {
                try {
                    if (vocab && idf) {
                        const { tokens, vector, featureNames: names } = vecFrozenFallback(text);
                        postMessage({ type: 'encoded', payload: { tokens, vector, usedTFIDF: false, featureNames: names, methodUsed: 'bow' } });
                    } else {
                        const tmp = isolatedPreviewVector(text);
                        postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'bow' } });
                    }
                } catch {
                    const tmp = isolatedPreviewVector(text);
                    postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'bow' } });
                }
                return;
            }

            try {
                if (tfidf) {
                    const vec = tfidf.transform([text])[0];
                    const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                    const names = featureNames || (typeof tfidf.getFeatureNames === 'function' ? tfidf.getFeatureNames() : null);
                    postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true, featureNames: names, methodUsed: 'tfidf' } });
                } else {
                    const { tokens, vector, featureNames: names } = vecFrozenFallback(text);
                    postMessage({ type: 'encoded', payload: { tokens, vector, usedTFIDF: false, featureNames: names, methodUsed: 'tfidf' } });
                }
            } catch {
                const tmp = isolatedPreviewVector(text);
                postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'tfidf' } });
            }
            return;
        }

        // Basis NOT frozen yet — produce a meaningful preview.
        if (method === 'bow') {
            try {
                const toks = simpleTokens(text);
                const idx = new Map(); const names = [];
                for (const t of toks) if (!idx.has(t)) { idx.set(t, idx.size); names.push(t); }
                const v = new Float32Array(names.length);
                for (const t of toks) v[idx.get(t)] += 1;
                postMessage({ type: 'encoded', payload: { tokens: toks, vector: Array.from(v), usedTFIDF: false, featureNames: names, methodUsed: 'bow' } });
            } catch {
                const tmp = isolatedPreviewVector(text);
                postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'bow' } });
            }
            return;
        }

        // TF-IDF preview with a corpus (so IDF is informative)
        try {
            if (TFIDFVectorizer && Array.isArray(corpus) && corpus.length > 0) {
                const v = new TFIDFVectorizer();
                v.fit(corpus);
                const vec = v.transform([text])[0];
                const names = (typeof v.getFeatureNames === 'function') ? v.getFeatureNames() : null;
                const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true, featureNames: names, methodUsed: 'tfidf' } });
            } else if (TFIDFVectorizer) {
                // fallback: single-doc fit (will look flat, but still valid)
                const v = new TFIDFVectorizer();
                v.fit([text]);
                const vec = v.transform([text])[0];
                const names = (typeof v.getFeatureNames === 'function') ? v.getFeatureNames() : null;
                const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true, featureNames: names, methodUsed: 'tfidf' } });
            } else {
                // no TF-IDF library available → simple counts
                const tmp = isolatedPreviewVector(text);
                postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'tfidf' } });
            }
        } catch {
            const tmp = isolatedPreviewVector(text);
            postMessage({ type: 'encoded', payload: { ...tmp, methodUsed: 'tfidf' } });
        }
    }

    if (type === 'init_hidden') initHidden(payload);
    if (type === 'project_hidden') projectHidden(payload);

    if (type === 'train') {
        const { rows, hidden = 32 } = payload || {};
        try {
            const texts = rows.map(r => r.text);
            // freeze basis
            basisFrozen = true;
            let X = [];
            let note = '';
            if (TFIDFVectorizer) {
                try {
                    tfidf = new TFIDFVectorizer();
                    tfidf.fit(texts);
                    X = tfidf.transform(texts);
                    featureNames = (typeof tfidf.getFeatureNames === 'function') ? tfidf.getFeatureNames() : null;
                    vocab = null; idf = null;
                    note = 'TF-IDF basis';
                } catch {
                    tfidf = null;
                }
            }
            if (!tfidf) {
                buildFrozenFallback(texts);
                X = texts.map(t => vecFrozenFallback(t).vector);
                note = 'fallback vectorizer (frozen basis)';
            }

            labelSpace = Array.from(new Set(rows.map(r => r.y))).sort((a, b) => a - b);
            const Y = rows.map(r => {
                const k = labelSpace.length, v = Array(k).fill(0);
                const idx = labelSpace.indexOf(r.y); if (idx >= 0) v[idx] = 1;
                return v;
            });

            // UMD path
            usingFallback = false;
            if (ELM) {
                try {
                    model = new ELM({
                        inputSize: X[0].length,
                        hiddenSize: hidden,
                        outputSize: labelSpace.length,
                        activation: 'relu',
                        lambda: 1e-2
                    });
                    model.train(X, Y);

                    if (Array.isArray(model.hiddenW) && Array.isArray(model.hiddenB)) {
                        lastHidden.W = model.hiddenW; lastHidden.b = model.hiddenB;
                    }
                    const dims = {
                        H_rows: model?.lastH?.length || rows.length,
                        H_cols: model?.lastH?.[0]?.length || hidden,
                        Y_rows: Y.length, Y_cols: Y[0].length,
                        B_rows: model?.beta?.length || hidden,
                        B_cols: model?.beta?.[0]?.length || labelSpace.length
                    };
                    const betaSample = (model.beta || []).slice(0, 8)
                        .map(r => r.slice(0, 8).map(v => (Math.abs(v) < 1e-3 ? '0.000' : (+v).toFixed(3))).join(' '))
                        .join('\n');
                    const betaVis = model.beta ? downsampleBeta(model.beta) : null;

                    postMessage({ type: 'trained', payload: { dims, betaSample, note, labels: labelSpace, betaVis } });
                    return;
                } catch {
                    usingFallback = true;
                }
            } else {
                usingFallback = true;
            }

            // Fallback ELM
            const inputDim = X[0].length;
            const W = Array.from({ length: hidden }, () =>
                Array.from({ length: inputDim }, () => (Math.random() * 2 - 1) * 0.5)
            );
            const b = Array.from({ length: hidden }, () => (Math.random() * 2 - 1) * 0.1);
            lastHidden.W = W; lastHidden.b = b;

            // H: n x hidden
            const H = Array.from({ length: X.length }, () => Array(hidden));
            for (let n = 0; n < X.length; n++) {
                const x = X[n];
                for (let i = 0; i < hidden; i++) {
                    let s = b[i], Wi = W[i];
                    for (let j = 0; j < inputDim; j++) s += Wi[j] * x[j];
                    H[n][i] = Acts.relu(s);
                }
            }
            const beta = ridgeSolveBeta(H, Y, 1e-2); // h x k

            model = { kind: 'fallback', W, b, beta, activation: 'relu' };
            const dims = { H_rows: H.length, H_cols: H[0].length, Y_rows: Y.length, Y_cols: Y[0].length, B_rows: beta.length, B_cols: beta[0].length };
            const betaSample = beta.slice(0, 8)
                .map(r => r.slice(0, 8).map(v => (Math.abs(v) < 1e-3 ? '0.000' : (+v).toFixed(3))).join(' '))
                .join('\n');
            const betaVis = downsampleBeta(beta);

            postMessage({
                type: 'trained',
                payload: { dims, betaSample, note: note + (usingFallback ? '; fallback ELM' : ''), labels: labelSpace, betaVis }
            });
        } catch (err) {
            postMessage({
                type: 'trained',
                payload: {
                    dims: { H_rows: 0, H_cols: 0, Y_rows: 0, Y_cols: 0, B_rows: 0, B_cols: 0 },
                    betaSample: String(err), note: 'train error', labels: labelSpace, betaVis: null
                }
            });
        }
    }

    if (type === 'predict') {
        const { text } = payload || {};
        if (!model) { postMessage({ type: 'predicted', payload: { pred: null, scores: [] } }); return; }
        try {
            let v = [];
            if (tfidf) v = tfidf.transform([text])[0];
            else if (vocab) v = vecFrozenFallback(text).vector;
            else v = isolatedPreviewVector(text).vector;

            let logits = [];
            if (model.kind === 'fallback') {
                const hidden = model.W.length, inputDim = v.length;
                const Hx = new Array(hidden);
                for (let i = 0; i < hidden; i++) {
                    let s = model.b[i], Wi = model.W[i];
                    for (let j = 0; j < inputDim; j++) s += Wi[j] * v[j];
                    Hx[i] = Acts.relu(s);
                }
                const k = model.beta[0].length;
                logits = new Array(k).fill(0);
                for (let i = 0; i < hidden; i++) {
                    const hi = Hx[i], row = model.beta[i];
                    for (let j = 0; j < k; j++) logits[j] += hi * row[j];
                }
            } else {
                logits = model.predict([v])[0];
            }

            let best = -Infinity, bi = -1;
            for (let i = 0; i < logits.length; i++) if (logits[i] > best) { best = logits[i]; bi = i; }
            const pred = labelSpace[bi] ?? null;

            postMessage({ type: 'predicted', payload: { pred, scores: logits, labels: labelSpace, predIndex: bi } });
        } catch (err) {
            postMessage({ type: 'predicted', payload: { pred: null, scores: [], err: String(err) } });
        }
    }

    if (type === 'export_model') {
        const basis = {
            kind: tfidf ? 'tfidf' : (vocab ? 'fallback' : 'none'),
            featureNames: featureNames || null,
            vocab: vocab ? Array.from(vocab.entries()) : null,
            idf: idf ? Array.from(idf.entries()) : null
        };
        const payloadOut = {
            usingFallback,
            labels: labelSpace,
            model: model && model.kind === 'fallback'
                ? { kind: 'fallback', W: model.W, b: model.b, beta: model.beta, activation: model.activation }
                : (model ? { kind: 'umd', hiddenW: model.hiddenW, hiddenB: model.hiddenB, beta: model.beta } : null),
            lastHidden,
            basis
        };
        // Let the main thread handle download UI (if any)
        postMessage({ type: 'exported_model', payload: payloadOut });
    }

    if (type === 'reset') {
        basisFrozen = false;
        tfidf = null; vocab = null; idf = null; featureNames = null;
        model = null; usingFallback = false; labelSpace = [];
        lastHidden = { W: null, b: null };
        postMessage({ type: 'status', payload: 'reset complete' });
    }
};
