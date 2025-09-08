// elm-worker.js
// ---------- path + script ----------
function resolve(url) {
    try { return new URL(url, self.location.href).toString(); } catch { return url; }
}
importScripts(resolve('astermind.umd.js'));

const lib = self.astermind || {};
const { ELM, EncoderELM, TFIDFVectorizer } = lib;
postMessage({ type: 'status', payload: `UMD ${lib ? 'loaded' : 'missing'}` });

const LocalActs = {
    relu: x => Math.max(0, x),
    leakyRelu: (x, a = 0.01) => x >= 0 ? x : a * x,
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    tanh: x => Math.tanh(x),
};
postMessage({ type: 'actCurve', payload: { fnName: 'relu', fnList: Object.keys(LocalActs) } });

/* ---------- Vectorization (TF-IDF if available; fallback BOW/IDF) ---------- */
let tfidf = null;
let vocab = new Map();
let idf = new Map();
let basisFrozen = false;            // becomes true after 'train'
let currentFeatureNames = null;     // names for the frozen basis (after train)

function featureNamesFromTFIDF(v = tfidf) {
    try {
        if (!v) return null;
        if (typeof v.getFeatureNames === 'function') return v.getFeatureNames();
        if (Array.isArray(v.featureNames)) return v.featureNames;
        if (v.vocabulary && typeof v.vocabulary === 'object') {
            const arr = [];
            for (const [tok, idx] of Object.entries(v.vocabulary)) arr[idx] = tok;
            return arr;
        }
        if (v.vocab && typeof v.vocab === 'object') {
            const arr = [];
            for (const [tok, idx] of Object.entries(v.vocab)) arr[idx] = tok;
            return arr;
        }
    } catch { }
    return null;
}

// Isolated, per-text fallback preview (fresh vocab just for this one vector)
function isolatedFallbackVector(text) {
    const toks = simpleTokens(text);
    const idx = new Map(); const names = [];
    for (const t of toks) { if (!idx.has(t)) { idx.set(t, idx.size); names.push(t); } }
    const v = new Float32Array(names.length);
    for (const t of toks) { v[idx.get(t)] += 1; } // simple counts (ok for preview)
    return { tokens: toks, vector: Array.from(v), featureNames: names };
}

function featureNamesFromTFIDF() {
    try {
        if (!tfidf) return null;
        if (typeof tfidf.getFeatureNames === 'function') return tfidf.getFeatureNames();
        if (Array.isArray(tfidf.featureNames)) return tfidf.featureNames;
        if (tfidf.vocabulary && typeof tfidf.vocabulary === 'object') {
            const arr = [];
            for (const [tok, idx] of Object.entries(tfidf.vocabulary)) arr[idx] = tok;
            return arr;
        }
        if (tfidf.vocab && typeof tfidf.vocab === 'object') {
            const arr = [];
            for (const [tok, idx] of Object.entries(tfidf.vocab)) arr[idx] = tok;
            return arr;
        }
    } catch { }
    return null;
}

function featureNamesFromFallbackVocab() {
    const arr = [];
    for (const [tok, idx] of vocab.entries()) { arr[idx] = tok; }
    return arr;
}

function simpleTokens(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
}
function ensureTFIDF() {
    if (tfidf || !TFIDFVectorizer) return;
    try { tfidf = new TFIDFVectorizer(); } catch { }
}
function fitFallbackTFIDF(texts) {
    const df = new Map();
    const allTokens = texts.map(simpleTokens);
    const N = allTokens.length;
    let idx = 0;
    for (const toks of allTokens) {
        const seen = new Set();
        for (const t of toks) {
            if (!vocab.has(t)) vocab.set(t, idx++);
            if (!seen.has(t)) { seen.add(t); df.set(t, (df.get(t) || 0) + 1); }
        }
    }
    for (const [t, d] of df.entries()) {
        idf.set(t, Math.log((N + 1) / (d + 1)) + 1);
    }
}
function vecFallback(text) {
    const toks = simpleTokens(text);
    const v = new Float32Array(vocab.size);
    for (const t of toks) {
        const j = vocab.get(t);
        if (j != null) { v[j] += (idf.get(t) || 1); }
    }
    return { tokens: toks, vector: Array.from(v) };
}

/* ---------- Hidden layer for visualization ---------- */
let lastHidden = { W: null, b: null };
function initHidden({ inputDim = 512, hidden = 32 }) {
    const W = new Array(hidden).fill(0).map(() => new Array(inputDim).fill(0).map(() => (Math.random() * 2 - 1) * 0.5));
    const b = new Array(hidden).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);
    lastHidden.W = W; lastHidden.b = b;
    postMessage({ type: 'hidden_init', payload: { W, b } });
}
function projectHidden({ x }) {
    if (!lastHidden.W) { initHidden({ inputDim: x.length, hidden: 32 }); }
    const { W, b } = lastHidden;
    const Z = new Array(W.length);  // pre-activation: Wx+b
    const H = new Array(W.length);  // activation: g(Z)
    for (let i = 0; i < W.length; i++) {
        let s = b[i];
        const row = W[i];
        for (let j = 0; j < row.length; j++) s += row[j] * x[j];
        Z[i] = s;
        H[i] = Math.max(0, s); // ReLU for display
    }
    postMessage({ type: 'hidden_project', payload: { Hx: H, Z, activation: 'relu' } });
}


/* ---------- ELM training/prediction ---------- */
let model = null;
let labelSpace = [];
function oneHot(y, labels) {
    const k = labels.length;
    const vec = new Array(k).fill(0);
    const idx = labels.indexOf(y);
    if (idx >= 0) vec[idx] = 1;
    return vec;
}
function unHot(scores) {
    let best = -1, bi = -1;
    for (let i = 0; i < scores.length; i++) { if (scores[i] > best) { best = scores[i]; bi = i; } }
    return { idx: bi, best };
}

/* ---------- Message handling ---------- */
self.onmessage = async (e) => {
    const { type, payload } = e.data || {};
    if (type === 'hello') { postMessage({ type: 'status', payload: 'ready' }); }
    if (type === 'list_activations') {
        postMessage({ type: 'actCurve', payload: { fnName: 'relu', fnList: Object.keys(LocalActs) } });
    }

    if (type === 'encode') {
        const { text } = payload;

        // If we've trained, DO NOT change the basis; just transform
        if (basisFrozen) {
            try {
                if (tfidf) {
                    const vec = tfidf.transform([text])[0];
                    const feats = featureNamesFromTFIDF() || currentFeatureNames || null;
                    const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                    postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true, featureNames: feats } });
                } else {
                    // Use existing global fallback vocab (don’t mutate it)
                    const toks = simpleTokens(text);
                    const v = new Float32Array(vocab.size);
                    for (const t of toks) { const j = vocab.get(t); if (j != null) v[j] += (idf.get(t) || 1); }
                    postMessage({ type: 'encoded', payload: { tokens: toks, vector: Array.from(v), usedTFIDF: false, featureNames: featureNamesFromFallbackVocab() } });
                }
            } catch {
                // safe fallback to isolated preview
                const iso = isolatedFallbackVector(text);
                postMessage({ type: 'encoded', payload: { ...iso, usedTFIDF: false } });
            }
            return;
        }

        // PREVIEW mode (not trained yet): use an isolated basis per click
        try {
            if (TFIDFVectorizer) {
                const v = new TFIDFVectorizer(); // fresh vectorizer
                v.fit([text]);
                const vec = v.transform([text])[0];
                const feats = featureNamesFromTFIDF(v);
                const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true, featureNames: feats } });
            } else {
                const iso = isolatedFallbackVector(text);
                postMessage({ type: 'encoded', payload: { ...iso, usedTFIDF: false } });
            }
        } catch {
            const iso = isolatedFallbackVector(text);
            postMessage({ type: 'encoded', payload: { ...iso, usedTFIDF: false } });
        }
    }

    if (type === 'init_hidden') { initHidden(payload); }
    if (type === 'project_hidden') { projectHidden(payload); }

    if (type === 'train') {
        const { rows, hidden = 32 } = payload;
        if (!ELM) {
            postMessage({ type: 'trained', payload: { dims: { H_rows: 0, H_cols: 0, Y_rows: 0, Y_cols: 0, B_rows: 0, B_cols: 0 }, betaSample: '<ELM missing>', note: 'UMD not exposing ELM' } });
            return;
        }
        const texts = rows.map(r => r.text);
        ensureTFIDF();
        let usedTF = false;
        let X = [];
        try {
            basisFrozen = true; // lock feature space after training
            if (tfidf) {
                tfidf.fit(texts);
                X = tfidf.transform(texts);
                usedTF = true;
                currentFeatureNames = featureNamesFromTFIDF(); // keep frozen names
            } else {
                fitFallbackTFIDF(texts);                       // builds global vocab/idf
                X = texts.map(t => vecFallback(t).vector);
                currentFeatureNames = featureNamesFromFallbackVocab();
            }
        } catch {
            fitFallbackTFIDF(texts);
            X = texts.map(t => vecFallback(t).vector);
        }
        labelSpace = Array.from(new Set(rows.map(r => r.y))).sort((a, b) => a - b);
        const Y = rows.map(r => oneHot(r.y, labelSpace));

        try {
            model = new ELM({
                inputSize: X[0].length,
                hiddenSize: hidden,
                outputSize: labelSpace.length,
                activation: 'relu',
                lambda: 1e-2
            });
            model.train(X, Y);
            const dims = {
                H_rows: model?.lastH?.length || rows.length,
                H_cols: model?.lastH?.[0]?.length || hidden,
                Y_rows: Y.length,
                Y_cols: Y[0].length,
                B_rows: model?.beta?.length || hidden,
                B_cols: model?.beta?.[0]?.length || labelSpace.length
            };
            const betaSample = (model.beta || []).slice(0, 8).map(r => r.slice(0, 8).map(v => (Math.abs(v) < 1e-3 ? '0.000' : v.toFixed(3))).join(' ')).join('\n');
            postMessage({ type: 'trained', payload: { dims, betaSample, note: usedTF ? 'TF-IDF' : 'fallback vectorizer' } });
        } catch (err) {
            postMessage({ type: 'trained', payload: { dims: { H_rows: 0, H_cols: 0, Y_rows: 0, Y_cols: 0, B_rows: 0, B_cols: 0 }, betaSample: String(err), note: 'train error' } });
        }
    }

    if (type === 'predict') {
        const { text } = payload;
        if (!model) { postMessage({ type: 'predicted', payload: { pred: null, scores: [] } }); return; }
        let v = [];
        try {
            if (tfidf) { v = tfidf.transform([text])[0]; }
            else { v = vecFallback(text).vector; }
        } catch {
            v = vecFallback(text).vector;
        }
        try {
            const logits = model.predict([v])[0];
            const { idx } = unHot(logits);
            const pred = labelSpace[idx] ?? null;
            postMessage({ type: 'predicted', payload: { pred, scores: logits } });
        } catch (err) {
            postMessage({ type: 'predicted', payload: { pred: null, scores: [], err: String(err) } });
        }
    }
};
