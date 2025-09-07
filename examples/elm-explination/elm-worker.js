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
    const H = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
        let s = b[i];
        const row = W[i];
        for (let j = 0; j < row.length; j++) s += row[j] * x[j];
        H[i] = Math.max(0, s); // ReLU for display
    }
    postMessage({ type: 'hidden_project', payload: { Hx: H } });
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
        ensureTFIDF();
        try {
            if (tfidf && typeof tfidf.fit === 'function' && typeof tfidf.transform === 'function') {
                tfidf.fit([text]);
                const vec = tfidf.transform([text])[0];
                const toks = (EncoderELM && EncoderELM.tokenize) ? EncoderELM.tokenize(text) : simpleTokens(text);
                postMessage({ type: 'encoded', payload: { tokens: toks, vector: vec, usedTFIDF: true } });
            } else {
                if (vocab.size === 0) fitFallbackTFIDF([text]);
                const { tokens, vector } = vecFallback(text);
                postMessage({ type: 'encoded', payload: { tokens, vector, usedTFIDF: false } });
            }
        } catch {
            if (vocab.size === 0) fitFallbackTFIDF([text]);
            const { tokens, vector } = vecFallback(text);
            postMessage({ type: 'encoded', payload: { tokens, vector, usedTFIDF: false } });
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
            if (tfidf) {
                tfidf.fit(texts);
                X = tfidf.transform(texts);
                usedTF = true;
            } else {
                fitFallbackTFIDF(texts);
                X = texts.map(t => vecFallback(t).vector);
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
