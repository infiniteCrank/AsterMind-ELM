/* eslint-disable no-undef */
/* global importScripts, fetch, TextDecoder */

// -------- path + fetch helpers --------
function resolve(url) {
    // Resolve relative to this worker's URL; if it fails, return as-is
    try { return new URL(url, self.location.href).toString(); } catch { return url; }
}

// Load your UMD bundle inside the worker (adjust path if needed)
importScripts(resolve('astermind.umd.js'));

// Guard: only accept valid JSON bodies (not HTML fallbacks)
async function fetchModelJSON(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return null;
    const text = await res.text();
    const ct = (res.headers.get('content-type') || '').toLowerCase();
    if (!ct.includes('application/json') || text.trim().startsWith('<')) return null;
    return text;
}

const { EncoderELM, LanguageClassifier } = self.astermind || {};

// -------- progress + CSV stream --------
function postProgress(phase, pct, status) {
    self.postMessage({ type: 'progress', phase, pct, status });
}

async function streamCSVWithProgress(url, onRow, { hasHeader = true, progressSpan = [0, 100], labelMap } = {}) {
    url = resolve(url);
    const res = await fetch(url, { cache: 'no-store' });
    const total = Number(res.headers.get('Content-Length')) || 0;
    const [p0, p1] = progressSpan;
    let loaded = 0;

    if (!res.body) {
        const text = await res.text();
        postProgress('parse', p1, 'Parsing file…');
        const lines = text.split(/\r?\n/);
        let started = !hasHeader;
        for (const line of lines) {
            const rec = parseTwoColCSV(line);
            if (!rec) continue;
            if (!started) { started = true; continue; }
            if (labelMap) rec.label = normalizeLabel(rec.label, labelMap);
            onRow(rec);
        }
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let { value, done } = await reader.read();
    let buffer = '';
    let started = !hasHeader;

    while (!done) {
        loaded += value.byteLength;
        const pct = total ? p0 + ((loaded / total) * (p1 - p0)) : p0;
        postProgress('download', Math.min(pct, p1 - 1), 'Streaming data…');

        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            if (!line.trim()) continue;
            if (!started) { started = true; continue; }
            const rec = parseTwoColCSV(line);
            if (!rec) continue;
            if (labelMap) rec.label = normalizeLabel(rec.label, labelMap);
            onRow(rec);
        }

        ({ value, done } = await reader.read());
    }

    const tail = decoder.decode(value || new Uint8Array(), { stream: false });
    if (tail) buffer += tail;
    if (buffer.trim()) {
        if (started) {
            const rec = parseTwoColCSV(buffer.trim());
            if (rec) {
                if (labelMap) rec.label = normalizeLabel(rec.label, labelMap);
                onRow(rec);
            }
        }
    }
    postProgress('download', p1, 'Data stream complete.');
}

function parseTwoColCSV(line) {
    const firstComma = line.indexOf(',');
    if (firstComma < 0) return null;
    const rawLabel = line.slice(0, firstComma).trim().replace(/^"|"$/g, '');
    let text = line.slice(firstComma + 1).trim();
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
        text = text.slice(1, -1);
    }
    return { label: rawLabel, text: text.toLowerCase() };
}

function normalizeLabel(rawLabel, categories) {
    if (/^\d+$/.test(rawLabel)) {
        const n = parseInt(rawLabel, 10);
        const idx = (n >= 1 && n <= 4) ? (n - 1) : n;
        return categories[idx] ?? rawLabel;
    }
    return rawLabel;
}

function softmax(arr) {
    const m = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - m));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / s);
}

// -------- worker state --------
let state = {
    encoder: null,
    classifier: null,
    categories: []
};

// -------- message handler --------
self.onmessage = async (e) => {
    const { type, payload, text } = e.data || {};
    try {
        if (type === 'init') {
            const {
                dataUrl,
                batch,
                categories,
                files,
                encoderHidden = 64,
                classifierHidden = 128,
                activation = 'relu'
            } = payload;

            state.categories = categories.slice();

            // ----- 1) Encoder: load or train (online identity y=x) -----
            postProgress('encoder', 0, 'Loading encoder…');
            state.encoder = new EncoderELM({
                charSet: 'abcdefghijklmnopqrstuvwxyz0123456789 ,.;:\'"!?()-',
                maxLen: 50,
                hiddenUnits: encoderHidden,
                activation,
                useTokenizer: true,
                tokenizerDelimiter: /\s+/,
                exportFileName: files.encoder,
                categories,
                log: { verbose: false, modelName: 'agnews_encoder' }
            });

            const encUrl = resolve(`models/${files.encoder}`);
            const encJson = await fetchModelJSON(encUrl);
            if (encJson) {
                state.encoder.loadModelFromJSON(encJson);
                postProgress('encoder', 45, 'Encoder loaded.');
            } else {
                postProgress('encoder', 5, 'Training encoder online…');
                const probe = 'probe';
                const probeVec = state.encoder.elm.encoder.normalize(state.encoder.elm.encoder.encode(probe));
                const inputDim = probeVec.length;

                state.encoder.beginOnline({
                    outputDim: inputDim,
                    inputDim,
                    hiddenUnits: encoderHidden,
                    lambda: 1e-2,
                    activation
                });

                let encBatch = [];
                await streamCSVWithProgress(
                    dataUrl,
                    ({ text }) => {
                        const x = state.encoder.elm.encoder.normalize(state.encoder.elm.encoder.encode(text));
                        encBatch.push({ x, y: x }); // identity
                        if (encBatch.length >= batch) {
                            state.encoder.partialTrainOnlineVectors(encBatch);
                            encBatch = [];
                        }
                    },
                    { hasHeader: true, progressSpan: [5, 45] }
                );
                if (encBatch.length) state.encoder.partialTrainOnlineVectors(encBatch);

                state.encoder.endOnline();
                postProgress('encoder', 45, 'Encoder trained.');
            }

            // ----- 2) Classifier: load or train (online on encoded vectors) -----
            postProgress('classifier', 45, 'Loading classifier…');
            state.classifier = new LanguageClassifier({
                charSet: 'abcdefghijklmnopqrstuvwxyz0123456789 ,.;:\'"!?()-',
                maxLen: 50,
                hiddenUnits: classifierHidden,
                activation,
                useTokenizer: true,
                tokenizerDelimiter: /\s+/,
                exportFileName: files.classifier,
                categories,
                log: { verbose: false, modelName: 'agnews_classifier' },
                metrics: { accuracy: 0.80 }
            });

            const clsUrl = resolve(`models/${files.classifier}`);
            const clsJson = await fetchModelJSON(clsUrl);
            if (clsJson) {
                state.classifier.loadModelFromJSON(clsJson);
                postProgress('classifier', 100, 'Classifier loaded.');
            } else {
                postProgress('classifier', 50, 'Training classifier online…');

                const encodedProbe = state.encoder.encode('probe');
                const inputDimForClassifier = encodedProbe.length;

                state.classifier.beginOnline({
                    categories,
                    inputDim: inputDimForClassifier,
                    hiddenUnits: classifierHidden,
                    lambda: 1e-2,
                    activation
                });

                let clsBatch = []; // { vector:number[], label:string }[]
                await streamCSVWithProgress(
                    dataUrl,
                    ({ label, text }) => {
                        const v = state.encoder.encode(text);
                        clsBatch.push({ vector: v, label });
                        if (clsBatch.length >= batch) {
                            state.classifier.partialTrainVectorsOnline(clsBatch);
                            clsBatch = [];
                        }
                    },
                    { hasHeader: true, progressSpan: [50, 100], labelMap: categories }
                );
                if (clsBatch.length) state.classifier.partialTrainVectorsOnline(clsBatch);

                state.classifier.endOnline();
                postProgress('classifier', 100, 'Classifier trained.');
            }

            self.postMessage({ type: 'ready' });
        }

        if (type === 'predict') {
            if (!state.encoder || !state.classifier) {
                self.postMessage({ type: 'error', error: 'Models not ready yet.' });
                return;
            }
            const vec = state.encoder.encode(String(text || '').toLowerCase());
            const res = state.classifier.predictFromVector(vec, 4);
            // If your classifier already returns calibrated probs, this is fine.
            // Otherwise normalize defensively:
            let best = res[0] || { label: 'Unknown', prob: 0 };
            if (!Number.isFinite(best.prob) || best.prob <= 0 || best.prob >= 1) {
                const probs = softmax(res.map(r => r.prob));
                let idx = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[idx]) idx = i;
                best = { label: state.categories[idx] || res[idx]?.label || 'Unknown', prob: probs[idx] };
            }
            self.postMessage({ type: 'prediction', label: best.label, prob: best.prob });
        }
    } catch (err) {
        self.postMessage({ type: 'error', error: String((err && err.message) || err) });
    }
};
