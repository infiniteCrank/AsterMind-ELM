/* global window, document, fetch */

const { EncoderELM, LanguageClassifier } = window.astermind;

// ----------- small util: try load, else run provided trainer & save -----------
async function tryLoadElseTrain(model, key, trainer) {
    try {
        const res = await fetch(`/models/${key}.json`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Fetch failed for ${key}.json`);
        const json = await res.text();
        if (json.trim().startsWith('<!DOCTYPE')) throw new Error(`Received HTML instead of JSON`);
        model.loadModelFromJSON(json);
        console.log(`✅ Loaded ${key} from /models/${key}.json`);
        return true;
    } catch (e) {
        console.warn(`⚠️ Could not load trained model for ${key}. Will train from scratch. Reason: ${e.message}`);
        await trainer();
        const json = model.elm?.savedModelJSON || model.savedModelJSON;
        if (json) {
            model.saveModelAsJSONFile(`${key}.json`);
            console.log(`📦 Model saved locally as ${key}.json — please deploy to /models/ manually.`);
        }
        return false;
    }
}

// ----------- CSV streaming (incremental; low memory) -----------
async function streamCSV(url, onRow, { hasHeader = true } = {}) {
    const res = await fetch(url);
    if (!res.body) {
        // Fallback: not a stream (older/blocked env). Still process line-by-line to keep memory bounded.
        const text = await res.text();
        const lines = text.split(/\r?\n/);
        let started = !hasHeader;
        for (const line of lines) {
            const ln = line.trim();
            if (!ln) continue;
            if (!started) { started = true; continue; } // skip header
            const rec = parseTwoColCSV(ln);
            if (rec) onRow(rec);
            await microYield();
        }
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let { value, done } = await reader.read();
    let buffer = '';
    let started = !hasHeader;

    while (!done) {
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;
            if (!started) { started = true; continue; } // skip header
            const rec = parseTwoColCSV(line);
            if (rec) onRow(rec);
        }
        await microYield();
        ({ value, done } = await reader.read());
    }
    // flush remainder
    if (buffer.trim()) {
        if (started) {
            const rec = parseTwoColCSV(buffer.trim());
            if (rec) onRow(rec);
        }
    }
}

function parseTwoColCSV(line) {
    // Minimal parser for "label,text" (text may contain commas and quotes)
    // Strategy: split on the first comma; strip outer quotes.
    const firstComma = line.indexOf(',');
    if (firstComma < 0) return null;
    let rawLabel = line.slice(0, firstComma).trim().replace(/^"|"$/g, '');
    let text = line.slice(firstComma + 1).trim();
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
        text = text.slice(1, -1);
    }
    // normalize common numeric label variants (AG News variants sometimes use 1..4 or 0..3)
    if (/^\d+$/.test(rawLabel)) {
        const n = parseInt(rawLabel, 10);
        // map 1..4 -> 0..3 -> categories index
        const idx = (n >= 1 && n <= 4) ? (n - 1) : n;
        rawLabel = categories[idx] ?? rawLabel;
    }
    return { label: String(rawLabel), text: text.toLowerCase() };
}

function microYield() {
    return new Promise(r => queueMicrotask(r));
}

// ----------- Demo config -----------
const charSet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,.;:\'"!?()-';
const maxLen = 50;
const categories = ['World', 'Sports', 'Business', 'Sci/Tech'];
const BATCH = 256;

const baseConfig = (hiddenUnits, exportFileName) => ({
    charSet,
    maxLen,
    hiddenUnits,
    activation: 'relu',
    useTokenizer: true,
    tokenizerDelimiter: /\s+/,
    exportFileName,
    categories,
    log: {
        verbose: true,
        modelName: exportFileName
    },
    metrics: { accuracy: 0.80 }
});

// ----------- Main -----------
window.addEventListener('DOMContentLoaded', async () => {
    const input = document.getElementById('headlineInput');
    const output = document.getElementById('predictionOutput');
    const fill = document.getElementById('confidenceFill');

    const encoder = new EncoderELM(baseConfig(64, 'agnews_encoder.json'));
    const classifier = new LanguageClassifier(baseConfig(128, 'agnews_classifier.json'));

    // ---------- 1) Train/load ENCODER online (identity targets: y == x) ----------
    await tryLoadElseTrain(encoder, 'agnews_encoder', async () => {
        // Determine encoder dims via a probe
        const probe = 'probe';
        const probeVec = encoder.elm.encoder.normalize(encoder.elm.encoder.encode(probe));
        const inputDim = probeVec.length;

        encoder.beginOnline({
            outputDim: inputDim,   // identity mapping
            inputDim,
            hiddenUnits: 64,
            lambda: 1e-2,
            activation: 'relu'
        });

        let encBatch = []; // { x:number[], y:number[] }[]
        await streamCSV('/ag-news-classification-dataset/train.csv', ({ text }) => {
            // Encode with the tokenizer/featurizer (not the ELM model)
            const x = encoder.elm.encoder.normalize(encoder.elm.encoder.encode(text));
            encBatch.push({ x, y: x }); // identity target
            if (encBatch.length >= BATCH) {
                encoder.partialTrainOnlineVectors(encBatch);
                encBatch = [];
            }
        });
        if (encBatch.length) encoder.partialTrainOnlineVectors(encBatch);

        // Publish W,b,beta so encoder.encode() uses the trained mapping
        encoder.endOnline();
    });

    // Determine encoded vector size (classifier input dim)
    const encodedProbe = encoder.encode('probe');
    const inputDimForClassifier = encodedProbe.length;

    // ---------- 2) Train/load CLASSIFIER online on encoded vectors ----------
    await tryLoadElseTrain(classifier, 'agnews_classifier', async () => {
        classifier.beginOnline({
            categories,
            inputDim: inputDimForClassifier,
            hiddenUnits: 128,
            lambda: 1e-2,
            activation: 'relu'
        });

        let clsBatch = []; // { vector:number[], label:string }[]
        await streamCSV('/ag-news-classification-dataset/train.csv', ({ label, text }) => {
            const v = encoder.encode(text);                         // now uses trained encoder
            clsBatch.push({ vector: v, label });
            if (clsBatch.length >= BATCH) {
                classifier.partialTrainVectorsOnline(clsBatch);
                clsBatch = [];
            }
        });
        if (clsBatch.length) classifier.partialTrainVectorsOnline(clsBatch);

        classifier.endOnline();
    });

    // ---------- 3) Inference UI (unchanged) ----------
    input.addEventListener('input', () => {
        const val = input.value.trim().toLowerCase();
        if (!val) {
            output.textContent = '';
            fill.style.width = '0%';
            fill.textContent = '';
            fill.style.background = '#ccc';
            return;
        }

        const encoded = encoder.encode(val);
        const [result] = classifier.predictFromVector(encoded);

        const percent = Math.round(result.prob * 100);
        output.textContent = `🔍 Predicted: ${result.label}`;
        fill.style.width = `${percent}%`;
        fill.textContent = `${result.label} (${percent}%)`;
        fill.style.background = {
            World: 'linear-gradient(to right, teal, cyan)',
            Sports: 'linear-gradient(to right, green, lime)',
            Business: 'linear-gradient(to right, goldenrod, yellow)',
            'Sci/Tech': 'linear-gradient(to right, purple, magenta)'
        }[result.label] || '#999';
    });
});
