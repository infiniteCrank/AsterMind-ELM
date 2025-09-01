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

// ----------- CSV streaming (no libraries; incremental; low memory) -----------
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
    // Minimal parser for "label,text" where text may contain commas but is usually quoted.
    // Strategy: split on the first comma.
    // If your file has different quoting, swap this with a full parser (Papa, etc.).
    const firstComma = line.indexOf(',');
    if (firstComma < 0) return null;
    const label = line.slice(0, firstComma).trim().replace(/^"|"$/g, '');
    let text = line.slice(firstComma + 1).trim();
    // Strip surrounding quotes if present
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
        text = text.slice(1, -1);
    }
    return { label, text: text.toLowerCase() };
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

    // ---------- 1) Train/load ENCODER online (identity targets) ----------
    // We train EncoderELM to learn x -> x (identity). This lets us use its encode()
    // without holding the whole dataset (no big matrices).
    await tryLoadElseTrain(encoder, 'agnews_encoder', async () => {
        // Infer input dimension from a probe sample
        const probe = 'probe';
        encoder.beginOnline({ outputDim: undefined, sampleText: probe, hiddenUnits: 64, lambda: 1e-2, activation: 'relu' });
        // NOTE: beginOnline({ outputDim, sampleText }) in our integration infers inputDim from sample;
        // we want identity mapping: outputDim === inputDim.
        // If your beginOnline requires explicit outputDim, compute it here:
        // const inputDim = encoder.elm.encoder.normalize(encoder.elm.encoder.encode(probe)).length;
        // encoder.beginOnline({ outputDim: inputDim, inputDim, hiddenUnits: 64, lambda: 1e-2, activation: 'relu' });

        let batchTexts = [];
        let prepared = false;

        // If your EncoderELM.beginOnline doesn't auto-set outputDim=inputDim, uncomment the above lines and set prepared=true.
        // For portability across your code, re-probe once:
        if (!prepared) prepared = true;

        await streamCSV('/ag-news-classification-dataset/train.csv', ({ text }) => {
            batchTexts.push({ text, target: null }); // target filled inside partialTrainOnlineTexts
            if (batchTexts.length >= BATCH) {
                // In our EncoderELM integration, partialTrainOnlineTexts encodes text internally
                // and uses the encoded vector as target (identity). That keeps memory tiny.
                encoder.partialTrainOnlineTexts(
                    batchTexts.map(({ text }) => ({
                        text,
                        // Target = identity of encoded input (EncoderELM code computes this internally)
                        target: new Array(1).fill(0) // placeholder; implementation ignores this and builds T=X
                    }))
                );
                batchTexts = [];
            }
        });

        if (batchTexts.length) {
            encoder.partialTrainOnlineTexts(
                batchTexts.map(({ text }) => ({ text, target: new Array(1).fill(0) }))
            );
            batchTexts = [];
        }

        // Publish W,b,beta into the model so encoder.encode() works
        encoder.endOnline();
    });

    // Determine encoded vector size (classifier input dim)
    const probeVec = encoder.encode('probe');
    const inputDimForClassifier = probeVec.length;

    // ---------- 2) Train/load CLASSIFIER online on encoded vectors ----------
    await tryLoadElseTrain(classifier, 'agnews_classifier', async () => {
        classifier.beginOnline({
            categories,
            inputDim: inputDimForClassifier,
            hiddenUnits: 128,
            lambda: 1e-2,
            activation: 'relu'
        });

        let vecBatch = []; // { vector:number[], label:string }[]

        await streamCSV('/ag-news-classification-dataset/train.csv', ({ label, text }) => {
            // Encode using the already-trained encoder
            const v = encoder.encode(text);
            vecBatch.push({ vector: v, label });
            if (vecBatch.length >= BATCH) {
                classifier.partialTrainVectorsOnline(vecBatch);
                vecBatch = [];
            }
        });

        if (vecBatch.length) classifier.partialTrainVectorsOnline(vecBatch);

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
