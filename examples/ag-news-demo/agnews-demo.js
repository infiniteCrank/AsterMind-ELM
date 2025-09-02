/* global window, document, fetch, Worker */

const DATA_URL = '/ag-news-classification-dataset/train.csv';
const WORKER_URL = '/agnews-worker.js';   // <-- put your path here
const BATCH = 256;
const CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech'];

// Minimal UI overlay for training/progress
function injectProgressOverlay() {
    const style = document.createElement('style');
    style.textContent = `
    #trainerOverlay {
      position: fixed; inset: 0; background: rgba(10,10,12,0.75);
      display: flex; align-items: center; justify-content: center; z-index: 9999;
      color: #fff; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }
    .trainer-card {
      width: min(520px, 90vw); background: #111; border-radius: 14px; padding: 20px 22px;
      box-shadow: 0 10px 35px rgba(0,0,0,0.45);
    }
    .trainer-title { font-size: 18px; margin: 0 0 8px; opacity: 0.95; }
    .trainer-status { font-size: 14px; margin: 0 0 10px; opacity: 0.85; }
    .trainer-bar {
      height: 10px; background: #2a2a2a; border-radius: 999px; overflow: hidden;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }
    .trainer-fill {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, #00d4ff, #7a5cff);
      transition: width 120ms linear;
    }
  `;
    document.head.appendChild(style);

    const overlay = document.createElement('div');
    overlay.id = 'trainerOverlay';
    overlay.innerHTML = `
    <div class="trainer-card">
      <h3 class="trainer-title">Preparing models…</h3>
      <p class="trainer-status" id="trainerStatus">Loading…</p>
      <div class="trainer-bar"><div class="trainer-fill" id="trainerFill"></div></div>
    </div>
  `;
    document.body.appendChild(overlay);
}

function setProgress(pct, status) {
    const fill = document.getElementById('trainerFill');
    const statusEl = document.getElementById('trainerStatus');
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (statusEl && status) statusEl.textContent = status;
}

function hideOverlay() {
    const el = document.getElementById('trainerOverlay');
    if (el) el.remove();
}

// Debounce helper
function debounce(fn, delay = 180) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
    };
}

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('headlineInput');
    const output = document.getElementById('predictionOutput');
    const fill = document.getElementById('confidenceFill');

    // disable input while models prepare
    if (input) {
        input.disabled = true;
        input.placeholder = 'Training models… please wait';
    }

    injectProgressOverlay();

    // Spin up worker
    const worker = new Worker(WORKER_URL); // classic worker (uses importScripts)
    let modelsReady = false;

    worker.onmessage = (e) => {
        const msg = e.data || {};
        switch (msg.type) {
            case 'progress':
                // msg: { phase, pct, status }
                setProgress(msg.pct ?? 0, msg.status || '');
                break;

            case 'ready':
                modelsReady = true;
                // enable UI
                if (input) {
                    input.disabled = false;
                    input.placeholder = 'Type a headline…';
                }
                hideOverlay();
                break;

            case 'prediction':
                // msg: { label, prob }
                if (!output || !fill) return;
                const percent = Math.round((msg.prob ?? 0) * 100);
                output.textContent = `🔍 Predicted: ${msg.label}`;
                fill.style.width = `${percent}%`;
                fill.textContent = `${msg.label} (${percent}%)`;
                fill.style.background = ({
                    World: 'linear-gradient(to right, teal, cyan)',
                    Sports: 'linear-gradient(to right, green, lime)',
                    Business: 'linear-gradient(to right, goldenrod, yellow)',
                    'Sci/Tech': 'linear-gradient(to right, purple, magenta)'
                })[msg.label] || '#999';
                break;

            case 'model-json': {
                const { name, json } = msg;
                const blob = new Blob([json], { type: 'application/json' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = name;      // e.g., 'agnews_encoder.json'
                a.click();
                URL.revokeObjectURL(a.href);
                break;
            }

            case 'error':
                console.error('[Worker error]', msg.error);
                setProgress(0, `Error: ${msg.error}`);
                break;

            default:
                break;
        }
    };

    // Initialize worker (kick off load/train)
    worker.postMessage({
        type: 'init',
        payload: {
            dataUrl: DATA_URL,
            batch: BATCH,
            categories: CATEGORIES,
            files: {
                encoder: 'agnews_encoder.json',
                classifier: 'agnews_classifier.json'
            },
            encoderHidden: 64,
            classifierHidden: 128,
            activation: 'relu'
        }
    });

    // Debounced input → worker prediction
    const debouncedPredict = debounce((text) => {
        if (!modelsReady || !text.trim()) return;
        worker.postMessage({ type: 'predict', text });
        // optimistic UI while waiting
        if (output && fill) {
            output.textContent = '…';
            fill.style.width = '0%';
            fill.textContent = '';
            fill.style.background = '#ccc';
        }
    }, 200);

    if (input) {
        input.addEventListener('input', () => {
            debouncedPredict(input.value.toLowerCase());
        });
    }
});
