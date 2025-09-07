/* =========================================================
   One clean module scope. No duplicate global identifiers.
   ========================================================= */
document.addEventListener('DOMContentLoaded', () => {

    const S1 = {
        act: 'relu',
        availableActs: [],
        rafId: null,
        actSelect: null, wRange: null, bRange: null, wVal: null, bVal: null, canvas: null,
    };

    // ---------------- Navigation shell ----------------
    const slides = ['slide1', 'slide2', 'slide3', 'slide4'].map(id => document.getElementById(id));
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const slideLabel = document.getElementById('slideLabel');
    document.getElementById('toOne').onclick = () => show(0);
    document.getElementById('toTwo').onclick = () => show(1);
    document.getElementById('toThree').onclick = () => show(2);
    document.getElementById('toFour').onclick = () => show(3);

    let idx = 0;
    let s1Inited = false, s2Inited = false, s3Inited = false, s4Inited = false;

    function show(i) {
        const prevIdx = idx;
        idx = Math.max(0, Math.min(slides.length - 1, i));
        slides.forEach((s, j) => s.hidden = j !== idx);
        slideLabel.textContent = `Slide ${idx + 1} / ${slides.length}`;
        prevBtn.disabled = idx === 0;
        nextBtn.disabled = idx === slides.length - 1;
        location.hash = `#${idx + 1}`;

        if (prevIdx === 0 && idx !== 0) stopNeuronLoop();

        if (idx === 0) { if (!s1Inited) { ensureSlide1(); s1Inited = true; } drawNeuron(); }
        if (idx === 1) { if (!s2Inited) { ensureSlide2(); s2Inited = true; } drawEncode(); }
        if (idx === 2) { if (!s3Inited) { ensureSlide3(); s3Inited = true; } drawHidden(); }
        if (idx === 3) { if (!s4Inited) { ensureSlide4(); s4Inited = true; } drawBeta(); }
    }
    prevBtn.onclick = () => show(idx - 1);
    nextBtn.onclick = () => show(idx + 1);
    if (location.hash) {
        const n = parseInt(location.hash.replace('#', ''), 10);
        show((n >= 1 && n <= slides.length) ? (n - 1) : 0);
    } else {
        show(0);
    }

    // ---------------- Web Worker ----------------
    const workerStatus = document.getElementById('workerStatus');
    const worker = new Worker('./elm-worker.js');
    const post = (type, payload = {}) => worker.postMessage({ type, payload });

    worker.onmessage = (e) => {
        const { type, payload } = e.data || {};
        if (type === 'status') { workerStatus.textContent = payload; }
        if (type === 'actCurve') { S1.act = payload.fnName; S1.availableActs = payload.fnList || []; }
        if (type === 'encoded') { onEncoded(payload); }
        if (type === 'hidden_init') { onHiddenInit(payload); }
        if (type === 'hidden_project') { onHiddenProject(payload); }
        if (type === 'trained') { onTrained(payload); }
        if (type === 'predicted') { onPredicted(payload); }
    };
    post('hello');
    post('list_activations');

    // ---------------- Slide 1 ----------------

    function ensureSlide1() {
        S1.actSelect = document.getElementById('actSelect');
        S1.wRange = document.getElementById('wRange');
        S1.bRange = document.getElementById('bRange');
        S1.wVal = document.getElementById('wVal');
        S1.bVal = document.getElementById('bVal');
        S1.canvas = document.getElementById('neuronCanvas');

        S1.actSelect.addEventListener('change', () => { S1.act = S1.actSelect.value; drawNeuron(); });
        [S1.wRange, S1.bRange].forEach(r => {
            r.addEventListener('input', () => {
                S1.wVal.textContent = (+S1.wRange.value).toFixed(2);
                S1.bVal.textContent = (+S1.bRange.value).toFixed(2);
                drawNeuron();
            });
        });
        S1.wVal.textContent = (+S1.wRange.value).toFixed(2);
        S1.bVal.textContent = (+S1.bRange.value).toFixed(2);
    }

    function stopNeuronLoop() { if (S1.rafId) { cancelAnimationFrame(S1.rafId); S1.rafId = null; } }

    function drawNeuron() {
        if (!S1.canvas) return;
        stopNeuronLoop();

        const c = S1.canvas;
        const g = c.getContext('2d');
        const dpr = devicePixelRatio || 1;

        const act = (z) => {
            switch (S1.act) {
                case 'relu': return Math.max(0, z);
                case 'leakyRelu': return z >= 0 ? z : 0.01 * z;
                case 'sigmoid': return 1 / (1 + Math.exp(-z));
                case 'tanh': return Math.tanh(z);
                default: return z;
            }
        };

        function loop() {
            const W = c.clientWidth, H = c.clientHeight;
            c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.clearRect(0, 0, W, H);

            // axes
            g.strokeStyle = '#3857a8'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(10, H - 20); g.lineTo(W - 10, H - 20); g.stroke();
            g.beginPath(); g.moveTo(40, 10); g.lineTo(40, H - 10); g.stroke();

            const w = +S1.wRange.value, b = +S1.bRange.value;

            const toXY = (x, y) => {
                const xm = (x + 3) / 6, ym = (y - (-2)) / (2 - (-2));
                return [40 + xm * (W - 55), (H - 20) - ym * (H - 35)];
            };

            // curve
            g.strokeStyle = '#5ad1ff'; g.lineWidth = 2;
            g.beginPath();
            for (let i = 0; i <= 300; i++) {
                const x = -3 + (i / 300) * 6;
                const y = act(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
            }
            g.stroke();

            // moving samples
            const now = performance.now() / 1000;
            for (let k = 0; k < 7; k++) {
                const x = -3 + ((now * 0.6 + k / 7) % 1) * 6;
                const y = act(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                g.fillStyle = '#6ee7a2';
                g.beginPath(); g.arc(px, py, 3.2, 0, Math.PI * 2); g.fill();
            }

            S1.rafId = requestAnimationFrame(loop);
        }
        loop();
    }

    // ---------------- Slide 2 ----------------
    const S2 = {
        rowSelect: null, encodeBtn: null, canvas: null, tokensOut: null,
        dataRows: [], lastEncoded: null
    };

    const CSV_SNIPPET = `Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
3,Carlyle Looks Toward Commercial Aerospace (Reuters),"Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market."`;

    function parseCSVMini(s) {
        const lines = s.trim().split(/\r?\n/);
        lines.shift(); // header
        const out = [];
        for (const line of lines) {
            const m = line.match(/^(\d+),([^,]+),(.*)$/);
            if (!m) continue;
            const cls = +m[1];
            const title = m[2].trim();
            let desc = m[3].trim();
            if (desc.startsWith('"') && desc.endsWith('"')) desc = desc.slice(1, -1);
            desc = desc.replaceAll('\\b', ' ').replaceAll('\\which', ' which').replaceAll('\\', ' ');
            out.push({ cls, text: `${title}. ${desc}` });
        }
        return out;
    }

    function ensureSlide2() {
        S2.rowSelect = document.getElementById('rowSelect');
        S2.encodeBtn = document.getElementById('encodeBtn');
        S2.canvas = document.getElementById('encodeCanvas');
        S2.tokensOut = document.getElementById('tokensOut');

        S2.dataRows = parseCSVMini(CSV_SNIPPET);
        S2.rowSelect.innerHTML = '';
        for (let i = 0; i < S2.dataRows.length; i++) {
            const o = document.createElement('option');
            o.value = String(i);
            o.textContent = `[${S2.dataRows[i].cls}] ${S2.dataRows[i].text.slice(0, 80)}…`;
            S2.rowSelect.appendChild(o);
        }
        S2.encodeBtn.onclick = () => {
            const i = +S2.rowSelect.value || 0;
            post('encode', { text: S2.dataRows[i].text });
        };
    }

    function onEncoded({ tokens, vector, usedTFIDF }) {
        S2.lastEncoded = { tokens, vector };
        S2.tokensOut.textContent = (usedTFIDF
            ? `TF-IDF tokens (top):\n${tokens.slice(0, 25).join(' ')}\n\nvector length: ${vector.length}`
            : `Tokens (fallback BOW):\n${tokens.slice(0, 25).join(' ')}\n\nvector length: ${vector.length}`);
        drawEncode();
    }

    function drawEncode() {
        if (!S2.canvas) return;
        const c = S2.canvas; const dpr = devicePixelRatio || 1; const W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!S2.lastEncoded) { g.fillStyle = '#93a9e8'; g.fillText('Click “Encode →” to preview the input vector.', 12, 22); return; }
        const v = S2.lastEncoded.vector;
        const n = Math.min(128, v.length);
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const cw = Math.floor((W - 20) / cols);
        const ch = Math.floor((H - 20) / rows);
        const max = Math.max(1e-6, Math.max(...v.map(x => Math.abs(x))));
        let k = 0;
        for (let r = 0; r < rows; r++) {
            for (let ccol = 0; ccol < cols; ccol++) {
                if (k >= n) break;
                const val = v[k++];
                const alpha = Math.min(1, Math.abs(val) / max);
                const hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue}, 90%, 60%, ${0.15 + 0.85 * alpha})`;
                g.fillRect(10 + ccol * cw, 10 + r * ch, cw - 2, ch - 2);
            }
        }
    }

    // ---------------- Slide 3 ----------------
    const S3 = {
        hiddenSize: null, hiddenSizeVal: null, shuffleBtn: null, previewHBtn: null, canvas: null, WPreview: null,
        W: null, b: null, Hx: null
    };

    function ensureSlide3() {
        S3.hiddenSize = document.getElementById('hiddenSize');
        S3.hiddenSizeVal = document.getElementById('hiddenSizeVal');
        S3.shuffleBtn = document.getElementById('shuffleBtn');
        S3.previewHBtn = document.getElementById('previewHBtn');
        S3.canvas = document.getElementById('hiddenCanvas');
        S3.WPreview = document.getElementById('WPreview');

        S3.hiddenSize.oninput = () => { S3.hiddenSizeVal.textContent = S3.hiddenSize.value; };
        S3.shuffleBtn.onclick = () => {
            post('init_hidden', { hidden: +S3.hiddenSize.value, inputDim: S2.lastEncoded?.vector?.length || 512 });
        };
        S3.previewHBtn.onclick = () => {
            if (!S2.lastEncoded) { alert('Encode a row on Slide 2 first'); return; }
            post('project_hidden', { x: S2.lastEncoded.vector });
        };
    }

    function onHiddenInit({ W, b }) {
        S3.W = W; S3.b = b; S3.Hx = null;
        S3.WPreview.textContent = `W: ${W.length}x${W[0]?.length || 0}\n b: ${b.length}\n(Showing 8×8 sample)\n` +
            sampleMatrixText(W, 8, 8);
        drawHidden();
    }
    function onHiddenProject({ Hx }) { S3.Hx = Hx; drawHidden(); }
    function sampleMatrixText(M, r, c) {
        const R = Math.min(r, M.length);
        const C = Math.min(c, M[0]?.length || 0);
        let s = '';
        for (let i = 0; i < R; i++) {
            s += M[i].slice(0, C).map(v => (Math.abs(v) < 1e-3 ? '0.000' : v.toFixed(3))).join(' ') + '\n';
        }
        return s;
    }
    function drawHidden() {
        if (!S3.canvas) return;
        const c = S3.canvas; const dpr = devicePixelRatio || 1; const W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!S3.W) { g.fillStyle = '#93a9e8'; g.fillText('Click “Shuffle Hidden” to initialize W,b', 12, 22); return; }
        const ww = Math.floor(W * 0.66);
        const vW = S3.W;
        const rows = vW.length;
        const cols = vW[0].length;
        const cellW = Math.max(1, Math.floor((ww - 20) / cols));
        const cellH = Math.max(1, Math.floor((H - 20) / rows));
        let vmax = 1e-6;
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) vmax = Math.max(vmax, Math.abs(vW[i][j]));
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = vW[i][j];
                const alpha = Math.min(1, Math.abs(val) / vmax);
                const hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * alpha})`;
                g.fillRect(10 + j * cellW, 10 + i * cellH, cellW, cellH);
            }
        }
        if (S3.Hx) {
            const Hx = S3.Hx;
            const x0 = ww + 10;
            const barW = (W - x0 - 20);
            const n = Hx.length;
            const eachH = Math.max(1, Math.floor((H - 20) / n));
            const absmax = Math.max(1e-6, ...Hx.map(x => Math.abs(x)));
            for (let i = 0; i < n; i++) {
                const val = Hx[i];
                const frac = Math.min(1, Math.abs(val) / absmax);
                const len = Math.floor(frac * barW);
                g.fillStyle = val >= 0 ? '#6ee7a2' : '#fb7185';
                g.fillRect(x0, 10 + i * eachH, len, Math.max(1, eachH - 1));
            }
        } else {
            g.fillStyle = '#93a9e8'; g.fillText('Click “Project H = g(X·W + b)” after encoding.', Math.floor(W * 0.66) + 12, 22);
        }
    }

    // ---------------- Slide 4 ----------------
    const S4 = { trainBtn: null, predictBtn: null, solveOut: null, canvas: null, dims: null, betaSample: null };

    function ensureSlide4() {
        S4.trainBtn = document.getElementById('trainBtn');
        S4.predictBtn = document.getElementById('predictBtn');
        S4.solveOut = document.getElementById('solveOut');
        S4.canvas = document.getElementById('betaCanvas');

        S4.trainBtn.onclick = () => {
            post('train', {
                rows: S2.dataRows.map(r => ({ y: r.cls, text: r.text })),
                hidden: +S3.hiddenSize.value
            });
        };
        S4.predictBtn.onclick = () => {
            const i = +S2.rowSelect.value || 0;
            post('predict', { text: S2.dataRows[i].text });
        };
    }

    function onTrained({ dims, betaSample, note }) {
        S4.dims = dims; S4.betaSample = betaSample;
        S4.solveOut.textContent = `Solved β with pseudo-inverse${note ? ` (${note})` : ''}\n` +
            `H shape: ${dims.H_rows}×${dims.H_cols},  Y shape: ${dims.Y_rows}×${dims.Y_cols}\n` +
            `β shape: ${dims.B_rows}×${dims.B_cols}\n` +
            `β (8×8 sample):\n${betaSample}`;
        drawBeta();
    }
    function onPredicted({ pred, scores }) {
        const label = `Predicted class: ${pred}  |  scores: [${(scores || []).map(x => x.toFixed(2)).join(', ')}]`;
        S4.solveOut.textContent += `\n\n${label}`;
    }
    function drawBeta() {
        if (!S4.canvas) return;
        const c = S4.canvas; const dpr = devicePixelRatio || 1; const W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!S4.dims) { g.fillStyle = '#93a9e8'; g.fillText('Train first to visualize β.', 12, 22); return; }
        const rows = Math.min(24, S4.dims.B_rows);
        const cols = Math.min(24, S4.dims.B_cols);
        const cw = Math.floor((W - 20) / cols);
        const ch = Math.floor((H - 20) / rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const phase = (i * 13 + j * 7) % 100;
                const a = 0.25 + 0.7 * (phase / 100);
                g.fillStyle = `hsla(${180 + (j * 6) % 60},100%,60%,${a})`;
                g.fillRect(10 + j * cw, 10 + i * ch, cw - 2, ch - 2);
            }
        }
    }

});