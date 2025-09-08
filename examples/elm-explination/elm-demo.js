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
    /* ---------- Slide 2: Encoding (legend + tooltip) ---------- */
    let rowSelect, encodeBtn, encodeCanvas, tokensOut;
    const CSV_SNIPPET = `Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
3,Carlyle Looks Toward Commercial Aerospace (Reuters),"Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market."`;

    let dataRows = [];
    const S2 = {
        legend: null,
        legendInfo: null,
        tipEl: null,
        lastEncoded: null,
        grid: null, // {x0,y0,cols,rows,cw,ch,n}
    };

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
        rowSelect = document.getElementById('rowSelect');
        encodeBtn = document.getElementById('encodeBtn');
        encodeCanvas = document.getElementById('encodeCanvas');
        tokensOut = document.getElementById('tokensOut');

        // Build data rows
        dataRows = parseCSVMini(CSV_SNIPPET);
        rowSelect.innerHTML = '';
        for (let i = 0; i < dataRows.length; i++) {
            const o = document.createElement('option');
            o.value = String(i);
            o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
            rowSelect.appendChild(o);
        }
        encodeBtn.onclick = () => {
            const i = +rowSelect.value || 0;
            post('encode', { text: dataRows[i].text });
        };
        let basisFrozen = false; // UI flag only
        // Legend (explains colors + quick stats)
        if (!S2.legend) {
            S2.legend = document.createElement('div');
            S2.legend.className = 'heat-legend';
            S2.legend.innerHTML = `
      <span>Heatmap key:</span>
      <span>neg</span><span class="legend-bar"></span><span>pos</span>
      <span style="margin-left:8px;">|value| intensity</span>
    `;
            S2.legendInfo = document.createElement('span');
            S2.legendInfo.style.marginLeft = 'auto';
            S2.legend.appendChild(S2.legendInfo);
            // insert right under the canvas, before tokensOut
            tokensOut.before(S2.legend);
        }

        // Tooltip for cell hover
        if (!S2.tipEl) {
            S2.tipEl = document.createElement('div');
            S2.tipEl.className = 'encode-tip';
            encodeCanvas.parentElement.appendChild(S2.tipEl);
            encodeCanvas.addEventListener('mousemove', onEncodeHover);
            encodeCanvas.addEventListener('mouseleave', () => { S2.tipEl.style.display = 'none'; });
        }
    }

    function onEncoded({ tokens, vector, usedTFIDF, featureNames }) {
        S2.lastEncoded = { tokens, vector, usedTFIDF, featureNames };
        tokensOut.textContent = (usedTFIDF
            ? `TF-IDF tokens (top):\n${tokens.slice(0, 25).join(' ')}\n\nvector length: ${vector.length}`
            : `Tokens (fallback BOW):\n${tokens.slice(0, 25).join(' ')}\n\nvector length: ${vector.length}`);
        drawEncode();
    }

    function drawEncode() {
        if (!encodeCanvas) return;

        const c = encodeCanvas; const dpr = devicePixelRatio || 1; const W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);

        // No vector yet
        if (!S2.lastEncoded) {
            g.fillStyle = '#93a9e8'; g.fillText('Click “Encode →” to preview the input vector.', 12, 22);
            if (S2.legendInfo) S2.legendInfo.textContent = '';
            S2.grid = null;
            return;
        }

        const v = S2.lastEncoded.vector;
        const n = Math.min(128, v.length);          // show a manageable subset
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const margin = 10;
        const x0 = margin, y0 = margin;
        const cw = Math.floor((W - 2 * margin) / cols);
        const ch = Math.floor((H - 2 * margin) / rows);

        const maxAbs = Math.max(1e-6, Math.max(...v.slice(0, n).map(x => Math.abs(x))));
        const nnz = v.slice(0, n).reduce((a, x) => a + (Math.abs(x) > 1e-12 ? 1 : 0), 0);
        const l2 = Math.sqrt(v.slice(0, n).reduce((a, x) => a + x * x, 0));
        let basisFrozen = false;

        // Save grid for hit-testing
        S2.grid = { x0, y0, cols, rows, cw, ch, n };

        // Cells
        for (let r = 0, k = 0; r < rows; r++) {
            for (let ccol = 0; ccol < cols; ccol++, k++) {
                if (k >= n) break;
                const val = v[k];
                const alpha = Math.min(1, Math.abs(val) / maxAbs);
                const hue = val >= 0 ? 200 : 0; // blue for +, red for -
                g.fillStyle = `hsla(${hue}, 90%, 60%, ${0.15 + 0.85 * alpha})`;
                g.fillRect(x0 + ccol * cw, y0 + r * ch, cw - 2, ch - 2);
            }
        }

        // Grid lines for readability
        g.strokeStyle = 'rgba(255,255,255,0.06)';
        g.lineWidth = 1;
        for (let ccol = 0; ccol <= cols; ccol++) {
            g.beginPath(); g.moveTo(x0 + ccol * cw, y0); g.lineTo(x0 + ccol * cw, y0 + rows * ch); g.stroke();
        }
        for (let r = 0; r <= rows; r++) {
            g.beginPath(); g.moveTo(x0, y0 + r * ch); g.lineTo(x0 + cols * cw, y0 + r * ch); g.stroke();
        }

        // Small caption
        g.fillStyle = '#a7b8e8';
        g.fillText(`features 0..${n - 1} (subset) — nnz: ${nnz}, ‖v‖₂: ${l2.toFixed(2)}, max |v|: ${maxAbs.toFixed(2)}`, 12, H - 12);

        // Update legend stats line (right side)
        if (S2.legendInfo) {
            S2.legendInfo.textContent =
                `showing ${n} dims  |  nnz ${nnz}  |  ‖v‖₂ ${l2.toFixed(2)}  |  max |v| ${maxAbs.toFixed(2)}  |  basis: ${basisFrozen ? 'trained' : 'isolated'}`;
        }

    }

    function onEncodeHover(e) {
        if (!S2.grid || !S2.lastEncoded) return;
        const { x0, y0, cols, rows, cw, ch, n } = S2.grid;
        const rect = encodeCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const col = Math.floor((x - x0) / cw);
        const row = Math.floor((y - y0) / ch);
        const k = row * cols + col;

        if (row >= 0 && col >= 0 && row < rows && col < cols && k < n) {
            const val = S2.lastEncoded.vector[k];
            const names = S2.lastEncoded.featureNames || [];
            const token = names[k] || null;              // ← stable mapping from worker
            const label = token ? `#${k} “${token}”` : `feature #${k}`;
            S2.tipEl.textContent = `${label}: ${val.toExponential(3)}`;
            S2.tipEl.style.display = 'block';
            const tip = S2.tipEl.getBoundingClientRect();
            const left = Math.min(rect.width - tip.width - 6, Math.max(0, x + 8));
            const top = Math.min(rect.height - tip.height - 6, Math.max(0, y - 28));
            S2.tipEl.style.transform = `translate(${left}px, ${top}px)`;
        } else {
            S2.tipEl.style.display = 'none';
        }
    }

    // ---------------- Slide 3 ----------------
    const S3 = {
        hiddenSize: null, hiddenSizeVal: null, shuffleBtn: null, previewHBtn: null, canvas: null, WPreview: null,
        W: null, b: null, Hx: null, Z: null,
        legendEl: null, legendInfo: null, tipEl: null,
        gridW: null, gridBars: null  // hit-test rectangles
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

        // Legend (under canvas, above the numbers box)
        if (!S3.legendEl) {
            S3.legendEl = document.createElement('div');
            S3.legendEl.className = 'heat-legend';
            S3.legendEl.innerHTML = `
      <span class="legend-chip">W heatmap</span>
      <span>neg</span><span class="legend-bar"></span><span>pos</span>
      <span style="margin-left:8px;">|value| intensity</span>
      <span style="margin-left:12px;" class="legend-chip">Hx bars = g(Wx+b)</span>
    `;
            S3.legendInfo = document.createElement('span');
            S3.legendInfo.style.marginLeft = 'auto';
            S3.legendEl.appendChild(S3.legendInfo);
            S3.WPreview.before(S3.legendEl);

            // Explanations block (compact)
            const expl = document.createElement('div');
            expl.className = 'note';
            expl.style.marginTop = '6px';
            expl.innerHTML = `
      <strong>What is this?</strong> Each row of the heatmap is a <em>hidden neuron</em>, each column an input feature.
      Cell <code>W[i,j]</code> is the weight from feature <code>j</code> to neuron <code>i</code>. Colors: red = negative, blue = positive, brighter = larger magnitude.<br>
      The green bars on the right are the activations <code>H = g(Wx + b)</code> for the selected text vector <code>x</code> (here <code>g</code>=ReLU).
      Increasing <em>Hidden size</em> adds more random features (rows), which can capture more patterns but also increases <code>β</code>'s size and solve time.<br>
      The numbers box shows a small <em>8×8 sample</em> of <code>W</code> (first 8 neurons × first 8 features) so you can see concrete values.
    `;
            S3.legendEl.after(expl);
        }

        // Tooltip over the canvas
        if (!S3.tipEl) {
            S3.tipEl = document.createElement('div');
            S3.tipEl.className = 'hidden-tip';
            S3.canvas.parentElement.appendChild(S3.tipEl);
            S3.canvas.addEventListener('mousemove', onHiddenHover);
            S3.canvas.addEventListener('mouseleave', () => { S3.tipEl.style.display = 'none'; });
        }
    }

    function onHiddenInit({ W, b }) {
        S3.W = W; S3.b = b; S3.Hx = null; S3.Z = null;
        S3.WPreview.textContent = `W: ${W.length}x${W[0]?.length || 0}  b: ${b.length}  (Showing 8×8 sample)\n` +
            sampleMatrixText(W, 8, 8);
        updateHiddenLegend();
        drawHidden();
    }

    function onHiddenProject({ Hx, Z, activation }) {
        S3.Hx = Hx; S3.Z = Z || null;
        updateHiddenLegend(activation || 'relu');
        drawHidden();
    }

    function onHiddenHover(e) {
        const rect = S3.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Over heatmap?
        if (S3.gridW) {
            const { x: gx, y: gy, cols, rows, cellW, cellH } = S3.gridW;
            const j = Math.floor((x - gx) / cellW);
            const i = Math.floor((y - gy) / cellH);
            if (i >= 0 && j >= 0 && i < rows && j < cols) {
                const val = S3.W[i][j];
                S3.tipEl.textContent = `W[${i},${j}] = ${val.toFixed(3)} (feature ${j} → neuron ${i})`;
                S3.tipEl.style.display = 'block';
                const tip = S3.tipEl.getBoundingClientRect();
                S3.tipEl.style.transform = `translate(${Math.min(rect.width - tip.width - 6, Math.max(0, x + 8))}px, ${Math.min(rect.height - tip.height - 6, Math.max(0, y - 28))}px)`;
                return;
            }
        }
        // Over bars?
        if (S3.gridBars) {
            const { x: bx, y: by, n, eachH } = S3.gridBars;
            if (x >= bx) {
                const i = Math.floor((y - by) / eachH);
                if (i >= 0 && i < n) {
                    const h = S3.Hx[i];
                    const z = S3.Z ? S3.Z[i] : null;
                    S3.tipEl.textContent = z == null ? `H[${i}] = ${h.toFixed(3)}` : `H[${i}] = g(z) = ${h.toFixed(3)}  (z=${z.toFixed(3)})`;
                    S3.tipEl.style.display = 'block';
                    const tip = S3.tipEl.getBoundingClientRect();
                    S3.tipEl.style.transform = `translate(${Math.min(rect.width - tip.width - 6, Math.max(0, x + 8))}px, ${Math.min(rect.height - tip.height - 6, Math.max(0, y - 28))}px)`;
                    return;
                }
            }
        }
        S3.tipEl.style.display = 'none';
    }

    function updateHiddenLegend(act = 'relu') {
        if (!S3.legendInfo) return;
        const h = S3.W?.length || 0;
        const d = S3.W?.[0]?.length || (S2.lastEncoded?.vector?.length || 0);
        const hasHx = Array.isArray(S3.Hx);
        S3.legendInfo.textContent = `W shape: ${h}×${d}   |   Hx ${hasHx ? 'computed' : 'pending'} (g=${act})`;
    }

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

        if (!S3.W) {
            g.fillStyle = '#93a9e8'; g.fillText('Click “Shuffle Hidden” to initialize W,b', 12, 22);
            S3.gridW = S3.gridBars = null;
            return;
        }

        // Layout
        const pad = 10;
        const ww = Math.floor(W * 0.66);         // left heatmap area width
        const xHeat = pad, yHeat = pad;
        const xBars = ww + pad, yBars = pad;

        // --- Heatmap W ---
        const vW = S3.W;
        const rows = vW.length;
        const cols = vW[0].length;
        const cellW = Math.max(1, Math.floor((ww - 2 * pad) / cols));
        const cellH = Math.max(1, Math.floor((H - 2 * pad) / rows));
        let vmax = 1e-6;
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) vmax = Math.max(vmax, Math.abs(vW[i][j]));
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = vW[i][j];
                const alpha = Math.min(1, Math.abs(val) / vmax);
                const hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * alpha})`;
                g.fillRect(xHeat + j * cellW, yHeat + i * cellH, cellW, cellH);
            }
        }
        // subtle grid lines
        g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
        for (let j = 0; j <= cols; j++) {
            g.beginPath(); g.moveTo(xHeat + j * cellW, yHeat); g.lineTo(xHeat + j * cellW, yHeat + rows * cellH); g.stroke();
        }
        for (let i = 0; i <= rows; i++) {
            g.beginPath(); g.moveTo(xHeat, yHeat + i * cellH); g.lineTo(xHeat + cols * cellW, yHeat + i * cellH); g.stroke();
        }
        // label
        g.fillStyle = '#a7b8e8';
        g.fillText('W (hidden × input weights)', xHeat, H - 8);

        // Save hit-test rect
        S3.gridW = { x: xHeat, y: yHeat, cols, rows, cellW, cellH };

        // --- Bars: Hx ---
        if (S3.Hx) {
            const Hx = S3.Hx;
            const n = Hx.length;
            const barW = (W - xBars - pad);
            const eachH = Math.max(1, Math.floor((H - 2 * pad) / n));
            const absmax = Math.max(1e-6, ...Hx.map(x => Math.abs(x)));
            for (let i = 0; i < n; i++) {
                const val = Hx[i];
                const frac = Math.min(1, Math.abs(val) / absmax);
                const len = Math.floor(frac * barW);
                g.fillStyle = val >= 0 ? '#6ee7a2' : '#fb7185';
                g.fillRect(xBars, yBars + i * eachH, len, Math.max(1, eachH - 2));
            }
            // y ticks every ~5 bars
            g.fillStyle = '#a7b8e8';
            g.fillText('Hx = g(Wx + b)', xBars, H - 8);
            S3.gridBars = { x: xBars, y: yBars, n, eachH, barW };
        } else {
            g.fillStyle = '#93a9e8';
            g.fillText('Click “Project H = g(X·W + b)” after encoding.', xBars, 22);
            S3.gridBars = null;
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
        basisFrozen = true;
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