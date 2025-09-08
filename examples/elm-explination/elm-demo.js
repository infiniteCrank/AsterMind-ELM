/* =========================================================
   ELM Visual Primer — App Script (self-contained)
   - Mini-map of ELM (Input→Hidden→Output) with live highlight
   - Safer tooltips, guardrails, clear instructions
   - Slide registry for easy add/remove/reorder
   ========================================================= */
document.addEventListener('DOMContentLoaded', () => {
    /* ---------------- Small style additions (mini-map, tips) -------------- */
    (function injectStyles() {
        const css = `
    .mini-map{display:flex;align-items:center;gap:8px;padding:6px 8px;border:1px solid #203a7c;border-radius:10px;background:#0c1a3d}
    .mini-map canvas{display:block}
    .encode-tip,.hidden-tip{
      position:absolute; top:0; left:0; transform: translate(8px,-28px);
      background:#0c1a3d; border:1px solid #203a7c; padding:4px 6px; border-radius:6px;
      font-size:.8rem; color:#cfe1ff; display:none; pointer-events:none; white-space:nowrap;
    }
    .heat-legend{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:.85rem;color:var(--muted)}
    .legend-chip{padding:2px 8px;border:1px solid #203a7c;border-radius:999px}
    .legend-bar{width:160px;height:10px;border-radius:999px;border:1px solid #203a7c;
      background:linear-gradient(90deg, hsla(0,90%,60%,.85), hsla(210,30%,20%,.15) 50%, hsla(200,90%,60%,.85))}
    .helper{margin-top:8px;font-size:.92rem}
    .helper h3{margin:.2rem 0 .3rem 0;font-size:1rem;color:#e5efff}
    .helper ul{margin:.2rem 0 .4rem 1.1rem;line-height:1.35}
    `;
        const el = document.createElement('style');
        el.textContent = css;
        document.head.appendChild(el);
    })();

    /* ---------------- Utilities ---------------- */
    const AG_LABELS = { 1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech' };
    const softmax = (arr) => {
        if (!arr || !arr.length) return [];
        const m = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - m));
        const s = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / s);
    };
    const fmtVal = (v) => {
        if (!Number.isFinite(v)) return '0';
        const a = Math.abs(v);
        if (a === 0) return '0';
        if (a >= 1e3 || a < 1e-3) return v.toExponential(2);
        return v.toFixed(a < 1 ? 3 : 2);
    };

    /* ---------------- Slide registry (easy add/remove) ---------------- */
    // If you add/remove slides in HTML, this script will adapt automatically.
    const slideNodes = Array.from(document.querySelectorAll('section.slide'));
    const slides = slideNodes.map((node, idx) => {
        // You can still override with data-stage="neuron|input|hidden|output" on a <section>
        const stage = node.dataset.stage || (idx === 0 ? 'neuron' : idx === 1 ? 'input' : idx === 2 ? 'hidden' : 'output');
        return { id: node.id, node, stage };
    });

    // Header bits + dynamic numeric nav
    const header = document.querySelector('header');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const slideLabel = document.getElementById('slideLabel');
    const footerBar = document.querySelector('.footerBar');

    // Replace footer number buttons with dynamic ones
    if (footerBar) {
        footerBar.innerHTML = '';
        slides.forEach((_, i) => {
            const b = document.createElement('button');
            b.textContent = String(i + 1);
            b.onclick = () => show(i);
            footerBar.appendChild(b);
        });
    }

    // Mini-map UI (created once)
    const miniWrap = document.createElement('div');
    miniWrap.className = 'mini-map';
    const miniLabel = document.createElement('span');
    miniLabel.textContent = 'ELM map:';
    miniLabel.style.fontSize = '.9rem';
    miniLabel.style.color = 'var(--muted)';
    const mini = document.createElement('canvas');
    // wider & slightly taller to fit 4 boxes
    mini.width = 320;
    mini.height = 54;
    miniWrap.appendChild(miniLabel);
    miniWrap.appendChild(mini);
    header.appendChild(miniWrap);

    function drawMiniMap(stage) {
        const c = mini, g = c.getContext('2d'), W = c.width, H = c.height;
        g.clearRect(0, 0, W, H);

        // Layout: Neuron (standalone) + Input → Hidden → Output flow
        const boxes = [
            { x: 10, y: 14, w: 68, h: 26, label: 'Neuron', key: 'neuron', dashed: true },
            { x: 92, y: 14, w: 60, h: 26, label: 'Input', key: 'input' },
            { x: 170, y: 10, w: 66, h: 34, label: 'Hidden', key: 'hidden' },
            { x: 252, y: 14, w: 60, h: 26, label: 'Output', key: 'output' }
        ];

        // Arrows for the main ELM path: Input → Hidden → Output
        g.strokeStyle = '#5ad1ff'; g.lineWidth = 1.5; g.setLineDash([]);
        g.beginPath(); g.moveTo(92 + 60, 27); g.lineTo(170, 27); g.stroke();
        g.beginPath(); g.moveTo(170 + 66, 27); g.lineTo(252, 27); g.stroke();

        // Draw boxes
        boxes.forEach(b => {
            const active = (b.key === stage);
            g.fillStyle = active ? 'rgba(90,209,255,0.18)' : 'rgba(12,26,61,.8)';
            g.strokeStyle = active ? '#5ad1ff' : '#203a7c';
            g.lineWidth = active ? 2 : 1;

            if (b.dashed) g.setLineDash([5, 3]); else g.setLineDash([]);
            g.fillRect(b.x, b.y, b.w, b.h);
            g.strokeRect(b.x, b.y, b.w, b.h);
            g.setLineDash([]);

            g.fillStyle = active ? '#e5efff' : '#a7b8e8';
            g.font = '11px system-ui,Segoe UI,Roboto,Arial';
            g.fillText(b.label, b.x + 6, b.y + b.h - 8);
        });
    }

    /* ------- IMPORTANT: declare S1 and init flags BEFORE show() -------- */
    const S1 = { act: 'relu', availableActs: [], rafId: null, actSelect: null, wRange: null, bRange: null, wVal: null, bVal: null, canvas: null };
    let s1Inited = false, s2Inited = false, s3Inited = false, s4Inited = false;

    let idx = 0;
    let uiBasisFrozen = false; // becomes true after training

    function show(i) {
        const prevIdx = idx;
        idx = Math.max(0, Math.min(slides.length - 1, i));
        slides.forEach((s, j) => s.node.hidden = j !== idx);
        slideLabel.textContent = `Slide ${idx + 1} / ${slides.length}`;
        prevBtn.disabled = idx === 0;
        nextBtn.disabled = idx === slides.length - 1;
        location.hash = `#${idx + 1}`;
        drawMiniMap(slides[idx].stage);

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

    /* ---------------- Worker wiring ---------------- */
    const workerStatus = document.getElementById('workerStatus');
    const worker = new Worker('./elm-worker.js'); // classic worker
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

    /* ---------------- Slide 1: Single neuron ---------------- */
    function ensureSlide1() {
        S1.actSelect = document.getElementById('actSelect');
        S1.wRange = document.getElementById('wRange');
        S1.bRange = document.getElementById('bRange');
        S1.wVal = document.getElementById('wVal');
        S1.bVal = document.getElementById('bVal');
        S1.canvas = document.getElementById('neuronCanvas');

        // Add “Try this” helper (idempotent)
        addHelper('slide1', {
            title: 'Try this:',
            steps: [
                'Pick an activation (ReLU, tanh, …).',
                'Drag the weight (w) and bias (b).',
                'Watch y = g(w·x + b) and the moving dots react.'
            ],
            blurb: 'This is one neuron. “Affine” just means we compute z = w·x + b (a line), then apply a squiggle g(z) so the model can bend.'
        });

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
        const c = S1.canvas; const g = c.getContext('2d'); const dpr = devicePixelRatio || 1;
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
            const toXY = (x, y) => { const xm = (x + 3) / 6, ym = (y - (-2)) / (2 - (-2)); return [40 + xm * (W - 55), (H - 20) - ym * (H - 35)]; };
            // curve
            g.strokeStyle = '#5ad1ff'; g.lineWidth = 2; g.beginPath();
            for (let i = 0; i <= 300; i++) {
                const x = -3 + (i / 300) * 6, y = act(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
            }
            g.stroke();
            // moving samples
            const now = performance.now() / 1000;
            for (let k = 0; k < 7; k++) {
                const x = -3 + ((now * 0.6 + k / 7) % 1) * 6, y = act(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                g.fillStyle = '#6ee7a2'; g.beginPath(); g.arc(px, py, 3.2, 0, Math.PI * 2); g.fill();
            }
            S1.rafId = requestAnimationFrame(loop);
        }
        loop();
    }

    /* ---------------- Slide 2: Encoding ---------------- */
    let rowSelect, encodeBtn, encodeCanvas, tokensOut;
    const CSV_SNIPPET = `Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
3,Carlyle Looks Toward Commercial Aerospace (Reuters),"Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market."
1,UN Council Weighs Ceasefire Proposal,"Leaders from several nations met to discuss a draft resolution aimed at de-escalation in the region."
4,Chip Startup Unveils Faster AI Accelerator,"The company claims a 2× speedup on transformer inference with a new memory layout."
2,Local Club Wins Championship Final,"Fans celebrated after the underdogs clinched the title with a late goal."`;

    let dataRows = [];
    const S2 = { legend: null, legendInfo: null, tipEl: null, lastEncoded: null, grid: null, rowSelect: null, dataRows: [] };

    function parseCSVMini(s) {
        const lines = s.trim().split(/\r?\n/); lines.shift();
        const out = [];
        for (const line of lines) {
            const m = line.match(/^(\d+),([^,]+),(.*)$/); if (!m) continue;
            const cls = +m[1]; const title = m[2].trim(); let desc = m[3].trim();
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

        // Helpers block
        addHelper('slide2', {
            title: 'Try this:',
            steps: [
                'Pick one of the rows in the dropdown.',
                'Click “Encode →”.',
                'Hover the tiny squares to see which feature/token that cell is.'
            ],
            blurb: 'We turn text into a vector of numbers (features). Each square is one feature value. Blue = positive, red = negative, brighter = larger magnitude.'
        });

        dataRows = parseCSVMini(CSV_SNIPPET);
        rowSelect.innerHTML = '';
        for (let i = 0; i < dataRows.length; i++) {
            const o = document.createElement('option');
            o.value = String(i);
            o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
            rowSelect.appendChild(o);
        }
        S2.rowSelect = rowSelect; S2.dataRows = dataRows;

        encodeBtn.onclick = () => {
            const i = +rowSelect.value || 0;
            post('encode', { text: dataRows[i].text });
        };

        // Legend
        if (!S2.legend) {
            S2.legend = document.createElement('div');
            S2.legend.className = 'heat-legend';
            S2.legend.innerHTML = `
        <span>Heatmap key:</span>
        <span>neg</span><span class="legend-bar"></span><span>pos</span>
        <span style="margin-left:8px;">|value| intensity</span>
      `;
            S2.legendInfo = document.createElement('span'); S2.legendInfo.style.marginLeft = 'auto';
            S2.legend.appendChild(S2.legendInfo);
            tokensOut.before(S2.legend);
        }

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
        const c = encodeCanvas, dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);

        if (!S2.lastEncoded) {
            g.fillStyle = '#93a9e8'; g.fillText('Click “Encode →” to preview the input vector.', 12, 22);
            if (S2.legendInfo) S2.legendInfo.textContent = '';
            S2.grid = null; return;
        }

        const v = S2.lastEncoded.vector;
        const n = Math.min(128, v.length);
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const m = 10, x0 = m, y0 = m;
        const cw = Math.floor((W - 2 * m) / cols);
        const ch = Math.floor((H - 2 * m) / rows);

        const vSub = v.slice(0, n);
        const maxAbs = Math.max(1e-6, ...vSub.map(x => Math.abs(x)));
        const nnz = vSub.reduce((a, x) => a + (Math.abs(x) > 1e-12 ? 1 : 0), 0);
        const l2 = Math.hypot(...vSub);

        S2.grid = { x0: x0, y0: y0, cols, rows, cw, ch, n };

        for (let r = 0, k = 0; r < rows; r++) {
            for (let ccol = 0; ccol < cols; ccol++, k++) {
                if (k >= n) break;
                const val = vSub[k];
                const alpha = Math.min(1, Math.abs(val) / maxAbs);
                const hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue}, 90%, 60%, ${0.15 + 0.85 * alpha})`;
                g.fillRect(x0 + ccol * cw, y0 + r * ch, cw - 2, ch - 2);
            }
        }
        g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
        for (let ccol = 0; ccol <= cols; ccol++) { g.beginPath(); g.moveTo(x0 + ccol * cw, y0); g.lineTo(x0 + ccol * cw, y0 + rows * ch); g.stroke(); }
        for (let rr = 0; rr <= rows; rr++) { g.beginPath(); g.moveTo(x0, y0 + rr * ch); g.lineTo(x0 + cols * cw, y0 + rr * ch); g.stroke(); }

        g.fillStyle = '#a7b8e8';
        g.fillText(`features 0..${n - 1} (subset) — nnz: ${nnz}, ‖v‖₂: ${l2.toFixed(2)}, max |v|: ${maxAbs.toFixed(2)}`, 12, H - 12);

        if (S2.legendInfo) {
            S2.legendInfo.textContent = `showing ${n} dims  |  nnz ${nnz}  |  ‖v‖₂ ${l2.toFixed(2)}  |  max |v| ${maxAbs.toFixed(2)}  |  basis: ${uiBasisFrozen ? 'trained' : 'isolated'}`;
        }
    }

    function onEncodeHover(e) {
        if (!S2.grid || !S2.lastEncoded) return;
        const { x0, y0, cols, rows, cw, ch, n } = S2.grid;
        const rect = encodeCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const col = Math.floor((x - x0) / cw), row = Math.floor((y - y0) / ch);
        const k = row * cols + col;
        if (row >= 0 && col >= 0 && row < rows && col < cols && k < n) {
            const v = S2.lastEncoded.vector;
            const val = Number.isFinite(v[k]) ? v[k] : 0;
            const names = S2.lastEncoded.featureNames || [];
            const token = names[k] || null;
            const label = token ? `#${k} “${token}”` : `feature #${k}`;
            S2.tipEl.textContent = `${label}: ${fmtVal(val)}`;
            S2.tipEl.style.display = 'block';
            const tip = S2.tipEl.getBoundingClientRect();
            const left = Math.min(rect.width - tip.width - 6, Math.max(0, x + 8));
            const top = Math.min(rect.height - tip.height - 6, Math.max(0, y - 28));
            S2.tipEl.style.transform = `translate(${left}px, ${top}px)`;
        } else {
            S2.tipEl.style.display = 'none';
        }
    }

    /* ---------------- Slide 3: Hidden layer ---------------- */
    const S3 = {
        hiddenSize: null, hiddenSizeVal: null, shuffleBtn: null, previewHBtn: null, canvas: null, WPreview: null,
        W: null, b: null, Hx: null, Z: null, legendEl: null, legendInfo: null, tipEl: null,
        gridW: null, gridBars: null
    };

    function ensureSlide3() {
        S3.hiddenSize = document.getElementById('hiddenSize');
        S3.hiddenSizeVal = document.getElementById('hiddenSizeVal');
        S3.shuffleBtn = document.getElementById('shuffleBtn');
        S3.previewHBtn = document.getElementById('previewHBtn');
        S3.canvas = document.getElementById('hiddenCanvas');
        S3.WPreview = document.getElementById('WPreview');

        addHelper('slide3', {
            title: 'Try this:',
            steps: [
                'Set “Hidden size” (how many rows/neurons).',
                'Click “Shuffle Hidden” to make a new random W and b.',
                'Go back to Slide 2, hit Encode, then return and click “Project H = g(X·W + b)”.'
            ],
            blurb: 'Heatmap: each row is a hidden neuron, each column is a feature from the text vector. Right-side green bars are H = g(Wx + b) for your selected text.'
        });

        S3.hiddenSize.oninput = () => { S3.hiddenSizeVal.textContent = S3.hiddenSize.value; };
        S3.shuffleBtn.onclick = () => { post('init_hidden', { hidden: +S3.hiddenSize.value, inputDim: S2.lastEncoded?.vector?.length || 512 }); };
        S3.previewHBtn.onclick = () => {
            if (!S2.lastEncoded) { alert('Encode a row on Slide 2 first'); return; }
            post('project_hidden', { x: S2.lastEncoded.vector });
        };

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

            const expl = document.createElement('div');
            expl.className = 'note'; expl.style.marginTop = '6px';
            expl.innerHTML = `
        <strong>What is this?</strong> <em>W</em> is hidden×input weights. Colors show sign and magnitude. The numbers box prints an 8×8 slice of W (first 8 neurons × first 8 features).
      `;
            S3.legendEl.after(expl);
        }
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
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        if (S3.gridW) {
            const { x: gx, y: gy, cols, rows, cellW, cellH } = S3.gridW;
            const j = Math.floor((x - gx) / cellW), i = Math.floor((y - gy) / cellH);
            if (i >= 0 && j >= 0 && i < rows && j < cols) {
                const val = S3.W[i][j];
                S3.tipEl.textContent = `W[${i},${j}] = ${fmtVal(val)} (feature ${j} → neuron ${i})`;
                S3.tipEl.style.display = 'block';
                const tip = S3.tipEl.getBoundingClientRect();
                S3.tipEl.style.transform = `translate(${Math.min(rect.width - tip.width - 6, Math.max(0, x + 8))}px, ${Math.min(rect.height - tip.height - 6, Math.max(0, y - 28))}px)`;
                return;
            }
        }
        if (S3.gridBars) {
            const { x: bx, y: by, n, eachH } = S3.gridBars;
            if (x >= bx) {
                const i = Math.floor((y - by) / eachH);
                if (i >= 0 && i < n) {
                    const h = S3.Hx[i];
                    const z = S3.Z ? S3.Z[i] : null;
                    S3.tipEl.textContent = z == null ? `H[${i}] = ${fmtVal(h)}` : `H[${i}] = g(z) = ${fmtVal(h)}  (z=${fmtVal(z)})`;
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
        const h = S3.W?.length || 0, d = S3.W?.[0]?.length || (S2.lastEncoded?.vector?.length || 0);
        const hasHx = Array.isArray(S3.Hx);
        S3.legendInfo.textContent = `W shape: ${h}×${d}   |   Hx ${hasHx ? 'computed' : 'pending'} (g=${act})`;
    }
    function sampleMatrixText(M, r, c) {
        const R = Math.min(r, M.length), C = Math.min(c, M[0]?.length || 0);
        let s = ''; for (let i = 0; i < R; i++) { s += M[i].slice(0, C).map(v => (Math.abs(v) < 1e-3 ? '0.000' : (+v).toFixed(3))).join(' ') + '\n'; }
        return s;
    }
    function drawHidden() {
        if (!S3.canvas) return;
        const c = S3.canvas, dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!S3.W) { g.fillStyle = '#93a9e8'; g.fillText('Click “Shuffle Hidden” to initialize W,b', 12, 22); S3.gridW = S3.gridBars = null; return; }
        const pad = 10, ww = Math.floor(W * 0.66), xHeat = pad, yHeat = pad, xBars = ww + pad, yBars = pad;
        const vW = S3.W, rows = vW.length, cols = vW[0].length;
        const cellW = Math.max(1, Math.floor((ww - 2 * pad) / cols));
        const cellH = Math.max(1, Math.floor((H - 2 * pad) / rows));
        let vmax = 1e-6; for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) vmax = Math.max(vmax, Math.abs(vW[i][j]));
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = vW[i][j], alpha = Math.min(1, Math.abs(val) / vmax), hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * alpha})`;
                g.fillRect(xHeat + j * cellW, yHeat + i * cellH, cellW, cellH);
            }
        }
        g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
        for (let j = 0; j <= cols; j++) { g.beginPath(); g.moveTo(xHeat + j * cellW, yHeat); g.lineTo(xHeat + j * cellW, yHeat + rows * cellH); g.stroke(); }
        for (let i = 0; i <= rows; i++) { g.beginPath(); g.moveTo(xHeat, yHeat + i * cellH); g.lineTo(xHeat + cols * cellW, yHeat + i * cellH); g.stroke(); }
        g.fillStyle = '#a7b8e8'; g.fillText('W (hidden × input weights)', xHeat, H - 8);
        S3.gridW = { x: xHeat, y: yHeat, cols, rows, cellW, cellH };

        if (S3.Hx) {
            const Hx = S3.Hx, n = Hx.length, barW = (W - xBars - pad), eachH = Math.max(1, Math.floor((H - 2 * pad) / n));
            const absmax = Math.max(1e-6, ...Hx.map(x => Math.abs(x)));
            for (let i = 0; i < n; i++) {
                const val = Hx[i], frac = Math.min(1, Math.abs(val) / absmax), len = Math.floor(frac * barW);
                g.fillStyle = val >= 0 ? '#6ee7a2' : '#fb7185';
                g.fillRect(xBars, yBars + i * eachH, len, Math.max(1, eachH - 2));
            }
            g.fillStyle = '#a7b8e8'; g.fillText('Hx = g(Wx + b)', xBars, H - 8);
            S3.gridBars = { x: xBars, y: yBars, n, eachH, barW };
        } else {
            g.fillStyle = '#93a9e8'; g.fillText('Click “Project H = g(X·W + b)” after encoding.', xBars, 22);
            S3.gridBars = null;
        }
    }

    /* ---------------- Slide 4: Train & Predict ---------------- */
    const S4 = { trainBtn: null, predictBtn: null, solveOut: null, canvas: null, dims: null, betaSample: null, labels: [], lastSel: 0 };
    function ensureSlide4() {
        S4.trainBtn = document.getElementById('trainBtn');
        S4.predictBtn = document.getElementById('predictBtn');
        S4.solveOut = document.getElementById('solveOut');
        S4.canvas = document.getElementById('betaCanvas');

        // Guardrail: disable Predict until trained
        if (S4.predictBtn) {
            S4.predictBtn.disabled = true;
            S4.predictBtn.title = 'Train first to enable prediction';
        }

        addHelper('slide4', {
            title: 'Try this:',
            steps: [
                'Click “Train on snippet” — we solve β using the current hidden size.',
                'Pick a row (Slide 2 dropdown) and click “Predict selected row”.',
                'Read the output: predicted label vs. ground truth and per-class probabilities.'
            ],
            blurb: 'After training, inference is quick: Hx = g(x·W + b) then logits = Hx · β.'
        });

        // rows getter (works even if you never visited slide 2)
        const getRows = () => {
            if (S2.dataRows?.length) return S2.dataRows;
            if (dataRows?.length) return dataRows;
            dataRows = parseCSVMini(CSV_SNIPPET);
            S2.dataRows = dataRows;
            return dataRows;
        };
        const getSelectedIndex = () => {
            const sel = S2.rowSelect || document.getElementById('rowSelect');
            return +(sel?.value ?? 0);
        };

        S4.trainBtn.onclick = () => {
            const rows = getRows();
            if (!rows.length) { alert('No training rows available'); return; }
            const hidden = S3.hiddenSize ? +S3.hiddenSize.value : 32;
            post('train', { rows: rows.map(r => ({ y: r.cls, text: r.text })), hidden });
        };
        S4.predictBtn.onclick = () => {
            if (S4.predictBtn.disabled) return; // guardrail
            const rows = getRows();
            if (!rows.length) { alert('No rows to predict'); return; }
            const i = getSelectedIndex();
            S4.lastSel = i;
            post('predict', { text: rows[i].text });
        };

        // If user trained before opening slide 4, auto-enable Predict now.
        if (S4.dims && S4.predictBtn) { S4.predictBtn.disabled = false; S4.predictBtn.title = ''; }
    }

    function onTrained({ dims, betaSample, note, labels }) {
        uiBasisFrozen = true;
        S4.dims = dims; S4.betaSample = betaSample;
        S4.labels = labels || [];
        if (S4.predictBtn) { S4.predictBtn.disabled = false; S4.predictBtn.title = ''; }

        S4.solveOut.textContent = `Solved β with pseudo-inverse${note ? ` (${note})` : ''}\n` +
            `H shape: ${dims.H_rows}×${dims.H_cols},  Y shape: ${dims.Y_rows}×${dims.Y_cols}\n` +
            `β shape: ${dims.B_rows}×${dims.B_cols}\n` +
            `β (8×8 sample):\n${betaSample}`;
        if ((S4.labels || []).length <= 1) {
            S4.solveOut.textContent += `\n\nNote: training data contained a SINGLE class (${S4.labels[0]}). Predictions will always be that class. Add at least one row from another class to demo multi-class.`;
        }
        drawBeta();
        drawEncode(); // update slide2 legend ("basis: trained")
    }

    function onPredicted({ pred, scores, labels }) {
        const probs = softmax(scores || []);
        const lbls = labels || S4.labels || [];
        const predName = (pred != null) ? (AG_LABELS[pred] || String(pred)) : '—';
        const rows = S2.dataRows || dataRows || [];
        const truthId = rows[S4.lastSel]?.cls ?? null;
        const truthName = (truthId != null) ? (AG_LABELS[truthId] || String(truthId)) : 'N/A';
        const verdict = (truthId == null) ? 'truth unknown' : (pred === truthId ? '✓ correct' : '✗ incorrect');

        const probText = (lbls.length === probs.length && probs.length > 0)
            ? lbls.map((lab, i) => `${lab}(${AG_LABELS[lab] || ''}): ${probs[i].toFixed(2)}`).join('  |  ')
            : `[${probs.map(p => p.toFixed(2)).join(', ')}]`;

        S4.solveOut.textContent += `\n\nPredicted: ${predName} (${pred})  |  Truth: ${truthName} (${truthId})  |  ${verdict}\n` +
            `Probabilities: ${probText}`;
    }

    function drawBeta() {
        if (!S4.canvas) return;
        const c = S4.canvas, dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
        c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
        const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!S4.dims) { g.fillStyle = '#93a9e8'; g.fillText('Train first to visualize β.', 12, 22); return; }
        const rows = Math.min(24, S4.dims.B_rows), cols = Math.min(24, S4.dims.B_cols);
        const cw = Math.floor((W - 20) / cols), ch = Math.floor((H - 20) / rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const phase = (i * 13 + j * 7) % 100, a = 0.25 + 0.7 * (phase / 100);
                g.fillStyle = `hsla(${180 + (j * 6) % 60},100%,60%,${a})`;
                g.fillRect(10 + j * cw, 10 + i * ch, cw - 2, ch - 2);
            }
        }
    }

    /* ---------------- Helpers shared by slides ---------------- */
    function addHelper(slideId, { title, steps, blurb }) {
        const root = document.getElementById(slideId);
        if (!root) return;
        // Prefer the right panel if present, else append to first panel
        const rightPanel = root.querySelector('.right .panel') || root.querySelector('.panel');
        if (!rightPanel || rightPanel.querySelector('.helper')) return; // idempotent
        const box = document.createElement('div');
        box.className = 'helper';
        box.innerHTML = `
      <h3>${title}</h3>
      <ul>${steps.map(s => `<li>${s}</li>`).join('')}</ul>
      <p class="note" style="margin:.2rem 0 0 0">${blurb}</p>
    `;
        rightPanel.appendChild(box);
    }
});
