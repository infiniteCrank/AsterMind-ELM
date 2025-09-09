/* =========================================================
   elm-demo.js — UI/App script for the ELM primer deck
   Updated to add:
   - New slides (huang, grid, gps, why, compare)
   - Mini-map (stage chips + progress bar)
   - Speaker notes toggle (global, persists across slides)
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {
    /* ---------------- Hero parallax (kept light) ---------------- */
    (function initHeroImages() {
        const heroes = Array.from(document.querySelectorAll('.slide > .hero'));
        if (!heroes.length) return;

        const rafState = { ticking: false };
        function applyParallax() {
            rafState.ticking = false;
            const vh = window.innerHeight || 800;
            heroes.forEach(el => {
                const r = el.getBoundingClientRect();
                const h = r.height || el.offsetHeight || 240;
                const start = -h;
                const end = vh;
                const t = Math.min(1, Math.max(0, (r.top - start) / (end - start)));
                const px = Math.round((t - 0.5) * 24); // -12..+12
                el.style.backgroundPosition = `center calc(50% ${px >= 0 ? '+' : ''}${px}px)`;
            });
        }
        function onScroll() {
            if (!rafState.ticking) {
                rafState.ticking = true;
                requestAnimationFrame(applyParallax);
            }
        }
        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll);
        requestAnimationFrame(applyParallax);
    })();

    /* ---------------- Slide registry + nav ---------------- */
    const slideNodes = Array.from(document.querySelectorAll('section.slide'));
    const slides = slideNodes.map((node) => ({ id: node.id, node, stage: node.dataset.stage || 'misc' }));
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    // --- Make mini-map stick directly below the sticky header ---
    const headerEl = document.querySelector('header');
    function setMinimapTop() {
        const h = headerEl ? headerEl.offsetHeight : 0;
        document.documentElement.style.setProperty('--minimap-top', h + 'px');
    }
    // Set once and on resize (covers responsive layout/font changes)
    setMinimapTop();
    window.addEventListener('resize', setMinimapTop);

    const slideLabel = document.getElementById('slideLabel');
    const footerBar = document.querySelector('.footerBar');
    const notesToggle = document.getElementById('notesToggle');
    const progressBar = document.getElementById('progressBar');

    // Mini-map chips
    const stageChips = Array.from(document.querySelectorAll('[data-stagechip]'));

    // Build numbered nav buttons
    if (footerBar) {
        footerBar.innerHTML = '';
        slides.forEach((_, i) => {
            const b = document.createElement('button');
            b.textContent = String(i + 1);
            b.onclick = () => show(i);
            footerBar.appendChild(b);
        });
    }

    // keyboard arrows
    window.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') prevBtn.click();
        if (e.key === 'ArrowRight') nextBtn.click();
        if (e.key.toLowerCase() === 'n') notesToggle?.click();
    });

    // Notes toggle: persist in session
    const NOTES_KEY = 'elm_show_notes';
    const initialNotes = sessionStorage.getItem(NOTES_KEY);
    if (initialNotes === '1') document.body.classList.add('show-notes');
    notesToggle?.addEventListener('click', () => {
        document.body.classList.toggle('show-notes');
        sessionStorage.setItem(NOTES_KEY, document.body.classList.contains('show-notes') ? '1' : '0');
    });

    let idx = 0;

    function updateMinimap(stage, index) {
        // Highlight current stage chip
        stageChips.forEach(ch => {
            const s = ch.getAttribute('data-stagechip');
            ch.classList.toggle('active', s === stage);
        });
        // Update progress bar (based on slide index)
        const pct = Math.max(0, Math.min(100, Math.round(((index + 1) / slides.length) * 100)));
        if (progressBar) progressBar.style.width = pct + '%';
    }

    function show(i) {
        idx = Math.max(0, Math.min(slides.length - 1, i));
        slides.forEach((s, j) => s.node.hidden = j !== idx);
        prevBtn.disabled = idx === 0;
        nextBtn.disabled = idx === slides.length - 1;
        slideLabel.textContent = `Slide ${idx + 1} / ${slides.length}`;
        location.hash = `#${idx + 1}`;

        const { id, stage } = slides[idx];
        updateMinimap(stage, idx);

        // init on demand
        if (id === 'slide1') ensureNeuron();
        if (id === 'slide2') ensureVectorize();
        if (id === 'slideBP') ensureBackprop();
        if (id === 'slideHidden') ensureHidden();
        if (id === 'slide4') ensureELMTrain();
        if (id === 'slidePred') ensurePredict();
        if (id === 'slideIntroNeuron') ensureIceCone();
    }

    prevBtn.onclick = () => show(idx - 1);
    nextBtn.onclick = () => show(idx + 1);

    // Delay initial slide navigation until after all declarations have run.
    const navigateInitialSlide = () => {
        if (location.hash) {
            const n = parseInt(location.hash.replace('#', ''), 10);
            show(Number.isFinite(n) ? n - 1 : 0);
        } else {
            show(0);
        }
    };
    requestAnimationFrame(navigateInitialSlide);


    /* ---------------- Worker wiring ---------------- */
    const workerStatus = document.getElementById('workerStatus');
    const worker = new Worker('./elm-worker.js'); // same-origin served
    worker.onerror = (err) => {
        if (workerStatus) workerStatus.textContent = 'worker failed (serve via http://localhost)';
        console.error(err);
    };
    const post = (type, payload = {}) => worker.postMessage({ type, payload });

    // state shared across slides
    let uiBasisFrozen = false;
    const AG_LABELS = { 1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech' };

    // datasets (tiny)
    const CSV_SNIPPET = `Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
3,Carlyle Looks Toward Commercial Aerospace (Reuters),"Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market."
1,UN Council Weighs Ceasefire Proposal,"Leaders from several nations met to discuss a draft resolution aimed at de-escalation in the region."
4,Chip Startup Unveils Faster AI Accelerator,"The company claims a 2× speedup on transformer inference with a new memory layout."
2,Local Club Wins Championship Final,"Fans celebrated after the underdogs clinched the title with a late goal."`;

    function parseCSVMini(s) {
        const lines = s.trim().split(/\r?\n/); lines.shift();
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
    const dataRows = parseCSVMini(CSV_SNIPPET);

    function softmax(arr) {
        if (!arr || !arr.length) return [];
        const m = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - m));
        const s = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / s);
    }
    const fmtVal = (v) => {
        // Always show 3 decimals to highlight differences and avoid constant "1.00"
        if (!Number.isFinite(v)) return '0';
        const a = Math.abs(v);
        if (a === 0) return '0';
        if (a >= 1e3 || a < 1e-3) return v.toExponential(2);
        return v.toFixed(3);
    };

    /* ---------------- Slide: Neuron demo ---------------- */

    const S1 = { act: 'relu', rafId: null, canvas: null };
    function ensureNeuron() {
        if (S1.inited) return;
        S1.inited = true;
        S1.canvas = document.getElementById('neuronCanvas');
        const actSelect = document.getElementById('actSelect');
        const wRange = document.getElementById('wRange');
        const bRange = document.getElementById('bRange');
        const wVal = document.getElementById('wVal');
        const bVal = document.getElementById('bVal');
        const ampVal = document.getElementById('ampVal');
        const ampBar = document.getElementById('ampBar');
        const postAmpVal = document.getElementById('postAmpVal');
        const postAmpBar = document.getElementById('postAmpBar');


        const actFn = (z) => {
            switch (S1.act) {
                case 'relu': return Math.max(0, z);
                case 'leakyRelu': return z >= 0 ? z : 0.01 * z;
                case 'sigmoid': return 1 / (1 + Math.exp(-z));
                case 'tanh': return Math.tanh(z);
                default: return z;
            }
        };

        function updateAmplitude() {
            const w = +wRange.value;
            const b = +bRange.value;
            // For z = w·x + b over x ∈ [-3, 3], the extrema are at the endpoints.
            const amp = Math.max(Math.abs(w * 3 + b), Math.abs(w * -3 + b));
            ampVal.textContent = amp.toFixed(2);
            const maxAmp = 20;            // max possible amplitude for w,b ∈ [-5,5]
            const pct = Math.min(100, (amp / maxAmp) * 100);
            ampBar.style.width = pct + '%';
        }

        function updatePostAmplitude() {
            const w = +wRange.value;
            const b = +bRange.value;
            let maxAbsOutput = 0;

            // Sample across the x-range [-3, 3] to find the largest |g(w·x + b)|
            for (let i = 0; i <= 300; i++) {
                const x = -3 + (i / 300) * 6;
                const z = w * x + b;
                const y = actFn(z);
                maxAbsOutput = Math.max(maxAbsOutput, Math.abs(y));
            }

            // Display numeric amplitude
            postAmpVal.textContent = maxAbsOutput.toFixed(2);

            // Scale bar width relative to the maximum possible output for the current activation
            // For ReLU/LeakyReLU the output can reach up to 20 (given w,b ranges),
            // but sigmoid/tanh are bounded between 0..1 and -1..1 respectively.
            let maxPossible = (S1.act === 'sigmoid' || S1.act === 'tanh') ? 1 : 20;
            const pct = Math.min(100, (maxAbsOutput / maxPossible) * 100);
            postAmpBar.style.width = pct + '%';
        }

        function loop() {
            if (!S1.canvas) return;
            const c = S1.canvas, g = c.getContext('2d'), dpr = devicePixelRatio || 1;
            const W = c.clientWidth, H = c.clientHeight;
            c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.clearRect(0, 0, W, H);

            // axes
            g.strokeStyle = '#3857a8'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(10, H - 20); g.lineTo(W - 10, H - 20); g.stroke();
            g.beginPath(); g.moveTo(40, 10); g.lineTo(40, H - 10); g.stroke();

            const w = +wRange.value, b = +bRange.value;
            const toXY = (x, y) => {
                const xm = (x + 3) / 6, ym = (y - (-2)) / (2 - (-2));
                return [40 + xm * (W - 55), (H - 20) - ym * (H - 35)];
            };

            // curve
            g.strokeStyle = '#5ad1ff'; g.lineWidth = 2; g.beginPath();
            for (let i = 0; i <= 300; i++) {
                const x = -3 + (i / 300) * 6, y = actFn(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
            }
            g.stroke();

            // moving dots
            const now = performance.now() / 1000;
            for (let k = 0; k < 7; k++) {
                const x = -3 + ((now * 0.6 + k / 7) % 1) * 6, y = actFn(w * x + b);
                const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
                g.fillStyle = '#6ee7a2'; g.beginPath(); g.arc(px, py, 3.2, 0, Math.PI * 2); g.fill();
            }
            S1.rafId = requestAnimationFrame(loop);
        }

        actSelect.addEventListener('change', () => { S1.act = actSelect.value; updatePostAmplitude(); });
        [wRange, bRange].forEach(r => {
            r.addEventListener('input', () => {
                wVal.textContent = (+wRange.value).toFixed(2);
                bVal.textContent = (+bRange.value).toFixed(2);
                updateAmplitude();          // update amplitude gauge
                updatePostAmplitude();   // new post‑activation function
            });
        });
        wVal.textContent = (+wRange.value).toFixed(2);
        bVal.textContent = (+bRange.value).toFixed(2);
        updateAmplitude();
        updatePostAmplitude();

        // start animation
        if (S1.rafId) cancelAnimationFrame(S1.rafId);
        loop();
    }

    /* ---------------- Slide: Vectorization ---------------- */
    const S2 = { lastEncoded: null, grid: null, featureLimit: 128, lastMethod: 'tfidf' };

    function ensureVectorize() {
        if (S2.inited) return;
        S2.inited = true;

        const rowSelect = document.getElementById('rowSelect');
        const encodeBtn = document.getElementById('encodeBtn');
        const encodeCanvas = document.getElementById('encodeCanvas');
        const tokensOut = document.getElementById('tokensOut');
        const encodingSelect = document.getElementById('encodingSelect');
        const featureLimit = document.getElementById('featureLimit');
        const featureLimitVal = document.getElementById('featureLimitVal');

        // populate dropdown
        rowSelect.innerHTML = '';
        for (let i = 0; i < dataRows.length; i++) {
            const o = document.createElement('option');
            o.value = String(i);
            o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
            rowSelect.appendChild(o);
        }

        // controls
        S2.featureLimit = +featureLimit.value;
        featureLimit.addEventListener('input', () => {
            S2.featureLimit = +featureLimit.value;
            featureLimitVal.textContent = featureLimit.value;
            drawEncode();
        });
        encodingSelect.addEventListener('change', () => {
            S2.lastMethod = encodingSelect.value;
        });

        // request encoding; IMPORTANT: pass a small corpus so TF-IDF has real IDF
        encodeBtn.onclick = () => {
            const i = +rowSelect.value || 0;
            const method = encodingSelect.value || 'tfidf';
            const corpus = dataRows.map(r => r.text);     // gives meaningful IDF
            post('encode', { text: dataRows[i].text, method, corpus });
        };

        // tooltip element
        let tipEl = document.querySelector('#slide2 .encode-tip');
        if (!tipEl) {
            tipEl = document.createElement('div');
            tipEl.className = 'encode-tip';
            encodeCanvas.parentElement.appendChild(tipEl); // parent has position:relative
        }

        // tooltip behavior
        encodeCanvas.addEventListener('mousemove', (e) => {
            if (!S2.grid || !S2.lastEncoded) { tipEl.style.display = 'none'; return; }
            const { x0, y0, cols, rows, cw, ch, n } = S2.grid;
            const rect = encodeCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;

            const inside = cw > 0 && ch > 0 && x >= x0 && y >= y0 && x < x0 + cols * cw && y < y0 + rows * ch;
            if (!inside) { tipEl.style.display = 'none'; return; }

            const col = Math.floor((x - x0) / cw);
            const row = Math.floor((y - y0) / ch);
            const k = row * cols + col;
            if (k < 0 || k >= n) { tipEl.style.display = 'none'; return; }

            const v = S2.lastEncoded.vector;
            const names = S2.lastEncoded.featureNames || [];
            const token = names[k] || null;
            const val = Number.isFinite(v[k]) ? v[k] : 0;

            // small centered window preview around k for quick context
            const WN = 6; // ±3
            const start = Math.max(0, k - 3), end = Math.min(v.length, k + 4);
            const preview = v.slice(start, end).map(x => fmtVal(x)).join(', ');
            const dotsL = start > 0 ? '… ' : '';
            const dotsR = end < v.length ? ' …' : '';

            tipEl.textContent = `${token ? `#${k} “${token}”` : `feature #${k}`} = ${fmtVal(val)}`;
            tipEl.style.display = 'block';
            tipEl.style.transform = 'none';
            tipEl.style.zIndex = '10';

            const tbox = tipEl.getBoundingClientRect();
            const pad = 6;
            const left = Math.min(rect.width - tbox.width - pad, Math.max(pad, x + 10));
            const top = Math.min(rect.height - tbox.height - pad, Math.max(pad, y - 28));
            tipEl.style.left = `${left}px`;
            tipEl.style.top = `${top}px`;
        });
        encodeCanvas.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });

        // draw heatmap (with numbers when space permits)
        function drawEncode() {
            const c = encodeCanvas, dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
            c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
            const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.clearRect(0, 0, W, H);

            if (!S2.lastEncoded) {
                g.fillStyle = '#93a9e8';
                g.fillText('Click “Encode text” to preview the input vector.', 12, 22);
                S2.grid = null; return;
            }

            const v = S2.lastEncoded.vector;
            const n = Math.min(S2.featureLimit || 128, v.length);
            const cols = Math.ceil(Math.sqrt(n));
            const rows = Math.ceil(n / cols);
            const m = 10, x0 = m, y0 = m;
            const cw = Math.floor((W - 2 * m) / cols);
            const ch = Math.floor((H - 2 * m) / rows);

            const vSub = v.slice(0, n);
            const maxAbs = Math.max(1e-6, ...vSub.map(x => Math.abs(x)));
            const nnz = vSub.reduce((a, x) => a + (Math.abs(x) > 1e-12 ? 1 : 0), 0);
            const l2 = Math.hypot(...vSub);

            S2.grid = { x0, y0, cols, rows, cw, ch, n };

            for (let r = 0, k = 0; r < rows; r++) {
                for (let ccol = 0; ccol < cols; ccol++, k++) {
                    if (k >= n) break;
                    const val = vSub[k];
                    const alpha = Math.min(1, Math.abs(val) / maxAbs);
                    const hue = val >= 0 ? 200 : 0; // blue or red
                    const X = x0 + ccol * cw, Y = y0 + r * ch;

                    // cell color
                    const bg = `hsla(${hue},90%,60%,${0.15 + 0.85 * alpha})`;
                    g.fillStyle = bg; g.fillRect(X, Y, cw - 2, ch - 2);

                    // number overlay (only if space permits)
                    if (cw >= 56 && ch >= 36) {
                        g.save();
                        g.fillStyle = 'rgba(255,255,255,0.95)';
                        g.font = `600 ${Math.min(18, Math.floor(ch * 0.42))}px ui-sans-serif,system-ui`;
                        g.textAlign = 'center'; g.textBaseline = 'middle';
                        g.fillText(fmtVal(val), X + (cw - 2) / 2, Y + (ch - 2) / 2);
                        g.restore();
                    }
                }
            }
            // grid lines
            g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
            for (let ccol = 0; ccol <= cols; ccol++) { g.beginPath(); g.moveTo(x0 + ccol * cw, y0); g.lineTo(x0 + ccol * cw, y0 + rows * ch); g.stroke(); }
            for (let rr = 0; rr <= rows; rr++) { g.beginPath(); g.moveTo(x0, y0 + rr * ch); g.lineTo(x0 + cols * cw, y0 + rr * ch); g.stroke(); }

            g.fillStyle = '#a7b8e8';
            g.fillText(`features 0..${n - 1} — nnz: ${nnz}, ‖v‖₂: ${l2.toFixed(2)}, max |v|: ${maxAbs.toFixed(2)}  |  basis: ${uiBasisFrozen ? 'trained' : 'isolated'}`, 12, H - 12);
        }

        // worker events for this slide
        worker.addEventListener('message', (e) => {
            const { type, payload } = e.data || {};
            if (type === 'encoded') {
                S2.lastEncoded = payload;
                const chosen = payload.methodUsed || encodingSelect.value;
                const label =
                    chosen === 'tfidf' ? 'TF-IDF' :
                        chosen === 'bow' ? 'Bag-of-Words' : 'Isolated';

                tokensOut.textContent =
                    `${label} tokens (top):\n${payload.tokens.slice(0, 25).join(' ')}\n\nvector length: ${payload.vector.length}`;
                drawEncode();
            }
        });
    }

    /* ---------------- Slide: Backprop demo ---------------- */
    const BP = { canvas: null, lrRange: null, lrVal: null, raf: null, t0: 0, state: null };
    function ensureBackprop() {
        if (BP.inited) return;
        BP.inited = true;

        BP.canvas = document.getElementById('bpCanvas');
        BP.lrRange = document.getElementById('bpLR');
        BP.lrVal = document.getElementById('bpLRVal');
        BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2);

        // Convergence + auto-repeat controls
        const CONV_THRESH = 0.06;   // consider “converged” below this loss
        const CONV_FRAMES = 90;     // must stay under threshold this many frames
        const RESET_HOLD_MS = 400;  // short pause so the reset is noticeable

        function resetBackpropDemo() {
            const H = 24, W = 36; // visual grid size
            BP.state = {
                weights: Array.from({ length: H }, () =>
                    Array.from({ length: W }, () => (Math.random() * 2 - 1) * 0.6)
                ),
                loss: 1.0,
                noise: 0.0
            };
            BP.points = [];
            BP.convergedFrames = 0;
            BP.holdUntil = performance.now() + RESET_HOLD_MS; // small breathing room
        }
        resetBackpropDemo();

        BP.lrRange.addEventListener('input', () => {
            BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2);
        });

        function step(dt) {
            // respect brief hold between restarts
            if (performance.now() < (BP.holdUntil || 0)) return;

            const lr = +BP.lrRange.value;
            const kDecay = 0.9 + 0.08 * Math.exp(-dt * 0.001); // slow approach
            BP.state.loss = Math.max(0.02, BP.state.loss * (kDecay - 0.08 * lr * 0.1));
            BP.state.noise = 0.02 + lr * 0.25;

            const H = BP.state.weights.length;
            const W = BP.state.weights[0].length;
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    const w = BP.state.weights[i][j];
                    const grad = w; // pretend grad ~ w
                    const noise = (Math.random() * 2 - 1) * BP.state.noise * 0.02;
                    BP.state.weights[i][j] = w - lr * 0.02 * grad + noise;
                }
            }

            // detect convergence & restart
            if (BP.state.loss <= CONV_THRESH) {
                BP.convergedFrames = (BP.convergedFrames || 0) + 1;
                if (BP.convergedFrames >= CONV_FRAMES) resetBackpropDemo();
            } else {
                BP.convergedFrames = 0;
            }
        }

        function render() {
            const c = BP.canvas, g = c.getContext('2d'), dpr = devicePixelRatio || 1;
            const Wc = c.clientWidth, Hc = c.clientHeight;
            c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.clearRect(0, 0, Wc, Hc);

            const pad = 10, gridW = Math.floor(Wc * 0.66);

            // ✅ define H and W from current state
            const H = BP.state.weights.length;
            const W = BP.state.weights[0].length;

            const cellW = Math.floor((gridW - 2 * pad) / W);
            const cellH = Math.floor((Hc - 2 * pad) / H);

            // heatmap of weights
            let vmax = 1e-6;
            for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
                vmax = Math.max(vmax, Math.abs(BP.state.weights[i][j]));
            }
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    const val = BP.state.weights[i][j];
                    const a = Math.min(1, Math.abs(val) / vmax);
                    const hue = val >= 0 ? 200 : 0;
                    g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * a})`;
                    g.fillRect(pad + j * cellW, pad + i * cellH, cellW - 1, cellH - 1);
                }
            }
            g.fillStyle = '#a7b8e8';
            g.fillText('Hidden weights (changing each step)', pad, Hc - 8);

            // loss chart
            const x0 = gridW + 20, y0 = pad, w = Wc - x0 - pad, h = Hc - 2 * pad;
            g.strokeStyle = '#3857a8'; g.strokeRect(x0, y0, w, h);
            BP.points = BP.points || [];
            BP.points.push(BP.state.loss);
            if (BP.points.length > 240) BP.points.shift();
            const minL = Math.min(...BP.points, 0), maxL = Math.max(...BP.points, 1);
            g.strokeStyle = '#6ee7a2'; g.lineWidth = 2; g.beginPath();
            BP.points.forEach((L, i) => {
                const px = x0 + (i / (240 - 1)) * w;
                const py = y0 + h - ((L - minL) / (maxL - minL || 1)) * h;
                if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
            });
            g.stroke();

            g.fillStyle = '#a7b8e8';
            g.fillText(`loss ~ ${BP.state.loss.toFixed(3)}  (learning rate ${(+BP.lrRange.value).toFixed(2)})`, x0 + 6, y0 + 16);
        }

        function loop(ts) {
            if (!BP.t0) BP.t0 = ts;
            const dt = ts - BP.t0; BP.t0 = ts;
            step(dt);
            render();
            BP.raf = requestAnimationFrame(loop);
        }
        BP.raf && cancelAnimationFrame(BP.raf);
        BP.raf = requestAnimationFrame(loop);
    }

    /* ---------------- Hidden Layer: shuffle & project ---------------- */
    const HL = {
        canvas: null, hiddenSize: null, hiddenSizeVal: null,
        shuffleBtn: null, previewBtn: null, tipEl: null,
        W: null, b: null, Hx: null, Z: null, gridW: null, gridBars: null
    };

    function ensureHidden() {
        if (HL.inited) return;
        HL.inited = true;

        HL.canvas = document.getElementById('hiddenCanvas');
        HL.hiddenSize = document.getElementById('hiddenSizeHL');
        HL.hiddenSizeVal = document.getElementById('hiddenSizeHLVal');
        HL.shuffleBtn = document.getElementById('shuffleBtnHL');
        HL.previewBtn = document.getElementById('previewHBtnHL');
        HL.tipEl = document.querySelector('#slideHidden .hidden-tip');

        // Show value live
        HL.hiddenSize.addEventListener('input', () => {
            HL.hiddenSizeVal.textContent = HL.hiddenSize.value;
        });
        HL.hiddenSizeVal.textContent = HL.hiddenSize.value;

        // Shuffle (reseed W,b)
        HL.shuffleBtn.onclick = () => {
            const inputDim = (S2.lastEncoded?.vector?.length || 512);
            post('init_hidden', { inputDim, hidden: +HL.hiddenSize.value });
        };

        // Project H = g(Wx+b) for current encoded row
        HL.previewBtn.onclick = () => {
            if (!S2.lastEncoded) { alert('Encode a row on the Vectorization slide first'); return; }
            post('project_hidden', { x: S2.lastEncoded.vector });
        };

        // Tooltip over heatmap/bars
        HL.canvas.addEventListener('mousemove', (e) => {
            if (!HL.tipEl) return;
            const rect = HL.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;

            // over W heatmap?
            if (HL.gridW) {
                const { x: gx, y: gy, cols, rows, cellW, cellH } = HL.gridW;
                const j = Math.floor((x - gx) / cellW), i = Math.floor((y - gy) / cellH);
                if (i >= 0 && j >= 0 && i < rows && j < cols && HL.W && HL.W[i] && Number.isFinite(HL.W[i][j])) {
                    const val = HL.W[i][j];
                    HL.tipEl.textContent = `W[${i},${j}] = ${fmtVal(val)}`;
                    HL.tipEl.style.display = 'block';
                    HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width - 120, x + 8))}px, ${Math.max(0, y - 28)}px)`;
                    return;
                }
            }
            // over H bars?
            if (HL.gridBars) {
                const { x: bx, y: by, n, eachH } = HL.gridBars;
                if (x >= bx) {
                    const i = Math.floor((y - by) / eachH);
                    if (i >= 0 && i < n && HL.Hx) {
                        const h = HL.Hx[i];
                        const z = HL.Z ? HL.Z[i] : null;
                        HL.tipEl.textContent = z == null ? `H[${i}] = ${fmtVal(h)}` : `H[${i}] = g(z) = ${fmtVal(h)}  (z=${fmtVal(z)})`;
                        HL.tipEl.style.display = 'block';
                        HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width - 160, x + 8))}px, ${Math.max(0, y - 28)}px)`;
                        return;
                    }
                }
            }
            HL.tipEl.style.display = 'none';
        });
        HL.canvas.addEventListener('mouseleave', () => { if (HL.tipEl) HL.tipEl.style.display = 'none'; });

        // Draw once (empty state)
        drawHidden();
    }

    function onHiddenInit(payload) {
        HL.W = payload.W; HL.b = payload.b; HL.Hx = null; HL.Z = null;
        const WPreview = document.getElementById('WPreview');
        if (WPreview && HL.W?.length) {
            const sample = (M, r = 8, c = 8) => {
                const R = Math.min(r, M.length), C = Math.min(c, M[0]?.length || 0);
                let s = ''; for (let i = 0; i < R; i++) s += M[i].slice(0, C).map(v => (Math.abs(v) < 1e-3 ? '0.000' : (+v).toFixed(3))).join(' ') + '\n';
                return s;
            };
            WPreview.textContent = `W: ${HL.W.length}x${HL.W[0]?.length || 0}  b: ${HL.b.length}  (8×8 sample)\n` + sample(HL.W);
        }
        drawHidden();
    }

    function onHiddenProject(payload) {
        HL.Hx = payload.Hx || null;
        HL.Z = payload.Z || null;
        drawHidden();
    }

    function drawHidden() {
        const c = HL.canvas;
        if (!c) return;
        const g = c.getContext('2d'), dpr = devicePixelRatio || 1;
        const Wc = c.clientWidth, Hc = c.clientHeight;
        c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, Wc, Hc);

        const pad = 10, ww = Math.floor(Wc * 0.66);
        const xHeat = pad, yHeat = pad, xBars = ww + pad, yBars = pad;

        // Heatmap W
        if (!HL.W) {
            g.fillStyle = '#93a9e8';
            g.fillText('Click “Reseed hidden” to initialize W,b', 12, 22);
            HL.gridW = HL.gridBars = null;
            return;
        }

        const rows = HL.W.length, cols = HL.W[0].length;
        // downsample to ~256×256 for perf
        const MAXD = 256;
        const dr = Math.max(1, Math.ceil(rows / MAXD));
        const dc = Math.max(1, Math.ceil(cols / MAXD));
        const dsRows = Math.ceil(rows / dr);
        const dsCols = Math.ceil(cols / dc);

        let vmax = 1e-6;
        for (let i = 0; i < rows; i += dr)
            for (let j = 0; j < cols; j += dc)
                vmax = Math.max(vmax, Math.abs(HL.W[i][j]));

        const cellW = Math.max(1, Math.floor((ww - 2 * pad) / dsCols));
        const cellH = Math.max(1, Math.floor((Hc - 2 * pad) / dsRows));

        for (let i = 0, ri = 0; i < rows; i += dr, ri++) {
            for (let j = 0, rj = 0; j < cols; j += dc, rj++) {
                const val = HL.W[i][j];
                const a = Math.min(1, Math.abs(val) / vmax);
                const hue = val >= 0 ? 200 : 0;
                g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * a})`;
                g.fillRect(xHeat + rj * cellW, yHeat + ri * cellH, cellW, cellH);
            }
        }
        g.fillStyle = '#a7b8e8';
        g.fillText('W (hidden × input)', xHeat, Hc - 8);
        HL.gridW = { x: xHeat, y: yHeat, cols: dsCols, rows: dsRows, cellW, cellH };

        // Bars for Hx
        if (HL.Hx && HL.Hx.length) {
            const n = HL.Hx.length;
            const barW = (Wc - xBars - pad), eachH = Math.max(1, Math.floor((Hc - 2 * pad) / n));
            const absmax = Math.max(1e-6, ...HL.Hx.map(x => Math.abs(x)));
            for (let i = 0; i < n; i++) {
                const val = HL.Hx[i], frac = Math.min(1, Math.abs(val) / absmax), len = Math.floor(frac * barW);
                g.fillStyle = val >= 0 ? '#6ee7a2' : '#fb7185';
                g.fillRect(xBars, yBars + i * eachH, len, Math.max(1, eachH - 2));
            }
            g.fillStyle = '#a7b8e8';
            g.fillText('Hx = g(Wx + b)', xBars, Hc - 8);
            HL.gridBars = { x: xBars, y: yBars, n, eachH, barW };
        } else {
            g.fillStyle = '#93a9e8';
            g.fillText('Encode a row (Vectorization), then click “Project H”.', xBars, 22);
            HL.gridBars = null;
        }
    }

    /* ---------------- Slide 11: ELM Train (one-shot) ---------------- */
    const S4 = { canvas: null, betaVis: null, labels: [], dims: null };
    function ensureELMTrain() {
        if (S4.inited) return;
        S4.inited = true;

        const trainBtn = document.getElementById('trainBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const resetBtn = document.getElementById('resetBtn');
        const hiddenSize = document.getElementById('hiddenSize');
        const hiddenSizeVal = document.getElementById('hiddenSizeVal');
        const solveOut = document.getElementById('solveOut');
        S4.canvas = document.getElementById('betaCanvas');

        hiddenSize.addEventListener('input', () => hiddenSizeVal.textContent = hiddenSize.value);
        hiddenSizeVal.textContent = hiddenSize.value;

        trainBtn.onclick = () => {
            post('train', {
                rows: dataRows.map(r => ({ y: r.cls, text: r.text })),
                hidden: +hiddenSize.value
            });
            solveOut.textContent = 'Training… (freezing basis, solving β)…';
        };

        downloadBtn.onclick = () => post('export_model');

        resetBtn.onclick = () => {
            post('reset');
            uiBasisFrozen = false;
            S4.betaVis = null; S4.labels = []; S4.dims = null;
            drawBeta();
            solveOut.textContent = 'Reset complete. Re-train to continue.';
            const predictBtn = document.getElementById('predictBtn');
            if (predictBtn) { predictBtn.disabled = true; predictBtn.title = 'Train first'; }
        };

        function drawBeta() {
            const c = S4.canvas, g = c.getContext('2d'), dpr = devicePixelRatio || 1;
            const Wc = c.clientWidth, Hc = c.clientHeight;
            c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.clearRect(0, 0, Wc, Hc);
            if (!S4.betaVis) {
                g.fillStyle = '#93a9e8'; g.fillText('Train first to visualize β.', 12, 22);
                return;
            }
            const B = S4.betaVis, R = B.length, C = B[0].length;
            const m = 10, cw = Math.floor((Wc - 2 * m) / C), ch = Math.floor((Hc - 2 * m) / R);
            let vmax = 1e-6;
            for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) vmax = Math.max(vmax, Math.abs(B[i][j]));
            for (let i = 0; i < R; i++) {
                for (let j = 0; j < C; j++) {
                    const v = B[i][j], a = Math.min(1, Math.abs(v) / vmax);
                    const hue = v >= 0 ? 200 : 0;
                    g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * a})`;
                    g.fillRect(m + j * cw, m + i * ch, cw - 1, ch - 1);
                }
            }
        }
        S4.drawBeta = drawBeta;
    }

    /* ---------------- Slide 12: Prediction ---------------- */
    const SP = {};
    function ensurePredict() {
        if (SP.inited) return;
        SP.inited = true;

        const predRowSelect = document.getElementById('predRowSelect');
        const predictBtn = document.getElementById('predictBtn');
        const predOut = document.getElementById('predOut');

        // populate
        predRowSelect.innerHTML = '';
        for (let i = 0; i < dataRows.length; i++) {
            const o = document.createElement('option');
            o.value = String(i);
            o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
            predRowSelect.appendChild(o);
        }

        predictBtn.onclick = () => {
            const i = +predRowSelect.value || 0;
            post('predict', { text: dataRows[i].text });
            predOut.textContent = 'Predicting…';
            SP.lastIndex = i;
        };
    }

    /* ---------------- Worker events (shared) ---------------- */
    worker.onmessage = (e) => {
        const { type, payload } = e.data || {};
        if (type === 'status') {
            if (workerStatus) workerStatus.textContent = payload;
            return;
        }
        if (type === 'trained') {
            uiBasisFrozen = true;

            // enable Predict button
            const predictBtn = document.getElementById('predictBtn');
            if (predictBtn) { predictBtn.disabled = false; predictBtn.title = ''; }

            const solveOut = document.getElementById('solveOut');
            const lines = [];
            lines.push('Trained ELM ✓');
            if (payload.note) lines.push(`note: ${payload.note}`);
            if (payload.dims) {
                const d = payload.dims;
                lines.push(`H: ${d.H_rows}×${d.H_cols}  Y: ${d.Y_rows}×${d.Y_cols}  β: ${d.B_rows}×${d.B_cols}`);
            }
            if (payload.betaSample) {
                lines.push('\nβ (8×8 sample):\n' + payload.betaSample);
            }
            solveOut.textContent = lines.join('\n');

            // store β visualization + labels
            S4.betaVis = payload.betaVis || null;
            S4.labels = payload.labels || [];
            S4.dims = payload.dims || null;
            if (S4.drawBeta) S4.drawBeta();

            return;
        }
        if (type === 'predicted') {
            const probs = softmax(payload.scores || []);
            const labels = payload.labels || [];
            const pred = payload.pred;
            const predOut = document.getElementById('predOut') || document.getElementById('solveOut');

            const i = SP.lastIndex ?? 0;
            const truthId = dataRows[i]?.cls ?? null;
            const truthName = truthId != null ? (AG_LABELS[truthId] || String(truthId)) : 'N/A';
            const predName = (pred != null) ? (AG_LABELS[pred] || String(pred)) : '—';
            const verdict = (truthId == null) ? 'truth unknown' : (pred === truthId ? '✓ correct' : '✗ incorrect');

            const probText = (labels.length === probs.length && probs.length > 0)
                ? labels.map((lab, k) => `${lab}(${AG_LABELS[lab] || ''}): ${probs[k].toFixed(2)}`).join('  |  ')
                : `[${probs.map(p => p.toFixed(2)).join(', ')}]`;

            predOut.textContent =
                `Predicted: ${predName} (${pred})  |  Truth: ${truthName} (${truthId})  |  ${verdict}\n` +
                `Probabilities: ${probText}`;
            return;
        }
        if (type === 'exported_model') {
            // download already triggered inside worker
            return;
        }
        if (type === 'encoded') {
            // handled by ensureVectorize as well
            return;
        }
        if (type === 'actCurve') {
            // could reflect available activations
            return;
        }
        if (type === 'hidden_init') { onHiddenInit(payload); return; }
        if (type === 'hidden_project') { onHiddenProject(payload); return; }
    };

    // bootstrap worker
    post('hello');
    post('list_activations');
});
