/* =========================================================
   elm-demo.js — UI/App script for the ELM primer deck
   Clean init order + stable minimap hooks
   - JSON slide meta (optional)
   - Cyberpunk minimap (left rail)
   - One-screen slides + per-slide sizing
   - Worker-driven demos
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* ---------------- 0) Globals & helpers ---------------- */
  let SLIDE_META = null;          // optional, from slides.json
  let slides = [];                // [{id,node,stage}]
  let idx = 0;                    // current slide index

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* ---------------- 1) Header / layout refs ---------------- */
  const prevBtn = $('#prevBtn');
  const nextBtn = $('#nextBtn');
  const slideLabel = $('#slideLabel');
  const footerBar = $('.footerBar');
  const notesToggle = $('#notesToggle');
  const progressBar = $('#progressBar');
  const headerEl = $('header');

  /* Keep minimap pinned under sticky header */
  function setMinimapTop() {
    const h = headerEl ? headerEl.offsetHeight : 0;
    document.documentElement.style.setProperty('--minimap-top', h + 'px');
  }
  setMinimapTop();
  window.addEventListener('resize', setMinimapTop);

  /* Notes toggle (persist in session) */
  const NOTES_KEY = 'elm_show_notes';
  if (sessionStorage.getItem(NOTES_KEY) === '1') document.body.classList.add('show-notes');
  notesToggle?.addEventListener('click', () => {
    document.body.classList.toggle('show-notes');
    sessionStorage.setItem(NOTES_KEY, document.body.classList.contains('show-notes') ? '1' : '0');
  });

  /* ---------------- 2) Hero parallax (lightweight) ---------------- */
  (function initHeroImages() {
    const heroes = $$('.slide > .hero');
    if (!heroes.length) return;
    const rafState = { ticking: false };
    function applyParallax() {
      rafState.ticking = false;
      const vh = window.innerHeight || 800;
      heroes.forEach(el => {
        const r = el.getBoundingClientRect();
        const h = r.height || el.offsetHeight || 240;
        const start = -h, end = vh;
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

  /* ---------------- 3) Slide registry + basic nav ---------------- */
  const slideNodes = $$('.slide');
  slides = slideNodes.map(node => ({
    id: node.id,
    node,
    stage: node.dataset.stage || 'misc'
  }));

  /* Build simple footer number nav */
  if (footerBar) {
    footerBar.innerHTML = '';
    slides.forEach((_, i) => {
      const b = document.createElement('button');
      b.textContent = String(i + 1);
      b.onclick = () => show(i);
      footerBar.appendChild(b);
    });
  }

  /* Keyboard nav */
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevBtn?.click();
    if (e.key === 'ArrowRight') nextBtn?.click();
    if (e.key.toLowerCase() === 'n') notesToggle?.click();
  });

  /* Stage chips (if any in header) */
  const stageChips = $$('[data-stagechip]');

  /* ---------------- 4) Slide Atlas (left rail / mobile drawer) ---------------- */
  const minimapState = {
    host: null,              // <nav id="minimap">
    items: [],               // [{btn, index}]
    drawerBackdrop: null,    // #atlasBackdrop
    burger: null,            // #atlasToggle (in header)
    closeBtn: null,          // #mmClose
    compactBtn: null         // #mmToggle
  };

  /* Build list items from slides (or SLIDE_META if present) */
  function buildMinimap() {
    minimapState.host = $('#minimap');            // we reuse the same id
    minimapState.drawerBackdrop = $('#atlasBackdrop');
    minimapState.burger = $('#atlasToggle');
    minimapState.closeBtn = $('#mmClose');
    minimapState.compactBtn = $('#mmToggle');

    if (!minimapState.host) return;

    // Resolve titles from SLIDE_META or from the DOM
    const meta = SLIDE_META || slides.map((s, i) => {
      // Try slide caption, then first H2, then stage/id fallback
      const cap = s.node.querySelector(':scope > .hero .caption')?.textContent?.trim();
      const h2 = s.node.querySelector('h2')?.textContent?.trim();
      const title = cap || h2 || s.id || `Slide ${i + 1}`;
      return { id: s.id, stage: s.stage, title };
    });

    // Build the list
    minimapState.host.innerHTML = '';
    minimapState.items = meta.map((m, i) => {
      const btn = document.createElement('button');
      btn.className = 'atlas-item';
      btn.setAttribute('type', 'button');
      btn.setAttribute('data-index', String(i));
      btn.setAttribute('aria-label', `Slide ${i + 1}: ${m.title}`);
      btn.innerHTML = `
      <span class="num">${String(i + 1).padStart(2, '0')}</span>
      <span class="ttl">${m.title}</span>
      <span class="stage">${m.stage || '—'}</span>
    `;
      btn.addEventListener('click', () => {
        show(i);
        // If we’re in the mobile drawer, close it after navigating
        if (document.body.classList.contains('atlas-open')) closeAtlasDrawer();
      });
      minimapState.host.appendChild(btn);
      return { btn, index: i };
    });

    // Wire compact toggle (desktop)
    const COMPACT_KEY = 'elm_atlas_compact';
    if (minimapState.compactBtn) {
      // initialize from storage
      if (localStorage.getItem(COMPACT_KEY) === '1') document.body.classList.add('atlas-compact');
      minimapState.compactBtn.title = 'Collapse / Expand';
      minimapState.compactBtn.onclick = () => {
        document.body.classList.toggle('atlas-compact');
        localStorage.setItem(COMPACT_KEY, document.body.classList.contains('atlas-compact') ? '1' : '0');
      };
    }

    // Wire drawer open/close (mobile)
    minimapState.burger?.addEventListener('click', openAtlasDrawer);
    minimapState.closeBtn?.addEventListener('click', closeAtlasDrawer);
    minimapState.drawerBackdrop?.addEventListener('click', closeAtlasDrawer);
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && document.body.classList.contains('atlas-open')) closeAtlasDrawer();
    });

    function openAtlasDrawer() {
      document.body.classList.add('atlas-open');
      minimapState.drawerBackdrop?.removeAttribute('hidden');
    }
    function closeAtlasDrawer() {
      document.body.classList.remove('atlas-open');
      minimapState.drawerBackdrop?.setAttribute('hidden', '');
    }
    // Expose for use in click handler above
    window.openAtlasDrawer = openAtlasDrawer;
    window.closeAtlasDrawer = closeAtlasDrawer;

    // If viewport becomes desktop, ensure drawer is closed
    const mql = window.matchMedia('(min-width: 981px)');
    mql.addEventListener?.('change', () => closeAtlasDrawer());
  }

  // Keyboard nav within the atlas list (focus required)
  minimapState.host?.addEventListener('keydown', (e) => {
    const current = minimapState.items.findIndex(({ btn }) => btn.classList.contains('active'));
    if (e.key === 'ArrowDown' || e.key === 'j') {
      const next = Math.min(slides.length - 1, current + 1);
      minimapState.items[next]?.btn.focus();
      e.preventDefault();
    }
    if (e.key === 'ArrowUp' || e.key === 'k') {
      const prev = Math.max(0, current - 1);
      minimapState.items[prev]?.btn.focus();
      e.preventDefault();
    }
    if (e.key === 'Enter' || e.key === ' ') {
      const idx = minimapState.items.findIndex(({ btn }) => btn === document.activeElement);
      if (idx >= 0) show(idx);
    }
  });

  /* Called by show(i) to update highlight & auto-scroll selection into view */
  function updateMinimapHighlight(i) {
    if (!minimapState.items?.length) return;
    minimapState.items.forEach(({ btn }, k) => {
      btn.classList.toggle('active', k === i);
      if (k === i) {
        btn.setAttribute('aria-current', 'page');
        // gentle into view
        btn.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        btn.removeAttribute('aria-current');
      }
    });
  }

  /* ---------------- 5) Navigation core ---------------- */
  function updateHeaderAndProgress(i) {
    prevBtn && (prevBtn.disabled = (i === 0));
    nextBtn && (nextBtn.disabled = (i === slides.length - 1));
    slideLabel && (slideLabel.textContent = `Slide ${i + 1} / ${slides.length}`);
    const pct = Math.max(0, Math.min(100, Math.round(((i + 1) / slides.length) * 100)));
    if (progressBar) progressBar.style.width = pct + '%';
    // highlight stage chips (if you use them)
    stageChips.forEach(ch => ch.classList.toggle('active', ch.getAttribute('data-stagechip') === (slides[i]?.stage)));
  }

  function applyPerSlideSizing(i) {
    const s = slides[i].node;
    const hero = s.querySelector(':scope > .hero');
    const left = s.querySelector(':scope .left');
    const right = s.querySelector(':scope .right');
    const hHero = +s.dataset.heroH || 200;
    const hLeft = +s.dataset.leftH || 360;
    const hRight = +s.dataset.rightH || 360;
    if (hero) hero.style.height = hHero + 'px';
    if (left) left.style.minHeight = hLeft + 'px';
    if (right) right.style.minHeight = hRight + 'px';
  }

  /* Main show() */
  function show(i) {
    idx = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, j) => s.node.hidden = (j !== idx));
    updateHeaderAndProgress(idx);
    location.hash = `#${idx + 1}`;

    const { id } = slides[idx];
    // init on demand
    if (id === 'slide1') ensureNeuron();
    if (id === 'slide2a') ensureNNOverview();
    if (id === 'slide2') ensureVectorize();
    if (id === 'slideBP') ensureBackprop();
    if (id === 'slideHidden') ensureHidden();
    if (id === 'slide4') ensureELMTrain();
    if (id === 'slidePred') ensurePredict();
    if (id === 'slideIntroNeuron') ensureIceCone?.();
    if (id === 'slideGPS') ensurePseudoInverse();
    if (id === 'slideWhy') ensureWhyWorks();   // ← add this

    applyPerSlideSizing(idx);
    updateMinimapHighlight(idx);
  }

  /* Prev/Next buttons (null-safe) */
  prevBtn && (prevBtn.onclick = () => show(idx - 1));
  nextBtn && (nextBtn.onclick = () => show(idx + 1));

  /* Navigate to hash or 0 after everything is wired */
  function navigateInitialSlide() {
    if (location.hash) {
      const n = parseInt(location.hash.replace('#', ''), 10);
      show(Number.isFinite(n) ? n - 1 : 0);
    } else {
      show(0);
    }
  }

  /* ---------------- 6) Load slide meta AFTER nav is ready ---------------- */
  (async function loadSlideMetaThenBuild() {
    try {
      const res = await fetch('slides.json', { cache: 'no-store' });
      if (res.ok) SLIDE_META = await res.json();
    } catch { }
    //buildRoadmapFromMeta();
    buildMinimap();           // safe now — show() and slides exist
    requestAnimationFrame(navigateInitialSlide); // ensure first highlight syncs with minimap
  })();

  /* ---------------- 7) Worker wiring (shared) ---------------- */
  const workerStatus = $('#workerStatus');
  const worker = new Worker('./elm-worker.js');
  worker.onerror = (err) => {
    workerStatus && (workerStatus.textContent = 'worker failed (serve via http://localhost)');
    console.error(err);
  };
  const post = (type, payload = {}) => worker.postMessage({ type, payload });

  let uiBasisFrozen = false;
  const AG_LABELS = { 1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech' };

  /* Tiny dataset for vectorization/prediction */
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

  function softmax(arr) { if (!arr?.length) return []; const m = Math.max(...arr); const exps = arr.map(x => Math.exp(x - m)); const s = exps.reduce((a, b) => a + b, 0); return exps.map(e => e / s); }
  const fmtVal = (v) => {
    if (!Number.isFinite(v)) return '0';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e3 || a < 1e-3) return v.toExponential(2);
    return v.toFixed(3);
  };

  /* ---------------- 8) Moore–Penrose pseudoinverse (slide #slideGPS) ---------------- */
  function ensurePseudoInverse() {
    if (ensurePseudoInverse.inited) return;
    ensurePseudoInverse.inited = true;

    const c = document.querySelector('#mpPseudo');
    if (!c) return;
    const g = c.getContext('2d');
    const DPR = devicePixelRatio || 1;

    let hover = null;            // {side:'H'|'B', i, j, val, x, y, cx, cy, cw, ch}
    let showNumbers = false;     // checkbox controlled
    let HviewCache = null;       // last-drawn H heatmap matrix
    let betaCache = null;        // last-drawn β matrix
    let geomH = null, geomB = null; // geometry of heatmap cells for hit-testing

    // UI
    const ui = {
      h: document.getElementById('mp-h'),
      k: document.getElementById('mp-k'),
      noise: document.getElementById('mp-noise'),
      lam: document.getElementById('mp-lam'),
      rand: document.getElementById('mp-rand'),
      corr: document.getElementById('mp-corr'),
      loss: document.getElementById('mp-loss'),
      hint: document.getElementById('mp-hint'),
    };

    const chkNums = document.getElementById('mp-cellnums');
    if (chkNums) chkNums.addEventListener('change', () => {
      showNumbers = chkNums.checked;
    });

    c.addEventListener('mouseleave', () => { hover = null; });


    const CFG = {
      cssH: 280,
      colors: {
        panel: "rgba(10,18,40,.85)",
        wire: "#0c1a3d", rim: "#203a7c",
        cyan: "rgba(90,209,255,.9)", cyanSoft: "rgba(90,209,255,.45)",
        magenta: "rgba(255,149,255,.95)", magentaSoft: "rgba(255,149,255,.45)",
        ring: "#ad7bff", out: "#6ee7a2", yellow: "#ffd166",
        ink: "#e5efff", muted: "#a7b8e8"
      },
      fonts: {
        label: '700 13px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial',
        tiny: '600 11px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial',
        big: '800 16px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial'
      }
    };

    function resize() {
      const W = c.clientWidth, H = CFG.cssH;
      c.width = Math.max(1, W * DPR);
      c.height = Math.max(1, H * DPR);
      g.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    resize();
    addEventListener('resize', resize, { passive: true });

    // ---- Simple matrix helpers (small sizes for interactivity)
    const M = {
      zeros: (r, c) => Array.from({ length: r }, () => Array(c).fill(0)),
      randn: (r, c, scale = 1) => Array.from({ length: r }, () => Array.from({ length: c }, () => gauss() * scale)),
      eye: n => { const A = M.zeros(n, n); for (let i = 0; i < n; i++)A[i][i] = 1; return A; },
      T: A => A[0].map((_, j) => A.map(row => row[j])),
      mul: (A, B) => {
        const r = A.length, m = A[0].length, c = B[0].length;
        const out = M.zeros(r, c);
        for (let i = 0; i < r; i++) for (let k = 0; k < m; k++) {
          const aik = A[i][k];
          for (let j = 0; j < c; j++) out[i][j] += aik * B[k][j];
        }
        return out;
      },
      add: (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j])),
      addLamI: (A, lam) => {
        const n = A.length, B = A.map(r => r.slice());
        for (let i = 0; i < n; i++) B[i][i] += lam;
        return B;
      },
      inv: (A) => { // Gauss-Jordan (OK for small n)
        const n = A.length;
        const Mx = A.map(row => row.slice());
        const I = M.eye(n);
        for (let col = 0; col < n; col++) {
          // pivot
          let piv = col;
          for (let r = col + 1; r < n; r++) if (Math.abs(Mx[r][col]) > Math.abs(Mx[piv][col])) piv = r;
          if (Math.abs(Mx[piv][col]) < 1e-12) return null; // singular
          if (piv !== col) { [Mx[col], Mx[piv]] = [Mx[piv], Mx[col]];[I[col], I[piv]] = [I[piv], I[col]]; }
          const p = Mx[col][col];
          for (let j = 0; j < n; j++) { Mx[col][j] /= p; I[col][j] /= p; }
          for (let r = 0; r < n; r++) if (r !== col) {
            const f = Mx[r][col];
            for (let j = 0; j < n; j++) { Mx[r][j] -= f * Mx[col][j]; I[r][j] -= f * I[col][j]; }
          }
        }
        return I;
      },
      frobLoss: (A, B) => { // ||A-B||_F^2
        let s = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
          const d = A[i][j] - B[i][j]; s += d * d;
        } return s / A.length;
      },
      colCorrHint: (H) => { // crude “instability hint”: avg abs correlation of columns
        const n = H.length, h = H[0].length;
        const HT = M.T(H);
        const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
        const std = v => Math.sqrt(v.reduce((a, b) => a + (b - mean(v)) ** 2, 0) / v.length) || 1e-9;
        const zcols = HT.map(col => { const m = mean(col), s = std(col) || 1e-9; return col.map(x => (x - m) / s); });
        let s = 0, c = 0;
        for (let a = 0; a < h; a++) for (let b = a + 1; b < h; b++) {
          const va = zcols[a], vb = zcols[b];
          let dot = 0; for (let i = 0; i < n; i++) dot += va[i] * vb[i];
          s += Math.abs(dot / n); c++;
        }
        return c ? s / c : 0;
      }
    };

    function gauss() { // Box-Muller
      let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // ---- Demo data (n fixed small for speed)
    const n = 40; // training rows
    let h = +ui.h.value | 0, k = +ui.k.value | 0, noise = +ui.noise.value, lam = +ui.lam.value;
    let H = M.randn(n, h, 1.0); // hidden features
    let Y = oneHotY(n, k);      // labels (simple repeating classes)
    let correlated = false;

    function oneHotY(n, k) {
      const Y = M.zeros(n, k);
      for (let i = 0; i < n; i++) { Y[i][i % k] = 1; }
      return Y;
    }

    function makeCorrelated() {
      // Force columns of H to be linear combos of a small basis (rank-deficient-ish)
      const basis = M.randn(n, Math.max(2, Math.floor(h / 4)), 1.0);
      const W = M.randn(basis[0].length, h, 0.8);
      H = M.mul(basis, W);
      correlated = true;
    }

    function randomizeH() {
      H = M.randn(n, h, 1.0);
      correlated = false;
    }

    function ridgeSolve(H, Y, lam) {
      // β = (HᵀH + λI)^{-1} Hᵀ Y
      const HT = M.T(H);
      const A = M.addLamI(M.mul(HT, H), lam);
      const Ainv = M.inv(A);
      if (!Ainv) return null;
      return M.mul(M.mul(Ainv, HT), Y);
    }

    // ---- Layout helpers
    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
    }
    function neonBox(x, y, w, h, stroke) {
      g.save();
      g.fillStyle = CFG.colors.panel;
      g.shadowBlur = 14; g.shadowColor = stroke; g.lineWidth = 2;
      roundRect(g, x, y, w, h, 10); g.fill(); g.strokeStyle = stroke; g.stroke();
      g.restore();
    }
    function ring(cx, cy, rOuter, rInner, t, label) {
      g.save();
      g.shadowColor = CFG.colors.ring; g.shadowBlur = 22; g.lineWidth = 2.5; g.strokeStyle = CFG.colors.ring;
      g.beginPath(); g.arc(cx, cy, rOuter, 0, Math.PI * 2); g.stroke();
      g.lineWidth = 3; g.shadowBlur = 10;
      for (let i = 0; i < 10; i++) {
        const a = t * 1.2 + i * (Math.PI * 0.18);
        g.beginPath(); g.arc(cx, cy, rInner, a, a + 0.8, false); g.stroke();
      }
      g.restore();
      g.save();
      g.font = CFG.fonts.big; g.fillStyle = "white";
      g.textAlign = "center"; g.textBaseline = "middle";
      const flick = 0.8 + 0.2 * Math.abs(Math.sin(t * 6));
      g.globalAlpha = flick; g.fillText(label, cx, cy); g.globalAlpha = 1;
      g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.muted;
      g.fillText("OPTIMIZATION ENGINE", cx, cy - rOuter - 18);
      g.fillStyle = CFG.colors.ink;
      g.fillText("ONE SOLVE", cx, cy + rOuter + 16);
      g.restore();
    }
    function drawHeat(x, y, w, h, A, cmap, opts = { showNumbers: false }) {
      const rows = A.length, cols = A[0].length;
      const cellW = Math.max(1, Math.floor(w / cols));
      const cellH = Math.max(1, Math.floor(h / rows));

      // normalize for color mapping
      let max = 0;
      for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
          max = Math.max(max, Math.abs(A[i][j]));
      const norm = max > 0 ? v => v / max : _ => 0;

      // draw cells
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = A[i][j];
          const t = (norm(v) + 1) / 2; // [-1..1] -> [0..1]
          g.fillStyle = interpColor(cmap[0], cmap[1], t);
          const cx = x + j * cellW, cy = y + i * cellH;
          g.fillRect(cx, cy, cellW, cellH);

          // optional numbers (auto-hide if cells are tiny)
          if (opts.showNumbers && cellW >= 14 && cellH >= 14) {
            g.save();
            g.font = CFG.fonts.tiny;
            g.textAlign = 'center';
            g.textBaseline = 'middle';
            g.shadowColor = 'rgba(0,0,0,0.6)';
            g.shadowBlur = 4;
            g.fillStyle = '#fff';
            g.fillText(v.toFixed(2), cx + cellW / 2, cy + cellH / 2);
            g.restore();
          }
        }
      }

      // return geometry for hit-testing
      return { x, y, w, h, rows, cols, cellW, cellH };
    }

    function interpColor(a, b, t) {
      const ca = parseRGBA(a), cb = parseRGBA(b);
      const r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
      const g1 = Math.round(ca[1] + (cb[1] - ca[1]) * t);
      const b1 = Math.round(ca[2] + (cb[2] - ca[2]) * t);
      const a1 = (ca[3] + (cb[3] - ca[3]) * t).toFixed(3);
      return `rgba(${r},${g1},${b1},${a1})`;
    }
    function parseRGBA(str) {
      // expects rgba(r,g,b,a)
      const m = str.match(/rgba?\(([^)]+)\)/);
      if (!m) return [90, 209, 255, 1];
      const parts = m[1].split(',').map(s => +s.trim());
      if (parts.length === 3) parts.push(1);
      return parts;
    }

    function layout() {
      const W = c.clientWidth, H = CFG.cssH, pad = 16;
      const colW = Math.max(140, Math.floor(W * 0.28));
      return {
        W, H, pad,
        left: { x: pad, y: 56, w: colW, h: H - 96 },
        center: { cx: W / 2, cy: H / 2, rO: 58, rI: 34 },
        right: { x: W - pad - colW, y: 56, w: colW, h: H - 96 },
        labels: {
          left: { x: pad + 8, y: 34, txt: "H (hidden features)" },
          right: { x: W - pad - colW + 8, y: 34, txt: "β (output weights)" }
        },
        footer: { x: W / 2, y: H - 10 }
      };
    }

    // ---- Recompute + draw
    function recompute() {
      h = +ui.h.value | 0;
      k = +ui.k.value | 0;
      noise = +ui.noise.value;
      lam = +ui.lam.value;

      if (H[0].length !== h) randomizeH();
      Y = oneHotY(n, k);

      HviewCache = H.map(row => row.map(v => v + gauss() * noise * 0.2));
      betaCache = ridgeSolve(H, Y, lam);

      const pred = betaCache ? M.mul(H, betaCache) : M.zeros(n, k);
      const loss = betaCache ? M.frobLoss(pred, Y) : Infinity;

      const corrHint = M.colCorrHint(H);
      ui.loss.textContent = isFinite(loss) ? loss.toFixed(4) : 'singular';
      ui.hint.textContent = (corrHint > 0.45 ? 'high collinearity' : 'stable') + (lam > 0 ? ` • λ=${lam.toFixed(2)}` : '');

      drawScene(); // no need to pass matrices; we use caches
    }

    function drawScene() {
      const t = performance.now() / 1000;
      const L = layout();
      g.clearRect(0, 0, L.W, L.H);

      // labels
      g.save();
      g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.muted; g.textAlign = "left";
      g.fillText(L.labels.left.txt, L.labels.left.x, L.labels.left.y);
      g.fillText(L.labels.right.txt, L.labels.right.x, L.labels.right.y);
      g.restore();

      // left panel (H)
      neonBox(L.left.x, L.left.y, L.left.w, L.left.h, CFG.colors.magenta);
      geomH = drawHeat(L.left.x + 6, L.left.y + 6, L.left.w - 12, L.left.h - 12,
        HviewCache, [CFG.colors.magentaSoft, CFG.colors.cyanSoft],
        { showNumbers });

      // right panel (β)
      neonBox(L.right.x, L.right.y, L.right.w, L.right.h, CFG.colors.out);
      if (betaCache) {
        geomB = drawHeat(L.right.x + 6, L.right.y + 6, L.right.w - 12, L.right.h - 12,
          betaCache, [CFG.colors.magentaSoft, CFG.colors.cyanSoft],
          { showNumbers });
      } else {
        geomB = null;
        g.save(); g.fillStyle = CFG.colors.muted; g.font = CFG.fonts.tiny; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText("singular HᵀH (try λ > 0)", L.right.x + L.right.w / 2, L.right.y + L.right.h / 2);
        g.restore();
      }

      // engine + footer
      ring(L.center.cx, L.center.cy, L.center.rO, L.center.rI, t, "β = H⁺Y");
      g.save(); g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.ink; g.textAlign = "center";
      g.fillText("Solve Hβ ≈ Y  •  No gradient loops  •  Add λ for stability", L.footer.x, L.footer.y);
      g.restore();

      // hover highlight + tooltip (draw last so it sits on top)
      if (hover) {
        g.save();
        g.strokeStyle = CFG.colors.yellow; g.lineWidth = 2;
        g.shadowColor = CFG.colors.yellow; g.shadowBlur = 10;
        g.strokeRect(hover.cx, hover.cy, hover.cw, hover.ch);
        // tooltip
        const tip = `${hover.side} [${hover.i},${hover.j}] = ${hover.val.toFixed(4)}`;
        const pad = 6;
        g.font = CFG.fonts.tiny; g.textAlign = 'left'; g.textBaseline = 'top';
        const tw = g.measureText(tip).width;
        const tx = Math.min(L.W - (tw + 2 * pad), hover.cx + hover.cw + 8);
        const ty = Math.max(16, hover.cy - 8);
        g.fillStyle = "rgba(10,18,40,.92)";
        g.shadowColor = CFG.colors.rim; g.shadowBlur = 14;
        g.fillRect(tx, ty, tw + 2 * pad, 20);
        g.shadowBlur = 0; g.fillStyle = '#e5efff';
        g.fillText(tip, tx + pad, ty + 4);
        g.restore();
      }
    }

    // UI events
    ['h', 'k', 'noise', 'lam'].forEach(id => {
      ui[id].addEventListener('input', recompute);
    });
    ui.rand.addEventListener('click', () => { randomizeH(); recompute(); });
    ui.corr.addEventListener('click', () => { makeCorrelated(); recompute(); });

    // initial
    recompute();

    // keep animating ring
    let raf; (function loop() { recompute(); raf = requestAnimationFrame(loop); })();
    addEventListener('beforeunload', () => cancelAnimationFrame(raf));
  }

  function ensurePseudoInverseOld() {
    if (ensurePseudoInverse.inited) return;
    ensurePseudoInverse.inited = true;

    const c = document.querySelector('#mpPseudo');
    if (!c) return;
    const g = c.getContext('2d');

    const CFG = {
      cssH: 260,
      colors: {
        wire: "#0c1a3d", rim: "#203a7c",
        cyan: "rgba(90,209,255,.9)", cyanSoft: "rgba(90,209,255,.45)",
        magenta: "rgba(255,149,255,.95)", magentaSoft: "rgba(255,149,255,.45)",
        ring: "#ad7bff", ringSoft: "rgba(173,123,255,.6)",
        pulse: "#6ee7a2", out: "#6ee7a2", yellow: "#ffd166", ink: "#e5efff", muted: "#a7b8e8"
      },
      fonts: {
        label: '700 13px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial',
        tiny: '600 11px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial',
        big: '800 16px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial'
      }
    };

    const DPR = devicePixelRatio || 1;
    function resize() {
      const W = c.clientWidth, H = CFG.cssH;
      c.width = Math.max(1, W * DPR);
      c.height = Math.max(1, H * DPR);
      g.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    resize();
    addEventListener('resize', resize, { passive: true });

    function layout() {
      const W = c.clientWidth, H = CFG.cssH, pad = 36;
      const left = pad + 52, right = W - pad - 40;
      const cx = Math.round(W * 0.50), cy = Math.round(H * 0.52);
      const rOuter = 58, rInner = 34;
      return {
        W, H, pad, left, right, cx, cy, rOuter, rInner,
        Hbox: { x: pad - 6, y: cy - 70, w: 120, h: 42 },
        Ybox: { x: pad - 6, y: cy + 28, w: 120, h: 42 },
        OutBox: { x: right - 20, y: cy - 24, w: 170, h: 48 },
        bar: { x: W * 0.36, w: W * 0.28, y: H - 26, h: 8 }
      };
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
    }
    function neonBox(x, y, w, h, stroke) {
      g.save();
      g.fillStyle = "rgba(10,18,40,.85)";
      g.shadowBlur = 14; g.shadowColor = stroke; g.lineWidth = 2;
      roundRect(g, x, y, w, h, 10); g.fill(); g.strokeStyle = stroke; g.stroke();
      g.restore();
    }
    function cable(x1, y1, x2, y2, glow, color, body = true) {
      if (body) {
        g.save(); g.lineWidth = 7; g.strokeStyle = CFG.colors.wire;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        g.lineWidth = 1.5; g.strokeStyle = CFG.colors.rim; g.stroke(); g.restore();
      }
      g.save(); g.shadowColor = color; g.shadowBlur = glow; g.lineWidth = 2.3; g.strokeStyle = color;
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.restore();
    }
    function ring(cx, cy, rOuter, rInner, t) {
      g.save();
      g.shadowColor = CFG.colors.ring; g.shadowBlur = 22; g.lineWidth = 2.5; g.strokeStyle = CFG.colors.ring;
      g.beginPath(); g.arc(cx, cy, rOuter, 0, Math.PI * 2); g.stroke();
      g.lineWidth = 3; g.shadowBlur = 10;
      for (let i = 0; i < 10; i++) {
        const a = t * 1.2 + i * (Math.PI * 0.18);
        g.beginPath(); g.arc(cx, cy, rInner, a, a + 0.8, false); g.stroke();
      }
      for (let i = 0; i < 18; i++) {
        const a = t * 1.5 + i * (Math.PI * 2 / 18);
        const rr = rOuter - 6;
        const x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr;
        g.fillStyle = "rgba(255,255,255,.75)";
        g.beginPath(); g.arc(x, y, 1.6, 0, Math.PI * 2); g.fill();
      }
      g.restore();
      g.save();
      g.font = CFG.fonts.big; g.fillStyle = "white";
      g.textAlign = "center"; g.textBaseline = "middle";
      const flick = 0.8 + 0.2 * Math.abs(Math.sin(t * 6));
      g.globalAlpha = flick; g.fillText("β = H⁺Y", cx, cy); g.globalAlpha = 1;
      g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.muted;
      g.fillText("OPTIMIZATION ENGINE", cx, cy - rOuter - 18);
      g.fillStyle = CFG.colors.ink;
      g.fillText("ONE SOLVE", cx, cy + rOuter + 16);
      g.restore();
    }
    function pulse(tNorm, pts, color) {
      let L = 0, segs = [];
      for (let i = 0; i < pts.length - 1; i++) {
        const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
        const d = Math.hypot(x2 - x1, y2 - y1); segs.push({ x1, y1, x2, y2, d }); L += d;
      }
      let dWant = tNorm * L;
      for (const s of segs) {
        if (dWant <= s.d) {
          const k = dWant / s.d, x = s.x1 + (s.x2 - s.x1) * k, y = s.y1 + (s.y2 - s.y1) * k;
          g.save(); g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 16;
          g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fill(); g.restore(); return;
        }
        dWant -= s.d;
      }
    }

    function draw(ts) {
      const t = (ts || 0) / 1000;
      const L = layout();
      g.clearRect(0, 0, L.W, L.H);

      // Left boxes
      neonBox(L.Hbox.x, L.Hbox.y, L.Hbox.w, L.Hbox.h, CFG.colors.magenta);
      neonBox(L.Ybox.x, L.Ybox.y, L.Ybox.w, L.Ybox.h, CFG.colors.cyan);
      g.save();
      g.font = CFG.fonts.label; g.textBaseline = "middle"; g.textAlign = "left";
      g.shadowColor = CFG.colors.magenta; g.shadowBlur = 10; g.fillStyle = CFG.colors.magenta;
      g.fillText("INPUTS (H)", L.Hbox.x + 12, L.Hbox.y + L.Hbox.h / 2);
      g.shadowColor = CFG.colors.cyan; g.fillStyle = CFG.colors.cyan;
      g.fillText("LABELS (Y)", L.Ybox.x + 12, L.Ybox.y + L.Ybox.h / 2);
      g.restore();

      // Cables → engine
      const HmidY = L.Hbox.y + L.Hbox.h / 2, YmidY = L.Ybox.y + L.Ybox.h / 2;
      cable(L.Hbox.x + L.Hbox.w, HmidY, L.cx - L.rOuter - 12, L.cy - 26, 10, CFG.colors.magenta);
      cable(L.Ybox.x + L.Ybox.w, YmidY, L.cx - L.rOuter - 12, L.cy + 26, 10, CFG.colors.cyan);

      // Engine
      ring(L.cx, L.cy, L.rOuter, L.rInner, t);

      // Engine → Output box
      cable(L.cx + L.rOuter + 8, L.cy, L.OutBox.x - 12, L.cy, 12, CFG.colors.out);
      neonBox(L.OutBox.x, L.OutBox.y, L.OutBox.w, L.OutBox.h, CFG.colors.out);
      g.save();
      g.font = CFG.fonts.label; g.textBaseline = "middle"; g.textAlign = "left";
      g.shadowColor = CFG.colors.out; g.shadowBlur = 12; g.fillStyle = CFG.colors.out;
      g.fillText("BEST OUTPUT", L.OutBox.x + 12, L.OutBox.y + 14);
      g.fillText("WEIGHTS (β)", L.OutBox.x + 12, L.OutBox.y + L.OutBox.h - 14);
      g.restore();

      // Progress bar
      g.save();
      const bx = L.bar.x, by = L.bar.y, bw = L.bar.w, bh = L.bar.h;
      g.strokeStyle = CFG.colors.ring; g.lineWidth = 2; g.shadowColor = CFG.colors.ring; g.shadowBlur = 10;
      roundRect(g, bx, by, bw, bh, 6); g.stroke();
      const prog = (t % 2) / 2; const fillW = Math.max(18, bw * prog);
      g.fillStyle = "rgba(173,123,255,.35)"; roundRect(g, bx, by, bw, bh, 6); g.fill();
      g.fillStyle = "rgba(110,231,162,.9)"; roundRect(g, bx, by, fillW, bh, 6); g.fill();
      g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.muted; g.textAlign = "center"; g.textBaseline = "bottom";
      g.fillText("ONE SOLVE", bx + bw / 2, by - 4);
      g.restore();

      // Pulses
      const pH = (t % 3) / 3, pY = ((t + 1.2) % 3) / 3;
      pulse(pH, [[L.Hbox.x + L.Hbox.w, HmidY], [L.cx - L.rOuter - 12, L.cy - 26], [L.cx - 2, L.cy - 2]], CFG.colors.magenta);
      pulse(pY, [[L.Ybox.x + L.Ybox.w, YmidY], [L.cx - L.rOuter - 12, L.cy + 26], [L.cx - 2, L.cy + 2]], CFG.colors.cyan);

      // Footer
      g.save();
      g.font = CFG.fonts.tiny; g.fillStyle = CFG.colors.ink; g.textAlign = "center";
      g.fillText("Solve Hβ ≈ Y  •  β via Moore–Penrose pseudoinverse  •  No gradient loops", L.W / 2, L.H - 8);
      g.restore();
    }

    let raf; (function loop(ts) { draw(ts); raf = requestAnimationFrame(loop); })(0);
    addEventListener('beforeunload', () => cancelAnimationFrame(raf));
  }

  /* ---------------- X) Slide: Why This Works (random projection) ---------------- */
  function ensureWhyWorks() {
    if (ensureWhyWorks.inited) return;
    ensureWhyWorks.inited = true;

    const c = document.querySelector('#whyWorks');
    if (!c) return;
    const g = c.getContext('2d');
    const DPR = window.devicePixelRatio || 1;

    function resize() {
      const W = c.clientWidth, H = c.clientHeight || 260;
      c.width = Math.max(1, W * DPR);
      c.height = Math.max(1, H * DPR);
      g.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    resize();
    addEventListener('resize', resize, { passive: true });

    // Build a "crumpled" blob of two interleaved classes
    const N = 36;
    const basePts = Array.from({ length: N }, (_, i) => {
      const a = Math.random() * Math.PI * 2;
      const r = 40 + Math.random() * 38;
      return { x: Math.cos(a) * r, y: Math.sin(a) * r, cls: i % 2 };
    });

    // Helpers
    function neonLine(x1, y1, x2, y2, color = 'rgba(110,231,162,.95)', width = 2, glow = 12) {
      g.save();
      g.lineWidth = width;
      g.shadowColor = color;
      g.shadowBlur = glow;
      g.strokeStyle = color;
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
      g.restore();
    }

    function draw(ts) {
      const t = (ts || 0) / 1000;
      const W = c.clientWidth, H = c.clientHeight || 260;
      g.clearRect(0, 0, W, H);

      const cxL = Math.round(W * 0.25), cxR = Math.round(W * 0.75), cy = Math.round(H * 0.5);

      // Phase smoothly oscillates: crumpled ↔ projected
      const phase = (Math.sin(t * 0.6) + 1) / 2; // 0..1

      // Titles
      g.fillStyle = '#5ad1ff';
      g.font = '600 13px ui-sans-serif,system-ui';
      g.textAlign = 'center';
      g.fillText('Crumpled data', cxL, 18);
      g.fillText('Higher-dimensional projection', cxR, 18);

      // --- Left: crumpled blob -------------------------------------------------
      g.save();
      g.translate(cxL, cy);
      // wireframe-ish outline
      g.strokeStyle = 'rgba(173,123,255,.35)';
      g.lineWidth = 1.5;
      g.beginPath();
      for (let i = 0; i < basePts.length; i++) {
        const p = basePts[i], q = basePts[(i + 1) % basePts.length];
        if (i === 0) g.moveTo(p.x * 0.8, p.y * 0.8);
        g.lineTo(q.x * 0.8, q.y * 0.8);
      }
      g.stroke();
      // points
      basePts.forEach(p => {
        g.fillStyle = p.cls ? 'rgba(255,149,255,.95)' : 'rgba(90,209,255,.95)';
        g.beginPath(); g.arc(p.x * 0.8, p.y * 0.8, 4, 0, Math.PI * 2); g.fill();
      });
      g.restore();

      // --- Right: projected & separated ---------------------------------------
      g.save();
      g.translate(cxR, cy);

      // Draw a soft grid pad
      const grid = 88;
      g.strokeStyle = 'rgba(255,255,255,0.06)';
      for (let x = -grid; x <= grid; x += 22) { g.beginPath(); g.moveTo(x, -grid); g.lineTo(x, grid); g.stroke(); }
      for (let y = -grid; y <= grid; y += 22) { g.beginPath(); g.moveTo(-grid, y); g.lineTo(grid, y); g.stroke(); }

      // Separating line (neon)
      neonLine(-96, 0, 96, 0, '#6ee7a2', 2.2, 14);

      // Project each point from crumpled → spread clusters across the line
      basePts.forEach((p, i) => {
        const tx = (p.cls ? 1 : -1) * 58;                 // target side
        const ty = (i % 6 - 3) * 10 + (p.cls ? 6 : -6);         // small spread
        const x = p.x * 0.5 * (1 - phase) + tx * phase;
        const y = p.y * 0.5 * (1 - phase) + ty * phase;

        // glow dot
        const col = p.cls ? 'rgba(255,149,255,.95)' : 'rgba(90,209,255,.95)';
        g.save();
        g.shadowColor = col; g.shadowBlur = 12; g.fillStyle = col;
        g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fill();
        g.restore();
      });

      g.restore();

      // Footer
      g.fillStyle = '#a7b8e8';
      g.font = '600 12px ui-sans-serif,system-ui';
      g.textAlign = 'center';
      g.fillText('Random projection into a richer space → simple linear split', W / 2, H - 8);
    }

    let raf; (function loop(ts) { draw(ts); raf = requestAnimationFrame(loop); })(0);
    addEventListener('beforeunload', () => cancelAnimationFrame(raf));
  }

  /* ---------------- 8) Slide 2a: overview animation ---------------- */
  function ensureNNOverview() {
    if (ensureNNOverview.inited) return; ensureNNOverview.inited = true;
    const c = $('#nnCanvas'); if (!c) return;
    const g = c.getContext('2d');
    const text = $('#nnText'); const run = $('#nnRun');
    const nodes = { in: [], hid: [], out: [] }, particles = [];
    const DPR = devicePixelRatio || 1;

    function resize() {
      const W = c.clientWidth, H = c.clientHeight; c.width = W * DPR; c.height = H * DPR; g.setTransform(DPR, 0, 0, DPR, 0, 0);
      layout();
    }
    const colX = () => [30, c.clientWidth / 2, c.clientWidth - 30];
    function layout() {
      const H = c.clientHeight; const [x0, x1, x2] = colX();
      const yFor = (n, i) => 20 + (i * (H - 40)) / (n - 1);
      nodes.in = Array.from({ length: 5 }, (_, i) => ({ x: x0, y: yFor(5, i), r: 10, stage: 'input' }));
      nodes.hid = Array.from({ length: 8 }, (_, i) => ({ x: x1, y: yFor(8, i), r: 11, stage: 'neuron' }));
      nodes.out = Array.from({ length: 3 }, (_, i) => ({ x: x2, y: yFor(3, i), r: 10, stage: 'output' }));
    }
    function spawnParticles() {
      particles.length = 0;
      for (let k = 0; k < 18; k++) {
        const s = nodes.in[k % nodes.in.length];
        const h = nodes.hid[Math.floor(Math.random() * nodes.hid.length)];
        const o = nodes.out[Math.floor(Math.random() * nodes.out.length)];
        particles.push({ path: [s, h, o], t: 0 });
      }
    }
    function step(dt) { particles.forEach(p => { p.t = Math.min(1, p.t + dt * 0.0004); }); }
    function draw() {
      const W = c.clientWidth, H = c.clientHeight; g.clearRect(0, 0, W, H);
      // wires
      g.strokeStyle = 'rgba(90,209,255,.35)';
      nodes.in.forEach(a => nodes.hid.forEach(b => { g.beginPath(); g.moveTo(a.x, a.y); g.bezierCurveTo(a.x + 40, a.y, b.x - 40, b.y, b.x, b.y); g.stroke(); }));
      nodes.hid.forEach(a => nodes.out.forEach(b => { g.beginPath(); g.moveTo(a.x, a.y); g.bezierCurveTo(a.x + 40, a.y, b.x - 40, b.y, b.x, b.y); g.stroke(); }));
      // nodes
      const dot = (n, glow) => {
        g.beginPath(); g.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        g.fillStyle = glow ? '#103a6b' : '#0c1a3d'; g.fill();
        g.lineWidth = 1.5; g.strokeStyle = '#203a7c'; g.stroke();
        if (glow) { g.strokeStyle = 'rgba(90,209,255,.9)'; g.stroke(); }
      };
      nodes.in.forEach(n => dot(n, false));
      nodes.hid.forEach(n => dot(n, true));
      nodes.out.forEach(n => dot(n, false));
      // particles
      particles.forEach(p => {
        const [a, b, c3] = p.path; const t = p.t;
        const seg = t < .6 ? [a, b, t / .6] : [b, c3, (t - .6) / .4];
        const [p0, p1, tt] = seg;
        const x = p0.x + (p1.x - p0.x) * tt, y = p0.y + (p1.y - p0.y) * tt;
        g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fillStyle = '#6ee7a2'; g.fill();
      });
    }
    let last = 0; function loop(ts) { const dt = ts - (last || ts); last = ts; step(dt); draw(); requestAnimationFrame(loop); }

    function hitNeuron(x, y) {
      return nodes.hid.find(n => (x - n.x) ** 2 + (y - n.y) ** 2 <= (n.r + 2) ** 2);
    }

    window.addEventListener('resize', resize);
    resize(); spawnParticles(); requestAnimationFrame(loop);

    // click any neuron to zoom → Neuron slide
    c.addEventListener('click', (e) => {
      const r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      if (hitNeuron(x, y)) {
        const k = slides.findIndex(s => s.id === 'slide1');
        show(k >= 0 ? k : 0);
      }
    });

    run?.addEventListener('click', () => spawnParticles());
  }

  /* ---------------- 9) Slide: small neuron micro-anim ---------------- */
  /**
   * Cyberpunk micro-neuron animation — styled like the reference image.
   * Draws 5 inputs -> weighted sum (Σ) -> + bias -> activation box (f) -> output (y)
   * Includes glowing neon cables and a traveling pulse.
   */
  (function microNeuron() {
    // Tiny helper
    const $ = (sel) => document.querySelector(sel);

    // ---- Config (tweak freely) ----------------------------------------------
    const CFG = {
      heightCSS: 220,         // CSS pixel height for the canvas
      inputs: 5,              // number of input lines
      colors: {
        cable: "rgba(90,209,255,0.75)",   // cyan
        cableSoft: "rgba(90,209,255,0.35)",
        output: "rgba(255,149,255,0.9)",  // pink/magenta
        sumCore: "#fbd38d",               // warm glow
        sumRim: "#ff9f6e",
        bias: "#ffd166",                  // yellow
        actBox: "rgba(173,123,255,0.85)", // purple
        actCore: "rgba(255,255,255,0.8)",
        bgWire: "#0c1a3d",
        bgWireRim: "#203a7c",
        pulse: "#6ee7a2",
        ink: "#e5efff",
        muted: "#a7b8e8"
      },
      fonts: {
        label: '600 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial',
        tiny: '500 10px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial',
        big: '700 16px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial'
      }
    };

    function init() {
      const c = $('#neuronMicro');
      if (!c) return;
      const g = c.getContext('2d');

      const DPR = window.devicePixelRatio || 1;
      function resize() {
        const W = c.clientWidth;
        const H = CFG.heightCSS;
        c.width = Math.max(1, W * DPR);
        c.height = Math.max(1, H * DPR);
        g.setTransform(DPR, 0, 0, DPR, 0, 0);
      }
      resize();
      window.addEventListener('resize', resize, { passive: true });

      // Geometry layout (in CSS pixels)
      function layout() {
        const W = c.clientWidth, H = CFG.heightCSS;
        const pad = 40;
        const leftX = pad + 50;           // inputs end
        const sumX = Math.round(W * 0.38);
        const actX = Math.round(W * 0.60);
        const outX = W - pad - 30;

        const centerY = Math.round(H * 0.55);
        const spread = 58;                // vertical fan of inputs

        const inStartX = pad;
        const inYs = [];
        for (let i = 0; i < CFG.inputs; i++) {
          const t = (i - (CFG.inputs - 1) / 2);
          inYs.push(centerY + t * (spread / ((CFG.inputs - 1) / 2 || 1)));
        }

        return {
          W, H, pad,
          inStartX, leftX,
          sumX, actX, outX,
          centerY, inYs,
          sumR: 22,
          actW: 86, actH: 54
        };
      }

      // Draw glowing line
      function neonLine(ctx, x1, y1, x2, y2, color, width = 2, glow = 8) {
        ctx.save();
        ctx.lineWidth = width;
        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = glow;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
      }

      // Base cable (dark body with rim)
      function cableBody(ctx, x1, y1, x2, y2) {
        ctx.save();
        ctx.lineWidth = 6;
        ctx.strokeStyle = CFG.colors.bgWire;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = CFG.colors.bgWireRim;
        ctx.stroke();
        ctx.restore();
      }

      // Σ node
      function drawSum(ctx, x, y, r) {
        const g1 = ctx.createRadialGradient(x, y, 2, x, y, r + 3);
        g1.addColorStop(0, CFG.colors.sumCore);
        g1.addColorStop(1, CFG.colors.sumRim);
        ctx.save();
        ctx.shadowColor = CFG.colors.sumRim;
        ctx.shadowBlur = 20;
        ctx.fillStyle = g1;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();

        // Rim
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.stroke();

        // Σ label
        ctx.font = CFG.fonts.big;
        ctx.fillStyle = "#1b243b";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Σ", x, y + 1);
        ctx.restore();
      }

      // Activation box with swirling core
      function drawActivation(ctx, x, y, w, h, t) {
        const r = 10;
        ctx.save();
        ctx.shadowColor = CFG.colors.actBox;
        ctx.shadowBlur = 18;
        ctx.fillStyle = "rgba(26, 18, 51, 0.9)";
        // rounded rect
        roundRect(ctx, x, y, w, h, r);
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = CFG.colors.actBox;
        ctx.stroke();

        // Swirl
        ctx.save();
        ctx.translate(x + w / 2, y + h / 2);
        ctx.rotate(0.2);
        const loops = 24;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < loops; i++) {
          const p = (i + (t * 24 % 1)) / loops;
          const rad = 2 + p * (Math.min(w, h) * 0.38);
          ctx.strokeStyle = `rgba(255,255,255,${0.9 - p})`;
          ctx.beginPath();
          ctx.arc(0, 0, rad, p * Math.PI * 2, p * Math.PI * 2 + 0.7);
          ctx.stroke();
        }
        ctx.restore();

        // f label
        ctx.font = CFG.fonts.tiny;
        ctx.fillStyle = CFG.colors.ink;
        ctx.textAlign = "center";
        ctx.fillText("ACTIVATION  f", x + w / 2, y + h + 14);
        ctx.restore();
      }

      function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
      }

      // Labels with subtle glow
      function glowText(ctx, text, x, y, color, font) {
        ctx.save();
        ctx.font = font || CFG.fonts.label;
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x, y);
        ctx.restore();
      }

      // Traveling pulse along segmented path (input → sum → bias/act → out)
      function drawPulse(ctx, tNorm, pts) {
        // pts: array of [x,y] defining piecewise linear path
        // Compute total length
        let segs = [];
        let total = 0;
        for (let i = 0; i < pts.length - 1; i++) {
          const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
          const d = Math.hypot(x2 - x1, y2 - y1);
          segs.push({ x1, y1, x2, y2, d });
          total += d;
        }
        let dist = tNorm * total;
        for (const s of segs) {
          if (dist <= s.d) {
            const k = dist / s.d;
            const x = s.x1 + (s.x2 - s.x1) * k;
            const y = s.y1 + (s.y2 - s.y1) * k;
            // pulse
            ctx.save();
            ctx.fillStyle = CFG.colors.pulse;
            ctx.shadowColor = CFG.colors.pulse;
            ctx.shadowBlur = 14;
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
            return;
          }
          dist -= s.d;
        }
      }

      function draw(ts) {
        const L = layout();
        const t = (ts || 0) / 1000;

        // Clear
        g.clearRect(0, 0, L.W, L.H);

        // INPUT cables to sum
        for (let i = 0; i < CFG.inputs; i++) {
          const y = L.inYs[i];
          cableBody(g, L.inStartX, y, L.leftX, y);
          neonLine(g, L.inStartX, y, L.leftX, y, CFG.colors.cable, 2.2, 10);

          // Input labels
          g.font = CFG.fonts.tiny;
          g.fillStyle = CFG.colors.muted;
          g.textAlign = "left";
          g.textBaseline = "middle";
          g.fillText(`INPUT ${i + 1}`, L.inStartX + 4, y - 12);
        }

        // Lines from input end to the Σ center (slanted fan)
        for (let i = 0; i < CFG.inputs; i++) {
          const y = L.inYs[i];
          cableBody(g, L.leftX, y, L.sumX - L.sumR, L.centerY);
          neonLine(g, L.leftX, y, L.sumX - L.sumR, L.centerY, CFG.colors.cableSoft, 1.8, 8);
        }

        // SUM node
        drawSum(g, L.sumX, L.centerY, L.sumR);
        glowText(g, "WEIGHTS (w)", L.leftX - 2, Math.min(...L.inYs) - 18, CFG.colors.muted, CFG.fonts.tiny);
        glowText(g, "SUM (Σ)", L.sumX - 22, L.centerY + L.sumR + 16, CFG.colors.muted, CFG.fonts.tiny);

        // Sum → Activation cable
        cableBody(g, L.sumX + L.sumR, L.centerY, L.actX - 10, L.centerY);
        neonLine(g, L.sumX + L.sumR, L.centerY, L.actX - 10, L.centerY, CFG.colors.cable, 2.2, 12);

        // Bias branch
        const biasY = L.centerY - 34;
        neonLine(g, L.sumX + 8, biasY, L.sumX + L.sumR + 8, biasY, CFG.colors.bias, 1.6, 6);
        neonLine(g, L.sumX + L.sumR + 8, biasY, L.sumX + L.sumR + 8, L.centerY, CFG.colors.bias, 1.6, 6);
        glowText(g, "BIAS (b)", L.sumX + 12, biasY - 10, CFG.colors.bias, CFG.fonts.tiny);

        // Activation box
        drawActivation(g, L.actX, L.centerY - L.actH / 2, L.actW, L.actH, t);

        // Activation → Output cable
        const ax = L.actX + L.actW;
        cableBody(g, ax, L.centerY, L.outX, L.centerY);
        neonLine(g, ax, L.centerY, L.outX, L.centerY, CFG.colors.output, 2.4, 14);

        // Output label
        g.font = CFG.fonts.big;
        g.fillStyle = CFG.colors.output;
        g.shadowColor = CFG.colors.output;
        g.shadowBlur = 12;
        g.textAlign = "left";
        g.textBaseline = "middle";
        g.fillText("OUTPUT (y)", L.outX, L.centerY);
        g.shadowBlur = 0;

        // Small “PARAMETERS” footer
        g.font = CFG.fonts.tiny;
        g.fillStyle = CFG.colors.ink;
        g.textAlign = "center";
        g.fillText("PARAMETERS:  Weights (w), Bias (b)", L.W * 0.5, L.H - 14);

        // Traveling pulse: construct path (choose one input that cycles)
        const iPick = Math.floor((t * 0.5) % CFG.inputs);
        const yIn = L.inYs[iPick];
        const path = [
          [L.inStartX, yIn],
          [L.leftX, yIn],
          [L.sumX - L.sumR, L.centerY],
          [L.sumX + L.sumR, L.centerY],
          [L.actX - 10, L.centerY],
          [L.actX + L.actW, L.centerY],
          [L.outX, L.centerY]
        ];
        const tPulse = (t % 2.8) / 2.8;
        drawPulse(g, tPulse, path);
      }

      // Animation loop
      let raf;
      function loop(ts) {
        draw(ts);
        raf = requestAnimationFrame(loop);
      }
      loop();

      // Clean up on unload (optional)
      window.addEventListener('beforeunload', () => cancelAnimationFrame(raf));
    }

    // Kick off when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
      init();
    }
  })();

  /* ---------------- 10) Slide: Neuron demo ---------------- */
  const S1 = { act: 'relu', rafId: null, canvas: null };
  function ensureNeuron() {
    if (S1.inited) return; S1.inited = true;
    S1.canvas = $('#neuronCanvas'); if (!S1.canvas) return;

    const actSelect = $('#actSelect');
    const wRange = $('#wRange'); const bRange = $('#bRange');
    const wVal = $('#wVal'); const bVal = $('#bVal');
    const ampVal = $('#ampVal'); const ampBar = $('#ampBar');
    const postAmpVal = $('#postAmpVal'); const postAmpBar = $('#postAmpBar');

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
      const w = +wRange.value, b = +bRange.value;
      const amp = Math.max(Math.abs(w * 3 + b), Math.abs(w * -3 + b));
      ampVal.textContent = amp.toFixed(2);
      const pct = Math.min(100, (amp / 20) * 100);
      ampBar.style.width = pct + '%';
    }
    function updatePostAmplitude() {
      const w = +wRange.value, b = +bRange.value;
      let maxAbsOutput = 0;
      for (let i = 0; i <= 300; i++) {
        const x = -3 + (i / 300) * 6;
        const y = actFn(w * x + b);
        maxAbsOutput = Math.max(maxAbsOutput, Math.abs(y));
      }
      postAmpVal.textContent = maxAbsOutput.toFixed(2);
      const maxPossible = (S1.act === 'sigmoid' || S1.act === 'tanh') ? 1 : 20;
      const pct = Math.min(100, (maxAbsOutput / maxPossible) * 100);
      postAmpBar.style.width = pct + '%';
    }

    function loop() {
      if (!S1.canvas) return;
      const c = S1.canvas, g = c.getContext('2d'), dpr = devicePixelRatio || 1;
      const W = c.clientWidth, H = c.clientHeight;
      c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);

      // axes
      g.strokeStyle = '#3857a8'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(10, H - 20); g.lineTo(W - 10, H - 20); g.stroke();
      g.beginPath(); g.moveTo(40, 10); g.lineTo(40, H - 10); g.stroke();

      const w = +wRange.value, b = +bRange.value;
      const toXY = (x, y) => { const xm = (x + 3) / 6, ym = (y - (-2)) / (2 - (-2)); return [40 + xm * (W - 55), (H - 20) - ym * (H - 35)]; };

      g.strokeStyle = '#5ad1ff'; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = -3 + (i / 300) * 6, y = actFn(w * x + b);
        const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
        if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();

      const now = performance.now() / 1000;
      for (let k = 0; k < 7; k++) {
        const x = -3 + ((now * 0.6 + k / 7) % 1) * 6, y = actFn(w * x + b);
        const [px, py] = toXY(x, Math.max(-2, Math.min(2, y)));
        g.fillStyle = '#6ee7a2'; g.beginPath(); g.arc(px, py, 3.2, 0, Math.PI * 2); g.fill();
      }
      S1.rafId = requestAnimationFrame(loop);
    }

    actSelect?.addEventListener('change', () => { S1.act = actSelect.value; updatePostAmplitude(); });
    [wRange, bRange].forEach(r => r?.addEventListener('input', () => {
      wVal.textContent = (+wRange.value).toFixed(2);
      bVal.textContent = (+bRange.value).toFixed(2);
      updateAmplitude(); updatePostAmplitude();
    }));
    wVal.textContent = (+wRange.value).toFixed(2);
    bVal.textContent = (+bRange.value).toFixed(2);
    updateAmplitude(); updatePostAmplitude();
    if (S1.rafId) cancelAnimationFrame(S1.rafId);
    loop();
  }

  /* ---------------- 11) Slide: Vectorization ---------------- */
  const S2 = { lastEncoded: null, grid: null, featureLimit: 128, lastMethod: 'tfidf' };

  function ensureVectorize() {
    if (S2.inited) return; S2.inited = true;
    const rowSelect = $('#rowSelect');
    const encodeBtn = $('#encodeBtn');
    const encodeCanvas = $('#encodeCanvas');
    const tokensOut = $('#tokensOut');
    const encodingSelect = $('#encodingSelect');
    const featureLimit = $('#featureLimit');
    const featureLimitVal = $('#featureLimitVal');

    // populate
    if (rowSelect) {
      rowSelect.innerHTML = '';
      for (let i = 0; i < dataRows.length; i++) {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
        rowSelect.appendChild(o);
      }
    }

    S2.featureLimit = +(featureLimit?.value || 128);
    featureLimit?.addEventListener('input', () => {
      S2.featureLimit = +featureLimit.value;
      featureLimitVal.textContent = featureLimit.value;
      drawEncode();
    });
    encodingSelect?.addEventListener('change', () => { S2.lastMethod = encodingSelect.value; });

    encodeBtn && (encodeBtn.onclick = () => {
      const i = +(rowSelect?.value || 0);
      const method = encodingSelect?.value || 'tfidf';
      const corpus = dataRows.map(r => r.text);
      post('encode', { text: dataRows[i].text, method, corpus });
    });

    // tooltip
    let tipEl = $('#slide2 .encode-tip');
    if (!tipEl && encodeCanvas?.parentElement) {
      tipEl = document.createElement('div'); tipEl.className = 'encode-tip';
      encodeCanvas.parentElement.appendChild(tipEl);
    }

    encodeCanvas?.addEventListener('mousemove', (e) => {
      if (!S2.grid || !S2.lastEncoded || !tipEl) { if (tipEl) tipEl.style.display = 'none'; return; }
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
      tipEl.textContent = `${token ? `#${k} “${token}”` : `feature #${k}`} = ${fmtVal(v[k])}`;
      tipEl.style.display = 'block';
      const tbox = tipEl.getBoundingClientRect(), pad = 6;
      const left = Math.min(rect.width - tbox.width - pad, Math.max(pad, x + 10));
      const top = Math.min(rect.height - tbox.height - pad, Math.max(pad, y - 28));
      tipEl.style.left = `${left}px`;
      tipEl.style.top = `${top}px`;
    });
    encodeCanvas?.addEventListener('mouseleave', () => tipEl && (tipEl.style.display = 'none'));

    function drawEncode() {
      if (!encodeCanvas) return;
      const c = encodeCanvas, dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
      c.width = Math.max(1, W * dpr); c.height = Math.max(1, H * dpr);
      const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      if (!S2.lastEncoded) {
        g.fillStyle = '#93a9e8'; g.fillText('Click “Encode text” to preview the input vector.', 12, 22);
        S2.grid = null; return;
      }
      const v = S2.lastEncoded.vector.slice();
      // normalize for contrast
      const norm = Math.hypot(...v) || 1; for (let i = 0; i < v.length; i++) v[i] /= norm;

      const n = Math.min(S2.featureLimit || 128, v.length);
      const cols = Math.ceil(Math.sqrt(n)); const rows = Math.ceil(n / cols);
      const m = 10, x0 = m, y0 = m; const cw = Math.floor((W - 2 * m) / cols), ch = Math.floor((H - 2 * m) / rows);
      const vSub = v.slice(0, n);
      const maxAbs = Math.max(1e-6, ...vSub.map(x => Math.abs(x)));
      let k = 0;
      for (let r = 0; r < rows; r++) {
        for (let ccol = 0; ccol < cols; ccol++, k++) {
          if (k >= n) break;
          const val = vSub[k]; const alpha = Math.min(1, Math.abs(val) / maxAbs);
          const hue = val >= 0 ? 200 : 0; const X = x0 + ccol * cw, Y = y0 + r * ch;
          const bg = `hsla(${hue},90%,60%,${0.15 + 0.85 * alpha})`;
          g.fillStyle = bg; g.fillRect(X, Y, cw - 2, ch - 2);
          if (cw >= 56 && ch >= 36) {
            g.save(); g.fillStyle = 'rgba(255,255,255,0.95)';
            g.font = `600 ${Math.min(18, Math.floor(ch * 0.42))}px ui-sans-serif,system-ui`;
            g.textAlign = 'center'; g.textBaseline = 'middle';
            g.fillText(fmtVal(val), X + (cw - 2) / 2, Y + (ch - 2) / 2); g.restore();
          }
        }
      }
      g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
      for (let ccol = 0; ccol <= cols; ccol++) { g.beginPath(); g.moveTo(x0 + ccol * cw, y0); g.lineTo(x0 + ccol * cw, y0 + rows * ch); g.stroke(); }
      for (let rr = 0; rr <= rows; rr++) { g.beginPath(); g.moveTo(x0, y0 + rr * ch); g.lineTo(x0 + cols * cw, y0 + rr * ch); g.stroke(); }
      g.fillStyle = '#a7b8e8';
      g.fillText(`features 0..${n - 1} — basis: ${uiBasisFrozen ? 'trained' : 'isolated'}`, 12, H - 12);

      S2.grid = { x0, y0, cols, rows, cw, ch, n };
    }

    // worker event for this slide
    worker.addEventListener('message', (e) => {
      const { type, payload } = e.data || {};
      if (type !== 'encoded') return;
      S2.lastEncoded = payload;
      const chosen = payload.methodUsed || encodingSelect?.value || 'tfidf';
      const label = chosen === 'tfidf' ? 'TF-IDF' : chosen === 'bow' ? 'Bag-of-Words' : 'Isolated';
      if (tokensOut) {
        tokensOut.textContent = `${label} tokens (top):\n${(payload.tokens || []).slice(0, 25).join(' ')}\n\nvector length: ${payload.vector.length}`;
      }
      // redraw
      drawEncode();
    });
  }

  /* ---------------- 12) Slide: Backprop demo ---------------- */
  const BP = { canvas: null, lrRange: null, lrVal: null, raf: null, t0: 0, state: null };
  function ensureBackprop() {
    if (BP.inited) return; BP.inited = true;
    BP.canvas = $('#bpCanvas'); if (!BP.canvas) return;
    BP.lrRange = $('#bpLR'); BP.lrVal = $('#bpLRVal');
    BP.lrVal && (BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2));

    const restartBtn = $('#bpRestart');
    const scenarioSel = $('#bpScenario');

    function reset(kind = 'converge') {
      const H = 24, W = 36;
      BP.state = { weights: Array.from({ length: H }, () => Array.from({ length: W }, () => (Math.random() * 2 - 1) * 0.6)), loss: 1.0, noise: 0.0, kind };
      BP.points = []; BP.convergedFrames = 0; BP.holdUntil = performance.now() + 400;
    }
    reset('converge');
    restartBtn?.addEventListener('click', () => reset(scenarioSel?.value || 'converge'));
    scenarioSel?.addEventListener('change', () => reset(scenarioSel.value));

    // tooltip
    let tip = $('#slideBP .bp-tip');
    if (!tip && BP.canvas.parentElement) {
      tip = document.createElement('div'); tip.className = 'bp-tip';
      BP.canvas.parentElement.appendChild(tip);
    }

    BP.lrRange?.addEventListener('input', () => BP.lrVal && (BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2)));

    function step(dt) {
      if (performance.now() < (BP.holdUntil || 0)) return;
      const lr = +(BP.lrRange?.value || 0.5);
      const kind = BP.state.kind || 'converge';
      const base = kind === 'converge' ? (0.985 - 0.25 * lr * 0.01)
        : kind === 'vanish' ? 0.997
          : 0.99;
      BP.state.loss = Math.max(0.02, BP.state.loss * base);
      const H = BP.state.weights.length, W = BP.state.weights[0].length;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let w = BP.state.weights[i][j];
        let grad = w;
        if (kind === 'vanish') grad *= 0.02;
        if (kind === 'explode') grad *= 3.0;
        const noise = (Math.random() * 2 - 1) * 0.02 * (kind === 'explode' ? 2 : 1);
        BP.state.weights[i][j] = w - lr * 0.01 * grad + noise;
      }
    }

    let cellGeom = null;
    function render() {
      const c = BP.canvas, g = c.getContext('2d'), dpr = devicePixelRatio || 1;
      const Wc = c.clientWidth, Hc = c.clientHeight;
      c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, Wc, Hc);

      const H = BP.state.weights.length, W = BP.state.weights[0].length;
      const pad = 10, gridW = Math.floor(Wc * 0.66);
      const cellW = Math.floor((gridW - 2 * pad) / W);
      const cellH = Math.floor((Hc - 2 * pad) / H);
      cellGeom = { x0: pad, y0: pad, cw: cellW, ch: cellH, rows: H, cols: W };

      let vmax = 1e-6;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) vmax = Math.max(vmax, Math.abs(BP.state.weights[i][j]));
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          const val = BP.state.weights[i][j];
          const a = Math.min(1, Math.abs(val) / vmax);
          const hue = val >= 0 ? 200 : 0;
          g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * a})`;
          g.fillRect(pad + j * cellW, pad + i * cellH, cellW - 1, cellH - 1);
        }
      }
      g.fillStyle = '#a7b8e8'; g.fillText('Hidden weights (changing each step)', pad, Hc - 8);

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
      const lrTxt = (+(BP.lrRange?.value || 0.5)).toFixed(2);
      g.fillText(`loss ~ ${BP.state.loss.toFixed(3)}  (learning rate ${lrTxt})`, x0 + 6, y0 + 16);
    }

    BP.canvas.addEventListener('mousemove', (e) => {
      if (!cellGeom || !tip) { tip && (tip.style.display = 'none'); return; }
      const rect = BP.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const { x0, y0, cw, ch, rows, cols } = cellGeom;
      if (x < x0 || y < y0) { tip.style.display = 'none'; return; }
      const c = Math.floor((x - x0) / cw), r = Math.floor((y - y0) / ch);
      if (c < 0 || c >= cols || r < 0 || r >= rows) { tip.style.display = 'none'; return; }
      const v = BP.state.weights[r][c];
      tip.textContent = `w[${r},${c}] = ${v.toFixed(3)}`;
      const tbox = tip.getBoundingClientRect(), pad2 = 6;
      tip.style.display = 'block';
      tip.style.left = Math.min(rect.width - tbox.width - pad2, Math.max(pad2, x + 10)) + 'px';
      tip.style.top = Math.min(rect.height - tbox.height - pad2, Math.max(pad2, y - 28)) + 'px';
    });
    BP.canvas.addEventListener('mouseleave', () => tip && (tip.style.display = 'none'));

    function loop(ts) { if (!BP.t0) BP.t0 = ts; const dt = ts - BP.t0; BP.t0 = ts; step(dt); render(); BP.raf = requestAnimationFrame(loop); }
    BP.raf && cancelAnimationFrame(BP.raf);
    BP.raf = requestAnimationFrame(loop);
  }

  /* ---------------- 13) Hidden Layer demo ---------------- */
  const HL = { canvas: null, hiddenSize: null, hiddenSizeVal: null, shuffleBtn: null, previewBtn: null, tipEl: null, W: null, b: null, Hx: null, Z: null, gridW: null, gridBars: null };

  function ensureHidden() {
    if (HL.inited) return; HL.inited = true;
    HL.canvas = $('#hiddenCanvas'); if (!HL.canvas) return;
    HL.hiddenSize = $('#hiddenSizeHL'); HL.hiddenSizeVal = $('#hiddenSizeHLVal');
    HL.shuffleBtn = $('#shuffleBtnHL'); HL.previewBtn = $('#previewHBtnHL');
    HL.tipEl = $('#slideHidden .hidden-tip');

    HL.hiddenSize?.addEventListener('input', () => HL.hiddenSizeVal && (HL.hiddenSizeVal.textContent = HL.hiddenSize.value));
    HL.hiddenSizeVal && (HL.hiddenSizeVal.textContent = HL.hiddenSize?.value || '');

    HL.shuffleBtn && (HL.shuffleBtn.onclick = () => {
      const inputDim = (S2.lastEncoded?.vector?.length || 512);
      post('init_hidden', { inputDim, hidden: +(HL.hiddenSize?.value || 256) });
    });
    HL.previewBtn && (HL.previewBtn.onclick = () => {
      if (!S2.lastEncoded) { alert('Encode a row on the Vectorization slide first'); return; }
      post('project_hidden', { x: S2.lastEncoded.vector });
    });

    HL.canvas.addEventListener('mousemove', (e) => {
      if (!HL.tipEl) return;
      const rect = HL.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;

      if (HL.gridW) {
        const { x: gx, y: gy, cols, rows, cellW, cellH } = HL.gridW;
        const j = Math.floor((x - gx) / cellW), i = Math.floor((y - gy) / cellH);
        if (i >= 0 && j >= 0 && i < rows && j < cols && HL.W?.[i]?.[j] != null) {
          const val = HL.W[i][j];
          HL.tipEl.textContent = `W[${i},${j}] = ${fmtVal(val)}`;
          HL.tipEl.style.display = 'block';
          HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width - 120, x + 8))}px, ${Math.max(0, y - 28)}px)`;
          return;
        }
      }
      if (HL.gridBars) {
        const { x: bx, y: by, n, eachH } = HL.gridBars;
        if (x >= bx) {
          const i = Math.floor((y - by) / eachH);
          if (i >= 0 && i < n && HL.Hx) {
            const h = HL.Hx[i]; const z = HL.Z ? HL.Z[i] : null;
            HL.tipEl.textContent = z == null ? `H[${i}] = ${fmtVal(h)}` : `H[${i}] = g(z) = ${fmtVal(h)}  (z=${fmtVal(z)})`;
            HL.tipEl.style.display = 'block';
            HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width - 160, x + 8))}px, ${Math.max(0, y - 28)}px)`;
            return;
          }
        }
      }
      HL.tipEl.style.display = 'none';
    });
    HL.canvas.addEventListener('mouseleave', () => HL.tipEl && (HL.tipEl.style.display = 'none'));

    drawHidden();
  }

  function onHiddenInit(payload) {
    HL.W = payload.W; HL.b = payload.b; HL.Hx = null; HL.Z = null;
    const WPreview = $('#WPreview');
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
    const c = HL.canvas; if (!c) return;
    const g = c.getContext('2d'), dpr = devicePixelRatio || 1;
    const Wc = c.clientWidth, Hc = c.clientHeight;
    c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, Wc, Hc);

    const pad = 10, ww = Math.floor(Wc * 0.66);
    const xHeat = pad, yHeat = pad, xBars = ww + pad, yBars = pad;

    if (!HL.W) {
      g.fillStyle = '#93a9e8'; g.fillText('Click “Reseed hidden” to initialize W,b', 12, 22);
      HL.gridW = HL.gridBars = null; return;
    }
    const rows = HL.W.length, cols = HL.W[0].length;
    const MAXD = 256, dr = Math.max(1, Math.ceil(rows / MAXD)), dc = Math.max(1, Math.ceil(cols / MAXD));
    const dsRows = Math.ceil(rows / dr), dsCols = Math.ceil(cols / dc);

    let vmax = 1e-6;
    for (let i = 0; i < rows; i += dr) for (let j = 0; j < cols; j += dc) vmax = Math.max(vmax, Math.abs(HL.W[i][j]));

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
    g.fillStyle = '#a7b8e8'; g.fillText('W (hidden × input)', xHeat, Hc - 8);
    HL.gridW = { x: xHeat, y: yHeat, cols: dsCols, rows: dsRows, cellW, cellH };

    if (HL.Hx?.length) {
      const n = HL.Hx.length;
      const barW = (Wc - xBars - pad), eachH = Math.max(1, Math.floor((Hc - 2 * pad) / n));
      const absmax = Math.max(1e-6, ...HL.Hx.map(x => Math.abs(x)));
      for (let i = 0; i < n; i++) {
        const val = HL.Hx[i], frac = Math.min(1, Math.abs(val) / absmax), len = Math.floor(frac * barW);
        g.fillStyle = val >= 0 ? '#6ee7a2' : '#fb7185';
        g.fillRect(xBars, yBars + i * eachH, len, Math.max(1, eachH - 2));
      }
      g.fillStyle = '#a7b8e8'; g.fillText('Hx = g(Wx + b)', xBars, Hc - 8);
      HL.gridBars = { x: xBars, y: yBars, n, eachH, barW };
    } else {
      g.fillStyle = '#93a9e8'; g.fillText('Encode a row (Vectorization), then click “Project H”.', xBars, 22);
      HL.gridBars = null;
    }
  }

  /* ---------------- 14) Slide 11: ELM train (one-shot β) ---------------- */
  const S4 = { canvas: null, betaVis: null, labels: [], dims: null, classIds: null };
  function ensureELMTrain() {
    if (S4.inited) return; S4.inited = true;

    const trainBtn = $('#trainBtn');
    const downloadBtn = $('#downloadBtn');
    const resetBtn = $('#resetBtn');
    const hiddenSize = $('#hiddenSize');
    const hiddenSizeVal = $('#hiddenSizeVal');
    const solveOut = $('#solveOut');
    S4.canvas = $('#betaCanvas');

    hiddenSize?.addEventListener('input', () => hiddenSizeVal && (hiddenSizeVal.textContent = hiddenSize.value));
    hiddenSizeVal && (hiddenSizeVal.textContent = hiddenSize?.value || '');

    trainBtn && (trainBtn.onclick = () => {
      post('train', { rows: dataRows.map(r => ({ y: r.cls, text: r.text })), hidden: +(hiddenSize?.value || 256) });
      solveOut && (solveOut.textContent = 'Training… (freezing basis, solving β)…');
    });

    downloadBtn && (downloadBtn.onclick = () => post('export_model'));

    resetBtn && (resetBtn.onclick = () => {
      post('reset'); uiBasisFrozen = false;
      S4.betaVis = null; S4.labels = []; S4.dims = null;
      S4.drawBeta && S4.drawBeta();
      solveOut && (solveOut.textContent = 'Reset complete. Re-train to continue.');
      const predictBtn = $('#predictBtn'); if (predictBtn) { predictBtn.disabled = true; predictBtn.title = 'Train first'; }
    });

    function drawBeta() {
      const c = S4.canvas; if (!c) return;
      const g = c.getContext('2d'), dpr = devicePixelRatio || 1;
      const Wc = c.clientWidth, Hc = c.clientHeight;
      c.width = Math.max(1, Wc * dpr); c.height = Math.max(1, Hc * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, Wc, Hc);

      g.font = '600 11px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial';
      const INK = '#e5efff', MUTED = '#a7b8e8';

      if (!S4.betaVis) { g.fillStyle = MUTED; g.fillText('Train first to visualize β.', 12, 22); return; }

      const B = S4.betaVis, R = B.length, C = B[0].length;

      // Build labels
      const colLabels = (
        S4.labels && Array.isArray(S4.labels.cols) && S4.labels.cols.length === C
      ) ? S4.labels.cols
        : Array.from({ length: C }, (_, j) => `class ${j}`);

      const rowLabels = (
        S4.labels && Array.isArray(S4.labels.rows) && S4.labels.rows.length === R
      ) ? S4.labels.rows
        : Array.from({ length: R }, (_, i) => `h${i}`);

      // Layout with margins
      const mL = 68, mR = 16, mT = 42, mB = 42;
      const innerW = Math.max(40, Wc - mL - mR);
      const innerH = Math.max(40, Hc - mT - mB);
      const cw = Math.max(6, Math.floor(innerW / C));
      const ch = Math.max(6, Math.floor(innerH / R));
      const gridW = cw * C, gridH = ch * R;
      const x0 = mL + Math.floor((innerW - gridW) / 2);
      const y0 = mT + Math.floor((innerH - gridH) / 2);

      // Color scale
      let vmax = 1e-6;
      for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) vmax = Math.max(vmax, Math.abs(B[i][j]));
      for (let i = 0; i < R; i++) {
        for (let j = 0; j < C; j++) {
          const v = B[i][j];
          const a = Math.min(1, Math.abs(v) / vmax);
          const hue = v >= 0 ? 200 : 0; // blue for +, red for -
          g.fillStyle = `hsla(${hue},90%,60%,${0.15 + 0.85 * a})`;
          g.fillRect(x0 + j * cw, y0 + i * ch, cw - 1, ch - 1);
        }
      }

      // Row labels
      g.fillStyle = MUTED;
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      for (let i = 0; i < R; i++) {
        const ty = y0 + i * ch + ch / 2;
        g.fillText(rowLabels[i], x0 - 10, ty);
      }

      // Column labels (rotated to fit)
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      for (let j = 0; j < C; j++) {
        const tx = x0 + j * cw + cw / 2;
        const ty = y0 - 8;
        g.save();
        g.translate(tx, ty);
        g.rotate(-Math.PI / 4);
        g.fillText(colLabels[j], 0, 0);
        g.restore();
      }

      // Titles
      g.fillStyle = INK;
      g.textAlign = 'center';
      g.textBaseline = 'alphabetic';
      g.font = '700 13px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial';
      g.fillText('β heatmap', x0 + gridW / 2, y0 - 24);
      g.font = '600 11px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial';
      g.fillStyle = MUTED;
      g.fillText('classes (columns)', x0 + gridW / 2, y0 + gridH + 20);

      // Left vertical title
      g.save();
      g.translate(x0 - 52, y0 + gridH / 2);
      g.rotate(-Math.PI / 2);
      g.fillText('hidden features (rows)', 0, 0);
      g.restore();

      // Legend
      const lgW = 120, lgH = 10, lgX = x0 + gridW - lgW, lgY = y0 + gridH + 26;
      const grad = g.createLinearGradient(lgX, 0, lgX + lgW, 0);
      grad.addColorStop(0, 'hsl(0,90%,60%)');
      grad.addColorStop(0.5, 'hsl(0,0%,40%)');
      grad.addColorStop(1, 'hsl(200,90%,60%)');
      g.fillStyle = grad; g.fillRect(lgX, lgY, lgW, lgH);
      g.fillStyle = MUTED; g.textAlign = 'center'; g.textBaseline = 'top';
      g.fillText('neg', lgX, lgY + lgH + 2);
      g.fillText('0', lgX + lgW / 2, lgY + lgH + 2);
      g.fillText('pos', lgX + lgW, lgY + lgH + 2);
    }

    S4.drawBeta = drawBeta;
    window.addEventListener('resize', () => S4.drawBeta && S4.drawBeta(), { passive: true });

  }

  /* ---------------- 15) Slide 12: Prediction ---------------- */
  const SP = {};
  function ensurePredict() {
    if (SP.inited) return; SP.inited = true;
    const predRowSelect = $('#predRowSelect');
    const predictBtn = $('#predictBtn');
    const predOut = $('#predOut');

    if (predRowSelect) {
      predRowSelect.innerHTML = '';
      for (let i = 0; i < dataRows.length; i++) {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = `[${dataRows[i].cls}] ${dataRows[i].text.slice(0, 80)}…`;
        predRowSelect.appendChild(o);
      }
    }
    predictBtn && (predictBtn.onclick = () => {
      const i = +(predRowSelect?.value || 0);
      post('predict', { text: dataRows[i].text });
      predOut && (predOut.textContent = 'Predicting…');
      SP.lastIndex = i;
    });
  }

  /* ---------------- 16) Worker events (shared) ---------------- */
  worker.onmessage = (e) => {
    const { type, payload } = e.data || {};
    if (type === 'status') { workerStatus && (workerStatus.textContent = payload); return; }

    if (type === 'trained') {
      uiBasisFrozen = true;
      const predictBtn = $('#predictBtn'); if (predictBtn) { predictBtn.disabled = false; predictBtn.title = ''; }
      const solveOut = $('#solveOut');
      const lines = ['Trained ELM ✓'];
      if (payload.note) lines.push(`note: ${payload.note}`);
      if (payload.dims) {
        const d = payload.dims;
        lines.push(`H: ${d.H_rows}×${d.H_cols}  Y: ${d.Y_rows}×${d.Y_cols}  β: ${d.B_rows}×${d.B_cols}`);
      }
      if (payload.betaSample) lines.push('\nβ (8×8 sample):\n' + payload.betaSample);
      solveOut && (solveOut.textContent = lines.join('\n'));

      // --- set visualization + labels ---
      S4.betaVis = payload.betaVis || null;
      S4.dims = payload.dims || null;

      // Column ID order: prefer worker-provided payload.labels; fallback to dataset
      const C = S4.betaVis?.[0]?.length || 0;
      const fromWorker = Array.isArray(payload.labels) && payload.labels.length === C ? payload.labels : null;
      const fromData = [...new Set(dataRows.map(r => r.cls))].sort((a, b) => a - b).slice(0, C);
      S4.classIds = fromWorker || fromData;

      // Map IDs to display names using AG_LABELS, fallback to raw id
      const colNames = (S4.classIds || []).map(id => AG_LABELS[id] || String(id));

      // Row labels from β rows (hidden features)
      const R = S4.betaVis ? S4.betaVis.length : 0;
      S4.labels = {
        rows: Array.from({ length: R }, (_, i) => `h${i}`),
        cols: colNames
      };

      S4.drawBeta && S4.drawBeta();
      return;
    }

    if (type === 'predicted') {
      const probs = softmax(payload.scores || []);
      const labels = payload.labels || [];
      const pred = payload.pred;
      const predOut = $('#predOut') || $('#solveOut');

      const i = SP.lastIndex ?? 0;
      const truthId = dataRows[i]?.cls ?? null;
      const truthName = truthId != null ? (AG_LABELS[truthId] || String(truthId)) : 'N/A';
      const predName = (pred != null) ? (AG_LABELS[pred] || String(pred)) : '—';
      const verdict = (truthId == null) ? 'truth unknown' : (pred === truthId ? '✓ correct' : '✗ incorrect');

      const probText = (labels.length === probs.length && probs.length > 0)
        ? labels.map((lab, k) => `${lab}(${AG_LABELS[lab] || ''}): ${probs[k].toFixed(2)}`).join('  |  ')
        : `[${probs.map(p => p.toFixed(2)).join(', ')}]`;

      predOut && (predOut.textContent =
        `Predicted: ${predName} (${pred})  |  Truth: ${truthName} (${truthId})  |  ${verdict}\n` +
        `Probabilities: ${probText}`);
      return;
    }

    if (type === 'hidden_init') { onHiddenInit(payload); return; }
    if (type === 'hidden_project') { onHiddenProject(payload); return; }

    // 'encoded', 'exported_model', 'actCurve' are handled elsewhere or ignored
  };

  /* ---------------- 17) Bootstrap worker ---------------- */
  post('hello');
  post('list_activations');

}); // DOMContentLoaded
