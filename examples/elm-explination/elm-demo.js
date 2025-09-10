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

const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

/* ---------------- 1) Header / layout refs ---------------- */
const prevBtn     = $('#prevBtn');
const nextBtn     = $('#nextBtn');
const slideLabel  = $('#slideLabel');
const footerBar   = $('.footerBar');
const notesToggle = $('#notesToggle');
const progressBar = $('#progressBar');
const headerEl    = $('header');

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
  const rafState = { ticking:false };
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
  window.addEventListener('scroll', onScroll, { passive:true });
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
  if (e.key === 'ArrowLeft')  prevBtn?.click();
  if (e.key === 'ArrowRight') nextBtn?.click();
  if (e.key.toLowerCase() === 'n') notesToggle?.click();
});

/* Stage chips (if any in header) */
const stageChips = $$('[data-stagechip]');

/* ---------------- 4) Minimap (left SVG) ---------------- */
/* We build once, keep refs to columns, and expose an update call */
const minimapState = {
  svg: null, colIn: null, colH: null, colOut: null, dotsHost: null
};

function buildRoadmapFromMeta() {
  const roadmap = $('#slide0 .right ol');
  if (!roadmap || !SLIDE_META) return;
  roadmap.innerHTML = '';
  SLIDE_META.forEach(s => {
    const li = document.createElement('li');
    li.textContent = s.title || s.id;
    roadmap.appendChild(li);
  });
}

function buildMinimap() {
  const host = $('#minimap');
  minimapState.dotsHost = $('#mmDots');
  if (!host) return;

  const width = 220, height = 360, pad = 14;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <defs>
      <linearGradient id="mmWires" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#2a64ff"/><stop offset="1" stop-color="#5ad1ff"/>
      </linearGradient>
      <filter id="mmGlow" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="0" stdDeviation="2" flood-color="#5ad1ff" flood-opacity=".8"/>
      </filter>
    </defs>
    <g id="col-input"></g>
    <g id="col-hidden"></g>
    <g id="col-output"></g>
    <g id="wire-group"></g>
  `;
  host.innerHTML = '';
  host.appendChild(svg);

  const N_in=4,N_h=7,N_out=3;
  const xIn = pad + 20, xH = width/2, xOut = width - pad - 20;
  const yFor = (n,idx)=> pad + 24 + (idx * (height - 2*pad - 48)) / (n - 1);

  const colIn  = svg.querySelector('#col-input');
  const colH   = svg.querySelector('#col-hidden');
  const colOut = svg.querySelector('#col-output');
  const wiresG = svg.querySelector('#wire-group');

  function makeNode(x,y, stageTag){
    const g = document.createElementNS(svg.namespaceURI,'g');
    g.classList.add('mm-node');
    g.innerHTML = `
      <circle cx="${x}" cy="${y}" r="9" fill="#0c1a3d" stroke="#203a7c" stroke-width="1.5"/>
      <circle cx="${x}" cy="${y}" r="9" fill="transparent" stroke="url(#mmWires)" stroke-width="1.5" opacity=".75"/>
    `;
    g.style.cursor='pointer';
    g.addEventListener('click', () => {
      /* map stage to a slide index, if present in meta; fallback to first */
      const data = (SLIDE_META || slides.map(s=>({id:s.id,title:s.id,stage:s.stage})));
      const byStage = (name) => {
        const k = data.findIndex(d => d.stage === name);
        return (k >= 0) ? k : 0;
      };
      const target =
        stageTag === 'input'  ? byStage('input')  :
        stageTag === 'hidden' ? byStage('hidden') :
        stageTag === 'output' ? byStage('output') :
        byStage('overview');
      show(target);
    });
    return g;
  }

  for (let i=0;i<N_in;i++)  colIn.appendChild(makeNode(xIn,  yFor(N_in,i),  'input'));
  for (let i=0;i<N_h;i++)   colH.appendChild(makeNode(xH,   yFor(N_h,i),   'hidden'));
  for (let i=0;i<N_out;i++) colOut.appendChild(makeNode(xOut, yFor(N_out,i), 'output'));

  // wires: input→hidden
  for (let i=0;i<N_in;i++) for (let j=0;j<N_h;j++){
    const p = document.createElementNS(svg.namespaceURI,'path');
    p.setAttribute('d', `M ${xIn+9} ${yFor(N_in,i)} C ${xIn+40} ${yFor(N_in,i)}, ${xH-40} ${yFor(N_h,j)}, ${xH-9} ${yFor(N_h,j)}`);
    p.setAttribute('stroke','url(#mmWires)');
    p.setAttribute('stroke-width','1'); p.setAttribute('fill','none'); p.setAttribute('opacity','.35');
    wiresG.appendChild(p);
  }
  // wires: hidden→output
  for (let i=0;i<N_h;i++) for (let j=0;j<N_out;j++){
    const p = document.createElementNS(svg.namespaceURI,'path');
    p.setAttribute('d', `M ${xH+9} ${yFor(N_h,i)} C ${xH+40} ${yFor(N_h,i)}, ${xOut-40} ${yFor(N_out,j)}, ${xOut-9} ${yFor(N_out,j)}`);
    p.setAttribute('stroke','url(#mmWires)');
    p.setAttribute('stroke-width','1'); p.setAttribute('fill','none'); p.setAttribute('opacity','.35');
    wiresG.appendChild(p);
  }

  // mobile dots
  if (minimapState.dotsHost) {
    minimapState.dotsHost.innerHTML = '';
    (SLIDE_META || slides).forEach((_,i)=>{
      const b = document.createElement('button'); b.textContent = String(i+1);
      b.onclick = ()=> show(i);
      minimapState.dotsHost.appendChild(b);
    });
  }

  // keep refs
  minimapState.svg    = svg;
  minimapState.colIn  = colIn;
  minimapState.colH   = colH;
  minimapState.colOut = colOut;
}

/* Called by show(i) to update glow + dot state */
function updateMinimapHighlight(i) {
  const stage = (slides[i]||{}).stage;
  const svg = minimapState.svg;
  if (!svg) return;
  svg.querySelectorAll('circle').forEach(c => c.removeAttribute('filter'));
  if (stage === 'input')  minimapState.colIn?.querySelectorAll('circle').forEach(c => c.setAttribute('filter','url(#mmGlow)'));
  if (stage === 'hidden') minimapState.colH?.querySelectorAll('circle').forEach(c => c.setAttribute('filter','url(#mmGlow)'));
  if (stage === 'output') minimapState.colOut?.querySelectorAll('circle').forEach(c => c.setAttribute('filter','url(#mmGlow)'));
  // mobile dots active state
  $$('#mmDots button').forEach((b,k)=> b.classList.toggle('active', k===i));
}

/* ---------------- 5) Navigation core ---------------- */
function updateHeaderAndProgress(i){
  prevBtn && (prevBtn.disabled = (i === 0));
  nextBtn && (nextBtn.disabled = (i === slides.length - 1));
  slideLabel && (slideLabel.textContent = `Slide ${i + 1} / ${slides.length}`);
  const pct = Math.max(0, Math.min(100, Math.round(((i + 1) / slides.length) * 100)));
  if (progressBar) progressBar.style.width = pct + '%';
  // highlight stage chips (if you use them)
  stageChips.forEach(ch => ch.classList.toggle('active', ch.getAttribute('data-stagechip') === (slides[i]?.stage)));
}

function applyPerSlideSizing(i){
  const s = slides[i].node;
  const hero  = s.querySelector(':scope > .hero');
  const left  = s.querySelector(':scope .left');
  const right = s.querySelector(':scope .right');
  const hHero = +s.dataset.heroH  || 200;
  const hLeft = +s.dataset.leftH  || 360;
  const hRight= +s.dataset.rightH || 360;
  if (hero)  hero.style.height = hHero + 'px';
  if (left)  left.style.minHeight = hLeft + 'px';
  if (right) right.style.minHeight = hRight + 'px';
}

/* Main show() */
function show(i){
  idx = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s,j)=> s.node.hidden = (j !== idx));
  updateHeaderAndProgress(idx);
  location.hash = `#${idx + 1}`;

  const { id } = slides[idx];
  // init on demand
  if (id === 'slide1')      ensureNeuron();
  if (id === 'slide2a')     ensureNNOverview();
  if (id === 'slide2')      ensureVectorize();
  if (id === 'slideBP')     ensureBackprop();
  if (id === 'slideHidden') ensureHidden();
  if (id === 'slide4')      ensureELMTrain();
  if (id === 'slidePred')   ensurePredict();
  if (id === 'slideIntroNeuron') ensureIceCone?.();

  applyPerSlideSizing(idx);
  updateMinimapHighlight(idx);
}

/* Prev/Next buttons (null-safe) */
prevBtn && (prevBtn.onclick = () => show(idx - 1));
nextBtn && (nextBtn.onclick = () => show(idx + 1));

/* Navigate to hash or 0 after everything is wired */
function navigateInitialSlide(){
  if (location.hash) {
    const n = parseInt(location.hash.replace('#',''), 10);
    show(Number.isFinite(n) ? n - 1 : 0);
  } else {
    show(0);
  }
}

/* ---------------- 6) Load slide meta AFTER nav is ready ---------------- */
(async function loadSlideMetaThenBuild(){
  try {
    const res = await fetch('slides.json', { cache: 'no-store' });
    if (res.ok) SLIDE_META = await res.json();
  } catch {}
  buildRoadmapFromMeta();
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
const post = (type, payload={}) => worker.postMessage({ type, payload });

let uiBasisFrozen = false;
const AG_LABELS = { 1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech' };

/* Tiny dataset for vectorization/prediction */
const CSV_SNIPPET = `Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
3,Carlyle Looks Toward Commercial Aerospace (Reuters),"Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market."
1,UN Council Weighs Ceasefire Proposal,"Leaders from several nations met to discuss a draft resolution aimed at de-escalation in the region."
4,Chip Startup Unveils Faster AI Accelerator,"The company claims a 2× speedup on transformer inference with a new memory layout."
2,Local Club Wins Championship Final,"Fans celebrated after the underdogs clinched the title with a late goal."`;

function parseCSVMini(s){
  const lines = s.trim().split(/\r?\n/); lines.shift();
  const out=[];
  for (const line of lines) {
    const m = line.match(/^(\d+),([^,]+),(.*)$/);
    if (!m) continue;
    const cls = +m[1];
    const title = m[2].trim();
    let desc = m[3].trim();
    if (desc.startsWith('"') && desc.endsWith('"')) desc = desc.slice(1,-1);
    desc = desc.replaceAll('\\b',' ').replaceAll('\\which',' which').replaceAll('\\',' ');
    out.push({ cls, text: `${title}. ${desc}` });
  }
  return out;
}
const dataRows = parseCSVMini(CSV_SNIPPET);

function softmax(arr){ if(!arr?.length) return []; const m=Math.max(...arr); const exps=arr.map(x=>Math.exp(x-m)); const s=exps.reduce((a,b)=>a+b,0); return exps.map(e=>e/s); }
const fmtVal = (v) => {
  if (!Number.isFinite(v)) return '0';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e3 || a < 1e-3) return v.toExponential(2);
  return v.toFixed(3);
};

/* ---------------- 8) Slide 2a: overview animation ---------------- */
function ensureNNOverview(){
  if (ensureNNOverview.inited) return; ensureNNOverview.inited = true;
  const c = $('#nnCanvas'); if (!c) return;
  const g = c.getContext('2d');
  const text = $('#nnText'); const run = $('#nnRun');
  const nodes = { in:[], hid:[], out:[] }, particles = [];
  const DPR = devicePixelRatio || 1;

  function resize(){
    const W=c.clientWidth, H=c.clientHeight; c.width=W*DPR; c.height=H*DPR; g.setTransform(DPR,0,0,DPR,0,0);
    layout();
  }
  const colX = ()=> [30, c.clientWidth/2, c.clientWidth-30];
  function layout(){
    const H=c.clientHeight; const [x0,x1,x2] = colX();
    const yFor = (n,i)=> 20 + (i*(H-40))/(n-1);
    nodes.in  = Array.from({length:5}, (_,i)=>({x:x0,y:yFor(5,i), r:10, stage:'input'}));
    nodes.hid = Array.from({length:8}, (_,i)=>({x:x1,y:yFor(8,i), r:11, stage:'neuron'}));
    nodes.out = Array.from({length:3}, (_,i)=>({x:x2,y:yFor(3,i), r:10, stage:'output'}));
  }
  function spawnParticles(){
    particles.length=0;
    for(let k=0;k<18;k++){
      const s = nodes.in[k%nodes.in.length];
      const h = nodes.hid[Math.floor(Math.random()*nodes.hid.length)];
      const o = nodes.out[Math.floor(Math.random()*nodes.out.length)];
      particles.push({ path:[s,h,o], t:0 });
    }
  }
  function step(dt){ particles.forEach(p=>{ p.t = Math.min(1, p.t + dt*0.0004); }); }
  function draw(){
    const W=c.clientWidth, H=c.clientHeight; g.clearRect(0,0,W,H);
    // wires
    g.strokeStyle='rgba(90,209,255,.35)';
    nodes.in.forEach(a=> nodes.hid.forEach(b=>{ g.beginPath(); g.moveTo(a.x,a.y); g.bezierCurveTo(a.x+40,a.y, b.x-40,b.y, b.x,b.y); g.stroke(); }));
    nodes.hid.forEach(a=> nodes.out.forEach(b=>{ g.beginPath(); g.moveTo(a.x,a.y); g.bezierCurveTo(a.x+40,a.y, b.x-40,b.y, b.x,b.y); g.stroke(); }));
    // nodes
    const dot=(n,glow)=>{ g.beginPath(); g.arc(n.x,n.y,n.r,0,Math.PI*2);
      g.fillStyle=glow?'#103a6b':'#0c1a3d'; g.fill();
      g.lineWidth=1.5; g.strokeStyle='#203a7c'; g.stroke();
      if(glow){ g.strokeStyle='rgba(90,209,255,.9)'; g.stroke(); }
    };
    nodes.in.forEach(n=>dot(n,false));
    nodes.hid.forEach(n=>dot(n,true));
    nodes.out.forEach(n=>dot(n,false));
    // particles
    particles.forEach(p=>{
      const [a,b,c3]=p.path; const t=p.t;
      const seg = t<.6 ? [a,b, t/.6] : [b,c3, (t-.6)/.4];
      const [p0,p1,tt] = seg;
      const x = p0.x + (p1.x-p0.x)*tt, y = p0.y + (p1.y-p0.y)*tt;
      g.beginPath(); g.arc(x,y,3,0,Math.PI*2); g.fillStyle='#6ee7a2'; g.fill();
    });
  }
  let last=0; function loop(ts){ const dt= ts-(last||ts); last=ts; step(dt); draw(); requestAnimationFrame(loop); }

  function hitNeuron(x,y){
    return nodes.hid.find(n => (x-n.x)**2+(y-n.y)**2 <= (n.r+2)**2);
  }

  window.addEventListener('resize', resize);
  resize(); spawnParticles(); requestAnimationFrame(loop);

  // click any neuron to zoom → Neuron slide
  c.addEventListener('click', (e)=>{
    const r=c.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
    if (hitNeuron(x,y)) {
      const k = slides.findIndex(s=>s.id==='slide1');
      show(k>=0 ? k : 0);
    }
  });

  run?.addEventListener('click', ()=> spawnParticles());
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
      big:  '700 16px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial'
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
      c.width  = Math.max(1, W * DPR);
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
      const sumX  = Math.round(W * 0.38);
      const actX  = Math.round(W * 0.60);
      const outX  = W - pad - 30;

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
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
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
      ctx.translate(x + w/2, y + h/2);
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
      ctx.fillText("ACTIVATION  f", x + w/2, y + h + 14);
      ctx.restore();
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r, y);
      ctx.lineTo(x+w-r, y);
      ctx.quadraticCurveTo(x+w, y, x+w, y+r);
      ctx.lineTo(x+w, y+h-r);
      ctx.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
      ctx.lineTo(x+r, y+h);
      ctx.quadraticCurveTo(x, y+h, x, y+h-r);
      ctx.lineTo(x, y+r);
      ctx.quadraticCurveTo(x, y, x+r, y);
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
        const [x1,y1] = pts[i], [x2,y2] = pts[i+1];
        const d = Math.hypot(x2-x1, y2-y1);
        segs.push({x1,y1,x2,y2,d});
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
          ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.fill();
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
        g.fillText(`INPUT ${i+1}`, L.inStartX + 4, y - 12);
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
      drawActivation(g, L.actX, L.centerY - L.actH/2, L.actW, L.actH, t);

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
      g.fillText("OUTPUT (y)", L.outX , L.centerY);
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
const S1 = { act:'relu', rafId:null, canvas:null };
function ensureNeuron(){
  if (S1.inited) return; S1.inited = true;
  S1.canvas = $('#neuronCanvas'); if (!S1.canvas) return;

  const actSelect = $('#actSelect');
  const wRange = $('#wRange'); const bRange = $('#bRange');
  const wVal = $('#wVal');     const bVal = $('#bVal');
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

  function updateAmplitude(){
    const w=+wRange.value, b=+bRange.value;
    const amp = Math.max(Math.abs(w*3 + b), Math.abs(w*-3 + b));
    ampVal.textContent = amp.toFixed(2);
    const pct = Math.min(100, (amp / 20) * 100);
    ampBar.style.width = pct + '%';
  }
  function updatePostAmplitude(){
    const w=+wRange.value, b=+bRange.value;
    let maxAbsOutput = 0;
    for (let i=0; i<=300; i++){
      const x = -3 + (i/300)*6;
      const y = actFn(w*x + b);
      maxAbsOutput = Math.max(maxAbsOutput, Math.abs(y));
    }
    postAmpVal.textContent = maxAbsOutput.toFixed(2);
    const maxPossible = (S1.act === 'sigmoid' || S1.act === 'tanh') ? 1 : 20;
    const pct = Math.min(100, (maxAbsOutput / maxPossible) * 100);
    postAmpBar.style.width = pct + '%';
  }

  function loop(){
    if (!S1.canvas) return;
    const c=S1.canvas, g=c.getContext('2d'), dpr=devicePixelRatio||1;
    const W=c.clientWidth, H=c.clientHeight;
    c.width=Math.max(1,W*dpr); c.height=Math.max(1,H*dpr);
    g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);

    // axes
    g.strokeStyle='#3857a8'; g.lineWidth=1;
    g.beginPath(); g.moveTo(10,H-20); g.lineTo(W-10,H-20); g.stroke();
    g.beginPath(); g.moveTo(40,10); g.lineTo(40,H-10); g.stroke();

    const w=+wRange.value, b=+bRange.value;
    const toXY = (x,y)=>{ const xm=(x+3)/6, ym=(y-(-2))/(2-(-2)); return [40 + xm*(W-55), (H-20) - ym*(H-35)]; };

    g.strokeStyle='#5ad1ff'; g.lineWidth=2; g.beginPath();
    for (let i=0;i<=300;i++){
      const x = -3 + (i/300)*6, y = actFn(w*x + b);
      const [px,py] = toXY(x, Math.max(-2, Math.min(2, y)));
      if (i===0) g.moveTo(px,py); else g.lineTo(px,py);
    }
    g.stroke();

    const now = performance.now()/1000;
    for (let k=0;k<7;k++){
      const x = -3 + ((now*0.6 + k/7) % 1) * 6, y = actFn(w*x + b);
      const [px,py] = toXY(x, Math.max(-2, Math.min(2, y)));
      g.fillStyle='#6ee7a2'; g.beginPath(); g.arc(px,py,3.2,0,Math.PI*2); g.fill();
    }
    S1.rafId = requestAnimationFrame(loop);
  }

  actSelect?.addEventListener('change', ()=>{ S1.act = actSelect.value; updatePostAmplitude(); });
  [wRange,bRange].forEach(r => r?.addEventListener('input', ()=>{
    wVal.textContent = (+wRange.value).toFixed(2);
    bVal.textContent = (+bRange.value).toFixed(2);
    updateAmplitude(); updatePostAmplitude();
  }));
  wVal.textContent=(+wRange.value).toFixed(2);
  bVal.textContent=(+bRange.value).toFixed(2);
  updateAmplitude(); updatePostAmplitude();
  if (S1.rafId) cancelAnimationFrame(S1.rafId);
  loop();
}

/* ---------------- 11) Slide: Vectorization ---------------- */
const S2 = { lastEncoded:null, grid:null, featureLimit:128, lastMethod:'tfidf' };

function ensureVectorize(){
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
    rowSelect.innerHTML='';
    for (let i=0;i<dataRows.length;i++){
      const o=document.createElement('option');
      o.value=String(i);
      o.textContent=`[${dataRows[i].cls}] ${dataRows[i].text.slice(0,80)}…`;
      rowSelect.appendChild(o);
    }
  }

  S2.featureLimit = +(featureLimit?.value || 128);
  featureLimit?.addEventListener('input', ()=>{
    S2.featureLimit = +featureLimit.value;
    featureLimitVal.textContent = featureLimit.value;
    drawEncode();
  });
  encodingSelect?.addEventListener('change', ()=>{ S2.lastMethod = encodingSelect.value; });

  encodeBtn && (encodeBtn.onclick = ()=>{
    const i = +(rowSelect?.value || 0);
    const method = encodingSelect?.value || 'tfidf';
    const corpus = dataRows.map(r=>r.text);
    post('encode', { text:dataRows[i].text, method, corpus });
  });

  // tooltip
  let tipEl = $('#slide2 .encode-tip');
  if (!tipEl && encodeCanvas?.parentElement) {
    tipEl = document.createElement('div'); tipEl.className='encode-tip';
    encodeCanvas.parentElement.appendChild(tipEl);
  }

  encodeCanvas?.addEventListener('mousemove', (e)=>{
    if (!S2.grid || !S2.lastEncoded || !tipEl) { if (tipEl) tipEl.style.display='none'; return; }
    const { x0,y0,cols,rows,cw,ch,n } = S2.grid;
    const rect = encodeCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y=e.clientY - rect.top;
    const inside = cw>0 && ch>0 && x>=x0 && y>=y0 && x<x0+cols*cw && y<y0+rows*ch;
    if (!inside){ tipEl.style.display='none'; return; }
    const col = Math.floor((x-x0)/cw);
    const row = Math.floor((y-y0)/ch);
    const k = row*cols + col;
    if (k<0 || k>=n){ tipEl.style.display='none'; return; }
    const v = S2.lastEncoded.vector;
    const names = S2.lastEncoded.featureNames || [];
    const token = names[k] || null;
    tipEl.textContent = `${token ? `#${k} “${token}”` : `feature #${k}`} = ${fmtVal(v[k])}`;
    tipEl.style.display='block';
    const tbox = tipEl.getBoundingClientRect(), pad=6;
    const left = Math.min(rect.width - tbox.width - pad, Math.max(pad, x + 10));
    const top  = Math.min(rect.height - tbox.height - pad, Math.max(pad, y - 28));
    tipEl.style.left = `${left}px`;
    tipEl.style.top  = `${top}px`;
  });
  encodeCanvas?.addEventListener('mouseleave', ()=> tipEl && (tipEl.style.display='none'));

  function drawEncode(){
    if (!encodeCanvas) return;
    const c=encodeCanvas, dpr=devicePixelRatio||1, W=c.clientWidth, H=c.clientHeight;
    c.width=Math.max(1,W*dpr); c.height=Math.max(1,H*dpr);
    const g=c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
    if (!S2.lastEncoded){
      g.fillStyle='#93a9e8'; g.fillText('Click “Encode text” to preview the input vector.',12,22);
      S2.grid=null; return;
    }
    const v = S2.lastEncoded.vector.slice();
    // normalize for contrast
    const norm = Math.hypot(...v) || 1; for (let i=0;i<v.length;i++) v[i]/=norm;

    const n = Math.min(S2.featureLimit || 128, v.length);
    const cols = Math.ceil(Math.sqrt(n)); const rows = Math.ceil(n/cols);
    const m=10, x0=m, y0=m; const cw=Math.floor((W-2*m)/cols), ch=Math.floor((H-2*m)/rows);
    const vSub = v.slice(0,n);
    const maxAbs = Math.max(1e-6, ...vSub.map(x=>Math.abs(x)));
    let k=0;
    for (let r=0; r<rows; r++){
      for (let ccol=0; ccol<cols; ccol++, k++){
        if (k>=n) break;
        const val=vSub[k]; const alpha=Math.min(1, Math.abs(val)/maxAbs);
        const hue = val>=0 ? 200 : 0; const X=x0+ccol*cw, Y=y0+r*ch;
        const bg = `hsla(${hue},90%,60%,${0.15 + 0.85*alpha})`;
        g.fillStyle = bg; g.fillRect(X, Y, cw-2, ch-2);
        if (cw>=56 && ch>=36){
          g.save(); g.fillStyle='rgba(255,255,255,0.95)';
          g.font=`600 ${Math.min(18, Math.floor(ch*0.42))}px ui-sans-serif,system-ui`;
          g.textAlign='center'; g.textBaseline='middle';
          g.fillText(fmtVal(val), X+(cw-2)/2, Y+(ch-2)/2); g.restore();
        }
      }
    }
    g.strokeStyle='rgba(255,255,255,0.06)'; g.lineWidth=1;
    for (let ccol=0; ccol<=cols; ccol++){ g.beginPath(); g.moveTo(x0+ccol*cw, y0); g.lineTo(x0+ccol*cw, y0+rows*ch); g.stroke(); }
    for (let rr=0; rr<=rows; rr++){ g.beginPath(); g.moveTo(x0, y0+rr*ch); g.lineTo(x0+cols*cw, y0+rr*ch); g.stroke(); }
    g.fillStyle='#a7b8e8';
    g.fillText(`features 0..${n-1} — basis: ${uiBasisFrozen ? 'trained' : 'isolated'}`, 12, H-12);

    S2.grid = { x0,y0,cols,rows,cw,ch,n };
  }

  // worker event for this slide
  worker.addEventListener('message', (e)=>{
    const { type, payload } = e.data || {};
    if (type !== 'encoded') return;
    S2.lastEncoded = payload;
    const chosen = payload.methodUsed || encodingSelect?.value || 'tfidf';
    const label = chosen === 'tfidf' ? 'TF-IDF' : chosen === 'bow' ? 'Bag-of-Words' : 'Isolated';
    if (tokensOut) {
      tokensOut.textContent = `${label} tokens (top):\n${(payload.tokens||[]).slice(0,25).join(' ')}\n\nvector length: ${payload.vector.length}`;
    }
    // redraw
    drawEncode();
  });
}

/* ---------------- 12) Slide: Backprop demo ---------------- */
const BP = { canvas:null, lrRange:null, lrVal:null, raf:null, t0:0, state:null };
function ensureBackprop(){
  if (BP.inited) return; BP.inited = true;
  BP.canvas = $('#bpCanvas'); if (!BP.canvas) return;
  BP.lrRange = $('#bpLR'); BP.lrVal = $('#bpLRVal');
  BP.lrVal && (BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2));

  const restartBtn = $('#bpRestart');
  const scenarioSel = $('#bpScenario');

  function reset(kind='converge'){
    const H=24,W=36;
    BP.state = { weights:Array.from({length:H},()=>Array.from({length:W},()=> (Math.random()*2-1)*0.6)), loss:1.0, noise:0.0, kind };
    BP.points=[]; BP.convergedFrames=0; BP.holdUntil = performance.now()+400;
  }
  reset('converge');
  restartBtn?.addEventListener('click', ()=> reset(scenarioSel?.value || 'converge'));
  scenarioSel?.addEventListener('change', ()=> reset(scenarioSel.value));

  // tooltip
  let tip = $('#slideBP .bp-tip');
  if (!tip && BP.canvas.parentElement) {
    tip = document.createElement('div'); tip.className='bp-tip';
    BP.canvas.parentElement.appendChild(tip);
  }

  BP.lrRange?.addEventListener('input', ()=> BP.lrVal && (BP.lrVal.textContent = (+BP.lrRange.value).toFixed(2)));

  function step(dt){
    if (performance.now() < (BP.holdUntil||0)) return;
    const lr = +(BP.lrRange?.value || 0.5);
    const kind = BP.state.kind || 'converge';
    const base = kind==='converge' ? (0.985 - 0.25*lr*0.01)
               : kind==='vanish'   ? 0.997
               :                      0.99;
    BP.state.loss = Math.max(0.02, BP.state.loss * base);
    const H=BP.state.weights.length, W=BP.state.weights[0].length;
    for (let i=0;i<H;i++) for (let j=0;j<W;j++){
      let w = BP.state.weights[i][j];
      let grad = w;
      if (kind==='vanish')  grad *= 0.02;
      if (kind==='explode') grad *= 3.0;
      const noise = (Math.random()*2-1) * 0.02 * (kind==='explode'?2:1);
      BP.state.weights[i][j] = w - lr * 0.01 * grad + noise;
    }
  }

  let cellGeom=null;
  function render(){
    const c=BP.canvas, g=c.getContext('2d'), dpr=devicePixelRatio||1;
    const Wc=c.clientWidth, Hc=c.clientHeight;
    c.width=Math.max(1,Wc*dpr); c.height=Math.max(1,Hc*dpr);
    g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,Wc,Hc);

    const H=BP.state.weights.length, W=BP.state.weights[0].length;
    const pad=10, gridW=Math.floor(Wc*0.66);
    const cellW=Math.floor((gridW-2*pad)/W);
    const cellH=Math.floor((Hc-2*pad)/H);
    cellGeom={ x0:pad,y0:pad,cw:cellW,ch:cellH,rows:H,cols:W };

    let vmax=1e-6;
    for (let i=0;i<H;i++) for (let j=0;j<W;j++) vmax=Math.max(vmax, Math.abs(BP.state.weights[i][j]));
    for (let i=0;i<H;i++){
      for (let j=0;j<W;j++){
        const val=BP.state.weights[i][j];
        const a=Math.min(1, Math.abs(val)/vmax);
        const hue= val>=0 ? 200 : 0;
        g.fillStyle=`hsla(${hue},90%,60%,${0.15+0.85*a})`;
        g.fillRect(pad + j*cellW, pad + i*cellH, cellW-1, cellH-1);
      }
    }
    g.fillStyle='#a7b8e8'; g.fillText('Hidden weights (changing each step)', pad, Hc-8);

    // loss chart
    const x0=gridW + 20, y0=pad, w=Wc - x0 - pad, h=Hc - 2*pad;
    g.strokeStyle='#3857a8'; g.strokeRect(x0,y0,w,h);
    BP.points = BP.points || [];
    BP.points.push(BP.state.loss);
    if (BP.points.length > 240) BP.points.shift();
    const minL=Math.min(...BP.points,0), maxL=Math.max(...BP.points,1);
    g.strokeStyle='#6ee7a2'; g.lineWidth=2; g.beginPath();
    BP.points.forEach((L,i) => {
      const px = x0 + (i / (240-1)) * w;
      const py = y0 + h - ((L - minL)/(maxL - minL || 1))*h;
      if (i===0) g.moveTo(px,py); else g.lineTo(px,py);
    });
    g.stroke();

    g.fillStyle='#a7b8e8';
    const lrTxt = (+(BP.lrRange?.value || 0.5)).toFixed(2);
    g.fillText(`loss ~ ${BP.state.loss.toFixed(3)}  (learning rate ${lrTxt})`, x0+6, y0+16);
  }

  BP.canvas.addEventListener('mousemove', (e)=>{
    if (!cellGeom || !tip) { tip && (tip.style.display='none'); return; }
    const rect=BP.canvas.getBoundingClientRect();
    const x=e.clientX-rect.left, y=e.clientY-rect.top;
    const {x0,y0,cw,ch,rows,cols} = cellGeom;
    if (x<x0 || y<y0){ tip.style.display='none'; return; }
    const c = Math.floor((x-x0)/cw), r = Math.floor((y-y0)/ch);
    if (c<0 || c>=cols || r<0 || r>=rows){ tip.style.display='none'; return; }
    const v = BP.state.weights[r][c];
    tip.textContent = `w[${r},${c}] = ${v.toFixed(3)}`;
    const tbox=tip.getBoundingClientRect(), pad2=6;
    tip.style.display='block';
    tip.style.left = Math.min(rect.width - tbox.width - pad2, Math.max(pad2, x+10))+'px';
    tip.style.top  = Math.min(rect.height - tbox.height - pad2, Math.max(pad2, y-28))+'px';
  });
  BP.canvas.addEventListener('mouseleave', ()=> tip && (tip.style.display='none'));

  function loop(ts){ if(!BP.t0) BP.t0=ts; const dt=ts-BP.t0; BP.t0=ts; step(dt); render(); BP.raf=requestAnimationFrame(loop); }
  BP.raf && cancelAnimationFrame(BP.raf);
  BP.raf = requestAnimationFrame(loop);
}

/* ---------------- 13) Hidden Layer demo ---------------- */
const HL = { canvas:null, hiddenSize:null, hiddenSizeVal:null, shuffleBtn:null, previewBtn:null, tipEl:null, W:null, b:null, Hx:null, Z:null, gridW:null, gridBars:null };

function ensureHidden(){
  if (HL.inited) return; HL.inited = true;
  HL.canvas = $('#hiddenCanvas'); if (!HL.canvas) return;
  HL.hiddenSize = $('#hiddenSizeHL'); HL.hiddenSizeVal = $('#hiddenSizeHLVal');
  HL.shuffleBtn = $('#shuffleBtnHL'); HL.previewBtn = $('#previewHBtnHL');
  HL.tipEl = $('#slideHidden .hidden-tip');

  HL.hiddenSize?.addEventListener('input', ()=> HL.hiddenSizeVal && (HL.hiddenSizeVal.textContent = HL.hiddenSize.value));
  HL.hiddenSizeVal && (HL.hiddenSizeVal.textContent = HL.hiddenSize?.value || '');

  HL.shuffleBtn && (HL.shuffleBtn.onclick = ()=>{
    const inputDim = (S2.lastEncoded?.vector?.length || 512);
    post('init_hidden', { inputDim, hidden: +(HL.hiddenSize?.value || 256) });
  });
  HL.previewBtn && (HL.previewBtn.onclick = ()=>{
    if (!S2.lastEncoded) { alert('Encode a row on the Vectorization slide first'); return; }
    post('project_hidden', { x: S2.lastEncoded.vector });
  });

  HL.canvas.addEventListener('mousemove', (e)=>{
    if (!HL.tipEl) return;
    const rect = HL.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;

    if (HL.gridW){
      const { x:gx,y:gy,cols,rows,cellW,cellH } = HL.gridW;
      const j = Math.floor((x-gx)/cellW), i = Math.floor((y-gy)/cellH);
      if (i>=0 && j>=0 && i<rows && j<cols && HL.W?.[i]?.[j] != null){
        const val = HL.W[i][j];
        HL.tipEl.textContent = `W[${i},${j}] = ${fmtVal(val)}`;
        HL.tipEl.style.display='block';
        HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width-120, x+8))}px, ${Math.max(0, y-28)}px)`;
        return;
      }
    }
    if (HL.gridBars){
      const { x:bx, y:by, n, eachH } = HL.gridBars;
      if (x >= bx){
        const i = Math.floor((y-by)/eachH);
        if (i>=0 && i<n && HL.Hx){
          const h = HL.Hx[i]; const z = HL.Z ? HL.Z[i] : null;
          HL.tipEl.textContent = z == null ? `H[${i}] = ${fmtVal(h)}` : `H[${i}] = g(z) = ${fmtVal(h)}  (z=${fmtVal(z)})`;
          HL.tipEl.style.display='block';
          HL.tipEl.style.transform = `translate(${Math.max(0, Math.min(rect.width-160, x+8))}px, ${Math.max(0, y-28)}px)`;
          return;
        }
      }
    }
    HL.tipEl.style.display='none';
  });
  HL.canvas.addEventListener('mouseleave', ()=> HL.tipEl && (HL.tipEl.style.display='none'));

  drawHidden();
}

function onHiddenInit(payload){
  HL.W = payload.W; HL.b = payload.b; HL.Hx = null; HL.Z = null;
  const WPreview = $('#WPreview');
  if (WPreview && HL.W?.length){
    const sample=(M,r=8,c=8)=>{ const R=Math.min(r,M.length), C=Math.min(c,M[0]?.length||0);
      let s=''; for (let i=0;i<R;i++) s += M[i].slice(0,C).map(v=> (Math.abs(v)<1e-3 ? '0.000' : (+v).toFixed(3))).join(' ') + '\n';
      return s;
    };
    WPreview.textContent = `W: ${HL.W.length}x${HL.W[0]?.length||0}  b: ${HL.b.length}  (8×8 sample)\n` + sample(HL.W);
  }
  drawHidden();
}

function onHiddenProject(payload){
  HL.Hx = payload.Hx || null;
  HL.Z  = payload.Z  || null;
  drawHidden();
}

function drawHidden(){
  const c = HL.canvas; if (!c) return;
  const g=c.getContext('2d'), dpr=devicePixelRatio||1;
  const Wc=c.clientWidth, Hc=c.clientHeight;
  c.width=Math.max(1,Wc*dpr); c.height=Math.max(1,Hc*dpr);
  g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,Wc,Hc);

  const pad=10, ww=Math.floor(Wc*0.66);
  const xHeat=pad, yHeat=pad, xBars=ww+pad, yBars=pad;

  if (!HL.W){
    g.fillStyle='#93a9e8'; g.fillText('Click “Reseed hidden” to initialize W,b',12,22);
    HL.gridW = HL.gridBars = null; return;
  }
  const rows=HL.W.length, cols=HL.W[0].length;
  const MAXD=256, dr=Math.max(1, Math.ceil(rows/MAXD)), dc=Math.max(1, Math.ceil(cols/MAXD));
  const dsRows=Math.ceil(rows/dr), dsCols=Math.ceil(cols/dc);

  let vmax=1e-6;
  for (let i=0;i<rows;i+=dr) for (let j=0;j<cols;j+=dc) vmax=Math.max(vmax, Math.abs(HL.W[i][j]));

  const cellW=Math.max(1, Math.floor((ww-2*pad)/dsCols));
  const cellH=Math.max(1, Math.floor((Hc-2*pad)/dsRows));

  for (let i=0,ri=0;i<rows;i+=dr,ri++){
    for (let j=0,rj=0;j<cols;j+=dc,rj++){
      const val=HL.W[i][j];
      const a=Math.min(1, Math.abs(val)/vmax);
      const hue= val>=0 ? 200 : 0;
      g.fillStyle=`hsla(${hue},90%,60%,${0.15+0.85*a})`;
      g.fillRect(xHeat + rj*cellW, yHeat + ri*cellH, cellW, cellH);
    }
  }
  g.fillStyle='#a7b8e8'; g.fillText('W (hidden × input)', xHeat, Hc-8);
  HL.gridW = { x:xHeat, y:yHeat, cols:dsCols, rows:dsRows, cellW, cellH };

  if (HL.Hx?.length){
    const n=HL.Hx.length;
    const barW = (Wc - xBars - pad), eachH = Math.max(1, Math.floor((Hc - 2*pad)/n));
    const absmax = Math.max(1e-6, ...HL.Hx.map(x=>Math.abs(x)));
    for (let i=0;i<n;i++){
      const val=HL.Hx[i], frac=Math.min(1, Math.abs(val)/absmax), len=Math.floor(frac*barW);
      g.fillStyle = val>=0 ? '#6ee7a2' : '#fb7185';
      g.fillRect(xBars, yBars + i*eachH, len, Math.max(1, eachH-2));
    }
    g.fillStyle='#a7b8e8'; g.fillText('Hx = g(Wx + b)', xBars, Hc-8);
    HL.gridBars = { x:xBars, y:yBars, n, eachH, barW };
  } else {
    g.fillStyle='#93a9e8'; g.fillText('Encode a row (Vectorization), then click “Project H”.', xBars, 22);
    HL.gridBars = null;
  }
}

/* ---------------- 14) Slide 11: ELM train (one-shot β) ---------------- */
const S4 = { canvas:null, betaVis:null, labels:[], dims:null };
function ensureELMTrain(){
  if (S4.inited) return; S4.inited = true;

  const trainBtn    = $('#trainBtn');
  const downloadBtn = $('#downloadBtn');
  const resetBtn    = $('#resetBtn');
  const hiddenSize  = $('#hiddenSize');
  const hiddenSizeVal = $('#hiddenSizeVal');
  const solveOut    = $('#solveOut');
  S4.canvas = $('#betaCanvas');

  hiddenSize?.addEventListener('input', ()=> hiddenSizeVal && (hiddenSizeVal.textContent = hiddenSize.value));
  hiddenSizeVal && (hiddenSizeVal.textContent = hiddenSize?.value || '');

  trainBtn && (trainBtn.onclick = ()=>{
    post('train', { rows: dataRows.map(r=>({y:r.cls, text:r.text})), hidden: +(hiddenSize?.value || 256) });
    solveOut && (solveOut.textContent = 'Training… (freezing basis, solving β)…');
  });

  downloadBtn && (downloadBtn.onclick = ()=> post('export_model'));

  resetBtn && (resetBtn.onclick = ()=>{
    post('reset'); uiBasisFrozen = false;
    S4.betaVis=null; S4.labels=[]; S4.dims=null;
    S4.drawBeta && S4.drawBeta();
    solveOut && (solveOut.textContent = 'Reset complete. Re-train to continue.');
    const predictBtn = $('#predictBtn'); if (predictBtn) { predictBtn.disabled = true; predictBtn.title = 'Train first'; }
  });

  function drawBeta(){
    const c=S4.canvas; if (!c) return;
    const g=c.getContext('2d'), dpr=devicePixelRatio||1;
    const Wc=c.clientWidth, Hc=c.clientHeight;
    c.width=Math.max(1,Wc*dpr); c.height=Math.max(1,Hc*dpr);
    g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,Wc,Hc);
    if (!S4.betaVis){ g.fillStyle='#93a9e8'; g.fillText('Train first to visualize β.',12,22); return; }
    const B=S4.betaVis, R=B.length, C=B[0].length;
    const m=10, cw=Math.floor((Wc-2*m)/C), ch=Math.floor((Hc-2*m)/R);
    let vmax=1e-6;
    for (let i=0;i<R;i++) for (let j=0;j<C;j++) vmax=Math.max(vmax, Math.abs(B[i][j]));
    for (let i=0;i<R;i++){
      for (let j=0;j<C;j++){
        const v=B[i][j], a=Math.min(1, Math.abs(v)/vmax), hue = v>=0 ? 200 : 0;
        g.fillStyle = `hsla(${hue},90%,60%,${0.15+0.85*a})`;
        g.fillRect(m + j*cw, m + i*ch, cw-1, ch-1);
      }
    }
  }
  S4.drawBeta = drawBeta;
}

/* ---------------- 15) Slide 12: Prediction ---------------- */
const SP = {};
function ensurePredict(){
  if (SP.inited) return; SP.inited = true;
  const predRowSelect = $('#predRowSelect');
  const predictBtn = $('#predictBtn');
  const predOut = $('#predOut');

  if (predRowSelect){
    predRowSelect.innerHTML='';
    for (let i=0;i<dataRows.length;i++){
      const o=document.createElement('option');
      o.value=String(i);
      o.textContent=`[${dataRows[i].cls}] ${dataRows[i].text.slice(0,80)}…`;
      predRowSelect.appendChild(o);
    }
  }
  predictBtn && (predictBtn.onclick = ()=>{
    const i = +(predRowSelect?.value || 0);
    post('predict', { text: dataRows[i].text });
    predOut && (predOut.textContent = 'Predicting…');
    SP.lastIndex = i;
  });
}

/* ---------------- 16) Worker events (shared) ---------------- */
worker.onmessage = (e)=>{
  const { type, payload } = e.data || {};
  if (type === 'status') { workerStatus && (workerStatus.textContent = payload); return; }

  if (type === 'trained'){
    uiBasisFrozen = true;
    const predictBtn = $('#predictBtn'); if (predictBtn) { predictBtn.disabled = false; predictBtn.title = ''; }
    const solveOut = $('#solveOut');
    const lines = ['Trained ELM ✓'];
    if (payload.note) lines.push(`note: ${payload.note}`);
    if (payload.dims){
      const d=payload.dims;
      lines.push(`H: ${d.H_rows}×${d.H_cols}  Y: ${d.Y_rows}×${d.Y_cols}  β: ${d.B_rows}×${d.B_cols}`);
    }
    if (payload.betaSample) lines.push('\nβ (8×8 sample):\n' + payload.betaSample);
    solveOut && (solveOut.textContent = lines.join('\n'));

    S4.betaVis = payload.betaVis || null;
    S4.labels  = payload.labels || [];
    S4.dims    = payload.dims || null;
    S4.drawBeta && S4.drawBeta();
    return;
  }

  if (type === 'predicted'){
    const probs = softmax(payload.scores || []);
    const labels = payload.labels || [];
    const pred   = payload.pred;
    const predOut = $('#predOut') || $('#solveOut');

    const i = SP.lastIndex ?? 0;
    const truthId = dataRows[i]?.cls ?? null;
    const truthName = truthId != null ? (AG_LABELS[truthId] || String(truthId)) : 'N/A';
    const predName  = (pred != null) ? (AG_LABELS[pred] || String(pred)) : '—';
    const verdict   = (truthId == null) ? 'truth unknown' : (pred === truthId ? '✓ correct' : '✗ incorrect');

    const probText = (labels.length === probs.length && probs.length > 0)
      ? labels.map((lab,k)=> `${lab}(${AG_LABELS[lab] || ''}): ${probs[k].toFixed(2)}`).join('  |  ')
      : `[${probs.map(p=>p.toFixed(2)).join(', ')}]`;

    predOut && (predOut.textContent =
      `Predicted: ${predName} (${pred})  |  Truth: ${truthName} (${truthId})  |  ${verdict}\n` +
      `Probabilities: ${probText}`);
    return;
  }

  if (type === 'hidden_init'){ onHiddenInit(payload); return; }
  if (type === 'hidden_project'){ onHiddenProject(payload); return; }

  // 'encoded', 'exported_model', 'actCurve' are handled elsewhere or ignored
};

/* ---------------- 17) Bootstrap worker ---------------- */
post('hello');
post('list_activations');

}); // DOMContentLoaded
