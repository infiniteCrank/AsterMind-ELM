/* Uses window.astermind on the main thread — no worker, no CORS gotchas. */

const TOK = { WAIT_1:0, ON_K:1, ON_S:2, ON_H:3 };
const ID2STR = ['WAIT_1','ON_K','ON_S','ON_H']; const V = ID2STR.length;

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function softmax(logits, temp=1){ let m=-Infinity; for(const v of logits) if(v>m) m=v;
  const ex=logits.map(v=>Math.exp((v-m)/temp)); const s=ex.reduce((a,b)=>a+b,0)||1; return ex.map(v=>v/s); }
function sampleTopK(probs,k){ const arr=probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0,k);
  const sum=arr.reduce((s,[p])=>s+p,0)||1; let r=Math.random()*sum; for(const [p,i] of arr){ r-=p; if(r<=0) return i; } return arr[arr.length-1][1]; }

function grooveFour(){ const seq=[]; for(let bar=0;bar<4;bar++){ for(let step=0;step<16;step++){ const pos=step%16;
  if(pos===0||pos===4||pos===8||pos===12) seq.push(TOK.ON_K);
  else if(pos===4||pos===12) seq.push(TOK.ON_S);
  else if(pos%2===0) seq.push(TOK.ON_H);
  else seq.push(TOK.WAIT_1);
}} return seq; }
function grooveBack(){ const seq=[]; for(let bar=0;bar<4;bar++){ for(let step=0;step<16;step++){ const pos=step%16;
  if(pos===0||pos===7) seq.push(TOK.ON_K);
  else if(pos===4||pos===12) seq.push(TOK.ON_S);
  else seq.push(TOK.ON_H);
}} return seq; }
const TRAIN_SEQS=[grooveFour(),grooveBack()];

function buildInputWindow(tokens, pos, N){ const out=new Array(N*V).fill(0); let o=0;
  for (let i=pos-N;i<pos;i++){ const t=i>=0?tokens[i]:TOK.WAIT_1; out[o+t]=1; o+=V; } return out; }
function makeXY(seqs,N=16){ const X=[],y=[]; for(const seq of seqs){ for(let t=1;t<seq.length;t++){ X.push(buildInputWindow(seq,t,N)); y.push(seq[t]); } } return {X,y}; }

/* Canvas */
const canvas=document.getElementById('c'); const g=canvas.getContext('2d');
function resizeCanvas(){ const DPR=window.devicePixelRatio||1; const W=canvas.clientWidth,H=canvas.clientHeight;
  canvas.width=Math.floor(W*DPR); canvas.height=Math.floor(H*DPR); g.setTransform(DPR,0,0,DPR,0,0); }
window.addEventListener('resize', resizeCanvas); resizeCanvas();
const colors={kick:'var(--accent)', snare:'#6ee7a2', hat:'#facc15'};
function drawTimeline(tokens, cursor, stepsPerBar=16){
  const W=canvas.clientWidth, H=canvas.clientHeight;
  g.clearRect(0,0,W,H);
  const pad=20, usableW=W-2*pad, laneH=(H-2*pad)/3;
  const totalSteps=stepsPerBar*4; const barWidth=usableW/4;
  for (let b=0;b<4;b++){ const xs=pad+b*barWidth; g.fillStyle='rgba(90,209,255,0.04)'; g.fillRect(xs,pad,barWidth,H-2*pad); }
  g.strokeStyle='rgba(90,209,255,0.25)';
  for (let i=0;i<=totalSteps;i++){ const x=pad+(i/totalSteps)*usableW; g.beginPath(); g.moveTo(x,pad); g.lineTo(x,H-pad); g.stroke(); }
  g.fillStyle='var(--muted)'; g.font='12px system-ui';
  ['KICK','SNARE','HAT'].forEach((L,i)=>{ g.fillText(L,6,pad+i*laneH+14); });
  const windowTokens=tokens.slice(-totalSteps);
  for (let i=0;i<windowTokens.length;i++){
    const t=windowTokens[i]; const x=pad+(i/totalSteps)*usableW;
    const yK=pad+laneH*0.5, yS=pad+laneH*1.5, yH=pad+laneH*2.5;
    if (t===TOK.ON_K){ g.fillStyle=colors.kick; g.beginPath(); g.arc(x,yK,6,0,Math.PI*2); g.fill(); }
    if (t===TOK.ON_S){ g.fillStyle=colors.snare; g.beginPath(); g.arc(x,yS,6,0,Math.PI*2); g.fill(); }
    if (t===TOK.ON_H){ g.fillStyle=colors.hat;  g.beginPath(); g.arc(x,yH,4,0,Math.PI*2); g.fill(); }
  }
  const idx=(cursor%totalSteps);
  const playX=pad+(idx/totalSteps)*usableW;
  g.strokeStyle='rgba(250,250,255,0.85)'; g.beginPath(); g.moveTo(playX,pad); g.lineTo(playX,H-pad); g.stroke();
}

function AM(){ return (window.astermind || window.Astermind || window.AstermindELM || {}); }

function createConfig(engine,inputSize,hiddenUnits,ridgeLambda,activation){
  if (engine==='kernel'){
    return { outputDim: V, mode:'nystrom',
      kernel:{ type:'rbf', gamma:1/Math.max(1,inputSize) },
      nystrom:{ m: Math.min(256,Math.max(32,hiddenUnits)), strategy:'kmeans++', whiten:true, jitter:1e-9 },
      ridgeLambda, task:'classification', log:{ verbose:false, modelName:'DrumKernelELM' } };
  }
  if (engine==='online'){
    return { inputDim:inputSize, outputDim:V, hiddenUnits, activation, ridgeLambda, weightInit:'he',
      forgettingFactor:0.997, log:{ verbose:false, modelName:'DrumOnlineELM' } };
  }
  return { inputSize, hiddenUnits, activation, ridgeLambda, task:'classification',
    categories: ID2STR.slice(), useTokenizer:false, maxLen:16, weightInit:'xavier', log:{ verbose:false, modelName:'DrumELM' } };
}

async function trainOnMain(engine, model, X, y){
  if (engine==='kernel' || engine==='online'){
    const Y = y.map(id => { const row=new Array(V).fill(0); row[id|0]=1; return row; });
    if (engine==='online'){
      if (typeof model.init==='function') model.init(X, Y, { task:'classification' });
      else if (typeof model.train==='function') model.train(X, Y);
    } else {
      if (typeof model.fit==='function') model.fit(X, Y, { task:'classification' });
      else if (typeof model.train==='function') model.train(X, Y);
    }
    return;
  }
  if (typeof model.trainFromData==='function'){
    try { model.trainFromData(X, y, { task:'classification' }); }
    catch { const Y = y.map(id => { const row=new Array(V).fill(0); row[id|0]=1; return row; });
            model.trainFromData(X, Y, { task:'classification' }); }
  } else if (typeof model.train==='function'){
    try { model.train(X, y); }
    catch { const Y = y.map(id => { const row=new Array(V).fill(0); row[id|0]=1; return row; });
            model.train(X, Y); }
  } else {
    throw new Error('Model has no train API');
  }
}

function predictLogitsOnMain(model, x){
  if (typeof model.predictLogitsFromVector==='function') return model.predictLogitsFromVector(x);
  if (typeof model.predictLogits==='function') return model.predictLogits(x);
  if (typeof model.predictProbaFromVector==='function'){ const p=model.predictProbaFromVector(x); return p.map(v=>Math.log((v||1e-12))); }
  if (typeof model.predictFromVector==='function'){ const r=model.predictFromVector([x])[0]; if (Array.isArray(r)) return r; if (r&&Array.isArray(r.probs)) return r.probs.map(v=>Math.log((v||1e-12))); }
  throw new Error('No vector-safe predict on model');
}

class DrumSynth {
  constructor(){ const AudioCtx=window.AudioContext||window.webkitAudioContext; this.ctx=new AudioCtx();
    this.master=this.ctx.createGain(); this.master.gain.value=0.25; this.master.connect(this.ctx.destination); }
  kick(t){ const c=this.ctx,o=c.createOscillator(),g=c.createGain(); o.type='sine'; o.frequency.setValueAtTime(120,t); o.frequency.exponentialRampToValueAtTime(40,t+0.15);
           g.gain.setValueAtTime(1,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.15); o.connect(g).connect(this.master); o.start(t); o.stop(t+0.16); }
  snare(t){ const c=this.ctx,b=c.createBuffer(1,c.sampleRate*0.2,c.sampleRate); const d=b.getChannelData(0); for(let i=0;i<d.length;i++) d[i]=Math.random()*2-1;
           const s=c.createBufferSource(); s.buffer=b; const bp=c.createBiquadFilter(); bp.type='bandpass'; bp.frequency.value=1800;
           const g=c.createGain(); g.gain.setValueAtTime(0.5,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.18); s.connect(bp).connect(g).connect(this.master); s.start(t); s.stop(t+0.2); }
  hat(t){ const c=this.ctx,b=c.createBuffer(1,c.sampleRate*0.05,c.sampleRate); const d=b.getChannelData(0); for(let i=0;i<d.length;i++) d[i]=Math.random()*2-1;
           const s=c.createBufferSource(); s.buffer=b; const hp=c.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=7000;
           const g=c.createGain(); g.gain.setValueAtTime(0.25,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.05); s.connect(hp).connect(g).connect(this.master); s.start(t); s.stop(t+0.06); }
  play(tok,when){ if(tok===TOK.ON_K) this.kick(when); else if(tok===TOK.ON_S) this.snare(when); else if(tok===TOK.ON_H) this.hat(when); }
}

function playSequence(seq){ const drums=new DrumSynth(); const bpm=120, spb=60/bpm, stepDur=spb/4; const ctx=drums.ctx; const start=ctx.currentTime+0.2;
  let step=0; currentTokens=[]; stopBtn.disabled=false;
  scheduler=setInterval(()=>{ const now=ctx.currentTime;
    while(step<seq.length && (start+step*stepDur)<now+0.15){
      const t=seq[step], when=start+step*stepDur; drums.play(t, when); currentTokens.push(t); drawTimeline(currentTokens, step%(16*4)); step++; }
    if(step>=seq.length){ clearInterval(scheduler); stopBtn.disabled=true; setStatus('Playback finished.'); }
  },20);
}

/* UI */
const trainBtn=document.getElementById('trainBtn');
const genBtn=document.getElementById('genBtn');
const stopBtn=document.getElementById('stopBtn');
const tempSlider=document.getElementById('temp');
const topkSlider=document.getElementById('topk');
const tempVal=document.getElementById('tempVal');
const topkVal=document.getElementById('topkVal');
const statusDiv=document.getElementById('status');
const engineSel=document.getElementById('engine');
const nWindowInp=document.getElementById('nWindow');
const hiddenInp=document.getElementById('hidden');
const lambdaInp=document.getElementById('lambda');
const activationSel=document.getElementById('activation');

function setStatus(msg){ statusDiv.textContent=msg; }
function updateSliders(){ tempVal.textContent=tempSlider.value; topkVal.textContent=topkSlider.value; }
tempSlider.addEventListener('input', updateSliders); topkSlider.addEventListener('input', updateSliders); updateSliders();

let MODEL=null; let scheduler=null; let currentTokens=[];

trainBtn.addEventListener('click', async ()=>{
  trainBtn.disabled=true; genBtn.disabled=true; stopBtn.disabled=true;
  setStatus('Training…');
  try{
    const engine=engineSel.value;
    const N=parseInt(nWindowInp.value,10)||16;
    const hidden=parseInt(hiddenInp.value,10)||128;
    const lambda=parseFloat(lambdaInp.value)||0.01;
    const activation=activationSel.value||'relu';
    const {X,y}=makeXY(TRAIN_SEQS, N);
    const am=AM();
    let model;
    if (engine==='kernel' && am.KernelELM) model=new am.KernelELM(createConfig(engine, N*V, hidden, lambda, activation));
    else if (engine==='online' && am.OnlineELM) model=new am.OnlineELM(createConfig(engine, N*V, hidden, lambda, activation));
    else if (am.ELM) model=new am.ELM(createConfig('elm', N*V, hidden, lambda, activation));
    else throw new Error('AsterMind UMD not exposing expected classes');
    const t0=performance.now();
    await trainOnMain(engine, model, X, y);
    const dt=((performance.now()-t0)/1000).toFixed(2);
    MODEL=model; setStatus(`Trained (${engine}) on main in ${dt}s. Ready to generate.`);
    currentTokens = TRAIN_SEQS[0].slice(0,64); drawTimeline(currentTokens,0); genBtn.disabled=false;
  }catch(err){ console.error(err); setStatus('Training failed: '+(err?.message||err)); }
  finally{ trainBtn.disabled=false; }
});

genBtn.addEventListener('click', async ()=>{
  if(!MODEL) return;
  const N=parseInt(nWindowInp.value,10)||16;
  const temp=parseFloat(tempSlider.value);
  const topk=parseInt(topkSlider.value,10);
  genBtn.disabled=true; trainBtn.disabled=true; setStatus('Generating and playing…');
  try{
    const seed = grooveBack().slice(0, N);
    const steps = 16*8;
    const out = seed.slice();
    for (let i=0;i<steps;i++){ const x=Array.from(buildInputWindow(out, out.length, N));
      const logits = predictLogitsOnMain(MODEL, x);
      const probs = softmax(Array.from(logits), temp);
      const id = sampleTopK(probs, topk|0);
      out.push(id);
    }
    playSequence(out);
  }catch(err){ console.error(err); setStatus('Generation error: '+(err?.message||err)); }
  finally{ genBtn.disabled=false; trainBtn.disabled=false; }
});

stopBtn.addEventListener('click',()=>{ if(scheduler){ clearInterval(scheduler); scheduler=null; } stopBtn.disabled=true; setStatus('Playback stopped.'); });

currentTokens = TRAIN_SEQS[0].slice(0,32); drawTimeline(currentTokens,0); setStatus('Load complete. Click “Train”.');
