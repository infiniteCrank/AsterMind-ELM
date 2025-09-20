import { TOK, ID2STR, V } from './drums/tokens.js?v=1007';
import { TRAIN_SEQS, grooveBackbeat } from './drums/grooves.js?v=1007';
import { DrumSynth } from './drums/synth.js?v=1007';
import { resizeCanvas, drawTimeline } from './drums/draw.js?v=1007';
import { softmax, sampleTopK, makeXY, buildInputWindow } from './utils.js?v=1007';
import { AM } from './engine/am.js?v=1007';
import { createConfig } from './engine/config.js?v=1007';
import { trainOnMain } from './engine/train.js?v=1007';
import { predictLogitsOnMain, generateTokensSync, generateTokensAsync } from './engine/predict.js?v=1007';
import { WorkerClient } from './engine/worker.js?v=1007';


// UI refs
const trainBtn = document.getElementById('trainBtn');
const genBtn = document.getElementById('genBtn');
const stopBtn = document.getElementById('stopBtn');
const tempSlider = document.getElementById('temp');
const topkSlider = document.getElementById('topk');
const tempVal = document.getElementById('tempVal');
const topkVal = document.getElementById('topkVal');
const statusDiv = document.getElementById('status');
const engineSel = document.getElementById('engine');
const nWindowInp = document.getElementById('nWindow');
const hiddenInp = document.getElementById('hidden');
const lambdaInp = document.getElementById('lambda');
const activationSel = document.getElementById('activation');
const useWorkerChk = document.getElementById('useWorker');
const kernelRow = document.getElementById('kernelRow');
const kernelTypeSel = document.getElementById('kernelType');
const gammaField = document.getElementById('gammaField');
const polyField = document.getElementById('polyField');
const gammaInp = document.getElementById('gamma');
const degreeInp = document.getElementById('degree');
const coef0Inp = document.getElementById('coef0');
const mLandmarksInp = document.getElementById('mLandmarks');
const whitenChk = document.getElementById('whiten');

function setStatus(msg) { statusDiv.textContent = msg; }
function updateSliders() { tempVal.textContent = tempSlider.value; topkVal.textContent = topkSlider.value; }
updateSliders();
tempSlider.addEventListener('input', updateSliders);
topkSlider.addEventListener('input', updateSliders);

engineSel.addEventListener('change', () => {
  const isKernel = engineSel.value === 'kernel';
  kernelRow.style.display = isKernel ? 'flex' : 'none';
});
kernelTypeSel.addEventListener('change', () => {
  const t = kernelTypeSel.value;
  gammaField.style.display = (t === 'rbf' || t === 'laplacian') ? 'flex' : 'none';
  polyField.style.display = (t === 'poly') ? 'flex' : 'none';
});

// Canvas
const canvas = document.getElementById('c');
const g = canvas.getContext('2d');
function doResize() { resizeCanvas(canvas, g); }
window.addEventListener('resize', doResize); doResize();

// App state
let MODEL = null, WORKER = null;
let currentTokens = []; let scheduler = null;
const uiRefs = { activationSel, kernelTypeSel, degreeInp, coef0Inp, gammaInp, mLandmarksInp, lambdaInp, whitenChk };

// Model creation
function createModel(engine, inputSize, hiddenUnits, ridgeLambda) {
  const am = AM();
  const cfg = createConfig(engine, inputSize, hiddenUnits, ridgeLambda, { ...uiRefs });
  if (engine === 'kernel' && am.KernelELM) return new am.KernelELM(cfg);
  if (engine === 'online' && am.OnlineELM) return new am.OnlineELM(cfg);
  if (am.ELM) return new am.ELM(createConfig('elm', inputSize, hiddenUnits, ridgeLambda, { ...uiRefs }));
  throw new Error('AsterMind UMD not exposing expected classes; found keys: ' + Object.keys(am));
}

// Playback
function playSequence(seq) {
  const drums = new DrumSynth();
  const bpm = 120, spb = 60 / bpm, stepDur = spb / 4; const ctx = drums.ctx; const start = ctx.currentTime + 0.2;
  let step = 0; currentTokens = []; stopBtn.disabled = false;
  scheduler = setInterval(() => {
    const now = ctx.currentTime;
    while (step < seq.length && (start + step * stepDur) < now + 0.15) {
      const t = seq[step], when = start + step * stepDur; drums.playToken(t, when);
      currentTokens.push(t); drawTimeline(canvas, g, currentTokens, step % (16 * 4)); step++;
    }
    if (step >= seq.length) { clearInterval(scheduler); stopBtn.disabled = true; setStatus('Playback finished.'); }
  }, 20);
}

// Train button

trainBtn.addEventListener('click', async () => {
  trainBtn.disabled = true; genBtn.disabled = true; stopBtn.disabled = true;
  setStatus('Preparing training…');
  try {
    const engine = engineSel.value;
    const N = parseInt(nWindowInp.value, 10) || 16;
    const hidden = parseInt(hiddenInp.value, 10) || 128;
    const lambda = parseFloat(lambdaInp.value) || 0.01;
    const { X, y } = makeXY(TRAIN_SEQS, N, V);
    console.log('y min/max:', Math.min(...y), Math.max(...y), 'V =', V);

    const useWorker = !!useWorkerChk.checked;
    if (useWorker && engine === 'elm') {
      // fresh worker each run to avoid stale state
      if (WORKER) { WORKER.terminate(); WORKER = null; }
      WORKER = new WorkerClient();

      const cfg = createConfig(engine, N * V, hidden, lambda, { ...uiRefs });

      // Older worker builds only support a generic "init"
      await WORKER.call('init', cfg);

      const t0 = performance.now();
      try {
        // Plain ELM can take labels; the worker/ELM is tolerant
        await WORKER.call('train', { X, y });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        setStatus(`Trained (elm) via Worker in ${dt}s. Ready to generate.`);
        MODEL = null; // use worker for generation
      } catch (err) {
        const msg = (err && err.message) ? err.message : String(err);
        const unknown = /Unknown action/i.test(msg) || /does not support action/i.test(msg);
        if (!unknown) throw err;

        console.warn('Worker cannot train, falling back to main thread:', msg);
        MODEL = createModel(engine, N * V, hidden, lambda);
        const t1 = performance.now();
        await trainOnMain(engine, MODEL, X, y); // labels OK; our library is tolerant now
        const dt = ((performance.now() - t1) / 1000).toFixed(2);
        setStatus(`Worker didn’t support "train" → trained on main in ${dt}s. Ready to generate.`);
        try { WORKER.terminate(); } catch { }
        WORKER = null;
      }
    } else {
      // Main-thread training for online or kernel (worker doesn’t support these)
      MODEL = createModel(engine, N * V, hidden, lambda);
      const t0 = performance.now();
      await trainOnMain(
        engine,
        MODEL,
        X,
        (engine === 'online' || engine === 'kernel') ? (await import('./utils.js')).toOneHot(y, V) : y
      );
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      setStatus(`Trained (${engine}) on main in ${dt}s. Ready to generate.`);
      if (WORKER) { WORKER.terminate(); WORKER = null; }
    }

    // Seed the timeline after training
    currentTokens = TRAIN_SEQS[0].slice(0, 64);
    drawTimeline(canvas, g, currentTokens, 0);
    genBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus('Training failed: ' + (err && err.message ? err.message : err));
  } finally {
    trainBtn.disabled = false;
  }
});

// Generate
genBtn.addEventListener('click', async () => {
  if (!MODEL && !WORKER) return;
  const N = parseInt(nWindowInp.value, 10) || 16;
  const temp = parseFloat(tempSlider.value);
  const topk = parseInt(topkSlider.value, 10);
  genBtn.disabled = true; trainBtn.disabled = true;
  setStatus('Generating and playing…');
  try {
    const seed = grooveBackbeat().slice(0, 16);
    const steps = 16 * 8;
    const generated = WORKER
      ? await generateTokensAsync(WORKER, seed, steps, buildInputWindow, N, V, softmax, sampleTopK, temp, topk)
      : generateTokensSync(MODEL, seed, steps, buildInputWindow, N, V, softmax, sampleTopK, temp, topk);
    playSequence(seed.concat(generated));
  } catch (err) {
    console.error(err); setStatus('Generation error: ' + (err && err.message ? err.message : err));
  } finally {
    genBtn.disabled = false; trainBtn.disabled = false;
  }
});

// Stop
stopBtn.addEventListener('click', () => {
  if (scheduler) { clearInterval(scheduler); scheduler = null; }
  stopBtn.disabled = true; setStatus('Playback stopped.');
});

// initial paint
currentTokens = TRAIN_SEQS[0].slice(0, 32);
drawTimeline(canvas, g, currentTokens, 0);
setStatus('Pick engine & click “Train”.');
