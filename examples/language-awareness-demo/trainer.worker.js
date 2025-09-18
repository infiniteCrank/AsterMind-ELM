/* global importScripts, self, performance */

// Load your UMD build into the worker's global scope
importScripts('/astermind.umd.js');
const { LanguageClassifier } = (self.astermind || self.window?.astermind || {});

function post(type, payload){ self.postMessage({ type, payload }); }
post('ready', null);

/* ===========================
   TUNING / TRAINING TOGGLES
   =========================== */
// Train with ALL data after tuning best config?
const RETRAIN_ON_ALL_AFTER_TUNING = true;

// Hold-out fraction for tuning (set to 0 if you truly want no split)
const VAL_FRACTION = 0.15;

// Cap per-class for speed (Infinity = no cap)
const MAX_PER_CLASS = Infinity;

// Optional class balancing before split (oversample the minority classes)
const BALANCE_OVERSAMPLE = false;

/* ===========================
   HELPERS
   =========================== */

// Keep useful punctuation + accents
const charSet = "abcdefghijklmnopqrstuvwxyzçàèéìíïòóôùúüñãõâêîôûáéíóúü¿¡!?',. ";

function normalize(s){
  const allowed = new Set(charSet.split(''));
  const n = s.normalize('NFC').toLowerCase();
  let out = '';
  for (const ch of n) if (allowed.has(ch)) out += ch;
  return out;
}

function parseCSV(text){
  const rows=[]; let f='', r=[], q=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i], nx=text[i+1];
    if(q){
      if(ch==='"' && nx === '"'){ f+='"'; i++; }
      else if(ch==='"'){ q=false; }
      else f+=ch;
    }else{
      if(ch==='"') q=true;
      else if(ch===','){ r.push(f); f=''; }
      else if(ch==='\n' || ch==='\r'){
        if(f.length||r.length){ r.push(f); rows.push(r); }
        f=''; r=[];
        if(ch==='\r' && nx==='\n') i++;
      }else f+=ch;
    }
  }
  if(f.length||r.length){ r.push(f); rows.push(r); }
  return rows.filter(r=>r.length>0);
}

function dedupBy(arr, key){
  const seen=new Set(), out=[];
  for(const x of arr){ const k=key(x); if(!seen.has(k)){ seen.add(k); out.push(x); } }
  return out;
}
function shuffle(a){ for(let i=a.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]]; } return a; }
function stratifiedSplit(data, p=0.85){
  const by=new Map(); for(const d of data){ if(!by.has(d.label)) by.set(d.label,[]); by.get(d.label).push(d); }
  const train=[], val=[]; for(const arr of by.values()){ shuffle(arr); const cut=Math.floor(arr.length*p);
    train.push(...arr.slice(0,cut)); val.push(...arr.slice(cut)); }
  return { train, val };
}
function capPerClass(data, maxPerClass=500){
  const by=new Map(); for(const d of data){ if(!by.has(d.label)) by.set(d.label,[]); by.get(d.label).push(d); }
  const out=[]; for(const arr of by.values()){ shuffle(arr); out.push(...arr.slice(0, Math.min(maxPerClass, arr.length))); }
  return shuffle(out);
}
function balanceOversample(data){
  const by=new Map();
  for (const d of data){ if(!by.has(d.label)) by.set(d.label,[]); by.get(d.label).push(d); }
  const maxN = Math.max(...[...by.values()].map(v=>v.length));
  const out = [];
  for (const arr of by.values()){
    let i=0; while(i<maxN){ out.push(arr[i % arr.length]); i++; }
  }
  return shuffle(out);
}
function accuracy(model, data){
  let ok=0; for(const d of data){ const [top]=model.predict(d.text,1); if(top?.label===d.label) ok++; }
  return ok / Math.max(1, data.length);
}
function fmtCounts(data){
  const c=data.reduce((a,d)=> (a[d.label]=(a[d.label]||0)+1, a), {});
  return Object.entries(c).map(([k,v])=>`${k}:${v}`).join(' • ');
}

/* ===========================
   BASE CONFIG + TUNER
   =========================== */

const BASE = {
  categories: ['English','French','Spanish'],
  maxLen: 40,
  activation: 'tanh',
  hiddenUnits: 300,
  charSet,
  useTokenizer: false,   // char mode (fast + effective)
  dropout: 0,            // IMPORTANT for closed-form ELM
  weightInit: 'xavier',
  ridgeLambda: 1e-3,
  log: { verbose: false },
};

function trainCandidate(cfg, train, val){
  const clf = new LanguageClassifier({ ...BASE, ...cfg });
  clf.train(train);
  const acc = accuracy(clf, val);
  return { acc, clf };
}

function tune(train, val){
  // Reasonable grid; increase if you want, worker can handle it
  const combos = [
    { activation:'tanh',      hiddenUnits:300, ridgeLambda:5e-4 },
    { activation:'tanh',      hiddenUnits:350, ridgeLambda:5e-3 },
    { activation:'leakyrelu', hiddenUnits:350, ridgeLambda:5e-4 },
    { activation:'leakyrelu', hiddenUnits:400, ridgeLambda:1e-3 },
    { activation:'tanh',      hiddenUnits:450, ridgeLambda:1e-3 },
    { activation:'leakyrelu', hiddenUnits:450, ridgeLambda:1e-2 },
  ];
  let best = { acc:-1, cfg:null, clf:null }, i=0;
  for (const cfg of combos){
    i++;
    post('progress', `Training ${i}/${combos.length}… <code>act=${cfg.activation}</code> <code>hu=${cfg.hiddenUnits}</code> <code>λ=${cfg.ridgeLambda}</code>`);
    const { acc, clf } = trainCandidate(cfg, train, val);
    if (acc > best.acc) best = { acc, cfg, clf };
  }
  return best;
}

/* ===========================
   WORKER MESSAGES
   =========================== */

let classifier = null;

self.onmessage = (e) => {
  const { type, csvText, text } = e.data || {};
  try{
    if (type === 'init') {
      const t0 = performance.now();
      const rows = parseCSV(csvText);
      const header = rows[0] && rows[0][0]?.toLowerCase().includes('text') ? 1 : 0;

      let data = rows.slice(header)
        .map(([t='',lab='']) => ({ text: normalize(t.trim()), label: lab.trim() }))
        .filter(d => d.text && d.label);

      data = dedupBy(data, d => d.text+'||'+d.label);
      if (BALANCE_OVERSAMPLE) data = balanceOversample(data);
      data = capPerClass(data, MAX_PER_CLASS);

      const { train, val } = stratifiedSplit(data, 1 - VAL_FRACTION);

      const best = tune(train, val);

      if (RETRAIN_ON_ALL_AFTER_TUNING) {
        // Retrain final model on 100% of the data with the best config
        const finalClf = new LanguageClassifier({ ...BASE, ...best.cfg });
        finalClf.train(data);
        classifier = finalClf;
      } else {
        classifier = best.clf; // uses 85% train if VAL_FRACTION = 0.15
      }

      const ms = Math.round(performance.now() - t0);
      post('trained', {
        html:
          `Best: <code>activation=${best.cfg.activation}</code> `+
          `<code>hiddenUnits=${best.cfg.hiddenUnits}</code> `+
          `<code>ridgeLambda=${best.cfg.ridgeLambda}</code> `+
          (RETRAIN_ON_ALL_AFTER_TUNING
            ? `| final: trained on <strong>100%</strong> of data`
            : `| val acc: <strong>${(best.acc*100).toFixed(1)}%</strong>`) +
          ` | data: ${fmtCounts(data)} | time: ${ms}ms`
      });
    }
    else if (type === 'predict') {
      if (!classifier) return;
      const typed = normalize(text || '');
      const results = classifier.predict(typed, 3);
      post('predictions', { top: results[0], second: results[1] });
    }
    else if (type === 'error') {
      post('error', e.data.message || 'Unknown error');
    }
  }catch(err){
    post('error', String(err && err.stack || err));
  }
};
