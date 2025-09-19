# AsterMind-ELM

[![npm version](https://img.shields.io/npm/v/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![npm downloads](https://img.shields.io/npm/dm/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

A modular Extreme Learning Machine (ELM) library for JS/TS (browser + Node).

---

# 🚀 What you can build — and why this is groundbreaking

AsterMind brings **instant, tiny, on-device ML** to the web. It lets you ship models that **train in milliseconds**, **predict with microsecond latency**, and **run entirely in the browser** — no GPU, no server, no tracking. With new **Kernel ELMs**, **streaming Online ELM**, and **Web Worker** offloading, you can create:

- **Private, on-device classifiers** (language, intent, toxicity, spam) that retrain on user feedback
- **Real-time retrieval & reranking** with compact embeddings (ELM, KernelELM, Nyström whitening) for search and RAG
- **Interactive creative tools** (music/drum pattern generators, autocompletes) that respond instantly
- **Edge analytics** for dashboards: learn lightweight regressors/classifiers from data that never leaves the page
- **Deep ELM chains**: stack encoders → embedders → classifiers for powerful pipelines, still tiny and transparent

**Why it matters:** ELMs give you **closed-form training** (no heavy SGD loops), **interpretable structure**, and **tiny memory footprints**. AsterMind modernizes ELM with kernels, online learning, workerized training, and robust text+numeric tooling — making **seriously fast ML** practical for every web app.

---

## 🆕 New in this release

- **Kernel ELMs (KELMs)** — exact and Nyström-approximated kernel methods (RBF/Linear/Poly/Laplacian/Custom) with ridge solve
- **Whitened Nyström** — optional \(K_{mm}^{-1/2}\) whitening via symmetric eigendecomposition for stable features
- **Online ELM (OS-ELM)** — streaming/online updates with RLS and optional forgetting factor (no full retrain)
- **Web Worker adapter** — train/predict off the main thread; message-based API for ELM and KernelELM
- **Matrix upgrades** — symmetric **Jacobi eig** + **invSqrtSym** for whitening; improved Cholesky solve
- **EmbeddingStore 2.0** — unit-norm storage, ring-buffer capacity, metadata filters, JSON import/export
- **ELMChain+Embeddings** — cleaner chaining, safer dimension checks, JSON I/O
- **Activations** — added **linear** and **GELU**; centralized activation registry
- **Configs** — clearer split between **Numeric** and **Text** configs; stronger typing for defaults
- **UMD exports** — `window.astermind` now exposes `ELM`, `OnlineELM`, `KernelELM`, `KernelRegistry`, `EmbeddingStore`, `ELMChain`, etc.
- **Robust text preprocessing** — safer encoder path (guards non-string inputs), better error messages

> See [Releases](#releases) for the full changelog and bug fixes.

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Kernel ELMs (KELM)](#kernel-elms-kelm)
4. [Online ELM (OS-ELM)](#online-elm-os-elm)
5. [Web Worker Adapter](#web-worker-adapter)
6. [Installation](#installation)
7. [Usage Examples](#usage-examples)
8. [AsterMind in Practice](#astermind-in-practice)
9. [Core API](#core-api)
10. [Config Reference](#config-reference)
11. [Embedding Store](#embedding-store)
12. [Releases](#releases)
13. [License](#license)

---

<a id="introduction"></a>
# 🌟 AsterMind: Decentralized ELM Framework Inspired by Nature

Welcome to **AsterMind**, a modular, decentralized machine learning framework built around small, cooperating Extreme Learning Machines (ELMs) that self-train, self-evaluate, and self-repair — like the decentralized nervous system of a starfish.

This library preserves the core ELM idea — **random hidden layer**, **nonlinear activation**, **closed-form output solve** — and modernizes it with: multiple activations (ReLU/LeakyReLU/Sigmoid/Tanh/**Linear/GELU**), initializer choices, dropout, sample weighting, metrics gating, JSON I/O, chaining, **kernel methods**, and **online** updates.

AsterMind is designed for:

* Lightweight, in-browser ML pipelines
* Transparent, interpretable predictions
* Continuous, incremental learning
* Resilient systems with no single point of failure

---

<a id="features"></a>
## ✨ Features

- ✅ **Closed-form training** (ridge / pseudoinverse) — instant fits
- ✅ **Activations**: relu, leakyrelu, sigmoid, tanh, **linear, gelu**
- ✅ **Initializers**: uniform, xavier (and **He** in OnlineELM)
- ✅ **Numeric + Text** inputs (clean config split)
- ✅ **Kernel ELM** with **Nyström** + **whitening**
- ✅ **Online ELM** (RLS) with forgetting factor
- ✅ **Web Worker** adapter for off-main-thread training
- ✅ **Embeddings & Chains** for retrieval and deep pipelines
- ✅ **UMD + ESM**: works in `<script>` and modern bundlers
- ✅ **Zero server/GPU** — private, on-device ML

---

<a id="kernel-elms-kelm"></a>
## 🧠 Kernel ELMs (KELM)

A drop-in kernelized variant supporting **Exact** and **Nyström** modes:

- Kernels: **RBF**, **Linear**, **Polynomial**, **Laplacian**, or **Custom** via `KernelRegistry`
- Nyström landmarks: `uniform`, `kmeans++`, or `preset`
- **Whitened Nyström**: \( \Phi = K_{nm}\,K_{mm}^{-1/2} \) for stable features (persisted `R` for inference parity)
- JSON save/load with persisted landmarks/weights/whitener

**Quick start:**
```ts
import { KernelELM } from '@astermind/astermind-elm';

const kelm = new KernelELM({
  outputDim: Y[0].length,
  kernel: { type: 'rbf', gamma: 1 / X[0].length },
  mode: 'nystrom',
  nystrom: { m: 256, strategy: 'kmeans++', whiten: true, jitter: 1e-9 },
  ridgeLambda: 1e-2,
  task: 'classification',
});
kelm.fit(X, Y);
const probs = kelm.predictProbaFromVectors(Xq);
const emb   = kelm.getEmbedding(Xq);
```

**Custom kernels:**
```ts
import { KernelRegistry } from '@astermind/astermind-elm';

KernelRegistry.register('cosine', (x, z) => {
  let d=0,nx=0,nz=0; for (let i=0;i<x.length;i++){ d+=x[i]*z[i]; nx+=x[i]*x[i]; nz+=z[i]*z[i]; }
  return d / (Math.sqrt(nx)||1) / (Math.sqrt(nz)||1);
});

const kelm = new KernelELM({
  outputDim: K,
  kernel: { type: 'custom', name: 'cosine' },
  mode: 'exact',
});
```

---

<a id="online-elm-os-elm"></a>
## 🔁 Online ELM (OS-ELM)

Stream updates without retraining from scratch. Uses **Recursive Least Squares (RLS)** with optional **forgetting factor** \( \rho \in (0,1] \). Supports **He/Xavier/Uniform** initializers.

```ts
import { OnlineELM } from '@astermind/astermind-elm';

const ol = new OnlineELM({
  inputDim: D,
  outputDim: K,
  hiddenUnits: 256,
  activation: 'relu',
  ridgeLambda: 1e-2,
  forgettingFactor: 0.995,
  weightInit: 'he',
  log: { verbose: true, modelName: 'OS-ELM' },
});

// First batch (init)
ol.init(X0, Y0);

// Stream updates
ol.update(Xt, Yt);

// Predict
const proba = ol.predictProbaFromVectors(Xq);
```

---

<a id="web-worker-adapter"></a>
## 🧵 Web Worker Adapter

Move training and heavy ops off the main thread.

- **Worker**: `ELMWorker` handles `init/trainFromData/predict/...` and `kelm.*` actions
- **Client**: `ELMWorkerClient` wraps `postMessage` with typed calls
- Works with both **ELM** and **KernelELM**; returns predictions, embeddings, JSON

**UMD usage (global):**
```html
<script src="/astermind.umd.js"></script>
<script>
  const {
    ELM, OnlineELM, KernelELM, KernelRegistry, ELMChain, EmbeddingStore
  } = window.astermind;
</script>
```

---

<a id="installation"></a>
## 🚀 Installation

**NPM (scoped package):**
```bash
npm install @astermind/astermind-elm
# or
pnpm add @astermind/astermind-elm
# or
yarn add @astermind/astermind-elm
```

**CDN / `<script>` (UMD global `astermind`):**
```html
<!-- jsDelivr -->
<script src="https://cdn.jsdelivr.net/npm/@astermind/astermind-elm/dist/astermind.umd.js"></script>

<!-- or unpkg -->
<script src="https://unpkg.com/@astermind/astermind-elm/dist/astermind.umd.js"></script>

<script>
  const { ELM } = window.astermind;
</script>
```

**Repository:**
- GitHub: https://github.com/infiniteCrank/AsterMind-ELM
- NPM: https://www.npmjs.com/package/@astermind/astermind-elm

---

<a id="usage-examples"></a>
## 🛠️ Usage Examples

**Simple text classifier (ELM):**
```ts
import { ELM } from "@astermind/astermind-elm";

const elm = new ELM({
  categories: ['English', 'French'],
  hiddenUnits: 128,
  activation: 'relu',
  // text mode defaults are applied automatically
});

elm.train(); // or elm.trainFromData(X, Y)
console.log(elm.predict("bonjour", 3));
```

**Kernel ELM (Nyström + whitening):**
```ts
const kelm = new KernelELM({
  outputDim: Y[0].length,
  kernel: { type: 'rbf' },
  mode: 'nystrom',
  nystrom: { m: 256, strategy: 'kmeans++', whiten: true },
  ridgeLambda: 1e-2,
});
kelm.fit(X, Y);
```

**ELMChain embeddings:**
```ts
import { ELMChain } from "@astermind/astermind-elm";
const chain = new ELMChain([encoderELM, kelm /* or another ELM */]);
const emb = chain.getEmbedding(Xq);
```

---

<a id="astermind-in-practice"></a>
## 🌿 AsterMind in Practice

Because you can build AI systems that:

* Are decentralized and **self-healing** (online updates)
* Run **fully in the browser**
* Are **transparent** and interpretable
* Train and retrain **in milliseconds**
* Offer strong **latency + privacy** guarantees

---

<a id="core-api"></a>
## 📚 Core API

### ELM
- `train(augmentationOptions?, weights?)`
- `trainFromData(X, Y, { reuseWeights?, weights? })`
- `predict(text, topK=5)`
- `predictFromVector(X: number[][], topK=5)`
- `getEmbedding(X)`
- `loadModelFromJSON(json)`, `saveModelAsJSONFile(name?)`

### OnlineELM
- `init(X0, Y0)`, `update(X, Y)`, `fit(X, Y)`
- `predictLogitsFromVector(x) / FromVectors(X)`
- `predictProbaFromVector(x) / FromVectors(X)`
- `predictTopKFromVector(x, k) / FromVectors(X, k)`
- `getEmbedding(X)`
- `toJSON(includeP?)`, `loadFromJSON(json)`

### KernelELM
- `fit(X, Y)` (exact or nystrom)
- `predictLogitsFromVectors(X)` / `predictProbaFromVectors(X)`
- `getEmbedding(X)`
- `toJSON()`, `fromJSON(json)`

### ELMChain
- `getEmbedding(X: number[][])` — sequentially passes data through all encoders

---

<a id="config-reference"></a>
## ⚙️ Config Reference

**Text vs Numeric config split (TypeScript):**

```ts
type Activation = 'tanh' | 'relu' | 'leakyrelu' | 'sigmoid' | 'linear' | 'gelu';

interface BaseConfig {
  hiddenUnits: number;
  activation?: Activation;
  ridgeLambda?: number;
  seed?: number;
  log?: { modelName?: string; verbose?: boolean; toFile?: boolean; level?: 'info' | 'debug' };
  logFileName?: string;
  dropout?: number;
  weightInit?: 'uniform' | 'xavier'; // OnlineELM also supports 'he'
  exportFileName?: string;
}

export interface NumericConfig extends BaseConfig {
  inputSize: number;
  useTokenizer?: false;
  categories: string[];
}

export interface TextConfig extends BaseConfig {
  useTokenizer: true;
  categories: string[];
  maxLen: number;
  charSet?: string;
  tokenizerDelimiter?: RegExp;
  encoder?: any;
}

export type ELMConfig = NumericConfig | TextConfig;
```

---

<a id="embedding-store"></a>
## 🧰 Embedding Store

A lightweight vector store with **cosine/dot/euclidean** KNN, **unit-norm** storage, and **ring buffer** capacity.

- `upsert({ id, vec, meta })`, `query(vec, k, { metric, filter, returnVectors })`
- `queryById(id, k, opts)`, `toJSON()`, `fromJSON(json)`

Use it to cache embeddings or build fast in-memory search.

---

<a id="releases"></a>
## 📦 Releases

### v2.1.0 — 2025-09-19
**New features**
- Kernel ELMs (Exact & Nyström) with RBF/Linear/Poly/Laplacian/Custom kernels
- **Whitened Nyström** \(K_{mm}^{-1/2}\) via `Matrix.invSqrtSym` (symmetric eigen)
- Online ELM (RLS) with optional forgetting factor and He/Xavier/Uniform inits
- Web Worker adapter for ELM + KELM (non-blocking training/prediction)
- EmbeddingStore 2.0: unit-norm storage, capacity (ring buffer), metadata filters, JSON I/O
- ELMChain: safer chaining with dim checks, toJSON/fromJSON helpers
- Activations: **linear** and **gelu**
- Matrix: **eigSym** (Jacobi), **invSqrtSym**, sturdier Cholesky solve

**Improvements**
- Stronger TypeScript configs: `NumericConfig` vs `TextConfig`, clearer defaults
- Better UMD bundle exports: `window.astermind.{ELM,OnlineELM,KernelELM,KernelRegistry,EmbeddingStore,ELMChain}`
- More informative logs, consistent weight init messaging

**Bug fixes**
- Fixed Xavier initializer formula/log message
- Guarded text encoders to avoid `.toLowerCase()` on non-string inputs
- Resolved edge cases in `trainFromData` when `X`/`Y` are empty or mismatched
- Minor math fixes in metrics and dropout scaling

**Potential breaking changes**
- Config split into `NumericConfig | TextConfig` (update custom TypeScript types)
- New activation names (`linear`, `gelu`) added to `Activation` union

---

<a id="license"></a>
## 📄 License

MIT License

---

> **“AsterMind doesn’t just mimic a brain—it functions more like a starfish: fully decentralized, self-evaluating, and self-repairing.”**
