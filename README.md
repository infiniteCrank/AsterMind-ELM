# AsterMind-ELM

[![npm version](https://img.shields.io/npm/v/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![npm downloads](https://img.shields.io/npm/dm/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

**A modular Extreme Learning Machine (ELM) library for JavaScript/TypeScript** that brings instant, on-device machine learning to browsers and Node.js.

---

## 🚀 Quick Start

```bash
npm install @astermind/astermind-elm
```

```typescript
import { ELM } from '@astermind/astermind-elm';

// Create and train a language classifier in milliseconds
const elm = new ELM({
  categories: ['English', 'French', 'Spanish'],
  hiddenUnits: 128
});

// Train with your data
elm.trainFromData(X, Y);

// Predict instantly
const prediction = elm.predict("bonjour");
console.log(prediction); // { category: 'French', confidence: 0.95 }
```

---

## 🌟 What Makes AsterMind Special

AsterMind brings **instant, tiny, on-device ML** to the web. Unlike traditional neural networks that require heavy training, ELMs provide:

- **⚡ Instant Training**: Train models in milliseconds, not minutes
- **🔒 Privacy First**: Everything runs locally - no data leaves your device
- **📱 Zero Dependencies**: No GPU, no server, no external APIs needed
- **🎯 Tiny Footprint**: Models are incredibly lightweight and fast
- **🔧 Modular Design**: Mix and match components for complex pipelines

### Perfect For:
- **Real-time classification** (language, sentiment, intent detection)
- **On-device search and retrieval** with compact embeddings
- **Interactive creative tools** (music generators, autocompletes)
- **Edge analytics** where data never leaves the device
- **Privacy-sensitive applications** requiring local ML

---

## 🆕 New in this release

- **Kernel ELMs (KELMs)** — exact and Nyström kernels (RBF/Linear/Poly/Laplacian/Custom) with ridge solve  
- **Whitened Nyström** — optional \(K_{mm}^{-1/2}\) whitening via symmetric eigendecomposition  
- **Online ELM (OS-ELM)** — streaming RLS updates with forgetting factor (no full retrain)  
- **DeepELM** — multi-layer stacked ELM with non-linear projections  
- **Web Worker adapter** — off-main-thread training/prediction for ELM and KELM  
- **Matrix upgrades** — Jacobi eigendecomp, invSqrtSym, improved Cholesky  
- **EmbeddingStore 2.0** — unit-norm vectors, ring buffer capacity, metadata filters  
- **ELMChain+Embeddings** — safer chaining with dimension checks, JSON I/O  
- **Activations** — added **linear** and **gelu**; centralized registry  
- **Configs** — split into **Numeric** and **Text** configs; stronger typing  
- **UMD exports** — `window.astermind` exposes `ELM`, `OnlineELM`, `KernelELM`, `DeepELM`, `KernelRegistry`, `EmbeddingStore`, `ELMChain`, etc.  
- **Robust preprocessing** — safer encoder path, improved error handling

See [Releases](#releases) for full changelog.

---

## 📑 Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Quick Examples](#quick-examples)
4. [Advanced Features](#advanced-features)
5. [API Reference](#api-reference)
6. [Examples & Demos](#examples--demos)
7. [Why ELMs?](#why-elms)
8. [Contributing](#contributing)
9. [License](#license)

---

## Core Concepts

### What is an Extreme Learning Machine (ELM)?

An ELM is a type of neural network that:
1. **Randomly initializes** the hidden layer weights (no backpropagation needed)
2. **Uses a closed-form solution** to find the output weights
3. **Trains in milliseconds** instead of minutes or hours
4. **Maintains high accuracy** for many classification and regression tasks

### Key Components

- **ELM**: Basic classifier/regressor with instant training
- **KernelELM**: Kernel-based ELM for non-linear problems
- **OnlineELM**: Stream learning for continuous updates
- **DeepELM**: Multi-layer stacked ELMs
- **EmbeddingStore**: Vector database for similarity search  

---

## Quick Examples

### 1. Language Classification

```typescript
import { ELM } from '@astermind/astermind-elm';

const elm = new ELM({
  categories: ['English', 'French', 'Spanish', 'German'],
  hiddenUnits: 256,
  activation: 'relu'
});

// Train with your data
const texts = ['hello', 'bonjour', 'hola', 'guten tag'];
const labels = ['English', 'French', 'Spanish', 'German'];
elm.trainFromData(texts, labels);

// Predict
const result = elm.predict("bonjour");
console.log(result.category); // 'French'
```

### 2. Real-time Sentiment Analysis

```typescript
import { OnlineELM } from '@astermind/astermind-elm';

const sentiment = new OnlineELM({
  inputDim: 100,
  outputDim: 3, // positive, negative, neutral
  hiddenUnits: 128
});

// Initialize with some data
sentiment.init(initialTexts, initialLabels);

// Update with new data as it comes in
sentiment.update(newTexts, newLabels);

// Get predictions
const prediction = sentiment.predictProbaFromVectors(textVector);
```

### 3. Vector Similarity Search

```typescript
import { EmbeddingStore, ELM } from '@astermind/astermind-elm';

// Create an embedding model
const embedder = new ELM({
  categories: ['dummy'], // We'll use it for embeddings
  hiddenUnits: 512
});

// Create vector store
const store = new EmbeddingStore({ capacity: 10000 });

// Add documents
const documents = [
  { id: 'doc1', text: 'Machine learning is fascinating' },
  { id: 'doc2', text: 'Neural networks are powerful' }
];

documents.forEach(doc => {
  const embedding = embedder.getEmbedding(doc.text);
  store.add({ id: doc.id, vector: embedding, meta: doc });
});

// Search
const query = 'artificial intelligence';
const queryEmbedding = embedder.getEmbedding(query);
const results = store.query({ vector: queryEmbedding, k: 5 });
```  

---

<a id="kernel-elms-kelm"></a>
## 🧠 Kernel ELMs (KELM)

Supports **Exact** and **Nyström** modes with RBF/Linear/Poly/Laplacian/Custom kernels.  
Includes **whitened Nyström** (persisted whitener for inference parity).

```ts
import { KernelELM, KernelRegistry } from '@astermind/astermind-elm';

const kelm = new KernelELM({
  outputDim: Y[0].length,
  kernel: { type: 'rbf', gamma: 1 / X[0].length },
  mode: 'nystrom',
  nystrom: { m: 256, strategy: 'kmeans++', whiten: true },
  ridgeLambda: 1e-2,
});
kelm.fit(X, Y);
```

---

<a id="online-elm-os-elm"></a>
## 🔁 Online ELM (OS-ELM)

Stream updates via **Recursive Least Squares (RLS)** with optional forgetting factor. Supports He/Xavier/Uniform initializers.

```ts
import { OnlineELM } from '@astermind/astermind-elm';
const ol = new OnlineELM({ inputDim: D, outputDim: K, hiddenUnits: 256 });
ol.init(X0, Y0);
ol.update(Xt, Yt);
ol.predictProbaFromVectors(Xq);
```

**Notes**  
- `forgettingFactor` controls how fast older observations decay (default 1.0).  
- Two natural embedding modes: **hidden** (activations) or **logits** (pre-softmax). Use with `ELMAdapter` (see below).

---

<a id="deepelm"></a>
## 🌊 DeepELM

Stack multiple ELM layers for deep nonlinear embeddings and an optional top ELM classifier.

```ts
import { DeepELM } from '@astermind/astermind-elm';
const deep = new DeepELM({
  inputDim: D,
  layers: [{ hiddenUnits: 128 }, { hiddenUnits: 64 }],
  numClasses: K
});
// 1) Unsupervised layer-wise training (autoencoders Y=X)
const X_L = deep.fitAutoencoders(X);
// 2) Supervised head (ELM) on last layer features
deep.fitClassifier(X_L, Y);
// 3) Predict
const probs = deep.predictProbaFromVectors(Xq);
```

**JSON I/O**  
`toJSON()` and `fromJSON()` persist the full stack (AEs + classifier).

---

<a id="web-worker-adapter"></a>
## 🧵 Web Worker Adapter

Move heavy ops off the main thread. Provides `ELMWorker` + `ELMWorkerClient` for RPC-style training/prediction with progress events.

- Initialize with `initELM(config)` or `initOnlineELM(config)`  
- Train via `train` / `trainFromData` / `fit` / `update`  
- Predict via `predict`, `predictFromVector`, or `predictLogits`  
- Subscribe to progress callbacks per call

See [Workers](#workers-elmworker--elmworkerclient) for full API.

---

## Installation

### NPM
```bash
npm install @astermind/astermind-elm
# or
pnpm add @astermind/astermind-elm
# or
yarn add @astermind/astermind-elm
```

### CDN (Browser)
```html
<script src="https://cdn.jsdelivr.net/npm/@astermind/astermind-elm/dist/astermind.umd.js"></script>
<script>
  const { ELM, KernelELM, OnlineELM } = window.astermind;
</script>
```

### TypeScript Support
Full TypeScript definitions are included. No additional `@types` package needed.

---

## Advanced Features

### Kernel ELMs for Complex Patterns

```typescript
import { KernelELM, KernelRegistry } from '@astermind/astermind-elm';

const kelm = new KernelELM({
  outputDim: 3,
  kernel: { type: 'rbf', gamma: 0.1 },
  mode: 'nystrom',
  nystrom: { m: 256, strategy: 'kmeans++', whiten: true },
  ridgeLambda: 1e-2
});

kelm.fit(X, Y);
const predictions = kelm.predictProbaFromVectors(X_test);
```

### Deep ELM Networks

```typescript
import { DeepELM } from '@astermind/astermind-elm';

const deep = new DeepELM({
  inputDim: 100,
  layers: [
    { hiddenUnits: 128 },
    { hiddenUnits: 64 },
    { hiddenUnits: 32 }
  ],
  numClasses: 5
});

// Train autoencoders layer by layer
const features = deep.fitAutoencoders(X);

// Train final classifier
deep.fitClassifier(features, Y);

// Predict
const predictions = deep.predictProbaFromVectors(X_test);
```

### Web Workers for Heavy Computation

```typescript
import { ELMWorkerClient } from '@astermind/astermind-elm/worker';

const client = new ELMWorkerClient(
  new Worker(new URL('./worker.js', import.meta.url))
);

await client.initELM({
  categories: ['A', 'B', 'C'],
  hiddenUnits: 256
});

// Training happens off-main-thread
await client.elmTrain({}, (progress) => {
  console.log(`Training: ${progress.pct}%`);
});

const predictions = await client.elmPredict('test input');
```  

---

## API Reference

### ELM Class

The core Extreme Learning Machine implementation.

```typescript
const elm = new ELM(config);
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `categories` | `string[]` | **Required** | List of classification categories |
| `hiddenUnits` | `number` | `50` | Number of hidden layer neurons |
| `activation` | `string` | `'relu'` | Activation function (`relu`, `sigmoid`, `tanh`, `linear`, `gelu`) |
| `weightInit` | `string` | `'xavier'` | Weight initialization (`uniform`, `xavier`, `he`) |
| `ridgeLambda` | `number` | `1e-6` | Ridge regularization parameter |
| `dropout` | `number` | `0` | Dropout rate (0-1) |
| `seed` | `number` | `null` | Random seed for reproducibility |

#### Methods

- `trainFromData(X, Y, options?)` - Train with data matrices
- `predict(text, topK?)` - Predict category for text input
- `predictFromVector(vector, topK?)` - Predict from feature vector
- `getEmbedding(text)` - Get embedding vector for text
- `saveModelAsJSONFile(filename?)` - Export model to JSON
- `loadModelFromJSON(json)` - Load model from JSON

### OnlineELM Class

For streaming/online learning scenarios.

```typescript
const online = new OnlineELM({
  inputDim: 100,
  outputDim: 3,
  hiddenUnits: 128,
  forgettingFactor: 0.99 // How fast to forget old data
});

online.init(initialX, initialY);
online.update(newX, newY);
```

### KernelELM Class

Kernel-based ELM for non-linear problems.

```typescript
const kelm = new KernelELM({
  outputDim: 3,
  kernel: { type: 'rbf', gamma: 0.1 },
  mode: 'exact' // or 'nystrom'
});
```

### EmbeddingStore Class

Vector database for similarity search.

```typescript
const store = new EmbeddingStore({
  capacity: 10000,
  normalize: true
});

store.add({ id: 'doc1', vector: [0.1, 0.2, ...], meta: { title: 'Doc 1' } });
const results = store.query({ vector: queryVec, k: 10, metric: 'cosine' });
```

---

## Examples & Demos

### Browser Demos
- **Language Classification**: `examples/language-awareness-demo/`
- **Autocomplete**: `examples/autocomplete-chain/`
- **News Classification**: `examples/ag-news-demo/`
- **Drum Generator**: `examples/elm-drum-demo-mainthread/`

### Node.js Examples
- **AG News Classification**: `node_examples/agnews-two-stage-retrieval.ts`
- **Book Indexing**: `node_examples/book-index-elm-tfidf.ts`
- **Deep ELM Retrieval**: `node_examples/deepelm-kelm-retrieval.ts`

Run demos with:
```bash
npm run dev:autocomplete
npm run dev:lang
npm run dev:chain
npm run dev:news
```

---

## Why ELMs?

### Traditional Neural Networks
- ❌ Require backpropagation (slow training)
- ❌ Need careful hyperparameter tuning
- ❌ Can overfit easily
- ❌ Require large datasets

### Extreme Learning Machines
- ✅ **Instant training** with closed-form solution
- ✅ **Robust** to hyperparameter choices
- ✅ **Less prone to overfitting**
- ✅ **Work well with small datasets**
- ✅ **Interpretable** and transparent
- ✅ **Perfect for edge devices**

### When to Use ELMs
- **Real-time applications** where speed matters
- **Privacy-sensitive** scenarios requiring local processing
- **Resource-constrained** environments (mobile, IoT)
- **Prototyping** and rapid experimentation
- **Ensemble methods** where you need many fast models  

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/infiniteCrank/AsterMind-ELM
cd AsterMind-ELM
npm install
npm run build
npm test
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

AsterMind is inspired by the decentralized, self-organizing nature of starfish nervous systems. Just as starfish can regenerate and adapt without a central brain, AsterMind enables ML systems that are resilient, distributed, and self-improving.

---

> **"AsterMind doesn't just mimic a brain—it functions more like a starfish: fully decentralized, self-evaluating, and self-repairing."**

---

## Changelog

### v2.1.0 (Latest)
- ✨ **Kernel ELMs** with RBF, Linear, Polynomial, and Laplacian kernels
- ✨ **Online ELM** for streaming learning
- ✨ **DeepELM** for multi-layer architectures
- ✨ **Web Worker support** for off-main-thread processing
- ✨ **EmbeddingStore 2.0** with improved performance
- ✨ **New activations**: Linear and GELU
- 🔧 **Improved matrix operations** with better numerical stability
- 📚 **Enhanced documentation** and examples

See [Releases](https://github.com/infiniteCrank/AsterMind-ELM/releases) for full changelog.
