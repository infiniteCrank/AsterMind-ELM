# AsterMind-ELM

[![npm version](https://img.shields.io/npm/v/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![npm downloads](https://img.shields.io/npm/dm/%40astermind/astermind-elm.svg)](https://www.npmjs.com/package/@astermind/astermind-elm)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

A modular Extreme Learning Machine (ELM) library for JS/TS (browser + Node).

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage Example](#usage-example)
5. [Suggested Experiments](#suggested-experiments)
6. [Why Use AsterMind](#why-use-astermind)
7. [Core API Documentation](#core-api-documentation)
8. [Method Options Reference](#method-options-reference)
9. [ELMConfig Options](#elmconfig-options-reference)
10. [Prebuilt Modules](#prebuilt-modules-and-custom-modules)
11. [Text Encoding Modules](#text-encoding-modules)
12. [UI Binding Utility](#ui-binding-utility)
13. [Data Augmentation Utilities](#data-augmentation-utilities)
14. [IO Utilities (Experimental)](#io-utilities-experimental)
15. [Example Demos and Scripts](#example-demos-and-scripts)
16. [Experiments and Results](#experiments-and-results)
17. [License](#license)

---

<a id="introduction"></a>
# 🌟 AsterMind: Decentralized ELM Framework Inspired by Nature

Welcome to **AsterMind**, a modular, decentralized machine learning framework built around small, cooperating Extreme Learning Machines (ELMs) that self-train, self-evaluate, and self-repair—just like the decentralized nervous system of a starfish.

**How This ELM Library Differs from a Traditional ELM**

This library preserves the core Extreme Learning Machine idea—randomized hidden layer weights and biases, a nonlinear activation, and a one-step closed-form solution for output weights using a pseudoinverse—but extends it with several modern enhancements. Unlike a “vanilla” ELM, it supports multiple activation functions (ReLU, LeakyReLU, Sigmoid, Tanh), Xavier or uniform initialization, optional dropout on hidden activations, and sample weighting. It also integrates a full metrics gate (RMSE, MAE, Accuracy, F1, Cross-Entropy, R²) to decide whether to persist the trained model, and produces softmax probabilities rather than raw outputs. The library further includes utilities for weight reuse (simulating fine-tuning), detailed logging, JSON export/import, and model lifecycle management.

In addition, this implementation is designed for end-to-end usability. It includes a UniversalEncoder for text preprocessing (character or token level), built-in augmentation utilities, and the ability to chain multiple ELMs (ELMChain) for stacked random projections and embeddings—something not found in classic ELMs. These features make the library practical for real-world use cases like browser-based ML apps, rapid prototyping, and lightweight experiments, while still retaining the speed and simplicity that make ELMs appealing.

AsterMind is designed for:

* Lightweight, in-browser ML pipelines
* Transparent, interpretable predictions
* Continuous, incremental learning
* Resilient systems with no single point of failure

---

<a id="features"></a>
## ✨ Features

- ✅ Modular Architecture
- ✅ Self-Governing Training
- ✅ Flexible Preprocessing
- ✅ Lightweight Deployment (ESM + UMD)
- ✅ Retrieval and Classification Utilities

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

<a id="usage-example"></a>
## 🛠️ Usage Example

Define config, initialize an ELM, load or train model, predict:

```ts
import { ELM } from "@astermind/astermind-elm";

const config = { categories: ['English', 'French'], hiddenUnits: 128 };
const elm = new ELM(config);

// Load or train logic here
const results = elm.predict("bonjour");
console.log(results);
```

**CommonJS / Node:**
```js
const { ELM } = require("@astermind/astermind-elm");
```

---

<a id="suggested-experiments"></a>
## 🧪 Suggested Experiments

* Compare retrieval performance with Sentence-BERT and TFIDF.
* Experiment with activations and token vs char encoding.
* Deploy in-browser retraining workflows.

---

<a id="why-use-astermind"></a>
## 🌿 Why Use AsterMind?

Because you can build AI systems that:

* Are decentralized.
* Self-heal and retrain independently.
* Run in the browser.
* Are transparent and interpretable.

---

<a id="core-api-documentation"></a>
## 📚 Core API Documentation

### ELM Class

**Constructor:**
```ts
new ELM(config: ELMConfig)
```
* `config`: Configuration object specifying categories, hidden units, activation, metrics, and more.

**Methods:**
* `train(augmentationOptions?, weights?)`: Trains the model using auto-generated training data.
* `trainFromData(X, Y, options?)`: Trains the model using provided matrices.
* `predict(text, topK)`: Predicts probabilities for each label.
* `predictFromVector(vector, topK)`: Predicts from a pre-encoded input.
* `loadModelFromJSON(json)`: Loads a model from saved JSON.
* `saveModelAsJSONFile(filename?)`: Saves the model to disk.
* `computeHiddenLayer(X)`: Computes hidden layer activations.
* `getEmbedding(X)`: Returns embeddings.
* `calculateRMSE`, `calculateMAE`, `calculateAccuracy`, `calculateF1Score`, `calculateCrossEntropy`, `calculateR2Score`: Evaluation metrics.

---

<a id="method-options-reference"></a>
### 📘 Method Options Reference

#### `train(augmentationOptions?, weights?)`
* `augmentationOptions`: An object `{ suffixes, prefixes, includeNoise }` to augment training data.
  * `suffixes`: Array of suffix strings to append.
  * `prefixes`: Array of prefix strings to prepend.
  * `includeNoise`: `boolean` to randomly perturb tokens.
* `weights`: Array of sample weights.

#### `trainFromData(X, Y, options?)`
* `X`: Input matrix.
* `Y`: Label matrix.
* `options`:
  * `reuseWeights`: `true` to reuse previous weights.
  * `weights`: Array of sample weights.

#### `predict(text, topK)`
* `text`: Input string.
* `topK`: How many predictions to return (default 5).

#### `predictFromVector(vector, topK)`
* `vector`: Pre-encoded numeric array.
* `topK`: Number of results.

#### `saveModelAsJSONFile(filename?)`
* `filename`: Optional custom file name.

---

<a id="elmconfig-options-reference"></a>
## ⚙️ ELMConfig Options Reference

| Option               | Type       | Description                                                   |
| -------------------- | ---------- | ------------------------------------------------------------- |
| `categories`         | `string[]` | List of labels the model should classify. *(Required)*        |
| `hiddenUnits`        | `number`   | Number of hidden layer units (default: 50).                   |
| `maxLen`             | `number`   | Max length of input sequences (default: 30).                  |
| `activation`         | `string`   | Activation function (`relu`, `tanh`, etc.) (default: `relu`). |
| `encoder`            | `any`      | Custom UniversalEncoder instance (optional).                  |
| `charSet`            | `string`   | Character set used for encoding (default: lowercase a-z).     |
| `useTokenizer`       | `boolean`  | Use token-level encoding (default: false).                    |
| `tokenizerDelimiter` | `RegExp`   | Custom tokenizer regex (default: `/\\s+/`).                   |
| `exportFileName`     | `string`   | Filename to export the model JSON.                            |
| `metrics`            | `object`   | Performance thresholds (`rmse`, `mae`, `accuracy`, etc.).     |
| `log`                | `object`   | Logging configuration: `modelName`, `verbose`, `toFile`.      |
| `logFileName`        | `string`   | File name for log exports.                                    |
| `dropout`            | `number`   | Dropout rate between 0 and 1.                                 |
| `weightInit`         | `string`   | Weight initializer (`uniform` or `xavier`).                   |

Refer to `ELMConfig.ts` for defaults and examples.

---

### ELMChain Class

**Constructor:**
```ts
new ELMChain(encoders: ELM[])
```

**Methods:**
* `getEmbedding(X)`: Sequentially passes data through all encoders.

---

### TFIDFVectorizer Class
* `vectorize(doc)`: Converts text into TFIDF vector.
* `vectorizeAll()`: Converts all training documents.

---

### KNN
* `KNN.find(queryVec, dataset, k, topX, metric)`: Finds k nearest neighbors.

For detailed examples, see `examples/` folder in the repository.

---

<a id="core-api-documentation-with-examples"></a>
## 📚 Core API Documentation with Examples

### ELM Class

**Constructor:**
```ts
const elm = new ELM({
  categories: ["English", "French"],
  hiddenUnits: 100,
  activation: "relu",
  log: { modelName: "LangModel" }
});
```

**Example Training:**
```ts
elm.train();
```

**Example Prediction:**
```ts
const results = elm.predict("bonjour");
console.log(results);
```

**Diagram:**
```
Input Text -> UniversalEncoder -> Hidden Layer -> Output Weights -> Probabilities
```

---

### ELMChain Class

**Constructor:**
```ts
const chain = new ELMChain([encoderELM, classifierELM]);
```

**Embedding Example:**
```ts
const embedding = chain.getEmbedding([vector]);
```

**Diagram:**
```
Input -> ELM1 -> Embedding -> ELM2 -> Final Embedding
```

---

<a id="prebuilt-modules-and-custom-modules"></a>
## 🧩 Prebuilt Modules and Custom Modules

AsterMind comes with a set of **prebuilt module classes** that wrap and extend `ELM` for specific use cases:

* `AutoComplete`: Learns to autocomplete inputs.
* `EncoderELM`: Encodes text into dense feature vectors.
* `CharacterLangEncoderELM`: Encodes character-level language representations.
* `FeatureCombinerELM`: Merges embedding vectors with metadata.
* `ConfidenceClassifierELM`: Classifies confidence levels.
* `IntentClassifier`: Classifies user intents.
* `LanguageClassifier`: Detects text language.
* `VotingClassifierELM`: Combines predictions from multiple ELMs.
* `RefinerELM`: Refines predictions based on low-confidence results.

These classes expose consistent methods like `.train()`, `.predict()`, `.loadModelFromJSON()`, `.saveModelAsJSONFile()`, and `.encode()` (for encoders).

**Custom Modules:**
```ts
class MyCustomELM {
  private elm: ELM;
  constructor(config: ELMConfig) {
    this.elm = new ELM(config);
  }
  train(pairs: { input: string; label: string }[]) {
    // your logic
  }
  predict(text: string) {
    return this.elm.predict(text);
  }
}
```
Each prebuilt module is an example of this pattern.

---

<a id="text-encoding-modules"></a>
## ✨ Text Encoding Modules

AsterMind includes several text encoding utilities:

* **TextEncoder**: Converts raw text to normalized one-hot vectors.
  * Supports character-level and token-level encoding.
  * Options: `charSet`, `maxLen`, `useTokenizer`, `tokenizerDelimiter`.
  * Methods:
    * `textToVector(text)`: Encodes text.
    * `normalizeVector(v)`: Normalizes vectors.
    * `getVectorSize()`: Returns the total length of output vectors.

* **Tokenizer**:
  * Splits text into tokens.
  * Methods:
    * `tokenize(text)`: Returns an array of tokens.
    * `ngrams(tokens, n)`: Generates n-grams.

* **UniversalEncoder**:
  * Automatically configures char vs token mode.
  * Simplifies encoding.
  * Methods:
    * `encode(text)`: Returns numeric vector.
    * `normalize(vector)`: Normalizes vector.

**Notes from Experiments:**
* Character-level encodings are more robust for small vocabularies.
* Token-level encodings improved retrieval accuracy on large datasets.
* Normalization is important for similarity searches.

---

<a id="ui-binding-utility"></a>
## 🖥️ UI Binding Utility

**bindAutocompleteUI** is a helper to wire an ELM model to HTML inputs and outputs.

**Options:**
* `model` (ELM): The trained ELM instance.
* `inputElement` (HTMLInputElement): Text input element.
* `outputElement` (HTMLElement): Element where predictions are rendered.
* `topK` (number, optional): How many predictions to show (default: 5).

**Behavior:**
* Listens to the `input` event.
* Runs `model.predict()` when typing.
* Displays predictions as a list with probabilities.
* If input is empty, shows a placeholder message.
* If prediction fails, shows error message in red.

**Usage Example:**
```ts
bindAutocompleteUI({
  model: myELM,
  inputElement: document.getElementById('query') as HTMLInputElement,
  outputElement: document.getElementById('results'),
  topK: 3
});
```

**Customization:** You can modify rendering logic or styling by editing `bindAutocompleteUI`. See `BindUI.ts` for full source.

---

<a id="data-augmentation-utilities"></a>
## ✨ Data Augmentation Utilities

**Augment** provides methods to enrich training data by generating new variants.

**Methods:**
* `addSuffix(text, suffixes)`: Appends each suffix to the text.
* `addPrefix(text, prefixes)`: Prepends each prefix to the text.
* `addNoise(text, charSet, noiseRate)`: Randomly replaces characters in `text` with characters from `charSet`. `noiseRate` controls the probability per character.
* `mix(text, mixins)`: Combines text with mixins.
* `generateVariants(text, charSet, options)`: Creates a list of augmented examples by applying suffixes, prefixes, and/or noise.

**Options for `generateVariants`:**
* `suffixes` (`string[]`): List of suffixes to append.
* `prefixes` (`string[]`): List of prefixes to prepend.
* `includeNoise` (`boolean`): Whether to add noisy variants.

**Example:**
```ts
const variants = Augment.generateVariants("hello", "abcdefghijklmnopqrstuvwxyz", {
  suffixes: ["world"],
  prefixes: ["greeting"],
  includeNoise: true
});
```

---

<a id="io-utilities-experimental"></a>
## ⚠️ IO Utilities (Experimental)

**IO** provides methods for importing, exporting, and inferring schemas of labeled training data. **Note:** These APIs are highly experimental and may be buggy.

**Methods:**
* `importJSON(json)`: Parse JSON array into labeled examples.
* `exportJSON(pairs)`: Serialize labeled examples into JSON.
* `importCSV(csv, hasHeader)`: Parse CSV into labeled examples.
* `exportCSV(pairs, includeHeader)`: Export to CSV string.
* `importTSV(tsv, hasHeader)`: Parse TSV into labeled examples.
* `exportTSV(pairs, includeHeader)`: Export to TSV string.
* `inferSchemaFromCSV(csv)`: Attempt to infer schema fields and suggest mappings from CSV.
* `inferSchemaFromJSON(json)`: Attempt to infer schema fields and suggest mappings from JSON.

**Caution:**
* Schema inference can fail or produce incorrect mappings.
* Delimited import assumes the first row is a header unless `hasHeader` is `false`.
* If a row has only one column, it will be used as both `text` and `label`.

**Example:**
```ts
const examples = IO.importCSV("text,label\nhello,greet\nbye,farewell");
const schema = IO.inferSchemaFromCSV("text,label\nhi,hello");
```

> **Tip:** In practice, importing and exporting **JSON** is the most reliable path. Prefer `importJSON()` and `exportJSON()` over CSV/TSV for production.

---

<a id="example-demos-and-scripts"></a>
## 🧪 Example Demos and Scripts

AsterMind includes multiple demo scripts you can launch via `npm run` commands:

* `dev:autocomplete`: Starts the autocomplete demo.
* `dev:lang`: Starts the language classification demo.
* `dev:chain`: Runs a pipeline chaining autocomplete and language classifier.
* `dev:news`: Trains on the AG News dataset (note: memory heavy).

**How to Run:**
```bash
npm install
npm run dev:autocomplete
```

**What You'll See:**
* A browser window with a live demo interface.
* Input box for typing test queries.
* Real-time predictions and confidence bars.

> These demos are fully in-browser and do not require any backend. Each script sets `DEMO` to load a different HTML+JavaScript pipeline.

---

<a id="experiments-and-results"></a>
## 🧪 Experiments and Results

AsterMind has been tested with a variety of automated experiments, including:

* **Dropout Tuning Experiments:** Scripts testing different dropout rates and activation functions.
* **Hybrid Retrieval Pipelines:** Combining dense embeddings and TFIDF.
* **Ensemble Knowledge Distillation:** Training ELMs to mimic ensembles.
* **Multi-Level Pipelines:** Chaining autocomplete, encoder, and classifier modules.

**Example Scripts:**
* `automated_experiment_dropout_fixedactivation.ts`
* `hybrid_retrieval.ts`
* `elm_ensemble_knowledge_distillation.ts`
* `train_hybrid_multilevel_pipeline.ts`
* `train_multi_encoder.ts` → `npx ts-node train_multi_encoder.ts`
* `train_weighted_hybrid_multilevel_pipeline.ts`

**Results Summary:**
| Experiment               | Dropout | Activation | Recall@1 | Recall@5 | MRR  |
| ------------------------ | ------- | ---------- | -------- | -------- | ---- |
| Dropout Fixed Activation | 0.05    | relu       | 0.42     | 0.75     | 0.61 |
| Hybrid Random Target     | 0.02    | tanh       | 0.46     | 0.78     | 0.65 |

> Results exported from CSV logs; see scripts to reproduce.

---

<a id="license"></a>
## 📄 License

MIT License

---

> **“AsterMind doesn’t just mimic a brain—it functions more like a starfish: fully decentralized, self-evaluating, and self-repairing.”**
