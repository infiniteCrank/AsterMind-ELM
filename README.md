## 📑 Table of Contents

1. [Introduction](#🌟-astermind-readme)
2. [Features](#✨-features)
3. [Installation](#🚀-installation)
4. [Usage Example](#🛠️-usage-example)
5. [Suggested Experiments](#🧪-suggested-experiments)
6. [Why Use AsterMind](#🌿-why-use-astermind)
7. [Core API Documentation](#📚-core-api-documentation)
8. [Method Options Reference](#📘-method-options-reference)
9. [ELMConfig Options](#⚙️-elmconfig-options-reference)
10. [Prebuilt Modules](#🧩-prebuilt-modules-and-custom-modules)
11. [Text Encoding Modules](#✨-text-encoding-modules)
12. [UI Binding Utility](#🖥️-ui-binding-utility)
13. [Data Augmentation Utilities](#✨-data-augmentation-utilities)
14. [IO Utilities](#⚠️-io-utilities-experimental)
15. [Example Demos and Scripts](#🧪-example-demos-and-scripts)
16. [Experiments and Results](#🧪-experiments-and-results)
17. [License](#📄-license)

# 🌟 AsterMind: Decentralized ELM Framework Inspired by Nature

Welcome to **AsterMind**, a modular, decentralized machine learning framework built around small, cooperating Extreme Learning Machines (ELMs) that self-train, self-evaluate, and self-repair—just like the decentralized nervous system of a starfish.

🔍 How This ELM Library Differs from a Traditional ELM

This library preserves the core Extreme Learning Machine idea—randomized hidden layer weights and biases, a nonlinear activation, and a one-step closed-form solution for output weights using a pseudoinverse—but extends it with several modern enhancements. Unlike a “vanilla” ELM, it supports multiple activation functions (ReLU, LeakyReLU, Sigmoid, Tanh), Xavier or uniform initialization, optional dropout on hidden activations, and sample weighting. It also integrates a full metrics gate (RMSE, MAE, Accuracy, F1, Cross-Entropy, R²) to decide whether to persist the trained model, and produces softmax probabilities rather than raw outputs. The library further includes utilities for weight reuse (simulating fine-tuning), detailed logging, JSON export/import, and model lifecycle management.

In addition, this implementation is designed for end-to-end usability. It includes a UniversalEncoder for text preprocessing (character or token level), built-in augmentation utilities, and the ability to chain multiple ELMs (ELMChain) for stacked random projections and embeddings—something not found in classic ELMs. These features make the library practical for real-world use cases like browser-based ML apps, rapid prototyping, and lightweight experiments, while still retaining the speed and simplicity that make ELMs appealing.

AsterMind is designed for:

* Lightweight, in-browser ML pipelines
* Transparent, interpretable predictions
* Continuous, incremental learning
* Resilient systems with no single point of failure

---

## ✨ Features

✅ Modular Architecture
✅ Self-Governing Training
✅ Flexible Preprocessing
✅ Lightweight Deployment
✅ Retrieval and Classification Utilities

---

## 🚀 Installation

Clone the repository and import modules:

```bash
https://github.com/infiniteCrank/astermind
```

---

## 🛠️ Usage Example

Define config, initialize an ELM, load or train model, predict:

```typescript
const config = { categories: ['English', 'French'], hiddenUnits: 128 };
const elm = new ELM(config);
// Load or train logic here
const results = elm.predict("bonjour");
```

---

## 🧪 Suggested Experiments

* Compare retrieval performance with Sentence-BERT and TFIDF.
* Experiment with activations and token vs char encoding.
* Deploy in-browser retraining workflows.

---

## 🌿 Why Use AsterMind?

Because you can build AI systems that:

* Are decentralized.
* Self-heal and retrain independently.
* Run in the browser.
* Are transparent and interpretable.

---
## 📚 Core API Documentation

### ELM Class

**Constructor:**

```typescript
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
| `tokenizerDelimiter` | `RegExp`   | Custom tokenizer regex (default: `/\s+/`).                    |
| `exportFileName`     | `string`   | Filename to export the model JSON.                            |
| `metrics`            | `object`   | Performance thresholds (`rmse`, `mae`, `accuracy`, etc.).     |
| `log`                | `object`   | Logging configuration: `modelName`, `verbose`, `toFile`.      |
| `logFileName`        | `string`   | File name for log exports.                                    |
| `dropout`            | `number`   | Dropout rate between 0 and 1.                                 |
| `weightInit`         | `string`   | Weight initializer (`uniform` or `xavier`).                   |

Refer to `ELMConfig.ts` for defaults and examples.

### ELMChain Class

**Constructor:**

```typescript
new ELMChain(encoders: ELM[])
```

**Methods:**

* `getEmbedding(X)`: Sequentially passes data through all encoders.

### TFIDFVectorizer Class

* `vectorize(doc)`: Converts text into TFIDF vector.
* `vectorizeAll()`: Converts all training documents.

### KNN

* `KNN.find(queryVec, dataset, k, topX, metric)`: Finds k nearest neighbors.

For detailed examples, see `examples/` folder in the repository.

## 📚 Core API Documentation with Examples

### ELM Class

**Constructor:**

```typescript
const elm = new ELM({
  categories: ["English", "French"],
  hiddenUnits: 100,
  activation: "relu",
  log: { modelName: "LangModel" }
});
```

**Example Training:**

```typescript
elm.train();
```

**Example Prediction:**

```typescript
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

```typescript
const chain = new ELMChain([encoderELM, classifierELM]);
```

**Embedding Example:**

```typescript
const embedding = chain.getEmbedding([vector]);
```

**Diagram:**

```
Input -> ELM1 -> Embedding -> ELM2 -> Final Embedding
```

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

You can build your own module by composing `ELM` in a similar way:

```typescript
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

Refer to `TextEncoder.ts`, `Tokenizer.ts`, and `UniversalEncoder.ts` for implementation details.

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

```typescript
bindAutocompleteUI({
  model: myELM,
  inputElement: document.getElementById('query') as HTMLInputElement,
  outputElement: document.getElementById('results'),
  topK: 3
});
```

**Customization:**

You can modify rendering logic or styling by editing `bindAutocompleteUI`.

Refer to `BindUI.ts` for full source.

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

**Example Usage:**

```typescript
const variants = Augment.generateVariants("hello", "abcdefghijklmnopqrstuvwxyz", {
  suffixes: ["world"],
  prefixes: ["greeting"],
  includeNoise: true
});
```
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

**Example Usage:**

```typescript
const examples = IO.importCSV("text,label\nhello,greet\nbye,farewell");
const schema = IO.inferSchemaFromCSV("text,label\nhi,hello");
```

**Tip:** In practice, importing and exporting **JSON** has been the most reliable and thoroughly tested method. If possible, prefer using `importJSON()` and `exportJSON()` over CSV or TSV.
## 🧪 Example Demos and Scripts

AsterMind includes multiple demo scripts you can launch via `npm run` commands:

* `dev:autocomplete`: Starts the autocomplete demo.
* `dev:lang`: Starts the language classification demo.
* `dev:chain`: Runs a pipeline chaining autocomplete and language classifier.
* `dev:news`: This model is trained on the ag news classification data set (there is memory problems currently)

**How to Run:**

```bash
npm install
npm run dev:autocomplete
```

**What You'll See:**

* A browser window with a live demo interface.
* Input box for typing test queries.
* Real-time predictions and confidence bars.

**Note:**

These demos are fully in-browser and do not require any backend. Each script sets `DEMO` to load a different HTML+JavaScript pipeline.

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
* `train_multi_encoder.ts`: Run with `npx ts-node train_multi_encoder.ts`
* `train_weighted_hybrid_multilevel_pipeline.ts`: 


### Run with `npx ts-node train_weighted_hybrid_multilevel_pipeline.ts`
### Also change tsconfig.json to the following:
```{
    "compilerOptions": {
        "target": "ES6",
        "module": "CommonJS",
        //"module": "esnext",
        ...
 ```

**Results Summary:**

| Experiment               | Dropout | Activation | Recall\@1 | Recall\@5 | MRR  |
| ------------------------ | ------- | ---------- | --------- | --------- | ---- |
| Dropout Fixed Activation | 0.05    | relu       | 0.42      | 0.75      | 0.61 |
| Hybrid Random Target     | 0.02    | tanh       | 0.46      | 0.78      | 0.65 |

**Note:** These results were exported from CSV logs and can be reproduced with the provided scripts.

### TFIDFVectorizer Class

**Example:**

```typescript
const vectorizer = new TFIDFVectorizer(["text one", "text two"]);
const vector = vectorizer.vectorize("text one");
```

**Diagram:**

```
Text -> Tokenization -> TFIDF Vector
```

---

### KNN

**Example:**

```typescript
const neighbors = KNN.find(queryVec, dataset, 5, 3, "cosine");
```

**Diagram:**

```
Query Vector -> Similarity -> Nearest Neighbors
```

---

For more examples, see the `examples/` folder.

## 📄 License

MIT License

---

**"AsterMind doesn’t just mimic a brain—it functions more like a starfish: fully decentralized, self-evaluating, and self-repairing."**
