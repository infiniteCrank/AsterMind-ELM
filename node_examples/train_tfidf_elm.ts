/**
 * Experiment: TF-IDF + ELM Classifier (Single Model, Direct Prediction)
 *
 * Goal
 *  - Train an Extreme Learning Machine (ELM) on TF-IDF vectors extracted
 *    from a small labeled text corpus, then use it to predict categories
 *    for unseen text.
 *
 * What it does
 *  - Defines a small training dataset across three categories (Go, Python, TypeScript).
 *  - Builds a TF-IDF vectorizer on training docs (vocab size = 500).
 *  - Converts docs into normalized TF-IDF vectors.
 *  - One-hot encodes labels for supervised training.
 *  - Instantiates an ELM with:
 *      hiddenUnits = 50,
 *      activation = sigmoid,
 *      weight initialization = Xavier.
 *  - Trains on TF-IDF vectors and one-hot labels.
 *  - Vectorizes a test sentence and predicts category probabilities.
 *
 * Why
 *  - Demonstrates a minimal supervised ELM pipeline with TF-IDF.
 *  - Suitable as a baseline text classification experiment.
 *
 * Pipeline Overview
 *
 *   Training Docs ──► TF-IDF ──► ELM (X→Y) ──► Train Weights
 *                                         │
 *   New Text ──► TF-IDF ──────────────────┘
 *                                         ▼
 *                                    Category Probs
 *
 * Notes
 *  - TF-IDF vocabulary capped at 500 features.
 *  - Uses sigmoid activation for hidden layer.
 *  - Outputs class probabilities per category.
 */

import { ELM } from "../src/core/ELM";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { ELMConfig } from "../src/core/ELMConfig";

/**
 * Example corpus and labels.
 */
const trainingData = [
    { text: "Go is a statically typed compiled language.", label: "go" },
    { text: "Python is dynamically typed and interpreted.", label: "python" },
    { text: "TypeScript adds types to JavaScript.", label: "typescript" },
    { text: "Go has goroutines and channels.", label: "go" },
    { text: "Python has dynamic typing and simple syntax.", label: "python" },
    { text: "TypeScript is popular for web development.", label: "typescript" }
];

/**
 * Extract raw text documents for TFIDF.
 */
const docs = trainingData.map(d => d.text);

/**
 * Build the vectorizer on all training docs.
 */
const vectorizer = new TFIDFVectorizer(docs, 500);

/**
 * Convert each doc into a TFIDF vector.
 */
const X: number[][] = docs.map(doc => vectorizer.vectorize(doc));

/**
 * Define unique categories.
 */
const categories = Array.from(new Set(trainingData.map(d => d.label)));

/**
 * Create one-hot encoded labels.
 */
const Y: number[][] = trainingData.map(d =>
    categories.map(c => (c === d.label ? 1 : 0))
);

/**
 * Build ELM configuration.
 */
const config: ELMConfig = {
    categories,
    hiddenUnits: 50,
    maxLen: X[0].length, // matches TFIDF vector length
    activation: "sigmoid",
    log: {
        verbose: true,
        toFile: false,
        modelName: "TFIDF_ELM"
    },
    weightInit: "xavier"
};

/**
 * Instantiate ELM.
 */
const elm = new ELM(config);

/**
 * Train using precomputed numeric vectors.
 */
elm.trainFromData(X, Y);

/**
 * Predict on new text.
 */
const testText = "Go uses goroutines for concurrency.";
const testVec = vectorizer.vectorize(testText);

const predictions = elm.predictFromVector([testVec])[0];

console.log(`🔍 Predictions for: "${testText}"`);
predictions.forEach(p =>
    console.log(`Label: ${p.label} — Probability: ${(p.prob * 100).toFixed(2)}%`)
);
