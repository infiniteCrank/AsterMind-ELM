/**
 * Train an ELM using TFIDF vectors and predict on new text.
 * 
 * Make sure you have the following imports available in your project:
 * - ELM
 * - TFIDFVectorizer
 * 
 * Assumes you have at least two categories to classify.
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
