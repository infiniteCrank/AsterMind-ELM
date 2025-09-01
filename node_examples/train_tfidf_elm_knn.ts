/**
 * Experiment: Hybrid Classifier with TF-IDF + ELM + KNN (Small Code Corpus)
 *
 * Goal
 *  - Train a simple hybrid classifier over a toy dataset of Go, Python, and
 *    TypeScript code snippets using:
 *    1) TF-IDF vectorization for lexical features,
 *    2) Extreme Learning Machine (ELM) for fast supervised classification,
 *    3) KNN for instance-based similarity,
 *    4) Averaging their scores for final predictions.
 *
 * What it does
 *  - Defines a tiny training corpus (6 sentences across 3 languages).
 *  - Builds a TF-IDF vocabulary and vectorizes samples.
 *  - Trains an ELM (X→one-hot labels) with 50 hidden units, ReLU activation.
 *  - Builds a KNN dataset over the same TF-IDF embeddings.
 *  - Runs inference on a test sentence:
 *    - ELM predicts softmax-like label probabilities,
 *    - KNN finds nearest neighbors and assigns cosine weights,
 *    - Results are averaged for a combined score.
 *  - Outputs ranked label predictions with scores.
 *
 * Why
 *  - ELM provides a fast, generalizable classifier,
 *  - KNN ensures similarity to exact known instances,
 *  - Combining both balances generalization and memorization on small datasets.
 *
 * Pipeline Overview
 *
 *   Train Samples ──► TF-IDF ──► L2 normalize ──► [ELM classifier] ──► Probabilities
 *                                   │
 *                                   └─► [KNN (cosine)] ──► Neighbor weights
 *
 *                                    ▼
 *                            Combine (avg) ──► Final Prediction
 *
 * Notes
 *  - With such a small dataset, performance is illustrative only.
 *  - Categories are auto-detected from unique labels.
 *  - Combine logic averages ELM probability and KNN cosine weight per label.
 */

import { KNN, KNNDataPoint } from "../src/ml/KNN";

import { ELM } from "../src/core/ELM";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { ELMConfig } from "../src/core/ELMConfig";

// Define training data
const trainSamples = [
    { text: "Go has goroutines and channels for concurrency.", label: "go" },
    { text: "The defer keyword is used in Go.", label: "go" },
    { text: "Python has list comprehensions.", label: "python" },
    { text: "Python supports generators and decorators.", label: "python" },
    { text: "TypeScript adds static typing to JavaScript.", label: "typescript" },
    { text: "TypeScript interfaces help define contracts.", label: "typescript" },
];

// Categories (labels)
const categories = Array.from(new Set(trainSamples.map(s => s.label)));

// Create TF-IDF vectorizer
const vectorizer = new TFIDFVectorizer(trainSamples.map(s => s.text));
const X_raw = trainSamples.map(s => vectorizer.vectorize(s.text));

// L2-normalize vectors
const X = X_raw.map(vec => TFIDFVectorizer.l2normalize(vec));

// Prepare one-hot labels
const Y = trainSamples.map(s => {
    const y = Array(categories.length).fill(0);
    y[categories.indexOf(s.label)] = 1;
    return y;
});

// Initialize ELM
const elm = new ELM({
    categories,
    hiddenUnits: 50,
    maxLen: X[0].length,
    activation: "relu",
    log: {
        modelName: "TFIDF-ELM-Ensemble",
        verbose: true,
        toFile: false,
    },
    weightInit: "xavier",
});

// Train ELM
elm.trainFromData(X, Y);

// Build KNN dataset
const knnData: KNNDataPoint[] = trainSamples.map((s, idx) => ({
    vector: X[idx],
    label: s.label,
}));

// Test input
const testText = "Go uses goroutines for concurrency.";
const testVec = TFIDFVectorizer.l2normalize(vectorizer.vectorize(testText));

// ELM prediction
const elmPreds = elm.predictFromVector([testVec], 3)[0];

// KNN prediction
const knnResults = KNN.find(testVec, knnData, 3, 3, "cosine");

// Combine predictions by averaging probabilities
const combined: { [label: string]: number } = {};
for (const cat of categories) {
    const elmProb = elmPreds.find(p => p.label === cat)?.prob ?? 0;
    const knnWeight = knnResults.find(k => k.label === cat)?.weight ?? 0;
    combined[cat] = (elmProb + knnWeight) / 2;
}

// Sort combined predictions
const finalPreds = Object.entries(combined)
    .map(([label, prob]) => ({ label, prob }))
    .sort((a, b) => b.prob - a.prob);

// Output
console.log(`\n🔍 Predictions for: "${testText}"\n`);
finalPreds.forEach(p => {
    console.log(`Label: ${p.label} — Score: ${p.prob.toFixed(4)}`);
});
