/**
 * First Experiment: Supervised ELM Retrieval Demo
 *
 * This script demonstrates how to train and use an Extreme Learning Machine (ELM)
 * in a supervised setting for simple question–answer retrieval.
 *
 * Steps:
 *  1. Define a small set of Go programming Q/A pairs.
 *  2. Encode both queries and targets using the UniversalEncoder (character-level).
 *  3. Train an ELM on these supervised pairs to map input text → output text.
 *  4. Compute hidden layer embeddings for each target (answers).
 *  5. For a new query, compute its embedding and retrieve the most similar target
 *     using cosine similarity.
 *
 * Purpose:
 *  - Shows the end-to-end pipeline (encoding → training → retrieval).
 *  - Provides a minimal, interpretable example for testing ELM + encoder integration.
 *  - Serves as a baseline for future experiments with larger datasets or
 *    different tasks (classification, language modeling, etc.).
 */
/**
 * First Experiment: Supervised ELM Retrieval Demo
 *
 * ...
 *
 * Pipeline Overview:
 *
 *            ┌────────────┐
 *            │   Q/A Pairs│
 *            └──────┬─────┘
 *                   │
 *                   ▼
 *        ┌───────────────────────┐
 *        │ UniversalEncoder       │
 *        │ (char-level encoding) │
 *        └──────┬────────────────┘
 *                   │
 *                   ▼
 *            ┌────────────┐
 *            │    ELM     │  (trained on queries → targets)
 *            └──────┬─────┘
 *                   │
 *          Hidden Layer Embeddings
 *                   │
 *                   ▼
 *        ┌───────────────────────┐
 *        │ Cosine Similarity     │
 *        │ (query vs. targets)   │
 *        └───────────────────────┘
 *                   │
 *                   ▼
 *             Retrieved Answer(s)
 *
 */

import { ELM } from "../src/core/ELM";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

// Simple L2 normalization
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// Define example Q/A pairs
const supervisedPairs = [
    {
        query: "How do you declare a map in Go?",
        target: "You declare a map with the syntax: var m map[keyType]valueType"
    },
    {
        query: "How do you create a slice?",
        target: "Slices are created using []type{}, for example: s := []int{1,2,3}"
    },
    {
        query: "How do you write a for loop?",
        target: "The for loop in Go looks like: for i := 0; i < n; i++ { ... }"
    }
];

// Initialize encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// Encode queries and targets
const X = supervisedPairs.map(p => encoder.normalize(encoder.encode(p.query)));
const Y = supervisedPairs.map(p => encoder.normalize(encoder.encode(p.target)));

// Train supervised ELM
const elm = new ELM({
    activation: "relu",
    hiddenUnits: 64,
    maxLen: X[0].length,
    categories: [],
    log: { modelName: "SupervisedELM", verbose: true },
    dropout: 0.02
});

console.log(`⚙️ Training supervised ELM on ${X.length} examples...`);
elm.trainFromData(X, Y);
console.log(`✅ Supervised ELM trained.`);

// Precompute target embeddings
const targetEmbeddings = Y.map(yVec => l2normalize(elm.computeHiddenLayer([yVec])[0]));

// -----------------------------------------------------------------------------
// Retrieval function:
// Given a query string, this encodes it with the UniversalEncoder, passes it
// through the trained ELM to get a hidden-layer embedding, and then compares it
// against precomputed target embeddings (answers). Similarity is measured using
// cosine similarity (dot product of L2-normalized vectors). The top-K most
// similar answers are returned.
// -----------------------------------------------------------------------------
function retrieve(query: string, topK = 3) {
    const qVec = encoder.normalize(encoder.encode(query));
    const qEmbedding = l2normalize(elm.computeHiddenLayer([qVec])[0]);

    const scored = targetEmbeddings.map((e, i) => ({
        text: supervisedPairs[i].target,
        similarity: e.reduce((s, v, j) => s + v * qEmbedding[j], 0)
    }));

    return scored.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
}

// Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(`${i + 1}. (Cosine=${r.similarity.toFixed(4)}) ${r.text}`)
);

console.log(`✅ Done.`);
