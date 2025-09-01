/**
 * Experiment: Token-Level UniversalEncoder → Unsupervised ELM → Indexer ELMChain (+ TF-IDF blend)
 *
 * Goal
 *  - Build a fast, lightweight retriever by:
 *    1) Encoding sections at the token level,
 *    2) Learning unsupervised dense embeddings with an ELM autoencoder,
 *    3) Refining them via a small Indexer ELMChain,
 *    4) Blending dense similarity with TF-IDF (50/50) for robustness.
 *
 * What it does
 *  - Parses markdown into (heading, content) sections.
 *  - Uses UniversalEncoder in token mode for richer lexical signals.
 *  - Trains or loads an unsupervised ELM (X→X) and a 2-layer Indexer ELMChain.
 *  - Computes TF-IDF vectors for lexical grounding + a sanity-check ranking.
 *  - Retrieval combines dense cosine and TF-IDF cosine equally.
 *
 * Why
 *  - Token-mode encoding captures word boundaries and punctuation patterns.
 *  - Unsupervised ELM + Indexer shapes a compact space without labels.
 *  - 50/50 blend balances semantic match with exact-term relevance.
 *
 * Pipeline Overview
 *
 *   Markdown ──► Sections ──► UniversalEncoder (token) ──► ELM (autoencoder) ──► Indexer ELMChain ──► Dense Embeds
 *                                                                                                  │
 *                                                                                                  ├─► TF-IDF (lexical)
 *                                                                                                  │
 *                                                                                                  └─► Blended Retrieval (0.5 dense + 0.5 tfidf)
 *
 * Notes
 *  - Weights are cached under ./elm_weights for reproducibility and speed.
 *  - All vectors are L2-normalized before cosine similarity.
 *  - Adjust Indexer depth/width and TF-IDF vocab (3000) to tune quality/latency.
 */

import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";

// ✅ Helpers
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 || !isFinite(norm) ? v.map(() => 0) : v.map(x => x / norm);
}

// ✅ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// ✅ Split sections (improved parsing)
// -----------------------------------------------------------------------------
// Section parsing (markdown headings #..###### → (heading, content) pairs):
// Filters out tiny sections to reduce retrieval noise.
// -----------------------------------------------------------------------------

const sectionRegex = /(^#{1,6}\s+.+$)/m;
const rawSections = rawText.split(/\n(?=#+ )/);
const sections = rawSections
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const headingLine = lines.find(l => /^#+ /.test(l)) || "";
        const contentLines = lines.filter(l => !/^#+ /.test(l));
        return {
            heading: headingLine.replace(/^#+ /, "").trim(),
            content: contentLines.join(" ").trim()
        };
    })
    .filter(s => s.content.length > 30);

console.log(`✅ Parsed ${sections.length} sections.`);

// ✅ Encoder with token-level settings
// -----------------------------------------------------------------------------
// UniversalEncoder (token mode):
// Token-level encoding with tokenizer enabled; richer than pure char mode.
// -----------------------------------------------------------------------------

const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "token",
    useTokenizer: true
});

// ✅ Compute embeddings
const paraVectors = sections.map(s =>
    l2normalize(encoder.normalize(encoder.encode(`${s.heading} ${s.content}`)))
);

// ✅ Load supervised pairs
const supervisedPaths = [
    "../public/supervised_pairs.csv",
    "../public/supervised_pairs_2.csv",
    "../public/supervised_pairs_3.csv",
    "../public/supervised_pairs_4.csv"
];
let supervisedPairs: { query: string, target: string }[] = [];
for (const path of supervisedPaths) {
    if (fs.existsSync(path)) {
        const csv = fs.readFileSync(path, "utf8");
        const rows = parse(csv, { skip_empty_lines: true });
        supervisedPairs.push(
            ...rows.map((r: [string, string]) => ({ query: r[0].trim(), target: r[1].trim() }))
        );
    }
}
console.log(`✅ Loaded ${supervisedPairs.length} supervised pairs.`);

// ✅ Encode supervised
const supQueryVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const supTargetVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// ✅ Unsupervised ELM
// -----------------------------------------------------------------------------
// Unsupervised ELM (autoencoder, X→X):
// Learns compact paragraph embeddings; weights cached to ./elm_weights.
// -----------------------------------------------------------------------------

const unsupELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: paraVectors[0].length,
    categories: [],
    log: { modelName: "UnsupervisedELM", verbose: true },
    dropout: 0.02
});
const unsupPath = "./elm_weights/unsupervised.json";
if (fs.existsSync(unsupPath)) {
    unsupELM.loadModelFromJSON(fs.readFileSync(unsupPath, "utf-8"));
    console.log(`✅ Loaded Unsupervised ELM weights.`);
} else {
    console.log(`⚙️ Training Unsupervised ELM...`);
    unsupELM.trainFromData(paraVectors, paraVectors);
    fs.writeFileSync(unsupPath, JSON.stringify(unsupELM.model));
    console.log(`💾 Saved Unsupervised ELM weights.`);
}

// ✅ Compute unsupervised embeddings
let embeddings = unsupELM.computeHiddenLayer(paraVectors).map(l2normalize);

// ✅ TFIDF
// -----------------------------------------------------------------------------
// TF-IDF baseline vectors (vocab=3000):
// Used for lexical grounding and a quick top-TFIDF sanity check.
// -----------------------------------------------------------------------------

console.log(`⏳ Computing TFIDF vectors...`);
const texts = sections.map(s => `${s.heading} ${s.content}`);
const vectorizer = new TFIDFVectorizer(texts, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// ✅ Indexer ELM chain
// -----------------------------------------------------------------------------
// Indexer ELMChain (2 layers):
// Further refines/whitens the unsupervised embeddings; each layer cached.
// -----------------------------------------------------------------------------

const indexerDims = [256, 128];
const chainELMs = indexerDims.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.02
    })
);
chainELMs.forEach((elm, i) => {
    const path = `./elm_weights/indexer_layer${i}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded Indexer layer ${i}.`);
    } else {
        console.log(`⚙️ Training Indexer layer ${i}...`);
        elm.trainFromData(embeddings, embeddings);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved Indexer layer ${i}.`);
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});
const indexerChain = new ELMChain(chainELMs);

// ✅ Save embeddings
// -----------------------------------------------------------------------------
// Persist final embeddings + metadata for downstream inspection.
// -----------------------------------------------------------------------------

const embeddingRecords: EmbeddingRecord[] = sections.map((s, i) => ({
    embedding: embeddings[i],
    metadata: { heading: s.heading, text: s.content }
}));
fs.writeFileSync("./embeddings/embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// ✅ Retrieval with blended scoring
// -----------------------------------------------------------------------------
// Retrieval:
// 1) Encode query → Unsupervised ELM → Indexer chain to get dense vector.
// 2) Compute TF-IDF vector for the query.
// 3) Rank sections by blended score = 0.5*dense_cosine + 0.5*tfidf_cosine.
// Also logs a pure-TFIDF top list for sanity checking.
// -----------------------------------------------------------------------------

function retrieve(query: string, topK = 5) {
    const qVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const unsupVec = unsupELM.computeHiddenLayer([qVec])[0];
    const finalVec = indexerChain.getEmbedding([l2normalize(unsupVec)])[0];
    const tfidfQ = l2normalize(vectorizer.vectorize(query));

    const scored = embeddingRecords.map((r, i) => ({
        heading: r.metadata.heading,
        snippet: r.metadata.text.slice(0, 100),
        dense: r.embedding.reduce((s, v, j) => s + v * finalVec[j], 0),
        tfidf: tfidfVectors[i].reduce((s, v, j) => s + v * tfidfQ[j], 0)
    }));

    // 🟢 Top TFIDF sanity check
    const tfidfRanked = [...scored]
        .sort((a, b) => b.tfidf - a.tfidf)
        .slice(0, 5);
    console.log(`\n🔍 Top TFIDF results:`);
    tfidfRanked.forEach((r, i) =>
        console.log(`${i + 1}. (TFIDF=${r.tfidf.toFixed(4)}) ${r.heading} – ${r.snippet}...`)
    );

    // 🟢 Blended score: 50% dense + 50% TFIDF
    return scored
        .sort((a, b) => (0.5 * b.dense + 0.5 * b.tfidf) - (0.5 * a.dense + 0.5 * a.tfidf))
        .slice(0, topK);
}

// ✅ Example retrieval
// -----------------------------------------------------------------------------
// Demo query: quick end-to-end smoke test of the blended retriever.
// -----------------------------------------------------------------------------

const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(`${i + 1}. [Dense+TFIDF] ${r.heading} – ${r.snippet}...`)
);
console.log(`✅ Done.`);
