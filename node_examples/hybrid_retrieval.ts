/**
 * Experiment: Chapter/Section Indexing with Paragraph ELM + Indexer ELMChain + TFIDF
 *
 * Goal
 *  - Build a dense retriever over a long markdown corpus (go_textbook.md) by:
 *    1) Splitting the book into sections at "### Day ..." headings,
 *    2) Learning paragraph-level embeddings with a Paragraph ELM,
 *    3) Refining those embeddings with a stacked Indexer ELMChain,
 *    4) Blending dense similarity with a TFIDF score for robust retrieval.
 *
 * What it does
 *  - Parses sections → encodes them (UniversalEncoder + TFIDF).
 *  - Trains or loads a Paragraph ELM (self-supervised: X→X) for paragraph embeddings.
 *  - Trains or loads an Indexer ELMChain to further densify/whiten embeddings.
 *  - Saves final embeddings + metadata to JSON.
 *  - At query time, computes a dense embedding (Paragraph ELM → Indexer ELMChain)
 *    and a TFIDF vector, then returns a weighted score (0.7*dense + 0.3*TFIDF).
 *
 * Why
 *  - Dense vectors capture semantic similarity across paraphrases,
 *    while TFIDF keeps strong lexical grounding. The blend improves stability
 *    on technical text where keywords matter but paraphrase should still match.
 *
 * Pipeline Overview
 *
 *   Markdown (### Day …) ──► Section Split ──► [UniversalEncoder] ──► Paragraph ELM ──► Indexer ELMChain ──► Dense Embeds
 *              │                                                                                                       │
 *              └──────────────────────────────────────────────► TFIDF Vectorizer ──────────────────────────────────────┘
 *                                                                                       │
 *                                                                                       ▼
 *                                                                         Weighted Retrieval (0.7*dense + 0.3*TFIDF)
 *
 * Notes
 *  - We cache weights in ./elm_weights to keep runs fast and reproducible.
 *  - Adjust `hiddenUnits`, chain depth, and TFIDF vocab size to tune quality/latency.
 *  - Section splitting assumes "### Day ..." headings; change the regex to match your TOC style.
 */

import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { TFIDFVectorizer } from "../src/ml/TFIDF";

// Helpers
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// ✅ New Section Splitting: each "### Day ..." heading defines a section
const rawSections = rawText.split(/\n(?=### Day )/);

const sections = rawSections
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const headingLine = lines.find(l => /^### Day /.test(l)) || "";
        const contentLines = lines.filter(l => !/^### Day /.test(l));
        return {
            heading: headingLine.replace(/^### /, "").trim(),
            content: contentLines.join(" ").trim()
        };
    })
    .filter(s => s.content.length > 30);

console.log(`✅ Parsed ${sections.length} sections.`);

// Prepare encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// -----------------------------------------------------------------------------
// TFIDF baseline features for lexical grounding (normalized):
// Used at retrieval time as a 30% weight to stabilize dense similarity.
// -----------------------------------------------------------------------------
const tfidfTexts = sections.map(s => `${s.heading}. ${s.content}`);
console.log(`⏳ Computing TFIDF vectors...`);
const vectorizer = new TFIDFVectorizer(tfidfTexts, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// Compute encoder embeddings
const sectionVectors = tfidfTexts.map(t =>
    l2normalize(encoder.normalize(encoder.encode(t)))
);

// -----------------------------------------------------------------------------
// Paragraph ELM (self-supervised X→X):
// Learns a compact paragraph representation from encoder outputs. We cache/load
// weights for reproducibility and faster iteration.
// -----------------------------------------------------------------------------
const paraELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: sectionVectors[0].length,
    categories: [],
    log: { modelName: "ParagraphELM", verbose: true },
    dropout: 0.02
});
const weightsPath = "./elm_weights/paragraphELM.json";
if (fs.existsSync(weightsPath)) {
    paraELM.loadModelFromJSON(fs.readFileSync(weightsPath, "utf-8"));
    console.log(`✅ Loaded ParagraphELM weights.`);
} else {
    console.log(`⚙️ Training ParagraphELM...`);
    paraELM.trainFromData(sectionVectors, sectionVectors);
    fs.mkdirSync("./elm_weights", { recursive: true });
    fs.writeFileSync(weightsPath, JSON.stringify(paraELM.model));
    console.log(`💾 Saved ParagraphELM weights.`);
}
let embeddings = paraELM.computeHiddenLayer(sectionVectors).map(l2normalize);

// -----------------------------------------------------------------------------
// Indexer ELMChain (stacked refinement):
// Further densifies/whitens paragraph embeddings layer-by-layer, normalizing
// after each layer. We persist each layer's weights under ./elm_weights.
// -----------------------------------------------------------------------------
const hiddenUnits = [256, 128];
const indexerELMs = hiddenUnits.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.02
    })
);

indexerELMs.forEach((elm, i) => {
    const path = `./elm_weights/indexerELM_${i}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded IndexerELM_${i}.`);
    } else {
        console.log(`⚙️ Training IndexerELM_${i}...`);
        elm.trainFromData(embeddings, embeddings);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved IndexerELM_${i} weights.`);
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(indexerELMs);
console.log(`✅ Indexer chain ready.`);

// Save embeddings
const embeddingRecords: EmbeddingRecord[] = sections.map((s, i) => ({
    embedding: embeddings[i],
    metadata: { heading: s.heading, text: s.content }
}));
fs.writeFileSync("./embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// -----------------------------------------------------------------------------
// Retrieval:
// 1) Encode query → Paragraph ELM → Indexer ELMChain for dense embedding.
// 2) Also compute TFIDF for the query.
// 3) Score each section with cosine(dense) and cosine(TFIDF), then combine
//    as totalScore = 0.7*dense + 0.3*TFIDF.
// -----------------------------------------------------------------------------
function retrieve(query: string, topK = 5) {
    const qVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const paraE = paraELM.computeHiddenLayer([qVec])[0];
    const denseVec = indexerChain.getEmbedding([l2normalize(paraE)])[0];
    const tfidfVec = l2normalize(vectorizer.vectorize(query));

    const scored = embeddingRecords.map((r, i) => {
        const denseScore = r.embedding.reduce((s, v, j) => s + v * denseVec[j], 0);
        const tfidfScore = tfidfVectors[i].reduce((s, v, j) => s + v * tfidfVec[j], 0);
        return {
            heading: r.metadata.heading,
            snippet: r.metadata.text.slice(0, 100),
            denseScore,
            tfidfScore,
            totalScore: 0.7 * denseScore + 0.3 * tfidfScore
        };
    });

    return scored.sort((a, b) => b.totalScore - a.totalScore).slice(0, topK);
}

// Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(
        `${i + 1}. [Dense=${r.denseScore.toFixed(4)} | TFIDF=${r.tfidfScore.toFixed(4)}] ${r.heading} – ${r.snippet}...`
    )
);
console.log(`✅ Done.`);
