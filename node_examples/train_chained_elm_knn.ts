/**
 * Experiment: UniversalEncoder → ELM Autoencoder → ELMChain + KNN (Markdown Corpus)
 *
 * Goal
 *  - Explore stacked ELM chains on top of a text autoencoder and measure
 *    retrieval quality (Recall@1/5, MRR) over a markdown textbook split by headings.
 *
 * What it does
 *  1) Parse markdown into (heading, content) sections.
 *  2) Encode each section with UniversalEncoder (char-level) → baseVectors.
 *  3) Train an ELM autoencoder (X→X) to get compact paragraph embeddings.
 *  4) Pass embeddings through a configurable ELMChain (hybrid activations).
 *     (Here the chain is trained layer-by-layer against random targets to
 *      probe how depth + normalization affect geometry—see Notes.)
 *  5) Evaluate retrieval (cosine) with query/reference split; log Recall@1/5, MRR.
 *  6) Save per-config embeddings JSON and a CSV summary of metrics.
 *  7) Demonstrate KNN retrieval for a sample query.
 *
 * Why
 *  - Autoencoder compresses raw encoder features into a smoother space.
 *  - ELMChain tests whether additional nonlinearity/whitening helps retrieval.
 *  - Random-target training isolates geometric effects (distribution shaping)
 *    from supervised signals; useful for ablation and intuition.
 *
 * Pipeline Overview
 *
 *   Markdown ──► Section Split ──► UniversalEncoder ──► ELM (autoencoder) ──► ELMChain ──► Embeddings
 *                                                                                         │
 *                                                                                         ├─► Metrics (Recall@1/5, MRR)
 *                                                                                         └─► KNN demo (query → top-K)
 *
 * Notes
 *  - Chain layers train on random targets (dimension-capped) purely to reshape the
 *    embedding distribution (an ablation). Swap in supervised targets later if desired.
 *  - Tune `hiddenUnitSequences`, `dropouts`, and `repeats` to scan the space.
 *  - All cosine similarities assume L2-normalized vectors.
 */

import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { KNN, KNNDataPoint } from "../src/ml/KNN";

// L2 normalize utility
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// Cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
    return normA && normB ? dot / (normA * normB) : 0;
}

// Evaluate Recall@1, Recall@5, MRR
function evaluateRecallMRR(query: number[][], reference: number[][], k = 5) {
    let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
    for (const q of query) {
        const scores = reference.map(r => cosineSimilarity(q, r));
        const sorted = scores.slice().sort((a, b) => b - a);
        const rank = sorted.findIndex(s => s === scores[0]);
        if (rank === 0) hitsAt1++;
        if (rank < k) hitsAtK++;
        reciprocalRanks += 1 / (rank + 1);
    }
    return { recall1: hitsAt1 / query.length, recallK: hitsAtK / query.length, mrr: reciprocalRanks / query.length };
}

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const rawSections = rawText.split(/\n(?=#{1,6}\s)/);
const sections = rawSections
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const headingLine = lines.find(l => /^#{1,6}\s/.test(l)) || "";
        const contentLines = lines.filter(l => !/^#{1,6}\s/.test(l));
        return {
            heading: headingLine.replace(/^#{1,6}\s/, "").trim(),
            content: contentLines.join(" ").trim()
        };
    })
    .filter(s => s.content.length > 30);

console.log(`✅ Parsed ${sections.length} sections.`);

// 2️⃣ Prepare encoder
// -----------------------------------------------------------------------------
// UniversalEncoder (char-level):
// Produces fixed-length baseVectors for each section. Acts as the raw feature
// space prior to any learned compression or chaining.
// -----------------------------------------------------------------------------
const encoder = new UniversalEncoder({
    maxLen: 150,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// 3️⃣ Encode sections
const texts = sections.map(s => `${s.heading} ${s.content}`);
const baseVectors = texts.map(t => encoder.normalize(encoder.encode(t)));
console.log(`✅ Encoded sections.`);

// 4️⃣ Define hyperparameters
const hiddenUnitSequences = [
    [256, 128, 64],
    [128, 64, 32],
    [64, 32, 16]
];
const dropouts = [0.0, 0.02, 0.05];
const repeats = 2;

// 5️⃣ Prepare output CSV
const csvLines = ["config,run,recall_at_1,recall_at_5,mrr"];

// 6️⃣ Experiment loop
for (const seq of hiddenUnitSequences) {
    for (const dropout of dropouts) {
        for (let run = 1; run <= repeats; run++) {
            console.log(`\n🔹 Config: ${seq.join("_")} dropout ${dropout} (Run ${run})`);

            // -----------------------------------------------------------------------------
            // ELM Autoencoder (X → X):
            // Learns compact paragraph embeddings from baseVectors. These embeddings are
            // L2-normalized and used as inputs to the deeper ELMChain.
            // -----------------------------------------------------------------------------
            const autoencoder = new ELM({
                activation: "relu",
                hiddenUnits: 128,
                maxLen: baseVectors[0].length,
                categories: [],
                log: { modelName: "AutoencoderELM", verbose: true },
                dropout
            });
            autoencoder.trainFromData(baseVectors, baseVectors);
            let embeddings = autoencoder.computeHiddenLayer(baseVectors).map(l2normalize);

            // -----------------------------------------------------------------------------
            // ELMChain construction (hybrid activations):
            // Sequence of ELM layers (e.g., relu/tanh alternation). We’ll train each layer
            // sequentially to reshape the embedding distribution.
            // -----------------------------------------------------------------------------
            const elms = seq.map((h, i) =>
                new ELM({
                    activation: i % 2 === 0 ? "relu" : "tanh",
                    hiddenUnits: h,
                    maxLen: embeddings[0].length,
                    categories: [],
                    log: { modelName: `IndexerELM_${i + 1}`, verbose: true },
                    dropout
                })
            );

            const chain = new ELMChain(elms);

            // Sequential training
            for (const elm of elms) {
                const targetDim = Math.min(embeddings[0].length, elm.hiddenUnits);
                // Layer-wise distribution shaping (ablation):
                // Train the current ELM against random targets (dim-capped) to encourage
                // spread/whitening without supervised labels; then L2-normalize outputs.
                // Swap in supervised targets later to test task-driven shaping.
                // -----------------------------------------------------------------------------
                const randomTargets = Array.from({ length: embeddings.length }, () =>
                    Array.from({ length: targetDim }, () => Math.random())
                );
                elm.trainFromData(embeddings, randomTargets);
                embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
            }

            // Evaluation
            // -----------------------------------------------------------------------------
            // Retrieval evaluation split + metrics:
            // 20% queries vs 80% references; cosine ranking; compute Recall@1/5 and MRR.
            // -----------------------------------------------------------------------------
            const splitIdx = Math.floor(embeddings.length * 0.2);
            const queryVecs = embeddings.slice(0, splitIdx);
            const refVecs = embeddings.slice(splitIdx);

            const { recall1, recallK, mrr } = evaluateRecallMRR(queryVecs, refVecs, 5);

            // -----------------------------------------------------------------------------
            // Persist embeddings for inspection/analysis:
            // Saves (embedding, heading, content) for downstream tooling or visualization.
            // -----------------------------------------------------------------------------
            const configName = `Chain_${seq.join("_")}_drop${dropout}`;
            csvLines.push(`${configName},${run},${recall1.toFixed(4)},${recallK.toFixed(4)},${mrr.toFixed(4)}`);
            console.log(`✅ Recall@1=${recall1.toFixed(4)}, Recall@5=${recallK.toFixed(4)}, MRR=${mrr.toFixed(4)}`);

            // Save embeddings for retrieval
            const embeddingRecords = sections.map((s, i) => ({
                embedding: embeddings[i],
                metadata: { heading: s.heading, content: s.content }
            }));
            const outFile = `./embeddings_${configName}_run${run}.json`;
            fs.writeFileSync(outFile, JSON.stringify(embeddingRecords, null, 2));
            console.log(`💾 Saved embeddings to ${outFile}.`);

            // Retrieval example
            const query = "How do you declare a map in Go?";
            const queryVec = encoder.normalize(encoder.encode(query));
            const autoVec = autoencoder.computeHiddenLayer([queryVec])[0];
            const chainVec = l2normalize(chain.getEmbedding([l2normalize(autoVec)])[0]);

            const knnDataset: KNNDataPoint[] = embeddingRecords.map(r => ({
                vector: r.embedding,
                label: r.metadata.heading
            }));
            // -----------------------------------------------------------------------------
            // KNN demo retrieval:
            // Encode the query via Autoencoder→ELMChain, then cosine KNN over final embeddings.
            // -----------------------------------------------------------------------------
            const knnResults = KNN.find(chainVec, knnDataset, 5, 5, "cosine");
            console.log(`\n🔍 Top 5 retrieval for query: "${query}"`);
            knnResults.forEach((r, i) => {
                console.log(`${i + 1}. [Score=${r.weight.toFixed(4)}] ${r.label}`);
            });
        }
    }
}

// 7️⃣ Save CSV summary
// -----------------------------------------------------------------------------
// CSV summary export (timestamped):
// Aggregates per-config metrics for quick comparison across runs.
// -----------------------------------------------------------------------------
const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const filename = `elm_chain_knn_experiment_${timestamp}.csv`;
fs.writeFileSync(filename, csvLines.join("\n"));
console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
