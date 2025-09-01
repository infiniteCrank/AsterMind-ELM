/**
 * Experiment: TFIDF + ELM Ensemble Retrieval on AG News
 *
 * This script demonstrates how to combine classic TFIDF embeddings with
 * stacked Extreme Learning Machines (ELMs) to improve text retrieval quality.
 *
 * Workflow:
 *  1. Load a subset of the AG News classification dataset from CSV.
 *  2. Compute TFIDF vectors as the baseline representation.
 *  3. Evaluate baseline retrieval performance using cosine similarity on raw TFIDF.
 *  4. Build an ensemble of ELMChains:
 *     - Each chain consists of multiple ELM layers with residual connections,
 *       normalization, and per-dimension scaling.
 *     - ELM weights are cached to JSON files for reuse between runs.
 *  5. Train or load the ensemble, distilling knowledge from TFIDF vectors
 *     into progressively deeper ELM embeddings.
 *  6. Compare retrieval performance (Recall@1, Recall@5, MRR) of the baseline
 *     vs. the ELM ensemble.
 *  7. Save all results as a timestamped CSV for later analysis.
 *
 * Purpose:
 *  - Establishes TFIDF as a baseline for text retrieval.
 *  - Tests whether layered ELM ensembles can refine or distill TFIDF embeddings
 *    into higher-quality semantic vectors.
 *  - Provides reusable weight persistence for faster experimentation.
 *
 * Notes:
 *  - Sample size is adjustable (default: 5000 records).
 *  - The ensemble size and hidden unit sequence can be tuned for performance.
 *  - Results are logged and written to disk in CSV format.
 */
/**
 * Experiment: TFIDF + ELM Ensemble Retrieval on AG News
 *
 * ...
 *
 * Pipeline Overview:
 *
 *            ┌────────────┐
 *            │   Raw Text │
 *            └──────┬─────┘
 *                   │
 *                   ▼
 *            ┌────────────┐
 *            │   TFIDF    │  (baseline embedding)
 *            └──────┬─────┘
 *                   │
 *          Baseline Retrieval
 *                   │
 *                   ▼
 *        ┌───────────────────────┐
 *        │  ELM Ensemble (stack) │
 *        │                       │
 *        │  ┌────┐   ┌────┐      │
 *        │  │ELM1│→→→│ELM2│→ ... │
 *        │  └────┘   └────┘      │
 *        │   │residual+normalize │
 *        └───────────────────────┘
 *                   │
 *                   ▼
 *          Refined Embeddings
 *                   │
 *                   ▼
 *          Retrieval + Metrics
 *     (Recall@1, Recall@5, MRR)
 *
 */

import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { evaluateEnsembleRetrieval } from "../src/core/evaluateEnsembleRetrieval";
import { TFIDFVectorizer } from "../src/ml/TFIDF";

(async () => {

    function l2normalize(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        return norm === 0 ? v : v.map(x => x / norm);
    }

    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));

    const sampleSize = 5000; // Feel free to scale
    const texts = records.slice(0, sampleSize).map(r => r.text);
    const labels = records.slice(0, sampleSize).map(r => r.label);

    console.log(`⏳ Computing TFIDF vectors...`);
    const vectorizer = new TFIDFVectorizer(texts, 2000);
    const tfidfVectors = vectorizer.vectorizeAll().map(v => TFIDFVectorizer.l2normalize(v));
    console.log(`✅ TFIDF vectors ready.`);

    // Baseline retrieval with raw TFIDF embeddings
    console.log(`\n🔍 Evaluating baseline retrieval using raw TFIDF cosine similarity...`);

    const rawTFIDFEmbeddings: EmbeddingRecord[] = records.slice(0, sampleSize).map((rec, i) => ({
        embedding: l2normalize(tfidfVectors[i]),
        metadata: { text: rec.text, label: rec.label }
    }));

    const splitIdx = Math.floor(texts.length * 0.2);
    const queryTFIDF = rawTFIDFEmbeddings.slice(0, splitIdx);
    const referenceTFIDF = rawTFIDFEmbeddings.slice(splitIdx);

    // -----------------------------------------------------------------------------
    // Baseline retrieval evaluation with TFIDF:
    // Here we test raw TFIDF embeddings by splitting the dataset into queries
    // (first 20%) and references (remaining 80%). Retrieval is performed via
    // cosine similarity, and metrics (Recall@1, Recall@5, MRR) are computed.
    // This establishes a baseline to compare against the ELM ensemble.
    // -----------------------------------------------------------------------------
    const tfidfResults = evaluateEnsembleRetrieval(queryTFIDF, referenceTFIDF, [], 5);

    console.log(
        `✅ TFIDF Baseline Results: Recall@1=${tfidfResults.recallAt1.toFixed(4)} ` +
        `Recall@5=${tfidfResults.recallAtK.toFixed(4)} MRR=${tfidfResults.mrr.toFixed(4)}`
    );

    const hiddenUnitSequence = [512, 256, 256, 128, 128, 64, 64, 32, 16, 8];
    const dropout = 0.02;
    const ensembleSize = 3;

    const csvLines = ["config,run,recall_at_1,recall_at_5,mrr"];

    csvLines.push(
        `TFIDF_Baseline,NA,${tfidfResults.recallAt1.toFixed(4)},` +
        `${tfidfResults.recallAtK.toFixed(4)},${tfidfResults.mrr.toFixed(4)}`
    );

    // Make sure the directory for weights exists
    const weightsDir = "./models";
    if (!fs.existsSync(weightsDir)) {
        fs.mkdirSync(weightsDir);
    }

    // -----------------------------------------------------------------------------
    // ELM Ensemble Training and Distillation:
    // For each run, we build an ensemble of ELMChains. Each chain is a sequence
    // of ELM layers with residual connections, normalization, and scaling.
    // - If saved weights exist, they are loaded from disk.
    // - Otherwise, new ELMs are trained and weights are persisted.
    // After training, embeddings from the ensemble are evaluated against
    // the TFIDF baseline using the same retrieval setup.
    // -----------------------------------------------------------------------------

    for (let run = 1; run <= 1; run++) {
        console.log(`\n🔹 Run ${run}`);

        const ensembleChains: ELMChain[] = [];

        for (let e = 0; e < ensembleSize; e++) {
            console.log(`  🎯 Preparing ELMChain #${e + 1}`);

            const elms = hiddenUnitSequence.map((h, i) =>
                new ELM({
                    activation: "relu",
                    hiddenUnits: h,
                    maxLen: 50,
                    categories: [],
                    log: { modelName: `ELM layer ${i + 1}`, verbose: false },
                    metrics: { accuracy: 0.01 },
                    dropout
                })
            );

            let inputs = tfidfVectors;

            for (const [layerIdx, elm] of elms.entries()) {
                const weightsPath = `${weightsDir}/elm_run${run}_chain${e}_layer${layerIdx}.json`;

                if (fs.existsSync(weightsPath)) {
                    const saved = fs.readFileSync(weightsPath, "utf-8");
                    elm.loadModelFromJSON(saved);
                    console.log(`✅ Loaded saved weights for ELM #${layerIdx + 1} in chain #${e + 1}`);
                } else {
                    // Train current ELM layer:
                    // Each layer learns to transform TFIDF vectors into refined embeddings.
                    // After training, outputs are combined with residual connections,
                    // normalized, and scaled to balance dimensions before feeding into
                    // the next layer. This stacking helps distill TFIDF information into
                    // progressively richer embeddings.
                    console.log(`⚙️ Training ELM #${layerIdx + 1} in chain #${e + 1}...`);
                    elm.trainFromData(inputs, tfidfVectors, { reuseWeights: false });

                    if (elm.model) {
                        const json = JSON.stringify(elm.model);
                        fs.writeFileSync(weightsPath, json);
                        console.log(`💾 Saved weights for ELM #${layerIdx + 1} in chain #${e + 1}`);
                    }
                }

                let outputs = inputs.map(vec => elm.getEmbedding([vec])[0]);

                outputs = outputs.map((o, i) => o.map((v, j) => v + inputs[i][j]));

                outputs = outputs.map(l2normalize);

                const dimMeans = Array(outputs[0].length).fill(0);
                for (const v of outputs)
                    for (let j = 0; j < dimMeans.length; j++)
                        dimMeans[j] += v[j];
                for (let j = 0; j < dimMeans.length; j++)
                    dimMeans[j] /= outputs.length;

                const scalingFactors = dimMeans.map(m => 1 / (Math.abs(m) + 1e-8));
                outputs = outputs.map(vec => vec.map((x, j) => x * scalingFactors[j]));

                outputs = outputs.map(l2normalize);

                inputs = outputs;

                const flat = outputs.flat();
                const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
                const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
                console.log(`    Layer ${layerIdx + 1}: mean=${mean.toFixed(4)} var=${variance.toFixed(4)}`);
            }

            ensembleChains.push(new ELMChain(elms));
        }

        const embeddingStore: EmbeddingRecord[] = records.slice(0, sampleSize).map((rec, i) => ({
            embedding: l2normalize(tfidfVectors[i]),
            metadata: { text: rec.text, label: rec.label }
        }));

        const queries = embeddingStore.slice(0, splitIdx);
        const reference = embeddingStore.slice(splitIdx);

        const results = evaluateEnsembleRetrieval(queries, reference, ensembleChains, 5);

        csvLines.push(
            `ELMEnsemble_TFIDFDistill_dropout${dropout}_run${run},` +
            `${results.recallAt1.toFixed(4)},${results.recallAtK.toFixed(4)},${results.mrr.toFixed(4)}`
        );
    }
    // -----------------------------------------------------------------------------
    // Results export:
    // After evaluation, metrics for both TFIDF baseline and ELM ensemble runs
    // are appended to a CSV file with a timestamped name. This allows easy
    // comparison of experiments over time.
    // -----------------------------------------------------------------------------
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `elm_tfidf_experiment_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
