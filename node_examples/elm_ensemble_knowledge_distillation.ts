import fs from "fs";
import { parse } from "csv-parse/sync";
import { pipeline } from "@xenova/transformers";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { evaluateRetrieval } from "../src/core/Evaluation";
import { evaluateEnsembleRetrieval } from "../src/core/evaluateEnsembleRetrieval";

(async () => {
    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));

    const sampleSize = 5000; // Scale up your data
    const texts = records.slice(0, sampleSize).map(r => r.text);
    const labels = records.slice(0, sampleSize).map(r => r.label);

    console.log(`⏳ Loading Sentence-BERT...`);
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    const bertTensor = await embedder(texts, { pooling: "mean" });
    const bertEmbeddings = bertTensor.tolist() as number[][];

    const splitIdx = Math.floor(texts.length * 0.2);
    const queryLabels = labels.slice(0, splitIdx);
    const refLabels = labels.slice(splitIdx);
    const queryBERT = bertEmbeddings.slice(0, splitIdx);
    const refBERT = bertEmbeddings.slice(splitIdx);

    const hiddenUnitSequence = [
        1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 16, 8
    ];

    const dropouts = [0.02];
    const ensembleSize = 3;
    const repeats = 1; // You can increase to average over repeats

    const csvLines = ["config,run,recall_at_1,recall_at_5,mrr"];

    function cosineSimilarity(a: number[], b: number[]): number {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return dot / (normA * normB);
    }

    function l2normalize(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        return norm === 0 ? v : v.map(x => x / norm);
    }

    for (const dropout of dropouts) {
        for (let run = 1; run <= repeats; run++) {
            console.log(`\n🔹 Run ${run} with dropout ${dropout}`);

            const ensembleChains: ELMChain[] = [];

            for (let e = 0; e < ensembleSize; e++) {
                const hybridActivations = hiddenUnitSequence.map((_, i) =>
                    i % 3 === 0 ? "relu" : i % 3 === 1 ? "tanh" : "leakyRelu"
                );

                console.log(`  🎯 Training ELMChain #${e + 1}`);

                const elms = hiddenUnitSequence.map((h, i) =>
                    new ELM({
                        activation: hybridActivations[i],
                        hiddenUnits: h,
                        maxLen: 50,
                        categories: [],
                        log: { modelName: `ELM layer ${i + 1}`, verbose: false },
                        metrics: { accuracy: 0.01 },
                        dropout
                    })
                );

                const encoder = elms[0].encoder;
                const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));
                let inputs = encodedVectors;

                // Learnable scaling factors initialized to 1
                let scalingFactors = inputs[0].map(() => 1);

                for (const [layerIdx, elm] of elms.entries()) {
                    // Train to regress BERT embeddings
                    elm.trainFromData(inputs, bertEmbeddings, { reuseWeights: false });

                    inputs = inputs.map(vec => elm.getEmbedding([vec])[0]);

                    // Learn scaling factor per dimension
                    const dimMeans = Array(inputs[0].length).fill(0);
                    for (const v of inputs) for (let j = 0; j < dimMeans.length; j++) dimMeans[j] += v[j];
                    for (let j = 0; j < dimMeans.length; j++) dimMeans[j] /= inputs.length;

                    const targetNorm = 1;
                    scalingFactors = dimMeans.map(m => targetNorm / (Math.abs(m) + 1e-8));
                    inputs = inputs.map(vec => vec.map((x, j) => x * scalingFactors[j]));

                    // L2 normalize
                    inputs = inputs.map(l2normalize);

                    // Diagnostic
                    const flat = inputs.flat();
                    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
                    const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
                    console.log(`    Layer ${layerIdx + 1} mean=${mean.toFixed(4)} var=${variance.toFixed(4)}`);
                }

                ensembleChains.push(new ELMChain(elms));
            }

            const embeddingStore: EmbeddingRecord[] = records.slice(0, sampleSize).map((rec, i) => ({
                embedding: l2normalize(bertEmbeddings[i]),
                metadata: { text: rec.text, label: rec.label }
            }));

            const queries = embeddingStore.slice(0, splitIdx);
            const reference = embeddingStore.slice(splitIdx);

            // Ensemble retrieval: average similarities across chains
            const recall1Results = evaluateEnsembleRetrieval(queries, reference, ensembleChains, 1);
            const recall5Results = evaluateEnsembleRetrieval(queries, reference, ensembleChains, 5);


            csvLines.push(
                `ELMEnsemble_${hiddenUnitSequence.join("_")}_dropout${dropout}_run${run},` +
                `${recall1Results.recallAtK.toFixed(4)},${recall5Results.recallAtK.toFixed(4)},${recall5Results.mrr.toFixed(4)}`
            );
        }
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `elm_ensemble_knowledge_distillation_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
