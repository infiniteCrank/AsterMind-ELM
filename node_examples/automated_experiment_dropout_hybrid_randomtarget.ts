import fs from "fs";
import { parse } from "csv-parse/sync";
import { pipeline } from "@xenova/transformers";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";

(async () => {
    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));
    const sampleSize = 1000;
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

    const hiddenUnitSequences = [
        [512, 256, 128],
        [256, 128, 64, 32],
        [256, 128, 64, 32, 16],
        [128, 64, 32, 16, 8, 4]
    ];

    const dropouts = [0.0, 0.02, 0.05];
    const repeats = 3;

    const csvLines = ["config,run,recall_at_1,recall_at_5,mrr"];

    function cosineSimilarity(a: number[], b: number[]): number {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return normA && normB ? dot / (normA * normB) : 0;
    }

    function evaluateRecallMRR(query: number[][], reference: number[][], qLabels: string[], rLabels: string[], k: number) {
        let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
        for (let i = 0; i < query.length; i++) {
            const scores = reference.map((emb, j) => ({
                label: rLabels[j],
                score: cosineSimilarity(query[i], emb)
            }));
            scores.sort((a, b) => b.score - a.score);
            const ranked = scores.map(s => s.label);
            if (ranked[0] === qLabels[i]) hitsAt1++;
            if (ranked.slice(0, k).includes(qLabels[i])) hitsAtK++;
            const rank = ranked.indexOf(qLabels[i]);
            reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
        }
        return { recall1: hitsAt1 / query.length, recallK: hitsAtK / query.length, mrr: reciprocalRanks / query.length };
    }

    const bertResults = evaluateRecallMRR(queryBERT, refBERT, queryLabels, refLabels, 5);
    csvLines.unshift(`SentenceBERT_Baseline,NA,${bertResults.recallK.toFixed(4)},${bertResults.mrr.toFixed(4)}`);

    function l2normalize(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        return norm === 0 ? v : v.map(x => x / norm);
    }

    for (const seq of hiddenUnitSequences) {
        for (const dropout of dropouts) {
            for (let run = 1; run <= repeats; run++) {
                const hybridActivations = seq.map((_, i) =>
                    i % 3 === 0 ? "relu" : i % 3 === 1 ? "tanh" : "leakyRelu"
                );
                console.log(`\n🔹 Testing config: ${seq.join("_")} hybrid dropout ${dropout} (Run ${run})`);

                const elms = seq.map((h, i) =>
                    new ELM({
                        activation: hybridActivations[i],
                        hiddenUnits: h,
                        maxLen: 50,
                        categories: [],
                        log: { modelName: `ELM layer ${i + 1}`, verbose: true, toFile: false },
                        metrics: { accuracy: 0.01 },
                        dropout
                    })
                );

                const chain = new ELMChain(elms);
                const encoder = elms[0].encoder;
                const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));
                let inputs = encodedVectors;

                for (const elm of elms) {
                    // Generate random target projections
                    const targetDim = Math.min(inputs[0].length, elm.hiddenUnits);
                    const randomTargetMatrix = Array.from({ length: inputs.length }, () =>
                        Array.from({ length: targetDim }, () => Math.random())
                    );

                    elm.trainFromData(inputs, randomTargetMatrix, { reuseWeights: false });

                    inputs = inputs.map(vec => elm.getEmbedding([vec])[0]);

                    // Diagnostics
                    const flat = inputs.flat();
                    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
                    const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
                    console.log(`  Layer embedding stats: mean=${mean.toFixed(6)}, variance=${variance.toFixed(6)}`);

                    // Sample cosine similarities
                    const pairA = inputs[Math.floor(Math.random() * inputs.length)];
                    const pairB = inputs[Math.floor(Math.random() * inputs.length)];
                    const cosSim = cosineSimilarity(pairA, pairB);
                    console.log(`  Sample cosine similarity: ${cosSim.toFixed(4)}`);
                }

                // Normalize only at the end
                const embeddingStore: EmbeddingRecord[] = records.slice(0, sampleSize).map((rec, i) => ({
                    embedding: l2normalize(inputs[i]),
                    metadata: { text: rec.text, label: rec.label }
                }));

                const queries = embeddingStore.slice(0, splitIdx);
                const reference = embeddingStore.slice(splitIdx);

                const recall1Results = evaluateRecallMRR(
                    queries.map(q => q.embedding),
                    reference.map(r => r.embedding),
                    queryLabels,
                    refLabels,
                    1
                );
                const recall5Results = evaluateRecallMRR(
                    queries.map(q => q.embedding),
                    reference.map(r => r.embedding),
                    queryLabels,
                    refLabels,
                    5
                );

                // Join activation names so it's clear which hybrid is used
                const activationPattern = hybridActivations.join("-");

                csvLines.push(
                    `ELMChain_${seq.join("_")}_${activationPattern}_dropout${dropout}_run${run},` +
                    `${recall1Results.recall1.toFixed(4)},${recall5Results.recallK.toFixed(4)},${recall5Results.mrr.toFixed(4)}`
                );
            }
        }
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `automated_experiment_dropout_hybrid_randomtarget_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
