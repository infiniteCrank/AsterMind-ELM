// automated_experiment_elm_transformer.ts
import fs from "fs";
import { parse } from "csv-parse/sync";
import { pipeline } from "@xenova/transformers";
import { ELMTransformer, ELMTransformerMode } from "../src/core/ELMTransformer";

(async () => {
    // Load dataset
    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));
    const sampleSize = 500;
    const texts = records.slice(0, sampleSize).map(r => r.text);
    const labels = records.slice(0, sampleSize).map(r => r.label);

    console.log(`⏳ Loading Sentence-BERT...`);
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    const bertTensor = await embedder(texts, { pooling: "mean" });
    const bertEmbeddings = bertTensor.tolist() as number[][];

    const splitIdx = Math.floor(texts.length * 0.2);
    const queryLabels = labels.slice(0, splitIdx);
    const refLabels = labels.slice(splitIdx);

    const modes = [
        ELMTransformerMode.ELM_TO_TRANSFORMER,
        ELMTransformerMode.TRANSFORMER_TO_ELM,
        ELMTransformerMode.PARAMETERIZE_ELM,
        ELMTransformerMode.ENSEMBLE
    ];

    const embedDims = [32, 64];
    const dropouts = [0.0, 0.05];
    const repeats = 2;

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
            // Diagnostic: cosine similarity distribution
            console.log(`\nQuery ${i} similarities:`);
            scores.slice(0, 10).forEach(s =>
                console.log(`  Label=${s.label}  Score=${s.score.toFixed(4)}`)
            );
            scores.sort((a, b) => b.score - a.score);
            const ranked = scores.map(s => s.label);
            console.log(`Top 5 labels for Query ${i}: ${ranked.slice(0, 5).join(", ")}`);
            if (ranked[0] === qLabels[i]) hitsAt1++;
            if (ranked.slice(0, k).includes(qLabels[i])) hitsAtK++;
            const rank = ranked.indexOf(qLabels[i]);
            reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
        }
        return { recall1: hitsAt1 / query.length, recallK: hitsAtK / query.length, mrr: reciprocalRanks / query.length };
    }

    function l2normalize(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        return norm === 0 ? v : v.map(x => x / norm);
    }

    function checkNaNMatrix(mat: number[][], label: string) {
        const anyNaN = mat.some(row => row.some(v => Number.isNaN(v)));
        if (anyNaN) {
            console.error(`❌ NaN detected in matrix: ${label}`);
            process.exit(1);
        }
    }

    const csvLines = ["mode,embedDim,dropout,run,recall_at_1,recall_at_5,mrr"];

    for (const mode of modes) {
        for (const embedDim of embedDims) {
            for (const dropout of dropouts) {
                for (let run = 1; run <= repeats; run++) {
                    console.log(`\n🔹 Testing Mode: ${mode}, embedDim=${embedDim}, dropout=${dropout}, Run=${run}`);

                    const transformer = new ELMTransformer({
                        mode,
                        embedDim,
                        seqLen: 8,
                        numHeads: 2,
                        numLayers: 1,
                        dropout,
                        elmConfig: {
                            categories: Array.from(new Set(labels)),
                            hiddenUnits: 64,
                            maxLen: 50,
                            activation: "relu",
                            metrics: { rmse: 1e6 },
                            log: { modelName: "ELM", verbose: true },
                        }
                    });

                    const trainPairs = records.slice(splitIdx).map(r => ({ input: r.text, label: r.label }));

                    console.log(`  ⏳ Training...`);
                    transformer.train(trainPairs);

                    console.log(`  ⏳ Generating query embeddings...`);
                    const queryEmbeddings = texts.slice(0, splitIdx).map(t => {
                        const e = l2normalize(transformer.getEmbedding(t));
                        // Diagnostic: embedding norm
                        const norm = Math.sqrt(e.reduce((s, x) => s + x * x, 0));
                        console.log(`Embedding norm: ${norm.toFixed(6)}`);
                        if (e.some(v => Number.isNaN(v))) {
                            console.error(`❌ NaN embedding detected in query: ${t}`);
                            process.exit(1);
                        }
                        console.log(`    Embedding sample: [${e.slice(0, 5).map(x => x.toFixed(4)).join(", ")}...]`);
                        return e;
                    });

                    console.log(`  ⏳ Generating reference embeddings...`);
                    const refEmbeddings = texts.slice(splitIdx).map(t => {
                        const e = l2normalize(transformer.getEmbedding(t));
                        if (e.some(v => Number.isNaN(v))) {
                            console.error(`❌ NaN embedding detected in reference: ${t}`);
                            process.exit(1);
                        }
                        return e;
                    });

                    const metrics = evaluateRecallMRR(queryEmbeddings, refEmbeddings, queryLabels, refLabels, 5);
                    csvLines.push(`${mode},${embedDim},${dropout},${run},${metrics.recall1.toFixed(4)},${metrics.recallK.toFixed(4)},${metrics.mrr.toFixed(4)}`);

                    console.log(`  ✅ Metrics: Recall@1=${metrics.recall1.toFixed(4)}, Recall@5=${metrics.recallK.toFixed(4)}, MRR=${metrics.mrr.toFixed(4)}`);
                }
            }
        }
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `automated_experiment_elm_transformer_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
