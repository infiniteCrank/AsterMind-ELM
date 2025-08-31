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

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `elm_tfidf_experiment_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
