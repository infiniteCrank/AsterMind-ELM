import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

// Utility: L2 normalize a vector
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// 1️⃣ Load the Markdown corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Parse sections intelligently (heading + content)
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

// 3️⃣ Prepare encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// 4️⃣ Encode all sections
const texts = sections.map(s => `${s.heading} ${s.content}`);
const baseVectors = texts.map(t => encoder.normalize(encoder.encode(t)));
console.log(`✅ Encoded sections.`);

// 5️⃣ Train or load Encoder ELM (autoencoder)
const encoderELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: baseVectors[0].length,
    categories: [],
    log: { modelName: "EncoderELM", verbose: true },
    dropout: 0.02
});

const encoderWeightsPath = "./elm_weights/encoderELM.json";
fs.mkdirSync("./elm_weights", { recursive: true });

if (fs.existsSync(encoderWeightsPath)) {
    encoderELM.loadModelFromJSON(fs.readFileSync(encoderWeightsPath, "utf-8"));
    console.log(`✅ Loaded Encoder ELM weights.`);
} else {
    console.log(`⚙️ Training Encoder ELM...`);
    encoderELM.trainFromData(baseVectors, baseVectors);
    fs.writeFileSync(encoderWeightsPath, JSON.stringify(encoderELM.model));
    console.log(`💾 Saved Encoder ELM weights.`);
}

// 6️⃣ Compute embeddings
let embeddings = encoderELM.computeHiddenLayer(baseVectors).map(l2normalize);
console.log(`✅ Encoder embeddings ready.`);

// 7️⃣ Train/load Indexer ELM chain
const hiddenUnits = [256, 128, 64];
const indexerELMs = hiddenUnits.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i + 1}`, verbose: true },
        dropout: 0.02
    })
);

indexerELMs.forEach((elm, i) => {
    const path = `./elm_weights/indexerELM_${i + 1}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded Indexer ELM ${i + 1} weights.`);
    } else {
        console.log(`⚙️ Training Indexer ELM ${i + 1}...`);
        elm.trainFromData(embeddings, embeddings);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved Indexer ELM ${i + 1} weights.`);
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(indexerELMs);
console.log(`✅ Indexer ELM chain ready.`);

// 8️⃣ Save embeddings
const embeddingRecords: EmbeddingRecord[] = sections.map((s, i) => ({
    embedding: embeddings[i],
    metadata: { heading: s.heading, text: s.content }
}));
fs.writeFileSync("./embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// 9️⃣ Retrieval
function retrieve(query: string, topK = 5) {
    const queryVec = encoder.normalize(encoder.encode(query));
    const encVec = encoderELM.computeHiddenLayer([queryVec])[0];
    const denseVec = l2normalize(indexerChain.getEmbedding([l2normalize(encVec)])[0]);

    const scored = embeddingRecords.map((r, i) => {
        if (r.embedding.length !== denseVec.length) {
            throw new Error(
                `Embedding length mismatch at index ${i}: ${r.embedding.length} vs ${denseVec.length}`
            );
        }
        if (r.embedding.some(x => !isFinite(x)) || denseVec.some(x => !isFinite(x))) {
            throw new Error(`NaN or Infinite values detected in embeddings at index ${i}`);
        }

        const score = r.embedding.reduce((s, v, j) => s + v * denseVec[j], 0);
        return {
            heading: r.metadata.heading,
            snippet: r.metadata.text.slice(0, 100),
            score
        };
    });

    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// 🔍 Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(`${i + 1}. [Score=${r.score.toFixed(4)}] ${r.heading} – ${r.snippet}...`)
);
console.log(`✅ Done.`);
