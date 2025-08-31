// retrieve_with_transformer.ts
import fs from "fs";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN, KNNDataPoint } from "../src/ml/KNN";
import { ELM } from "../src/core/ELM";
import { Vocab, MiniTransformer } from "../src/core/MiniTransformer";

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const sections = rawText.split(/\n(?=#{1,6}\s)/)
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const heading = lines.find(l => /^#{1,6}\s/.test(l)) || "";
        const content = lines.filter(l => !/^#{1,6}\s/.test(l)).join(" ").trim();
        return { heading: heading.replace(/^#{1,6}\s/, ""), content };
    })
    .filter(s => s.content.length > 30);

const texts = sections.map(s => `${s.heading} ${s.content}`);

// 2️⃣ TFIDF vectorizer
const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map(t =>
    TFIDFVectorizer.l2normalize(vectorizer.vectorize(t))
);

// 3️⃣ ELM embedding
const elm = new ELM({
    categories: [],
    hiddenUnits: 128,
    maxLen: tfidfVectors[0].length,
    activation: "relu",
    log: { modelName: "ELM-Autoencoder", verbose: true },
    dropout: 0.02
});
if (fs.existsSync("./elm_weights.json")) {
    elm.loadModelFromJSON(fs.readFileSync("./elm_weights.json", "utf8"));
    console.log("✅ Loaded saved ELM weights.");
} else {
    console.log("⚙️ Training ELM autoencoder...");
    elm.trainFromData(tfidfVectors, tfidfVectors);
    fs.writeFileSync("./elm_weights.json", JSON.stringify(elm.model));
}
const elmEmbeddings = elm.computeHiddenLayer(tfidfVectors);

// 4️⃣ KNN dataset
const knnData: KNNDataPoint[] = texts.map((_, i) => ({
    vector: elmEmbeddings[i],
    label: String(i)
}));

// 5️⃣ MiniTransformer setup with special tokens and lowercasing
const SPECIAL_TOKENS = ["<PAD>", "<UNK>"];
const allChars = new Set<string>(SPECIAL_TOKENS);
texts.forEach(t => t.toLowerCase().split("").forEach(c => allChars.add(c)));
const vocab = new Vocab([...allChars]);
const transformer = new MiniTransformer(vocab);

// Encode helper with fallback
function encodeSafe(text: string): number[] {
    return text
        .toLowerCase()
        .split("")
        .map(c => vocab.tokenToIdx[c] !== undefined ? vocab.tokenToIdx[c] : vocab.tokenToIdx["<UNK>"]);
}

const transformerEmbeddings = texts.map(t => transformer.encode(encodeSafe(t)));

// 6️⃣ Retrieval query
const query = "How do you declare a map in Go?";
const queryTFIDF = TFIDFVectorizer.l2normalize(vectorizer.vectorize(query));
const queryELM = elm.computeHiddenLayer([queryTFIDF])[0];
const knnResults = KNN.find(queryELM, knnData, 10, 10, "cosine");

// 7️⃣ Re-ranking using transformer embeddings
function cosine(a: number[], b: number[]): number {
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
    return normA && normB ? dot / (normA * normB) : 0;
}

const queryTransformerEmbedding = transformer.encode(encodeSafe(query));

const reranked = knnResults
    .map(r => {
        const idx = parseInt(r.label, 10);
        const sim = cosine(queryTransformerEmbedding, transformerEmbeddings[idx]);
        return { idx, sim };
    })
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 5);

// 8️⃣ Display results
console.log(`\n🔍 Top results for: "${query}"\n`);
reranked.forEach((r, i) => {
    const section = sections[r.idx];
    console.log(`${i + 1}. [Sim=${r.sim.toFixed(4)}] ${section.heading}`);
    console.log(`   ${section.content.slice(0, 120)}...`);
});
