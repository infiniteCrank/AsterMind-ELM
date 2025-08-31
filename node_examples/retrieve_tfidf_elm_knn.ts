// retrieve_tfidf_elm_knn.ts

import fs from "fs";
import { ELM } from "../src/core/ELM";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN, KNNDataPoint } from "../src/ml/KNN";

// 🟢 Utility to l2-normalize vectors
function l2normalize(vec: number[]): number[] {
    const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? vec : vec.map(x => x / norm);
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

// 3️⃣ Build TF-IDF vectorizer over all sections
const texts = sections.map(s => `${s.heading} ${s.content}`);
const vectorizer = new TFIDFVectorizer(texts);
const X_raw = texts.map(t => vectorizer.vectorize(t));
const X = X_raw.map(l2normalize);

// 4️⃣ Train ELM embedding model
const elm = new ELM({
    categories: [], // No classification—autoencoder
    hiddenUnits: 128,
    maxLen: X[0].length,
    activation: "relu",
    dropout: 0.02,
    weightInit: "xavier",
    log: { modelName: "ELM-Embedding", verbose: true, toFile: false },
});

// If weights file exists, load
const weightsFile = "./elm_embedding_model.json";
if (fs.existsSync(weightsFile)) {
    elm.loadModelFromJSON(fs.readFileSync(weightsFile, "utf-8"));
    console.log("✅ Loaded ELM weights.");
} else {
    console.log("⚙️ Training ELM autoencoder...");
    elm.trainFromData(X, X);
    fs.writeFileSync(weightsFile, JSON.stringify(elm.model));
    console.log("💾 Saved ELM weights.");
}

// 5️⃣ Compute ELM embeddings
const embeddings = elm.computeHiddenLayer(X).map(l2normalize);

// 6️⃣ Build KNN dataset
const knnData: KNNDataPoint[] = sections.map((s, i) => ({
    vector: embeddings[i],
    label: s.heading || `Section ${i + 1}`,
}));

console.log("✅ KNN dataset ready.");

// 7️⃣ Retrieval function
function retrieve(query: string, topK = 5) {
    // Vectorize query
    const tfidfVec = l2normalize(vectorizer.vectorize(query));
    const elmEmbedding = l2normalize(elm.computeHiddenLayer([tfidfVec])[0]);

    // KNN search
    const knnResults = KNN.find(elmEmbedding, knnData, 5, topK, "cosine");

    // Combine with cosine similarity to all sections
    const combinedScores = knnData.map((d, i) => ({
        index: i,
        score: d.vector.reduce((s, v, j) => s + v * elmEmbedding[j], 0),
    }));

    combinedScores.sort((a, b) => b.score - a.score);

    // Top results
    const topResults = combinedScores.slice(0, topK).map(r => ({
        heading: sections[r.index].heading,
        snippet: sections[r.index].content.slice(0, 150),
        score: r.score,
    }));

    return topResults;
}

// 8️⃣ Example query
const query = "How do you declare a map in Go?";
const results = retrieve(query);

console.log(`\n🔍 Retrieval results for query: "${query}"\n`);
results.forEach((r, i) => {
    console.log(`${i + 1}. [Score=${r.score.toFixed(4)}] ${r.heading}\n   ${r.snippet}\n`);
});
