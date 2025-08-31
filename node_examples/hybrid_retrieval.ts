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

// Compute TFIDF vectors
const tfidfTexts = sections.map(s => `${s.heading}. ${s.content}`);
console.log(`⏳ Computing TFIDF vectors...`);
const vectorizer = new TFIDFVectorizer(tfidfTexts, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// Compute encoder embeddings
const sectionVectors = tfidfTexts.map(t =>
    l2normalize(encoder.normalize(encoder.encode(t)))
);

// Train or load Paragraph ELM
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

// Train or load Indexer ELM chain
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

// Retrieval function
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
