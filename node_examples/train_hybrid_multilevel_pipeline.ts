import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";

// Helpers
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return !isFinite(norm) || norm === 0 ? v.map(() => 0) : v.map(x => x / norm);
}
function averageVectors(vectors: number[][]): number[] {
    return vectors[0].map((_, i) =>
        vectors.reduce((s, v) => s + v[i], 0) / vectors.length
    );
}
function zeroCenter(vectors: number[][]): number[][] {
    const mean = vectors[0].map((_, j) =>
        vectors.reduce((s, v) => s + v[j], 0) / vectors.length
    );
    return vectors.map(v =>
        v.map((x, j) => x - mean[j])
    );
}
function processEmbeddings(embs: number[][], label = "") {
    const centered = zeroCenter(embs);
    const normalized = centered.map(l2normalize);
    let sum = 0, count = 0, min = Infinity, max = -Infinity;
    for (const vec of normalized) {
        for (const x of vec) {
            sum += x;
            count++;
            if (x < min) min = x;
            if (x > max) max = x;
        }
    }
    console.log(`✅ [${label}] Embeddings stats: mean=${(sum / count).toFixed(6)} min=${min.toFixed(6)} max=${max.toFixed(6)}`);
    return normalized;
}

// Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// Improved Markdown parser
const rawSections = rawText.split(/\n(?=#+ )/);
const sections = rawSections
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const headingLine = lines.find(l => /^#+ /.test(l)) || "";
        const contentLines = lines.filter(l => !/^#+ /.test(l));
        return {
            heading: headingLine.replace(/^#+ /, "").trim(),
            content: contentLines.join(" ").trim()
        };
    })
    .filter(s => s.content.length > 30);
console.log(`✅ Parsed ${sections.length} sections.`);

// Use joined text
const paragraphs = sections.map(s => `${s.heading} ${s.content}`);

// Universal Encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// Compute embeddings
const wordVectors = paragraphs.map(p => {
    const tokens = p.split(/\s+/).filter(Boolean);
    return l2normalize(averageVectors(tokens.map(t => encoder.normalize(encoder.encode(t)))));
});
const sentenceVectors = paragraphs.map(p => {
    const sentences = p.split(/[.?!]\s+/).filter(s => s.length > 3);
    return l2normalize(averageVectors(sentences.map(s => encoder.normalize(encoder.encode(s)))));
});
const paragraphVectors = paragraphs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p)))
);
console.log(`✅ Computed word/sentence/paragraph vectors.`);

// Load supervised pairs
const supervisedPaths = [
    "../public/supervised_pairs.csv",
    "../public/supervised_pairs_2.csv",
    "../public/supervised_pairs_3.csv",
    "../public/supervised_pairs_4.csv"
];
let supervisedPairs: { query: string, target: string }[] = [];
for (const path of supervisedPaths) {
    if (fs.existsSync(path)) {
        const csv = fs.readFileSync(path, "utf8");
        const rows = parse(csv, { skip_empty_lines: true });
        supervisedPairs.push(
            ...rows.map((r: [string, string]) => ({ query: r[0].trim(), target: r[1].trim() }))
        );
    }
}
console.log(`✅ Loaded ${supervisedPairs.length} supervised pairs.`);
const supQueryVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const supTargetVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// Load negative pairs
const negativePaths = [
    "../public/negative_pairs.csv",
    "../public/negative_pairs_2.csv",
    "../public/negative_pairs_3.csv",
    "../public/negative_pairs_4.csv"
];
let negativePairs: { query: string, target: string }[] = [];
for (const path of negativePaths) {
    if (fs.existsSync(path)) {
        const csv = fs.readFileSync(path, "utf8");
        const rows = parse(csv, { skip_empty_lines: true });
        negativePairs.push(
            ...rows.map((r: [string, string]) => ({ query: r[0].trim(), target: r[1].trim() }))
        );
    }
}
console.log(`✅ Loaded ${negativePairs.length} negative pairs.`);
const negQueryVecs = negativePairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const negTargetVecs = negativePairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// Build ELM chains
function buildChain(
    name: string,
    vectors: number[][],
    hiddenDims: number[],
    activations: string[],
    dropout: number
): { chain: ELMChain, finalEmbeddings: number[][] } {
    const chain: ELM[] = [];
    let inputs = vectors;
    hiddenDims.forEach((h, i) => {
        const elm = new ELM({
            activation: activations[i],
            hiddenUnits: h,
            maxLen: inputs[0].length,
            categories: [],
            log: { modelName: `${name}_layer${i}`, verbose: true },
            dropout
        });
        const path = `./elm_weights/${name}_layer${i}.json`;
        if (fs.existsSync(path)) {
            elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
            console.log(`✅ Loaded ${name}_layer${i}`);
        } else {
            console.log(`⚙️ Training ${name}_layer${i}...`);
            elm.trainFromData(inputs, inputs);
            fs.writeFileSync(path, JSON.stringify(elm.model));
            console.log(`💾 Saved ${name}_layer${i}`);
        }
        inputs = processEmbeddings(elm.computeHiddenLayer(inputs), `${name}_layer${i}`);
        chain.push(elm);
    });
    return { chain: new ELMChain(chain), finalEmbeddings: inputs };
}

const wordResult = buildChain("word_encoder", wordVectors, [512, 256, 128], ["relu", "tanh", "leakyRelu"], 0.02);
const sentenceResult = buildChain("sentence_encoder", sentenceVectors, [512, 256, 128], ["relu", "tanh", "leakyRelu"], 0.02);
const paragraphResult = buildChain("paragraph_encoder", paragraphVectors, [512, 256, 128], ["relu", "tanh", "leakyRelu"], 0.02);

const combinedEmbeddings = wordResult.finalEmbeddings.map((_, i) =>
    l2normalize([
        ...wordResult.finalEmbeddings[i],
        ...sentenceResult.finalEmbeddings[i],
        ...paragraphResult.finalEmbeddings[i]
    ])
);

// Supervised / Negative
function trainSimpleELM(name: string, X: number[][], Y: number[][]) {
    const elm = new ELM({
        activation: "relu",
        hiddenUnits: 128,
        maxLen: X[0].length,
        categories: [],
        log: { modelName: name, verbose: true },
        dropout: 0.02
    });
    const path = `./elm_weights/${name}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded ${name}`);
    } else {
        console.log(`⚙️ Training ${name}...`);
        elm.trainFromData(X, Y);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved ${name}`);
    }
    return elm;
}
const supELM = trainSimpleELM("supervisedELM", supQueryVecs, supTargetVecs);
const negELM = trainSimpleELM("negativeELM", negQueryVecs, negTargetVecs);

// Build indexer chain manually
let indexerInputs = combinedEmbeddings;
const indexerDims = [512, 256, 128];
const indexerActs = ["relu", "tanh", "leakyRelu"];
const indexerChainEncoders: ELM[] = [];

indexerDims.forEach((h, i) => {
    const elm = new ELM({
        activation: indexerActs[i],
        hiddenUnits: h,
        maxLen: indexerInputs[0].length,
        categories: [],
        log: { modelName: `indexer_layer${i}`, verbose: true },
        dropout: 0.02
    });
    const path = `./elm_weights/indexer_layer${i}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded indexer_layer${i}`);
    } else {
        console.log(`⚙️ Training indexer_layer${i}...`);
        elm.trainFromData(indexerInputs, indexerInputs);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved indexer_layer${i}`);
    }
    indexerInputs = processEmbeddings(elm.computeHiddenLayer(indexerInputs), `indexer_layer${i}`);
    indexerChainEncoders.push(elm);
});
const indexerChain = new ELMChain(indexerChainEncoders);

// TFIDF
console.log(`⏳ Computing TFIDF vectors...`);
const vectorizer = new TFIDFVectorizer(paragraphs, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// Save
const embeddingRecords: EmbeddingRecord[] = sections.map((s, i) => ({
    embedding: indexerInputs[i],
    metadata: { heading: s.heading, text: s.content }
}));
fs.writeFileSync("./embeddings/combined_embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// Retrieval
function retrieve(query: string, topK = 5) {
    const tokens = query.split(/\s+/).filter(Boolean);
    const avgWord = l2normalize(averageVectors(tokens.map(t => encoder.normalize(encoder.encode(t)))));
    const sentVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const paraVec = sentVec;

    const wordE = wordResult.chain.getEmbedding([avgWord])[0];
    const sentE = sentenceResult.chain.getEmbedding([sentVec])[0];
    const paraE = paragraphResult.chain.getEmbedding([paraVec])[0];
    const supE = supELM.computeHiddenLayer([sentVec])[0];
    const negE = negELM.computeHiddenLayer([sentVec])[0];

    const combined = l2normalize([
        ...wordE, ...sentE, ...paraE, ...supE, ...negE.map(x => -0.5 * x)
    ]);
    const finalVec = indexerChain.getEmbedding([combined])[0];
    const tfidfQ = l2normalize(vectorizer.vectorize(query));

    const scored = embeddingRecords.map(r => ({
        heading: r.metadata.heading,
        text: r.metadata.text,
        dense: r.embedding.reduce((s, v, j) => s + v * finalVec[j], 0),
        tfidf: tfidfVectors.map(t => t.reduce((s, v, j) => s + v * tfidfQ[j], 0))
    }));

    return scored
        .sort((a, b) => b.dense - a.dense)
        .slice(0, topK);
}

const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(`${i + 1}. (Dense=${r.dense.toFixed(4)}) ${r.heading} – ${r.text.slice(0, 100)}...`)
);
console.log(`✅ Done.`);
