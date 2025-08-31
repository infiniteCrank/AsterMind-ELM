import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";

// Helpers
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 || !isFinite(norm) ? v.map(() => 0) : v.map(x => x / norm);
}
function averageVectors(vectors: number[][]): number[] {
    return vectors[0].map((_, i) => vectors.reduce((s, v) => s + v[i], 0) / vectors.length);
}
function zeroCenter(vectors: number[][]): number[][] {
    const mean = vectors[0].map((_, j) =>
        vectors.reduce((s, v) => s + v[j], 0) / vectors.length
    );
    return vectors.map(v => v.map((x, j) => x - mean[j]));
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

// Parse sections using markdown headings
const rawSections = rawText.split(/\n(?=#+ )/);
const paragraphs = rawSections
    .map(block => {
        const lines = block.split("\n").filter(Boolean);
        const heading = lines.find(l => /^#+ /.test(l)) || "";
        const contentLines = lines.filter(l => !/^#+ /.test(l));
        return (
            heading.replace(/^#+ /, "").trim() +
            " " +
            contentLines.join(" ").trim()
        );
    })
    .filter(p => p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} sections.`);

// Encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// Compute paragraph vectors
const paragraphVectors = paragraphs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p)))
);
console.log(`✅ Computed paragraph vectors.`);

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

// Encode supervised
const supQueryVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const supTargetVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// Build a simple encoder chain
function buildChain(
    name: string,
    vectors: number[][],
    hiddenDims: number[],
    activations: string[],
    dropout: number
): { chain: ELMChain, embeddings: number[][] } {
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
    return { chain: new ELMChain(chain), embeddings: inputs };
}

// Build paragraph encoder chain
const paragraphChainResult = buildChain(
    "paragraph_encoder",
    paragraphVectors,
    [256, 128],
    ["relu", "tanh"],
    0.02
);
const paragraphChain = paragraphChainResult.chain;

// Combine supervised query vectors with encoder
const supQueryCombined = supQueryVecs.map(vec =>
    l2normalize(paragraphChain.getEmbedding([vec])[0])
);

// Train supervised ELM
const supELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: supQueryCombined[0].length,
    categories: [],
    log: { modelName: "SupervisedELM", verbose: true },
    dropout: 0.02
});
console.log(`⚙️ Training SupervisedELM...`);
supELM.trainFromData(supQueryCombined, supTargetVecs);
console.log(`✅ SupervisedELM trained.`);

// Retrieval
function retrieve(query: string, topK = 5) {
    const qVec = encoder.normalize(encoder.encode(query));
    const emb = l2normalize(paragraphChain.getEmbedding([qVec])[0]);
    const pred = supELM.computeHiddenLayer([emb])[0];
    const scored = paragraphs.map((p, i) => ({
        text: p,
        score: pred.reduce((s, v, j) => s + v * paragraphChainResult.embeddings[i][j], 0)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(`${i + 1}. (Score=${r.score.toFixed(4)}) ${r.text.slice(0, 100)}...`)
);
console.log(`✅ Done.`);
