import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { CosineSimilarity } from "../src/core/Evaluation";

// Utility
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (!isFinite(norm) || norm === 0) return v.map(() => 0);
    return v.map(x => x / norm);
}

function processEmbeddings(embs: number[][], label = "") {
    let sum = 0, count = 0, min = Infinity, max = -Infinity;
    for (const vec of embs) {
        for (const x of vec) {
            sum += x;
            count++;
            if (x < min) min = x;
            if (x > max) max = x;
        }
    }
    console.log(`✅ [${label}] mean=${(sum / count).toFixed(6)} min=${min.toFixed(6)} max=${max.toFixed(6)}`);
    return embs.map(l2normalize);
}

// 1️⃣ Parse Markdown
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const sectionRegex = /^(#{2,3}) (.+)$/gm;

let match;
const sections: { heading: string, content: string, text: string }[] = [];
let lastIndex = 0;

while ((match = sectionRegex.exec(rawText)) !== null) {
    const heading = match[2].trim();
    const start = match.index + match[0].length;
    const end = sectionRegex.lastIndex;
    const content = rawText.slice(start, end).trim();
    sections.push({
        heading,
        content,
        text: heading + "\n\n" + content
    });
}
console.log(`✅ Parsed ${sections.length} sections.`);

// 2️⃣ Load supervised pairs
const csvData = fs.readFileSync("../public/supervised_pairs.csv", "utf8");
const rows = parse(csvData, { skip_empty_lines: true });
const supervisedPairs = rows.map((r: [string, string]) => ({
    query: r[0].trim(),
    target: r[1].trim()
}));
console.log(`✅ Loaded ${supervisedPairs.length} supervised pairs.`);

// 3️⃣ Initialize encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// 4️⃣ Encode supervised queries
const queryVectors = supervisedPairs.map((p: { query: string; }) => encoder.normalize(encoder.encode(p.query)));

// 5️⃣ Encode retrieval sections
const sectionVectors = sections.map(s => encoder.normalize(encoder.encode(s.text)));

// 6️⃣ Train query encoder
function buildChain(name: string, vectors: number[][]): ELMChain {
    const layers: ELM[] = [];
    let inputs = vectors;
    [512, 256, 128].forEach((h, i) => {
        const elm = new ELM({
            activation: i === 0 ? "relu" : "tanh",
            hiddenUnits: h,
            maxLen: inputs[0].length,
            categories: [],
            log: { modelName: `${name}_layer${i}`, verbose: true },
            dropout: 0.02
        });
        elm.trainFromData(inputs, inputs);
        inputs = processEmbeddings(elm.computeHiddenLayer(inputs), `${name}_layer${i}`);
        layers.push(elm);
    });
    return new ELMChain(layers);
}

console.log(`⚙️ Training query encoder...`);
const queryChain = buildChain("query_encoder", queryVectors);

console.log(`⚙️ Training target encoder...`);
const targetChain = buildChain("target_encoder", sectionVectors);

// 7️⃣ Encode all sections
const encodedSections = targetChain.getEmbedding(sectionVectors).map(l2normalize);

// 8️⃣ Retrieval
function retrieve(query: string, topK = 5) {
    const qVec = encoder.normalize(encoder.encode(query));
    const qEmb = l2normalize(queryChain.getEmbedding([qVec])[0]);

    const scored = encodedSections.map((e, i) => ({
        text: sections[i].text,
        score: CosineSimilarity(e, qEmb)
    }));

    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// 9️⃣ Test retrieval
const query = "How do I create arrays in Go?";
const results = retrieve(query, 5);
console.log(`\n🔍 Retrieval results for: "${query}"`);
results.forEach((r, i) =>
    console.log(`${i + 1}. (Cosine=${r.score.toFixed(4)}) ${r.text.slice(0, 100)}...`)
);

console.log("✅ Done.");
