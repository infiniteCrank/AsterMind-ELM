import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { PCA } from "ml-pca";
import { Matrix } from "ml-matrix";

// Helpers
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v.map(() => 0) : v.map(x => x / norm);
}

function loadPairs(paths: string[]) {
    const pairs: { query: string, target: string }[] = [];
    for (const path of paths) {
        if (fs.existsSync(path)) {
            const csv = fs.readFileSync(path, "utf8");
            const rows = parse(csv, { skip_empty_lines: true });
            pairs.push(...rows.map((r: [string, string]) => ({
                query: r[0].trim(),
                target: r[1].trim()
            })));
        }
    }
    return pairs;
}

// Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// Parse sections robustly
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

// Universal encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()[]{}<>+-=*/%\"'`_#|\\ \t",
    mode: "char",
    useTokenizer: false
});

// Encode sections
const sectionTexts = sections.map(s => `${s.heading} ${s.content}`);
const sectionVectors = sectionTexts.map(t => l2normalize(encoder.normalize(encoder.encode(t))));
console.log(`✅ Encoded sections.`);

// Load supervised and negative pairs
const supervisedPairs = loadPairs([
    "../public/supervised_pairs.csv",
    "../public/supervised_pairs_2.csv",
    "../public/supervised_pairs_3.csv",
    "../public/supervised_pairs_4.csv"
]);
const negativePairs = loadPairs([
    "../public/negative_pairs.csv",
    "../public/negative_pairs_2.csv",
    "../public/negative_pairs_3.csv",
    "../public/negative_pairs_4.csv"
]);
console.log(`✅ Loaded ${supervisedPairs.length} supervised and ${negativePairs.length} negative pairs.`);

// Encode pairs
const supQueryVecs = supervisedPairs.map(p => encoder.normalize(encoder.encode(p.query)));
const supTargetVecs = supervisedPairs.map(p => encoder.normalize(encoder.encode(p.target)));
const negQueryVecs = negativePairs.map(p => encoder.normalize(encoder.encode(p.query)));
const negTargetVecs = negativePairs.map(p => encoder.normalize(encoder.encode(p.target)));

// Baseline ELM (autoencoder)
const baselineELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: sectionVectors[0].length,
    categories: [],
    log: { modelName: "BaselineELM", verbose: true },
    dropout: 0.02
});
baselineELM.trainFromData(sectionVectors, sectionVectors);
const baselineEmbeddings = baselineELM.computeHiddenLayer(sectionVectors).map(l2normalize);
console.log(`✅ Baseline embeddings computed.`);

// Supervised ELM
const supervisedELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: supQueryVecs[0].length,
    categories: [],
    log: { modelName: "SupervisedELM", verbose: true },
    dropout: 0.02
});
supervisedELM.trainFromData(supQueryVecs, supTargetVecs);
console.log(`✅ Supervised ELM trained.`);

// Contrastive ELM
const contrastiveELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: supQueryVecs[0].length,
    categories: [],
    log: { modelName: "ContrastiveELM", verbose: true },
    dropout: 0.02
});
const allX = supQueryVecs.concat(negQueryVecs);
const allY = supTargetVecs.concat(negTargetVecs);
const weights = [
    ...supQueryVecs.map(() => 1.0),
    ...negQueryVecs.map(() => 0.25)
];
contrastiveELM.trainFromData(allX, allY, { weights });
console.log(`✅ Contrastive ELM trained.`);

// Embedding records
const embeddingRecords: EmbeddingRecord[] = sections.map((s, i) => ({
    embedding: baselineEmbeddings[i],
    metadata: { heading: s.heading, text: s.content }
}));

// Retrieval function
function retrieve(query: string, model: ELM, topK = 5) {
    const vec = encoder.normalize(encoder.encode(query));
    const emb = l2normalize(model.computeHiddenLayer([vec])[0]);

    const scored = embeddingRecords.map(r => ({
        heading: r.metadata.heading,
        snippet: r.metadata.text.slice(0, 100),
        similarity: r.embedding.reduce((s, v, j) => s + v * emb[j], 0)
    }));

    return scored.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
}

// Evaluate retrieval
const query = "How do you declare a map in Go?";
console.log(`\n🔍 Retrieval for: "${query}"`);

console.log(`\n⭐ Baseline:`);
retrieve(query, baselineELM).forEach((r, i) =>
    console.log(`${i + 1}. (Score=${r.similarity.toFixed(4)}) ${r.heading} – ${r.snippet}...`)
);

console.log(`\n⭐ Supervised:`);
retrieve(query, supervisedELM).forEach((r, i) =>
    console.log(`${i + 1}. (Score=${r.similarity.toFixed(4)}) ${r.heading} – ${r.snippet}...`)
);

console.log(`\n⭐ Contrastive:`);
retrieve(query, contrastiveELM).forEach((r, i) =>
    console.log(`${i + 1}. (Score=${r.similarity.toFixed(4)}) ${r.heading} – ${r.snippet}...`)
);

// PCA visualization
const m = new Matrix(baselineEmbeddings);
const pca = new PCA(m);
const reduced = pca.predict(m, { nComponents: 2 }).to2DArray();

const csvLines = [
    "x,y,heading",
    ...reduced.map((v, i) => `${v[0]},${v[1]},"${sections[i].heading.replace(/"/g, '""')}"`)
];
fs.writeFileSync("./embeddings_pca.csv", csvLines.join("\n"));
console.log(`💾 Saved PCA CSV.`);

const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Embedding PCA Visualization</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>Embedding PCA Projection</h1>
  <div id="plot" style="width:100%; height:90vh;"></div>
  <script>
    const points = ${JSON.stringify(reduced)};
    const texts = ${JSON.stringify(sections.map(s => s.heading))};
    Plotly.newPlot('plot', [{
      x: points.map(p => p[0]),
      y: points.map(p => p[1]),
      mode: 'markers',
      type: 'scatter',
      text: texts,
      marker: { size:6 }
    }], {
      title: 'PCA Projection',
      hovermode: 'closest'
    });
  </script>
</body>
</html>
`;
fs.writeFileSync("./embeddings_pca.html", html);
console.log(`💾 Saved PCA HTML.`);

console.log(`✅ All experiments complete.`);
