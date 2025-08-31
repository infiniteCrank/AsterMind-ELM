import fs from "fs";
import { PCA } from "ml-pca";
import { Matrix } from "ml-matrix";
import { tSNE } from "tsne-js";
import { UMAP } from "umap-js";

interface EmbeddingRecord {
  embedding: number[];
  metadata: {
    text: string;
    label?: string;
  };
}

// Load embeddings
const data: EmbeddingRecord[] = JSON.parse(
  fs.readFileSync("./embeddings/embeddings.json", "utf8")
);
console.log(`✅ Loaded ${data.length} embeddings.`);

const vectors = data.map(d => d.embedding);
const m = new Matrix(vectors);

// --- PCA ---
const pca = new PCA(m);
const pcaReduced = pca.predict(m, { nComponents: 2 }).to2DArray();
console.log(`✅ PCA completed.`);

// --- t-SNE ---
const tsne = new tSNE({
  dim: 2,
  perplexity: 30,
  earlyExaggeration: 4.0,
  learningRate: 100,
  nIter: 500,
  metric: "euclidean"
});
tsne.init({
  data: vectors,
  type: "dense"
});
console.log(`⏳ Running t-SNE...`);
for (let k = 0; k < 500; k++) {
  tsne.step();
  if (k % 100 === 0) console.log(`  t-SNE iteration ${k}`);
}
const tsneReduced = tsne.getOutputScaled();
console.log(`✅ t-SNE completed.`);

// --- UMAP ---
const umap = new UMAP({ nComponents: 2, nNeighbors: 15, minDist: 0.1 });
console.log(`⏳ Fitting UMAP...`);
const umapReduced = umap.fit(vectors);
console.log(`✅ UMAP completed.`);

// Prepare labels
const labels = data.map(d => d.metadata.label || "unknown");
const uniqueLabels = Array.from(new Set(labels));
const labelToColor: Record<string, string> = {};
const palette = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
  "#bcbd22", "#17becf"
];
uniqueLabels.forEach((label, i) => {
  labelToColor[label] = palette[i % palette.length];
});

// --- Write PCA CSV ---
const csvLines = [
  "method,x,y,label,text",
  ...pcaReduced.map((v, i) =>
    `PCA,${v[0]},${v[1]},"${labels[i]}","${data[i].metadata.text.replace(/"/g, '""')}"`
  ),
  ...tsneReduced.map((v, i) =>
    `t-SNE,${v[0]},${v[1]},"${labels[i]}","${data[i].metadata.text.replace(/"/g, '""')}"`
  ),
  ...umapReduced.map((v, i) =>
    `UMAP,${v[0]},${v[1]},"${labels[i]}","${data[i].metadata.text.replace(/"/g, '""')}"`
  )
];
fs.writeFileSync("./embeddings_projection.csv", csvLines.join("\n"));
console.log(`💾 Saved CSV to embeddings_projection.csv.`);

// --- Generate HTML visualization ---
const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Embedding Visualizations</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>Embeddings Visualization (PCA, t-SNE, UMAP)</h1>
  <div id="plot" style="width: 100%; height: 90vh;"></div>
  <script>
    const labels = ${JSON.stringify(labels)};
    const texts = ${JSON.stringify(data.map(d => d.metadata.text.slice(0, 100)))};
    const colors = ${JSON.stringify(labelToColor)};
    const pca = ${JSON.stringify(pcaReduced)};
    const tsne = ${JSON.stringify(tsneReduced)};
    const umap = ${JSON.stringify(umapReduced)};

    function makeTrace(method, points) {
      return [...new Set(labels)].map(label => {
        const indices = labels.map((l, i) => l === label ? i : -1).filter(i => i !== -1);
        return {
          x: indices.map(i => points[i][0]),
          y: indices.map(i => points[i][1]),
          mode: 'markers',
          type: 'scatter',
          name: label + ' (' + method + ')',
          text: indices.map(i => texts[i]),
          marker: { size: 6, color: colors[label] }
        };
      });
    }

    const traces = [
      ...makeTrace("PCA", pca),
      ...makeTrace("t-SNE", tsne),
      ...makeTrace("UMAP", umap)
    ];

    Plotly.newPlot('plot', traces, {
      title: 'Embeddings Projection (PCA, t-SNE, UMAP)',
      hovermode: 'closest'
    });
  </script>
</body>
</html>
`;
fs.writeFileSync("./embeddings_projection.html", html);
console.log(`💾 Saved HTML visualization to embeddings_projection.html.`);
