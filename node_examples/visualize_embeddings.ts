import fs from "fs";
import { PCA } from "ml-pca";
import { Matrix } from "ml-matrix";

interface EmbeddingRecord {
  embedding: number[];
  metadata: {
    text: string;
    label?: string; // make label optional
  };
}

// Load embeddings
const data: EmbeddingRecord[] = JSON.parse(
  fs.readFileSync("./embeddings/embeddings.json", "utf8")
);
console.log(`✅ Loaded ${data.length} embeddings.`);

// Prepare matrix
const vectors = data.map(d => d.embedding);
const m = new Matrix(vectors);

// Run PCA
const pca = new PCA(m);
const reduced = pca.predict(m, { nComponents: 2 }).to2DArray();
console.log(`✅ PCA completed.`);

// Prepare labels (default to "unknown")
const labels = data.map(d => d.metadata.label || "unknown");

// Make color map
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

// Write CSV
const csvLines = [
  "x,y,label,text",
  ...reduced.map((v, i) =>
    `${v[0]},${v[1]},"${labels[i]}","${data[i].metadata.text.replace(/"/g, '""')}"`
  )
];
fs.writeFileSync("./embeddings_pca.csv", csvLines.join("\n"));
console.log(`💾 Saved CSV to embeddings_pca.csv.`);

// Generate HTML visualization
const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Embedding PCA Visualization</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>Embedding Visualization (PCA)</h1>
  <div id="plot" style="width: 100%; height: 90vh;"></div>
  <script>
    const points = ${JSON.stringify(reduced)};
    const texts = ${JSON.stringify(data.map(d => d.metadata.text.slice(0, 100)))};
    const labels = ${JSON.stringify(labels)};
    const uniqueLabels = [...new Set(labels)];
    const labelColors = ${JSON.stringify(labelToColor)};

    const traces = uniqueLabels.map(label => {
      const indices = labels.map((l, i) => l === label ? i : -1).filter(i => i !== -1);
      return {
        x: indices.map(i => points[i][0]),
        y: indices.map(i => points[i][1]),
        mode: 'markers',
        type: 'scatter',
        name: label,
        text: indices.map(i => texts[i]),
        marker: { size: 6, color: labelColors[label] }
      };
    });

    Plotly.newPlot('plot', traces, {
      title: 'Embeddings PCA Projection (Colored by Label)',
      hovermode: 'closest'
    });
  </script>
</body>
</html>
`;

fs.writeFileSync("./embeddings_pca.html", html);
console.log(`💾 Saved HTML visualization to embeddings_pca.html.`);
