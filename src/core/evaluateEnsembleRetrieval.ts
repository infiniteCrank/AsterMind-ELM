import { ELMChain } from "./ELMChain";
import { EmbeddingRecord } from "./EmbeddingStore";

export function evaluateEnsembleRetrieval(
    queries: EmbeddingRecord[],
    reference: EmbeddingRecord[],
    chains: ELMChain[],
    k: number
) {
    let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;

    function cosine(a: number[], b: number[]) {
        const dot = a.reduce((s, ai, i) => s + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return dot / (normA * normB);
    }

    console.log("🔹 Precomputing embeddings...");

    // Precompute embeddings for each chain
    const chainQueryEmbeddings: number[][][] = chains.map(chain =>
        chain.getEmbedding(queries.map(q => q.embedding))
    );

    const chainReferenceEmbeddings: number[][][] = chains.map(chain =>
        chain.getEmbedding(reference.map(r => r.embedding))
    );

    console.log("✅ Precomputation complete. Starting retrieval evaluation...");

    queries.forEach((q, i) => {
        if (i % 10 === 0) console.log(`🔍 Query ${i + 1}/${queries.length}`);

        const ensembleScores: { label: string; score: number }[] = [];

        for (let j = 0; j < reference.length; j++) {
            let sum = 0;
            for (let c = 0; c < chains.length; c++) {
                const qEmb = chainQueryEmbeddings[c][i];
                const rEmb = chainReferenceEmbeddings[c][j];
                sum += cosine(qEmb, rEmb);
            }
            ensembleScores.push({
                label: reference[j].metadata.label || "",
                score: sum / chains.length
            });
        }

        ensembleScores.sort((a, b) => b.score - a.score);
        const ranked = ensembleScores.map(s => s.label);
        const correctLabel = q.metadata.label || "";

        if (ranked[0] === correctLabel) hitsAt1++;
        if (ranked.slice(0, k).includes(correctLabel)) hitsAtK++;
        const rank = ranked.indexOf(correctLabel);
        reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
    });

    return {
        recallAt1: hitsAt1 / queries.length,
        recallAtK: hitsAtK / queries.length,
        mrr: reciprocalRanks / queries.length
    };
}
