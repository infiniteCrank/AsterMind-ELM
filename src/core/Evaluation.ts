import { EmbeddingRecord } from './EmbeddingStore';
import { ELMChain } from './ELMChain';

export interface RetrievalMetricResult {
    recallAtK: number;
    mrr: number;
}

export function evaluateRetrieval(
    queries: EmbeddingRecord[],
    store: EmbeddingRecord[],
    chain: ELMChain,
    k: number = 5
): RetrievalMetricResult {
    let recallHits = 0;
    let reciprocalRanks: number[] = [];

    for (const query of queries) {
        const queryEmbedding = chain.getEmbedding([query.embedding])[0];
        const results = searchEmbeddings(queryEmbedding, k, store);

        const trueLabel = query.metadata.label;
        const labels = results.map(r => r.metadata.label);

        // Recall@K
        if (labels.includes(trueLabel)) {
            recallHits++;
        }

        // MRR
        const rank = labels.indexOf(trueLabel);
        if (rank !== -1) {
            reciprocalRanks.push(1 / (rank + 1));
        } else {
            reciprocalRanks.push(0);
        }
    }

    const recallAtK = recallHits / queries.length;
    const mrr = reciprocalRanks.reduce((a, b) => a + b, 0) / queries.length;

    return { recallAtK, mrr };
}

function searchEmbeddings(
    queryEmbedding: number[],
    topK: number,
    store: EmbeddingRecord[]
): EmbeddingRecord[] {
    const scored = store.map(r => ({
        ...r,
        score: CosineSimilarity(queryEmbedding, r.embedding)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

export function CosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
}


export interface RetrievalMetricResult {
    recallAtK: number;
    mrr: number;
}

export function evaluateRetrievalSimple(
    queries: EmbeddingRecord[],
    store: EmbeddingRecord[],
    embedFn: (vec: number[][]) => number[][],
    k: number = 5
): RetrievalMetricResult {
    let recallHits = 0;
    let reciprocalRanks: number[] = [];

    for (const query of queries) {
        const queryEmbedding = embedFn([query.embedding])[0];
        const results = searchEmbeddings(queryEmbedding, k, store);

        const trueLabel = query.metadata.label;
        const labels = results.map(r => r.metadata.label);

        if (labels.includes(trueLabel)) recallHits++;

        const rank = labels.indexOf(trueLabel);
        reciprocalRanks.push(rank !== -1 ? 1 / (rank + 1) : 0);
    }

    const recallAtK = recallHits / queries.length;
    const mrr = reciprocalRanks.reduce((a, b) => a + b, 0) / queries.length;
    return { recallAtK, mrr };
}

function cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
}
