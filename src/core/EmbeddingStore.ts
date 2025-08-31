export interface EmbeddingRecord {
    embedding: number[];
    metadata: {
        heading?: any;
        text: string;
        label?: string;
    };
}

export const embeddingStore: EmbeddingRecord[] = [];

export function addEmbedding(record: EmbeddingRecord): void {
    embeddingStore.push(record);
}

export function searchEmbeddings(queryEmbedding: number[], topK: number = 5): EmbeddingRecord[] {
    const scored = embeddingStore.map(r => ({
        ...r,
        score: cosineSimilarity(queryEmbedding, r.embedding)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

function cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
}
