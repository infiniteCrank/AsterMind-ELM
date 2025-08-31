export interface KNNDataPoint {
    vector: number[];
    label: string;
}

export interface KNNResult {
    label: string;
    weight: number;
}

export type KNNMetric = "cosine" | "euclidean";

export class KNN {
    /**
     * Compute cosine similarity between two numeric vectors.
     */
    static cosineSimilarity(vec1: number[], vec2: number[]): number {
        let dot = 0, norm1 = 0, norm2 = 0;
        for (let i = 0; i < vec1.length; i++) {
            dot += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        if (norm1 === 0 || norm2 === 0) return 0;
        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Compute Euclidean distance between two numeric vectors.
     */
    static euclideanDistance(vec1: number[], vec2: number[]): number {
        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            const diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Find k nearest neighbors.
     * @param queryVec - Query vector
     * @param dataset - Dataset to search
     * @param k - Number of neighbors
     * @param topX - Number of top results to return
     * @param metric - Similarity metric
     */
    static find(
        queryVec: number[],
        dataset: KNNDataPoint[],
        k: number = 5,
        topX: number = 3,
        metric: KNNMetric = "cosine"
    ): KNNResult[] {
        const similarities = dataset.map((item, idx) => {
            let score: number;
            if (metric === "cosine") {
                score = this.cosineSimilarity(queryVec, item.vector);
            } else {
                // For Euclidean, invert distance so higher = closer
                const dist = this.euclideanDistance(queryVec, item.vector);
                score = -dist;
            }
            return { index: idx, score };
        });

        similarities.sort((a, b) => b.score - a.score);

        const labelWeights: Record<string, number> = {};
        for (let i = 0; i < Math.min(k, similarities.length); i++) {
            const label = dataset[similarities[i].index].label;
            const weight = similarities[i].score;
            labelWeights[label] = (labelWeights[label] || 0) + weight;
        }

        const weightedLabels: KNNResult[] = Object.entries(labelWeights)
            .map(([label, weight]) => ({ label, weight }))
            .sort((a, b) => b.weight - a.weight);

        return weightedLabels.slice(0, topX);
    }
}
