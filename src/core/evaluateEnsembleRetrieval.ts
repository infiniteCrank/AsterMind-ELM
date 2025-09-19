// evaluateEnsembleRetrieval.ts
import { ELMChain } from "./ELMChain";

/** Minimal record shape for this evaluation (kept independent of the store). */
export interface EmbeddingRecord {
    id?: string;
    embedding: number[] | Float32Array;
    metadata: { label?: string;[k: string]: any };
}

export type EnsembleMetric = "cosine" | "dot";
export type ScoreAgg = "mean" | "sum" | "max" | "weighted";

export interface EnsembleEvalOptions {
    /** Similarity metric across chain embeddings (default: 'cosine') */
    metric?: EnsembleMetric;
    /** How to aggregate scores across chains (default: 'mean') */
    aggregate?: ScoreAgg;
    /** Per-chain weights (only used when aggregate='weighted'). Length must equal chains.length. */
    weights?: number[];
    /** K for Recall@K and Top-K lists (default: 5) */
    k?: number;
    /** If true, drop queries with missing/empty label from metrics (default: true) */
    ignoreUnlabeledQueries?: boolean;
    /** If true, also return per-label stats (slower) */
    reportPerLabel?: boolean;
    /** If true, attach top-K rankings per query to result (slower) */
    returnRankings?: boolean;
    /** Progress callback every N queries (default: 10) */
    logEvery?: number;
}

export interface PerLabelStats {
    count: number;
    hitsAt1: number;
    hitsAtK: number;
    mrrSum: number;
}

export interface QueryRanking {
    queryIndex: number;
    queryId?: string;
    label: string;
    topK: Array<{ label: string; score: number }>;
    correctRank: number; // -1 if not found
}

export interface EnsembleEvalResult {
    /** Queries that contributed to the metrics (after filtering) */
    usedQueries: number;
    recallAt1: number;
    recallAtK: number;
    mrr: number;
    /** Optional breakdown by label */
    perLabel?: Record<string, {
        support: number;
        recallAt1: number;
        recallAtK: number;
        mrr: number;
    }>;
    /** Optional top-K rankings per query (debug/inspection) */
    rankings?: QueryRanking[];
}

const EPS = 1e-12;

/* ---------- math helpers ---------- */

function l2Norm(v: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
}

function normalize(v: ArrayLike<number>): Float32Array {
    const out = new Float32Array(v.length);
    const n = l2Norm(v);
    if (n < EPS) return out; // keep zeros; cosine with zero gives 0
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
}

function dot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

/* ---------- main evaluation ---------- */

export function evaluateEnsembleRetrieval(
    queries: EmbeddingRecord[],
    reference: EmbeddingRecord[],
    chains: ELMChain[],
    k: number,
    options?: EnsembleEvalOptions
): EnsembleEvalResult {
    const metric = options?.metric ?? "cosine";
    const aggregate: ScoreAgg = options?.aggregate ?? "mean";
    const weights = options?.weights;
    const topK = Math.max(1, options?.k ?? k ?? 5);
    const ignoreUnlabeled = options?.ignoreUnlabeledQueries ?? true;
    const reportPerLabel = options?.reportPerLabel ?? false;
    const returnRankings = options?.returnRankings ?? false;
    const logEvery = Math.max(1, options?.logEvery ?? 10);

    if (chains.length === 0) {
        throw new Error("evaluateEnsembleRetrieval: 'chains' must be non-empty.");
    }
    if (aggregate === "weighted") {
        if (!weights || weights.length !== chains.length) {
            throw new Error(`aggregate='weighted' requires weights.length === chains.length`);
        }
        // normalize weights to sum=1 for interpretability
        const sumW = weights.reduce((s, w) => s + w, 0) || 1;
        for (let i = 0; i < weights.length; i++) weights[i] = weights[i] / sumW;
    }

    console.log("🔹 Precomputing embeddings...");

    // Pull raw embeddings from each chain
    const chainQueryEmb: Float32Array[][] = [];
    const chainRefEmb: Float32Array[][] = [];

    for (let c = 0; c < chains.length; c++) {
        const qMat = chains[c].getEmbedding(queries.map(q => {
            const v = q.embedding;
            if (!v || v.length === 0) throw new Error(`Query ${c} has empty embedding`);
            return Array.from(v);
        }));
        const rMat = chains[c].getEmbedding(reference.map(r => {
            const v = r.embedding;
            if (!v || v.length === 0) throw new Error(`Reference has empty embedding`);
            return Array.from(v);
        }));

        // Validate dims & normalize if cosine
        const qArr = qMat.map(row => Float32Array.from(row));
        const rArr = rMat.map(row => Float32Array.from(row));

        if (metric === "cosine") {
            chainQueryEmb.push(qArr.map(normalize));
            chainRefEmb.push(rArr.map(normalize));
        } else {
            chainQueryEmb.push(qArr);
            chainRefEmb.push(rArr);
        }

        // Basic safety: check dimensions match across Q/R for this chain
        const dimQ = qArr[0]?.length ?? 0;
        const dimR = rArr[0]?.length ?? 0;
        if (dimQ === 0 || dimR === 0 || dimQ !== dimR) {
            throw new Error(`Chain ${c}: query/ref embedding dims mismatch (${dimQ} vs ${dimR})`);
        }
    }

    console.log("✅ Precomputation complete. Starting retrieval evaluation...");

    let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
    let used = 0;

    const perLabelRaw: Record<string, PerLabelStats> = {};
    const rankings: QueryRanking[] = [];

    for (let i = 0; i < queries.length; i++) {
        if (i % logEvery === 0) console.log(`🔍 Query ${i + 1}/${queries.length}`);

        const correctLabel = (queries[i].metadata.label ?? "").toString();
        if (!correctLabel && ignoreUnlabeled) {
            continue; // skip this query entirely
        }

        // Accumulate ensemble scores per reference
        // We keep (label, score) per reference j
        const scores: { label: string; score: number }[] = new Array(reference.length);
        for (let j = 0; j < reference.length; j++) {
            let sAgg: number;

            if (aggregate === "max") {
                // Take max across chains
                let sMax = -Infinity;
                for (let c = 0; c < chains.length; c++) {
                    const q = chainQueryEmb[c][i];
                    const r = chainRefEmb[c][j];
                    const s = metric === "cosine" || metric === "dot" ? dot(q, r) : dot(q, r); // only cosine/dot supported
                    if (s > sMax) sMax = s;
                }
                sAgg = sMax;
            } else if (aggregate === "sum") {
                let sSum = 0;
                for (let c = 0; c < chains.length; c++) {
                    const q = chainQueryEmb[c][i];
                    const r = chainRefEmb[c][j];
                    sSum += (metric === "cosine" || metric === "dot") ? dot(q, r) : dot(q, r);
                }
                sAgg = sSum;
            } else if (aggregate === "weighted") {
                let sW = 0;
                for (let c = 0; c < chains.length; c++) {
                    const q = chainQueryEmb[c][i];
                    const r = chainRefEmb[c][j];
                    sW += ((metric === "cosine" || metric === "dot") ? dot(q, r) : dot(q, r)) * (weights as number[])[c];
                }
                sAgg = sW;
            } else { // "mean"
                let sSum = 0;
                for (let c = 0; c < chains.length; c++) {
                    const q = chainQueryEmb[c][i];
                    const r = chainRefEmb[c][j];
                    sSum += (metric === "cosine" || metric === "dot") ? dot(q, r) : dot(q, r);
                }
                sAgg = sSum / chains.length;
            }

            scores[j] = {
                label: (reference[j].metadata.label ?? "").toString(),
                score: sAgg
            };
        }

        // Sort by score desc
        scores.sort((a, b) => b.score - a.score);
        const rankedLabels = scores.map(s => s.label);

        // Update metrics
        const r1 = rankedLabels[0] === correctLabel ? 1 : 0;
        const rK = rankedLabels.slice(0, topK).includes(correctLabel) ? 1 : 0;
        const rank = rankedLabels.indexOf(correctLabel);
        const rr = rank === -1 ? 0 : 1 / (rank + 1);

        hitsAt1 += r1;
        hitsAtK += rK;
        reciprocalRanks += rr;
        used++;

        if (reportPerLabel) {
            const bucket = perLabelRaw[correctLabel] ?? (perLabelRaw[correctLabel] = { count: 0, hitsAt1: 0, hitsAtK: 0, mrrSum: 0 });
            bucket.count++;
            bucket.hitsAt1 += r1;
            bucket.hitsAtK += rK;
            bucket.mrrSum += rr;
        }

        if (returnRankings) {
            rankings.push({
                queryIndex: i,
                queryId: queries[i].id,
                label: correctLabel,
                topK: scores.slice(0, topK),
                correctRank: rank
            });
        }
    }

    const denom = used || 1;
    const result: EnsembleEvalResult = {
        usedQueries: used,
        recallAt1: hitsAt1 / denom,
        recallAtK: hitsAtK / denom,
        mrr: reciprocalRanks / denom
    };

    if (reportPerLabel) {
        const out: Record<string, { support: number; recallAt1: number; recallAtK: number; mrr: number }> = {};
        for (const [label, s] of Object.entries(perLabelRaw)) {
            out[label] = {
                support: s.count,
                recallAt1: s.hitsAt1 / (s.count || 1),
                recallAtK: s.hitsAtK / (s.count || 1),
                mrr: s.mrrSum / (s.count || 1)
            };
        }
        result.perLabel = out;
    }

    if (returnRankings) result.rankings = rankings;

    return result;
}
