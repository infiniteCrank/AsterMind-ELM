// ConfidenceClassifierELM.ts — numeric confidence classifier on top of ELM
// Upgrades:
//  • Numeric-only pipeline (no tokenizer)
//  • Proper trainFromData(X, Y) with one-hot labels
//  • Vector-safe prediction (predictFromVector)
//  • Score helpers, batch APIs, and simple evaluation
//  • Robust logging + safe handling of ELMConfig union

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';
import { FeatureCombinerELM } from './FeatureCombinerELM';

export type ConfidenceLabel = 'low' | 'high';

export interface ConfidenceClassifierOptions {
    /** Class names to use (default: ['low','high']) */
    categories?: [ConfidenceLabel, ConfidenceLabel] | string[];
    /** Activation for hidden units (default: 'relu') */
    activation?: Activation;
    /** Verbose console logs */
    verbose?: boolean;
    /** Export filename for saved JSON (optional) */
    exportFileName?: string;
}

/**
 * ConfidenceClassifierELM is a lightweight wrapper that classifies whether
 * an upstream model’s prediction is "low" or "high" confidence based on
 * (embedding, metadata) numeric features.
 */
export class ConfidenceClassifierELM {
    private elm: ELM;
    private categories: string[];
    private activation: Activation;

    constructor(private baseConfig: ELMConfig, opts: ConfidenceClassifierOptions = {}) {
        this.categories = (opts.categories as string[] | undefined) ?? ['low', 'high'];
        this.activation = opts.activation ?? ((baseConfig as any).activation ?? 'relu');

        // We force a numeric ELM config. Many ELM builds don’t require inputSize
        // at construction because trainFromData(X,Y) uses X[0].length to size W.
        // We still pass useTokenizer=false and categories to be explicit.
        const cfg: any = {
            ...this.baseConfig,
            useTokenizer: false,
            categories: this.categories,
            activation: this.activation,
            log: {
                modelName: 'ConfidenceClassifierELM',
                verbose: baseConfig.log?.verbose ?? opts.verbose ?? false,
                toFile: baseConfig.log?.toFile ?? false,
                level: (baseConfig.log as any)?.level ?? 'info',
            },
            // Optional passthroughs:
            exportFileName: opts.exportFileName ?? (this.baseConfig as any).exportFileName,
        };

        this.elm = new ELM(cfg);

        // Forward thresholds if present
        if ((this.baseConfig as any).metrics) {
            (this.elm as any).metrics = (this.baseConfig as any).metrics;
        }
    }

    /** One-hot helper */
    private oneHot(n: number, idx: number): number[] {
        const v = new Array(n).fill(0);
        if (idx >= 0 && idx < n) v[idx] = 1;
        return v;
    }

    /**
     * Train from numeric (vector, meta) → combined features + labels.
     * `vectors[i]` and `metas[i]` must be aligned with `labels[i]`.
     */
    public train(vectors: number[][], metas: number[][], labels: string[]): void {
        if (!vectors?.length || !metas?.length || !labels?.length) {
            throw new Error('train: empty inputs');
        }
        if (vectors.length !== metas.length || vectors.length !== labels.length) {
            throw new Error('train: vectors, metas, labels must have same length');
        }

        // Ensure categories include all observed labels (keeps order of existing categories first)
        const uniq = Array.from(new Set(labels));
        const merged = Array.from(new Set<string>([...this.categories, ...uniq]));
        this.categories = merged;
        this.elm.setCategories(this.categories);

        // Build X, Y
        const X: number[][] = new Array(vectors.length);
        const Y: number[][] = new Array(vectors.length);

        for (let i = 0; i < vectors.length; i++) {
            const x = FeatureCombinerELM.combineFeatures(vectors[i], metas[i]); // numeric feature vector
            X[i] = x;

            const li = this.categories.indexOf(labels[i]);
            Y[i] = this.oneHot(this.categories.length, li);
        }

        // Closed-form ELM training
        this.elm.trainFromData(X, Y);
    }

    /** Predict full distribution for a single (vec, meta). */
    public predict(vec: number[], meta: number[], topK = 2): Array<{ label: string; prob: number }> {
        const x = FeatureCombinerELM.combineFeatures(vec, meta);

        // Prefer vector-safe API; most Astermind builds expose predictFromVector([x], topK)
        const fn: any = (this.elm as any).predictFromVector;
        if (typeof fn === 'function') {
            const out = fn.call(this.elm, [x], topK); // PredictResult[][]
            return Array.isArray(out) && Array.isArray(out[0]) ? out[0] : (out ?? []);
        }

        // Fallback to predict() if it supports numeric vectors (some builds do)
        const maybe = (this.elm as any).predict?.(x, topK);
        if (Array.isArray(maybe)) return maybe;

        throw new Error('No vector-safe predict available on underlying ELM.');
    }

    /** Probability the label is "high" (or the second category by default). */
    public predictScore(vec: number[], meta: number[], positive: string = 'high'): number {
        const dist = this.predict(vec, meta, this.categories.length);
        const hit = dist.find(d => d.label === positive);
        return hit?.prob ?? 0;
    }

    /** Predicted top-1 label. */
    public predictLabel(vec: number[], meta: number[]): string {
        const dist = this.predict(vec, meta, 1);
        return dist[0]?.label ?? this.categories[0];
    }

    /** Batch prediction (distributions). */
    public predictBatch(vectors: number[][], metas: number[][], topK = 2):
        Array<Array<{ label: string; prob: number }>> {
        if (vectors.length !== metas.length) {
            throw new Error('predictBatch: vectors and metas must have same length');
        }
        return vectors.map((v, i) => this.predict(v, metas[i], topK));
    }

    /* ============ Simple evaluation helpers ============ */

    /** Compute accuracy and confusion counts for a labeled set. */
    public evaluate(
        vectors: number[][],
        metas: number[][],
        labels: string[],
    ): { accuracy: number; confusion: Record<string, Record<string, number>> } {
        if (vectors.length !== metas.length || vectors.length !== labels.length) {
            throw new Error('evaluate: inputs must have same length');
        }
        const confusion: Record<string, Record<string, number>> = {};
        for (const a of this.categories) {
            confusion[a] = {};
            for (const b of this.categories) confusion[a][b] = 0;
        }

        let correct = 0;
        for (let i = 0; i < vectors.length; i++) {
            const pred = this.predictLabel(vectors[i], metas[i]);
            const gold = labels[i];
            if (pred === gold) correct++;
            if (!confusion[gold]) confusion[gold] = {} as any;
            if (confusion[gold][pred] === undefined) confusion[gold][pred] = 0;
            confusion[gold][pred]++;
        }

        return { accuracy: correct / labels.length, confusion };
    }

    /* ============ I/O passthroughs ============ */

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    /** Access underlying ELM if needed */
    public getELM(): ELM {
        return this.elm;
    }

    /** Current category ordering used by the model */
    public getCategories(): string[] {
        return this.categories.slice();
    }
}
