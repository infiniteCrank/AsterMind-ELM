// RefinerELM.ts — numeric “refinement” classifier on top of arbitrary feature vectors

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';

type MetricsGate = {
    rmse?: number;
    mae?: number;
    accuracy?: number;
    f1?: number;
    crossEntropy?: number;
    r2?: number;
};

export interface RefinerELMOptions {
    /** REQUIRED: input vector length for numeric mode */
    inputSize: number;
    /** REQUIRED: hidden units for the ELM */
    hiddenUnits: number;
    /** Optional activation (defaults to 'relu') */
    activation?: Activation;
    /** Optional initial categories; can be overridden on train() */
    categories?: string[];
    /** Optional logging */
    log?: {
        modelName?: string;
        verbose?: boolean;
        toFile?: boolean;
        level?: 'info' | 'debug';
    };
    /** Optional export name */
    exportFileName?: string;
    /** Optional regularization / init knobs */
    ridgeLambda?: number;
    dropout?: number;
    weightInit?: 'uniform' | 'xavier' | 'he';
    /** Optional metric thresholds (set on the ELM instance, not in config) */
    metrics?: MetricsGate;
}

export class RefinerELM {
    private elm: ELM;

    constructor(opts: RefinerELMOptions) {
        if (!Number.isFinite(opts.inputSize) || opts.inputSize <= 0) {
            throw new Error('RefinerELM: opts.inputSize must be a positive number.');
        }
        if (!Number.isFinite(opts.hiddenUnits) || opts.hiddenUnits <= 0) {
            throw new Error('RefinerELM: opts.hiddenUnits must be a positive number.');
        }

        // Build a *numeric* ELM config (no text fields here)
        const numericConfig: ELMConfig = {
            // numeric discriminator:
            useTokenizer: false,
            inputSize: opts.inputSize,

            // required for ELM
            categories: opts.categories ?? [],

            // base config
            hiddenUnits: opts.hiddenUnits,
            activation: opts.activation ?? 'relu',
            ridgeLambda: opts.ridgeLambda,
            dropout: opts.dropout,
            weightInit: opts.weightInit,

            // misc
            exportFileName: opts.exportFileName,
            log: {
                modelName: opts.log?.modelName ?? 'RefinerELM',
                verbose: opts.log?.verbose ?? false,
                toFile: opts.log?.toFile ?? false,
                level: opts.log?.level ?? 'info',
            },
        };

        this.elm = new ELM(numericConfig);

        // Set metric thresholds on the instance (not inside the config)
        if (opts.metrics) {
            (this.elm as any).metrics = opts.metrics;
        }
    }

    /** Train from feature vectors + string labels. */
    train(
        inputs: number[][],
        labels: string[],
        opts?: { reuseWeights?: boolean; sampleWeights?: number[]; categories?: string[] }
    ): void {
        if (!inputs?.length || !labels?.length || inputs.length !== labels.length) {
            throw new Error('RefinerELM.train: inputs/labels must be non-empty and aligned.');
        }

        // Allow overriding categories at train time
        const categories = opts?.categories ?? Array.from(new Set(labels));
        this.elm.setCategories(categories);

        const Y: number[][] = labels.map((label) =>
            this.elm.oneHot(categories.length, categories.indexOf(label))
        );

        // Public training path; no 'task' key here
        const options: { reuseWeights?: boolean; weights?: number[] } = {};
        if (opts?.reuseWeights !== undefined) options.reuseWeights = opts.reuseWeights;
        if (opts?.sampleWeights) options.weights = opts.sampleWeights;

        this.elm.trainFromData(inputs, Y, options);
    }

    /** Full probability vector aligned to `this.elm.categories`. */
    predictProbaFromVector(vec: number[]): number[] {
        // Use the vector-safe path provided by the core ELM
        const out = this.elm.predictFromVector([vec], /*topK*/ this.elm.categories.length);
        // predictFromVector returns Array<PredictResult[]>, i.e., topK sorted.
        // We want a dense prob vector in category order, so map from topK back:
        const probs = new Array(this.elm.categories.length).fill(0);
        if (out && out[0]) {
            for (const { label, prob } of out[0]) {
                const idx = this.elm.categories.indexOf(label);
                if (idx >= 0) probs[idx] = prob;
            }
        }
        return probs;
    }

    /** Top-K predictions ({label, prob}) for a single vector. */
    predict(vec: number[], topK = 1): Array<{ label: string; prob: number }> {
        const [res] = this.elm.predictFromVector([vec], topK);
        return res;
    }

    /** Batch top-K predictions for an array of vectors. */
    predictBatch(
        vectors: number[][],
        topK = 1
    ): Array<Array<{ label: string; prob: number }>> {
        return this.elm.predictFromVector(vectors, topK);
    }

    /** Hidden-layer embedding(s) — useful for chaining. */
    embed(vec: number[]): number[] {
        return this.elm.getEmbedding([vec])[0];
    }
    embedBatch(vectors: number[][]): number[][] {
        return this.elm.getEmbedding(vectors);
    }

    /** Persistence passthroughs */
    loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }
    saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
