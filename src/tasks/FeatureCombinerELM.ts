// FeatureCombinerELM.ts — combine encoder vectors + metadata, train numeric ELM

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';

export class FeatureCombinerELM {
    private elm: ELM;
    private config: ELMConfig;
    private categories: string[] = [];

    constructor(config: ELMConfig & { inputSize?: number }) {
        const hidden = (config as any).hiddenUnits;
        const act = (config as any).activation as Activation | undefined;

        if (typeof hidden !== 'number') {
            throw new Error('FeatureCombinerELM requires config.hiddenUnits (number)');
        }
        if (!act) {
            throw new Error('FeatureCombinerELM requires config.activation');
        }

        // Force numeric mode (tokenizer off). Provide a safe inputSize placeholder;
        // ELM's trainFromData learns actual dims from X at train-time.
        this.config = {
            ...config,
            categories: (config as any).categories ?? [], // we set labels at train()
            useTokenizer: false,
            inputSize: (config as any).inputSize ?? 1,
            log: {
                modelName: 'FeatureCombinerELM',
                verbose: config.log?.verbose ?? false,
                toFile: config.log?.toFile ?? false,
                // @ts-ignore optional level passthrough
                level: (config.log as any)?.level ?? 'info',
            },
        } as any;

        this.elm = new ELM(this.config);

        // Optional thresholds/export passthrough
        if ((config as any).metrics) (this.elm as any).metrics = (config as any).metrics;
        if ((config as any).exportFileName) (this.elm as any).config.exportFileName = (config as any).exportFileName;
    }

    /** Concatenate encoder vector + metadata vector */
    static combineFeatures(encodedVec: number[], meta: number[]): number[] {
        // Fast path avoids spread copies in tight loops
        const out = new Array(encodedVec.length + meta.length);
        let i = 0;
        for (; i < encodedVec.length; i++) out[i] = encodedVec[i];
        for (let j = 0; j < meta.length; j++) out[i + j] = meta[j];
        return out;
    }

    /** Convenience for batch combination */
    static combineBatch(encoded: number[][], metas: number[][]): number[][] {
        if (encoded.length !== metas.length) {
            throw new Error(`combineBatch: encoded length ${encoded.length} != metas length ${metas.length}`);
        }
        const X = new Array(encoded.length);
        for (let i = 0; i < encoded.length; i++) {
            X[i] = FeatureCombinerELM.combineFeatures(encoded[i], metas[i]);
        }
        return X;
    }

    /** Train from encoder vectors + metadata + labels (classification) */
    train(encoded: number[][], metas: number[][], labels: string[]): void {
        if (!encoded?.length || !metas?.length || !labels?.length) {
            throw new Error('train: empty encoded/metas/labels');
        }
        if (encoded.length !== metas.length || encoded.length !== labels.length) {
            throw new Error('train: lengths must match (encoded, metas, labels)');
        }

        const X = FeatureCombinerELM.combineBatch(encoded, metas);

        this.categories = Array.from(new Set(labels));
        this.elm.setCategories(this.categories);

        const Y = labels.map((lab) => {
            const idx = this.categories.indexOf(lab);
            const row = new Array(this.categories.length).fill(0);
            if (idx >= 0) row[idx] = 1;
            return row;
        });

        // Closed-form solve via ELM; no private internals needed
        (this.elm as any).trainFromData(X, Y);
    }

    /** Predict top-K labels from a single (vec, meta) pair */
    predict(encodedVec: number[], meta: number[], topK = 1) {
        const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
        const batches = this.elm.predictFromVector(input, topK);
        return batches[0];
    }

    /** Predict the single best label + prob */
    predictLabel(encodedVec: number[], meta: number[]) {
        const [top] = this.predict(encodedVec, meta, 1);
        return top;
    }

    /** Get hidden embedding for (vec, meta) pair (useful for chaining) */
    getEmbedding(encodedVec: number[], meta: number[]): number[] {
        const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
        const H = this.elm.getEmbedding(input);
        return H[0];
    }

    loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
