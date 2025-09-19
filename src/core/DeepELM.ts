// DeepELM.ts — stacked ELM autoencoders + top ELM classifier

import { ELM } from './ELM';
import { ELMChain } from './ELMChain';
import { wrapELM } from './ELMAdapter';

export type Activation = 'linear' | 'tanh' | 'relu' | 'leakyrelu' | 'sigmoid';
export type WeightInit = 'uniform' | 'xavier' | 'he';

export interface DeepELMLayerSpec {
    hiddenUnits: number;
    activation?: Activation;
    ridgeLambda?: number;    // per-layer ridge (if your ELM uses it)
    weightInit?: WeightInit;
    dropout?: number;
    name?: string;
}

export interface DeepELMConfig {
    inputDim: number;                     // D_in of raw features
    layers: DeepELMLayerSpec[];           // stacked AE layers
    // Final classifier (ELM) trained on last layer features:
    numClasses: number;                   // K
    clfHiddenUnits?: number;              // default: 0 => “linear readout” via tiny ELM
    clfActivation?: Activation;           // default: 'linear'
    clfWeightInit?: WeightInit;           // default: 'xavier'
    normalizeEach?: boolean;              // L2-normalize after each layer
    normalizeFinal?: boolean;             // L2-normalize at the end
}

export class DeepELM {
    public cfg: DeepELMConfig;
    private aeLayers: ELM[] = [];
    private chain: ELMChain | null = null;
    private clf: ELM | null = null;

    constructor(cfg: DeepELMConfig) {
        this.cfg = {
            clfHiddenUnits: 0,
            clfActivation: 'linear',
            clfWeightInit: 'xavier',
            normalizeEach: false,
            normalizeFinal: true,
            ...cfg,
        };
    }

    /** Layer-wise unsupervised training with Y=X (autoencoder). Returns transformed X_L. */
    fitAutoencoders(X: number[][]): number[][] {
        let cur = X;
        this.aeLayers = [];

        for (let i = 0; i < this.cfg.layers.length; i++) {
            const spec = this.cfg.layers[i];
            // Minimal ELM config for numeric mode—categories aren’t used by trainFromData:
            const elm = new ELM({
                categories: ['ae'],                         // placeholder (unused in trainFromData)
                hiddenUnits: spec.hiddenUnits,
                activation: spec.activation ?? 'relu',
                weightInit: spec.weightInit ?? 'xavier',
                dropout: spec.dropout ?? 0,
                log: { modelName: spec.name ?? `AE#${i + 1}`, verbose: false },
            } as any);

            // Autoencoder: targets are the inputs
            elm.trainFromData(cur, cur);
            this.aeLayers.push(elm);

            // Forward to next layer using hidden activations
            cur = elm.getEmbedding(cur);
            if (this.cfg.normalizeEach) {
                cur = l2NormalizeRows(cur);
            }
        }

        // Build chain for fast forward passes
        this.chain = new ELMChain(this.aeLayers.map((m, i) => {
            const a = wrapELM(m, m['modelName'] || `AE#${i + 1}`);
            return a;
        }), {
            normalizeEach: !!this.cfg.normalizeEach,
            normalizeFinal: !!this.cfg.normalizeFinal,
            name: 'DeepELM-Chain',
        });

        return this.transform(X);
    }

    /** Supervised training of a top classifier ELM on last-layer features. */
    fitClassifier(X: number[][], yOneHot: number[][]): void {
        if (!this.chain) throw new Error('fitClassifier: call fitAutoencoders() first');
        const Z = this.chain.getEmbedding(X) as number[][];
        // If clfHiddenUnits === 0, we mimic a “linear readout” by using a very small hidden layer with linear activation.
        const hidden = Math.max(1, this.cfg.clfHiddenUnits || 1);

        this.clf = new ELM({
            categories: Array.from({ length: this.cfg.numClasses }, (_, i) => String(i)),
            hiddenUnits: hidden,
            activation: this.cfg.clfActivation ?? 'linear',
            weightInit: this.cfg.clfWeightInit ?? 'xavier',
            log: { modelName: 'DeepELM-Classifier', verbose: false },
        } as any);

        this.clf.trainFromData(Z, yOneHot);
    }

    /** One-shot convenience: train AEs then classifier. */
    fit(X: number[][], yOneHot: number[][]): void {
        this.fitAutoencoders(X);
        this.fitClassifier(X, yOneHot);
    }

    /** Forward through stacked AEs (no classifier). */
    transform(X: number[][]): number[][] {
        if (!this.chain) throw new Error('transform: model not fitted');
        const Z = this.chain.getEmbedding(X) as number[][];
        return Z;
    }

    /** Classifier probabilities (softmax) for a batch. */
    predictProba(X: number[][]): number[][] {
        if (!this.clf) throw new Error('predictProba: classifier not fitted');
        // Reuse existing ELM method on batch:
        const Z = this.transform(X);
        const res = this.clf.predictFromVector(Z, this.cfg.numClasses);
        // predictFromVector returns topK lists; convert back into dense probs when possible
        // If you’d rather have dense probs, expose a new method on ELM to return raw softmax scores for a batch.
        return topKListToDense(res, this.cfg.numClasses);
    }

    /** Utility: export all models for persistence. */
    toJSON(): any {
        return {
            cfg: this.cfg,
            layers: this.aeLayers.map(m => (m as any).savedModelJSON ?? JSON.stringify((m as any).model)),
            clf: this.clf ? ((this.clf as any).savedModelJSON ?? JSON.stringify((this.clf as any).model)) : null,
            __version: 'deep-elm-1.0.0',
        };
    }

    /** Utility: load from exported payload. */
    fromJSON(payload: any) {
        const { cfg, layers, clf } = payload ?? {};
        if (!Array.isArray(layers)) throw new Error('fromJSON: invalid payload');
        this.cfg = { ...this.cfg, ...cfg };

        this.aeLayers = layers.map((j: string, i: number) => {
            const m = new ELM({ categories: ['ae'], hiddenUnits: 1 } as any);
            m.loadModelFromJSON(j);
            return m;
        });

        this.chain = new ELMChain(this.aeLayers.map((m, i) => wrapELM(m, `AE#${i + 1}`)), {
            normalizeEach: !!this.cfg.normalizeEach,
            normalizeFinal: !!this.cfg.normalizeFinal,
            name: 'DeepELM-Chain',
        });

        if (clf) {
            const c = new ELM({ categories: Array.from({ length: this.cfg.numClasses }, (_, i) => String(i)), hiddenUnits: 1 } as any);
            c.loadModelFromJSON(clf);
            this.clf = c;
        }
    }
}

/* ---------- helpers ---------- */

function l2NormalizeRows(M: number[][]): number[][] {
    return M.map(r => {
        let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i];
        const inv = 1 / (Math.sqrt(s) || 1);
        return r.map(v => v * inv);
    });
}

function topKListToDense(list: Array<Array<{ label: string; prob: number }>>, K: number): number[][] {
    // Convert the ELM.predictFromVector top-K output back to dense [N x K] probs if needed.
    // (If your ELM exposes a dense “predictProbaFromVectors” for the batch, prefer that.)
    return list.map(row => {
        const out = new Array(K).fill(0);
        for (const { label, prob } of row) {
            const idx = Number(label);
            if (Number.isFinite(idx) && idx >= 0 && idx < K) out[idx] = prob;
        }
        return out;
    });
}
