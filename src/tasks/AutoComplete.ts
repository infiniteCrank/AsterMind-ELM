// ✅ AutoComplete.ts — ELM | KernelELM (Nyström+whiten) | OnlineELM
// Fixes:
//  • Avoids union narrowing on EnglishTokenPreset by shimming preset fields (no ExtendedELMConfig maxLen error)
//  • activation typed as Activation (not string)
//  • Removed non-existent "task" option in trainFromData()

import { ELM } from '../core/ELM';
import { KernelELM, KernelSpec } from '../core/KernelELM';
import { OnlineELM } from '../core/OnlineELM';
import { UniversalEncoder } from '../preprocessing/UniversalEncoder';

import { bindAutocompleteUI } from '../ui/components/BindUI';
import { EnglishTokenPreset } from '../config/Presets';

import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { Activation } from '../core/ELMConfig';

export interface TrainPair {
    input: string;
    label: string;
}

type Engine = 'elm' | 'kernel' | 'online';

export interface KernelOptions {
    type?: KernelSpec['type'];             // 'rbf' | 'linear' | 'poly' | 'laplacian' | 'custom'
    gamma?: number;
    degree?: number;
    coef0?: number;
    m?: number;
    strategy?: 'uniform' | 'kmeans++' | 'preset';
    seed?: number;
    preset?: { points?: number[][]; indices?: number[] };
    whiten?: boolean;
    jitter?: number;
}

interface AutoCompleteOptions {
    activation?: Activation;
    topK?: number;

    // UI
    inputElement: HTMLInputElement;
    outputElement: HTMLElement;

    // Training / gating
    metrics?: { rmse?: number; mae?: number; accuracy?: number; top1Accuracy?: number; crossEntropy?: number };
    verbose?: boolean;
    exportFileName?: string;
    augmentationOptions?: {
        suffixes?: string[];
        prefixes?: string[];
        includeNoise?: boolean;
    };

    // Engines
    engine?: Engine;
    hiddenUnits?: number;                  // ELM/Online
    ridgeLambda?: number;                  // all
    weightInit?: 'uniform' | 'xavier' | 'he'; // ELM/Online
    kernel?: KernelOptions;                // KernelELM
}

type AnyModel = ELM | KernelELM | OnlineELM;

/** Safe accessor for preset fields (avoids type errors on ExtendedELMConfig) */
const PRESET = (() => {
    const p = EnglishTokenPreset as any;
    return {
        maxLen: (p?.maxLen as number) ?? 30,
        charSet: (p?.charSet as string) ?? 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: (p?.useTokenizer as boolean) ?? true,
        tokenizerDelimiter: (p?.tokenizerDelimiter as RegExp) ?? /\s+/
    };
})();

function oneHot(idx: number, n: number): number[] {
    const v = new Array(n).fill(0);
    if (idx >= 0 && idx < n) v[idx] = 1;
    return v;
}

function sortTopK(labels: string[], probs: number[], k: number) {
    return probs
        .map((p, i) => ({ label: labels[i], prob: p }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, k);
}

export class AutoComplete {
    public model: AnyModel;
    private encoder: UniversalEncoder;
    public categories: string[];

    public activation: Activation;
    public engine: Engine;
    public topKDefault: number;

    private trainPairs: TrainPair[];

    constructor(pairs: TrainPair[], options: AutoCompleteOptions) {
        this.trainPairs = pairs;
        this.activation = options.activation ?? 'relu';
        this.engine = options.engine ?? 'elm';
        this.topKDefault = options.topK ?? 5;

        // Labels
        this.categories = Array.from(new Set(pairs.map(p => p.label)));

        // Text → numeric encoder (Kernel/Online need numeric; ELM can also consume numeric directly)
        this.encoder = new UniversalEncoder({
            charSet: PRESET.charSet,
            maxLen: PRESET.maxLen,
            useTokenizer: PRESET.useTokenizer,
            tokenizerDelimiter: PRESET.tokenizerDelimiter,
            mode: (PRESET.useTokenizer ? 'token' : 'char'),
        });

        const hiddenUnits = options.hiddenUnits ?? 128;
        const ridgeLambda = options.ridgeLambda ?? 1e-2;
        const weightInit = options.weightInit ?? 'xavier';
        const verbose = options.verbose ?? false;

        if (this.engine === 'kernel') {
            const D = this.encoder.getVectorSize();
            const ktype = options.kernel?.type ?? 'rbf';
            const kernel: KernelSpec =
                ktype === 'poly'
                    ? { type: 'poly', gamma: options.kernel?.gamma ?? (1 / Math.max(1, D)), degree: options.kernel?.degree ?? 2, coef0: options.kernel?.coef0 ?? 1 }
                    : ktype === 'linear'
                        ? { type: 'linear' }
                        : ktype === 'laplacian'
                            ? { type: 'laplacian', gamma: options.kernel?.gamma ?? (1 / Math.max(1, D)) }
                            : { type: 'rbf', gamma: options.kernel?.gamma ?? (1 / Math.max(1, D)) };

            this.model = new KernelELM({
                outputDim: this.categories.length,
                kernel,
                ridgeLambda,
                task: 'classification',
                mode: 'nystrom',
                nystrom: {
                    m: options.kernel?.m,
                    strategy: options.kernel?.strategy ?? 'uniform',
                    seed: options.kernel?.seed ?? 1337,
                    preset: options.kernel?.preset,
                    whiten: options.kernel?.whiten ?? true,
                    jitter: options.kernel?.jitter ?? 1e-10,
                },
                log: { modelName: 'AutoComplete-KELM', verbose }
            });

        } else if (this.engine === 'online') {
            const inputDim = this.encoder.getVectorSize();
            this.model = new OnlineELM({
                inputDim,
                outputDim: this.categories.length,
                hiddenUnits,
                activation: this.activation,
                ridgeLambda,
                weightInit: (weightInit as any) ?? 'he',
                forgettingFactor: 0.997,
                log: { modelName: 'AutoComplete-OnlineELM', verbose }
            });

        } else {
            // Classic ELM — use TextConfig branch explicitly
            this.model = new ELM({
                categories: this.categories,
                hiddenUnits,
                activation: this.activation,
                ridgeLambda,
                weightInit: weightInit === 'he' ? 'xavier' : weightInit, // map 'he' to 'xavier' if needed
                // Text branch fields:
                useTokenizer: true,
                maxLen: PRESET.maxLen,
                charSet: PRESET.charSet,
                tokenizerDelimiter: PRESET.tokenizerDelimiter,
                // Logging / export
                metrics: options.metrics,
                log: { modelName: 'AutoComplete', verbose },
                exportFileName: options.exportFileName
            } as any);
        }

        // Bind UI to a small adapter that calls our predict()
        bindAutocompleteUI({
            model: {
                predict: (text: string, k = this.topKDefault) => this.predict(text, k)
            } as any,
            inputElement: options.inputElement,
            outputElement: options.outputElement,
            topK: options.topK
        });
    }

    /* ============= Training ============= */

    public train(): void {
        // Build numeric X/Y
        const X: number[][] = [];
        const Y: number[][] = [];

        for (const { input, label } of this.trainPairs) {
            const vec = this.encoder.normalize(this.encoder.encode(input));
            const idx = this.categories.indexOf(label);
            if (idx === -1) continue;
            X.push(vec);
            Y.push(oneHot(idx, this.categories.length));
        }

        if (this.engine === 'kernel') {
            (this.model as KernelELM).fit(X, Y);
            return;
        }
        if (this.engine === 'online') {
            (this.model as OnlineELM).init(X, Y); // then .update() for new batches
            return;
        }

        // Classic ELM — options: { reuseWeights?, weights? }; do NOT pass "task"
        (this.model as ELM).trainFromData(X, Y);
    }

    /* ============= Prediction ============= */

    public predict(input: string, topN = 1): { completion: string; prob: number }[] {
        const k = Math.max(1, topN);

        if (this.engine === 'elm') {
            const out = (this.model as ELM).predict(input, k);
            return out.map(p => ({ completion: p.label, prob: p.prob }));
        }

        const x = this.encoder.normalize(this.encoder.encode(input));

        if (this.engine === 'kernel') {
            const probs = (this.model as KernelELM).predictProbaFromVectors([x])[0];
            return sortTopK(this.categories, probs, k).map(p => ({ completion: p.label, prob: p.prob }));
        }

        const probs = (this.model as OnlineELM).predictProbaFromVector(x);
        return sortTopK(this.categories, probs, k).map(p => ({ completion: p.label, prob: p.prob }));
    }

    /* ============= Persistence ============= */

    public getModel(): AnyModel { return this.model; }

    public loadModelFromJSON(json: string): void {
        if ((this.model as any).fromJSON) {
            (this.model as KernelELM).fromJSON(json);
        } else if ((this.model as any).loadModelFromJSON) {
            (this.model as ELM).loadModelFromJSON(json);
        } else if ((this.model as any).loadFromJSON) {
            (this.model as OnlineELM).loadFromJSON(json as any);
        } else {
            console.warn('No compatible load method found on model.');
        }
    }

    public saveModelAsJSONFile(filename = 'model.json'): void {
        let payload: any;
        if ((this.model as any).toJSON) {
            payload = (this.model as any).toJSON(true); // OnlineELM supports includeP; KernelELM ignores extra arg
        } else if ((this.model as any).savedModelJSON) {
            payload = (this.model as ELM).savedModelJSON;
        } else {
            console.warn('No compatible toJSON/savedModelJSON on model; skipping export.');
            return;
        }
        const blob = new Blob([typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /* ============= Evaluation helpers ============= */

    public top1Accuracy(pairs: TrainPair[]): number {
        let correct = 0;
        for (const { input, label } of pairs) {
            const [pred] = this.predict(input, 1);
            if (pred?.completion?.toLowerCase().trim() === label.toLowerCase().trim()) correct++;
        }
        return correct / Math.max(1, pairs.length);
    }

    public crossEntropy(pairs: TrainPair[]): number {
        let total = 0;
        for (const { input, label } of pairs) {
            const preds = this.predict(input, this.categories.length);
            const match = preds.find(p => p.completion.toLowerCase().trim() === label.toLowerCase().trim());
            const prob = match?.prob ?? 1e-12;
            total += -Math.log(prob);
        }
        return total / Math.max(1, pairs.length);
    }

    /** Internal CE via W/b/β (only for classic ELM); others fall back to external CE. */
    public internalCrossEntropy(verbose = false): number {
        if (!(this.model instanceof ELM)) {
            const ce = this.crossEntropy(this.trainPairs);
            if (verbose) console.log(`📏 Internal CE not applicable to ${this.engine}; external CE: ${ce.toFixed(4)}`);
            return ce;
        }

        const elm = this.model as ELM;
        const { model, categories } = elm;
        if (!model) {
            if (verbose) console.warn('⚠️ Cannot compute internal cross-entropy: model not trained.');
            return Infinity;
        }

        const X: number[][] = [];
        const Y: number[][] = [];
        for (const { input, label } of this.trainPairs) {
            const vec = this.encoder.normalize(this.encoder.encode(input));
            const idx = categories.indexOf(label);
            if (idx === -1) continue;
            X.push(vec);
            Y.push(oneHot(idx, categories.length));
        }

        const { W, b, beta } = model; // W: hidden x in, b: hidden x 1, beta: hidden x out
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const act = Activations.get(this.activation);
        const H = tempH.map(row => row.map((v, j) => act(v + b[j][0])));
        const logits = Matrix.multiply(H, beta);
        const probs = logits.map(row => Activations.softmax(row));

        let total = 0;
        for (let i = 0; i < Y.length; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                if (Y[i][j] === 1) {
                    const p = Math.min(Math.max(probs[i][j], 1e-15), 1 - 1e-15);
                    total += -Math.log(p);
                }
            }
        }
        const ce = total / Math.max(1, Y.length);
        if (verbose) console.log(`📏 Internal Cross-Entropy (ELM W/b/β): ${ce.toFixed(4)}`);
        return ce;
    }
}
