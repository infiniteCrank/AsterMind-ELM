// LanguageClassifier.ts — upgraded for new ELM/OnlineELM APIs (with requireEncoder guard)

import { ELM } from '../core/ELM';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { ELMConfig, Activation } from '../core/ELMConfig';
import { IO, LabeledExample } from '../utils/IO';
import { OnlineELM } from '../core/OnlineELM';

export class LanguageClassifier {
    private elm: ELM;
    private config: ELMConfig;

    // Online (incremental) state
    private onlineMdl?: OnlineELM;
    private onlineCats?: string[];
    private onlineInputDim?: number;

    constructor(config: ELMConfig) {
        this.config = {
            ...config,
            log: {
                modelName: 'LanguageClassifier',
                verbose: config.log?.verbose ?? false,
                toFile: config.log?.toFile ?? false,
                level: config.log?.level ?? 'info',
            },
        };

        this.elm = new ELM(this.config);

        if ((config as any).metrics) this.elm.metrics = (config as any).metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    /* ============== tiny helper to guarantee an encoder ============== */
    private requireEncoder(): { encode: (s: string) => number[]; normalize: (v: number[]) => number[] } {
        const enc = (this.elm as any).encoder as
            | { encode: (s: string) => number[]; normalize: (v: number[]) => number[] }
            | undefined;

        if (!enc) {
            throw new Error(
                'LanguageClassifier: encoder unavailable. Use text mode (useTokenizer=true with maxLen/charSet) ' +
                'or pass a UniversalEncoder in the ELM config.'
            );
        }
        return enc;
    }

    /* ================= I/O helpers ================= */

    loadTrainingData(raw: string, format: 'json' | 'csv' | 'tsv' = 'json'): LabeledExample[] {
        switch (format) {
            case 'csv': return IO.importCSV(raw);
            case 'tsv': return IO.importTSV(raw);
            case 'json':
            default: return IO.importJSON(raw);
        }
    }

    /* ================= Supervised training ================= */

    /** Train from labeled text examples (uses internal encoder). */
    train(data: LabeledExample[]): void {
        if (!data?.length) throw new Error('LanguageClassifier.train: empty dataset');

        const enc = this.requireEncoder();
        const categories = Array.from(new Set(data.map(d => d.label)));
        this.elm.setCategories(categories);

        const X: number[][] = [];
        const Y: number[][] = [];

        for (const { text, label } of data) {
            const x = enc.normalize(enc.encode(text));
            const yi = categories.indexOf(label);
            if (yi < 0) continue;
            X.push(x);
            Y.push(this.elm.oneHot(categories.length, yi));
        }

        this.elm.trainFromData(X, Y);
    }

    /** Predict from raw text (uses internal encoder). */
    predict(text: string, topK = 3) {
        // let ELM handle encode→predict (works in text mode)
        return this.elm.predict(text, topK);
    }

    /** Train using already-encoded numeric vectors (no text encoder). */
    trainVectors(data: { vector: number[]; label: string }[]): void {
        if (!data?.length) throw new Error('LanguageClassifier.trainVectors: empty dataset');

        const categories = Array.from(new Set(data.map(d => d.label)));
        this.elm.setCategories(categories);

        const X: number[][] = data.map(d => d.vector);
        const Y: number[][] = data.map(d =>
            this.elm.oneHot(categories.length, categories.indexOf(d.label))
        );

        if (typeof (this.elm as any).trainFromData === 'function') {
            (this.elm as any).trainFromData(X, Y);
            return;
        }

        // Fallback closed-form (compat)
        const hidden = this.config.hiddenUnits as number;
        const W = (this.elm as any).randomMatrix(hidden, X[0].length);
        const b = (this.elm as any).randomMatrix(hidden, 1);
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const act = Activations.get((this.config.activation as Activation) ?? 'relu');
        const H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            act
        );
        const Hpinv = (this.elm as any).pseudoInverse(H);
        const beta = Matrix.multiply(Hpinv, Y);
        (this.elm as any).model = { W, b, beta };
    }

    /** Predict from an already-encoded vector (no text encoder). */
    predictFromVector(vec: number[], topK = 1) {
        const out = this.elm.predictFromVector([vec], topK);
        return out[0];
    }

    /* ================= Online (incremental) API ================= */

    public beginOnline(opts: {
        categories: string[];
        inputDim: number;
        hiddenUnits?: number;
        lambda?: number;
        activation?: Activation;
        weightInit?: 'uniform' | 'xavier' | 'he';
        forgettingFactor?: number; // ρ in (0,1]; 1 = no forgetting
        seed?: number;
    }): void {
        const cats = opts.categories.slice();
        const D = opts.inputDim | 0;
        if (!cats.length) throw new Error('beginOnline: categories must be non-empty');
        if (D <= 0) throw new Error('beginOnline: inputDim must be > 0');

        const H = (opts.hiddenUnits ?? (this.config.hiddenUnits as number)) | 0;
        if (H <= 0) throw new Error('beginOnline: hiddenUnits must be > 0');

        const activation = opts.activation ?? (this.config.activation as Activation) ?? 'relu';
        const ridgeLambda = Math.max(opts.lambda ?? 1e-2, 1e-12);

        this.onlineMdl = new OnlineELM({
            inputDim: D,
            outputDim: cats.length,
            hiddenUnits: H,
            activation,
            ridgeLambda,
            seed: opts.seed ?? 1337,
            weightInit: opts.weightInit ?? 'xavier',
            forgettingFactor: opts.forgettingFactor ?? 1.0,
            log: { verbose: this.config.log?.verbose ?? false, modelName: 'LanguageClassifier/Online' },
        });

        this.onlineCats = cats;
        this.onlineInputDim = D;
    }

    public partialTrainVectorsOnline(batch: { vector: number[]; label: string }[]): void {
        if (!this.onlineMdl || !this.onlineCats || !this.onlineInputDim) {
            throw new Error('Call beginOnline() before partialTrainVectorsOnline().');
        }
        if (!batch.length) return;

        const D = this.onlineInputDim;
        const O = this.onlineCats.length;

        const X: number[][] = new Array(batch.length);
        const Y: number[][] = new Array(batch.length);

        for (let i = 0; i < batch.length; i++) {
            const { vector, label } = batch[i];
            if (vector.length !== D) throw new Error(`vector dim ${vector.length} != inputDim ${D}`);
            X[i] = vector.slice();

            const y = new Array(O).fill(0);
            const li = this.onlineCats.indexOf(label);
            if (li < 0) throw new Error(`Unknown label "${label}" for this online run.`);
            y[li] = 1;
            Y[i] = y;
        }

        if ((this.onlineMdl as any).beta && (this.onlineMdl as any).P) {
            this.onlineMdl.update(X, Y);
        } else {
            this.onlineMdl.init(X, Y);
        }
    }

    public endOnline(): void {
        if (!this.onlineMdl || !this.onlineCats) return;

        const W = (this.onlineMdl as any).W as number[][];
        const b = (this.onlineMdl as any).b as number[][];
        const B = (this.onlineMdl as any).beta as number[][];

        if (!W || !b || !B) throw new Error('endOnline: online model is not initialized.');

        this.elm.setCategories(this.onlineCats);
        (this.elm as any).model = { W, b, beta: B };

        this.onlineMdl = undefined;
        this.onlineCats = undefined;
        this.onlineInputDim = undefined;
    }

    /* ================= Persistence ================= */

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
