import { ELM } from '../core/ELM';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { ELMConfig } from '../core/ELMConfig';
import { IO, LabeledExample } from '../utils/IO';
import { OnlineELM, Activation as OnlineAct } from '../core/OnlineELM'; // adjust path


export class LanguageClassifier {
    private elm: ELM;
    private config: ELMConfig;
    private trainSamples: Record<string, string[]> = {};
    private onlineMdl?: OnlineELM;          // Online ELM engine (typed arrays)
    private onlineCats?: string[];          // fixed categories for the online run
    private onlineInputDim?: number;        // feature vector length for this run

    constructor(config: ELMConfig) {
        this.config = {
            ...config,
            log: {
                modelName: "IntentClassifier",
                verbose: config.log.verbose
            },
        };
        this.elm = new ELM(config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    loadTrainingData(raw: string, format: 'json' | 'csv' | 'tsv' = 'json'): LabeledExample[] {
        switch (format) {
            case 'csv':
                return IO.importCSV(raw);
            case 'tsv':
                return IO.importTSV(raw);
            case 'json':
            default:
                return IO.importJSON(raw);
        }
    }

    train(data: LabeledExample[]): void {
        const categories = [...new Set(data.map(d => d.label))];
        this.elm.setCategories(categories);
        data.forEach(({ text, label }) => {
            if (!this.trainSamples[label]) this.trainSamples[label] = [];
            this.trainSamples[label].push(text);
        });
        this.elm.train();
    }

    predict(text: string, topK = 3) {
        return this.elm.predict(text, topK);
    }

    /**
     * Train the classifier using already-encoded vectors.
     * Each vector must be paired with its label.
     */
    trainVectors(data: { vector: number[]; label: string }[]) {
        const categories = [...new Set(data.map(d => d.label))];
        this.elm.setCategories(categories);

        const X: number[][] = data.map(d => d.vector);
        const Y: number[][] = data.map(d =>
            this.elm.oneHot(categories.length, categories.indexOf(d.label))
        );

        const W = this.elm['randomMatrix'](this.config.hiddenUnits!, X[0].length);
        const b = this.elm['randomMatrix'](this.config.hiddenUnits!, 1);
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);
        const H_pinv = this.elm['pseudoInverse'](H);
        const beta = Matrix.multiply(H_pinv, Y);

        this.elm['model'] = { W, b, beta };
    }

    /**
     * Predict language directly from a dense vector representation.
     */
    predictFromVector(vec: number[], topK = 1) {
        const model = this.elm['model'];
        if (!model) {
            throw new Error('EncoderELM model has not been trained yet.');
        }

        const { W, b, beta } = model;
        const tempH = Matrix.multiply([vec], Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const rawOutput = Matrix.multiply(H, beta)[0];
        const probs = Activations.softmax(rawOutput);

        return probs
            .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    public beginOnline(opts: {
        categories: string[];
        inputDim: number;
        hiddenUnits?: number;
        lambda?: number;
        activation?: NonNullable<ELMConfig['activation']>;
    }): void {
        const categories = opts.categories.slice();
        const inputDim = opts.inputDim;
        const hiddenUnits = opts.hiddenUnits ?? (this.config.hiddenUnits as number);
        const outputDim = categories.length;
        const lambda = opts.lambda ?? 1e-2;
        const actName = opts.activation ?? (this.config.activation as NonNullable<ELMConfig['activation']>);

        // map your activation names to a function for OnlineELM
        const act: OnlineAct =
            actName === 'tanh' ? Math.tanh
                : actName === 'sigmoid' ? (x: number) => 1 / (1 + Math.exp(-x))
                    : (x: number) => (x > 0 ? x : 0); // relu default

        this.onlineMdl = new OnlineELM(inputDim, hiddenUnits, outputDim, act, lambda);
        this.onlineCats = categories;
        this.onlineInputDim = inputDim;
    }
    public partialTrainVectorsOnline(batch: { vector: number[]; label: string }[]): void {
        if (!this.onlineMdl || !this.onlineCats || !this.onlineInputDim) {
            throw new Error('Call beginOnline() before partialTrainVectorsOnline().');
        }
        if (!batch.length) return;

        const B = batch.length;
        const D = this.onlineInputDim;
        const O = this.onlineCats.length;
        const H = this.onlineMdl.hiddenUnits;

        // Pack X [B x D] and T [B x O] into typed arrays
        const X = new Float64Array(B * D);
        const T = new Float64Array(B * O);

        for (let i = 0; i < B; i++) {
            const { vector, label } = batch[i];
            if (vector.length !== D) {
                throw new Error(`vector dim ${vector.length} != inputDim ${D}`);
            }
            // X row
            X.set(vector, i * D);
            // one-hot T row
            const li = this.onlineCats.indexOf(label);
            if (li < 0) throw new Error(`Unknown label "${label}" for this online run.`);
            T[i * O + li] = 1;
        }

        this.onlineMdl.partialFit(X, T, B);
    }

    public endOnline(): void {
        if (!this.onlineMdl || !this.onlineCats) return;

        // Convert typed arrays -> number[][] expected by existing ELM paths
        const H = this.onlineMdl.hiddenUnits;
        const D = this.onlineMdl.inputDim;
        const O = this.onlineMdl.outputDim;

        const W = this.reshapeTo2D(this.onlineMdl.W, H, D);      // [H x D]
        const b = this.reshapeCol(this.onlineMdl.b, H);          // [H x 1]
        const beta = this.reshapeTo2D(this.onlineMdl.beta, H, O);// [H x O]

        // Activate categories for this classifier and publish the model
        this.elm.setCategories(this.onlineCats);
        this.elm['model'] = { W, b, beta };

        // clear online state
        this.onlineMdl = undefined;
        this.onlineCats = undefined;
        this.onlineInputDim = undefined;
    }
    private reshapeTo2D(buf: Float64Array, rows: number, cols: number): number[][] {
        const out: number[][] = new Array(rows);
        for (let r = 0; r < rows; r++) {
            const row: number[] = new Array(cols);
            for (let c = 0; c < cols; c++) row[c] = buf[r * cols + c];
            out[r] = row;
        }
        return out;
    }

    private reshapeCol(buf: Float64Array, rows: number): number[][] {
        const out: number[][] = new Array(rows);
        for (let r = 0; r < rows; r++) out[r] = [buf[r]];
        return out;
    }

}