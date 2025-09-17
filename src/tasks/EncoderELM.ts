import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { OnlineELM, Activation as OnlineAct } from '../core/OnlineELM';

/**
 * EncoderELM: Uses an ELM to convert strings into dense feature vectors.
 */
export class EncoderELM {
    public elm: ELM;
    private config: ELMConfig;
    // ===== NEW: online run state (purely additive) =====
    private onlineMdl?: OnlineELM;
    private onlineInputDim?: number;
    private onlineOutputDim?: number;

    constructor(config: ELMConfig) {
        if (typeof config.hiddenUnits !== 'number') {
            throw new Error('EncoderELM requires config.hiddenUnits to be defined as a number');
        }
        if (!config.activation) {
            throw new Error('EncoderELM requires config.activation to be defined');
        }

        this.config = {
            ...config,
            categories: [],
            useTokenizer: true,
            log: {
                modelName: "EncoderELM",
                verbose: config.log.verbose
            },
        };

        this.elm = new ELM(this.config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    /**
     * Custom training method for string → vector encoding.
     */
    train(inputStrings: string[], targetVectors: number[][]): void {
        const X: number[][] = inputStrings.map(s =>
            this.elm.encoder.normalize(this.elm.encoder.encode(s))
        );
        const Y = targetVectors;

        const hiddenUnits = this.config.hiddenUnits!;
        const inputDim = X[0].length;

        const W = this.elm['randomMatrix'](hiddenUnits, inputDim);
        const b = this.elm['randomMatrix'](hiddenUnits, 1);

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
     * Encodes an input string into a dense feature vector using the trained model.
     */
    encode(text: string): number[] {
        const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
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

        return Matrix.multiply(H, beta)[0];
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    // ===================== NEW: Online / Incremental encoding API =====================

    /**
     * Initialize an online (OS-ELM/RLS) run for string→vector encoding.
     *
     * Provide outputDim, and EITHER inputDim OR a sampleText we can encode to infer inputDim.
     * hiddenUnits defaults to config.hiddenUnits; activation defaults to config.activation; lambda defaults to 1e-2.
     */
    public beginOnline(opts: {
        outputDim: number;
        inputDim?: number;
        sampleText?: string;
        hiddenUnits?: number;
        lambda?: number;
        activation?: NonNullable<ELMConfig['activation']>;
    }): void {
        const outputDim = opts.outputDim;
        if (!Number.isFinite(outputDim) || outputDim <= 0) {
            throw new Error('beginOnline: outputDim must be a positive integer.');
        }

        let inputDim = opts.inputDim;
        if (inputDim == null) {
            if (!opts.sampleText) {
                throw new Error('beginOnline: provide either inputDim or sampleText to infer dimension.');
            }
            const probe = this.elm.encoder.normalize(this.elm.encoder.encode(opts.sampleText));
            inputDim = probe.length;
        }

        const hiddenUnits = opts.hiddenUnits ?? (this.config.hiddenUnits as number);
        const lambda = opts.lambda ?? 1e-2;
        const actName = opts.activation ?? (this.config.activation as NonNullable<ELMConfig['activation']>);

        const act: OnlineAct =
            actName === 'tanh' ? Math.tanh
                : actName === 'sigmoid' ? (x: number) => 1 / (1 + Math.exp(-x))
                    : (x: number) => (x > 0 ? x : 0); // relu default

        // Spin up the typed-array OnlineELM
        this.onlineMdl = new OnlineELM(inputDim, hiddenUnits, outputDim, act, lambda);
        this.onlineInputDim = inputDim;
        this.onlineOutputDim = outputDim;
    }

    /**
     * Online partial fit with pre-encoded vectors.
     * Each batch element supplies x (length = inputDim) and y (length = outputDim).
     * Memory-friendly: pack into typed arrays, update, discard.
     */
    public partialTrainOnlineVectors(batch: { x: number[]; y: number[] }[]): void {
        if (!this.onlineMdl || this.onlineInputDim == null || this.onlineOutputDim == null) {
            throw new Error('partialTrainOnlineVectors: call beginOnline() first.');
        }
        if (!batch.length) return;

        const B = batch.length;
        const D = this.onlineInputDim;
        const O = this.onlineOutputDim;

        const X = new Float64Array(B * D);
        const T = new Float64Array(B * O);

        for (let i = 0; i < B; i++) {
            const { x, y } = batch[i];
            if (x.length !== D) throw new Error(`x length ${x.length} != inputDim ${D}`);
            if (y.length !== O) throw new Error(`y length ${y.length} != outputDim ${O}`);
            X.set(x, i * D);
            T.set(y, i * O);
        }

        this.onlineMdl.partialFit(X, T, B);
    }

    /**
     * Online partial fit with raw texts.
     * We encode+normalize each text internally to build X, and use the given dense target vector y.
     */
    public partialTrainOnlineTexts(batch: { text: string; target: number[] }[]): void {
        if (!this.onlineMdl || this.onlineInputDim == null || this.onlineOutputDim == null) {
            throw new Error('partialTrainOnlineTexts: call beginOnline() first.');
        }
        if (!batch.length) return;

        const B = batch.length;
        const D = this.onlineInputDim;
        const O = this.onlineOutputDim;

        const X = new Float64Array(B * D);
        const T = new Float64Array(B * O);

        for (let i = 0; i < B; i++) {
            const { text, target } = batch[i];
            const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
            if (vec.length !== D) throw new Error(`encoded text dim ${vec.length} != inputDim ${D}`);
            if (target.length !== O) throw new Error(`target length ${target.length} != outputDim ${O}`);
            X.set(vec, i * D);
            T.set(target, i * O);
        }

        this.onlineMdl.partialFit(X, T, B);
    }

    /**
     * Finalize the online run by publishing learned weights into the standard model shape.
     * After this, the normal encode() path works unchanged.
     */
    public endOnline(): void {
        if (!this.onlineMdl) return;

        const H = this.onlineMdl.hiddenUnits;
        const D = this.onlineMdl.inputDim;
        const O = this.onlineMdl.outputDim;

        const W = this.reshapeTo2D(this.onlineMdl.W, H, D);       // [H x D]
        const b = this.reshapeCol(this.onlineMdl.b, H);           // [H x 1]
        const beta = this.reshapeTo2D(this.onlineMdl.beta, H, O); // [H x O]

        this.elm['model'] = { W, b, beta };

        // clear online state
        this.onlineMdl = undefined;
        this.onlineInputDim = undefined;
        this.onlineOutputDim = undefined;
    }

    // ===================== small helpers =====================

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
