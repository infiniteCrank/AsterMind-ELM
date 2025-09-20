// EncoderELM.ts — string→vector encoder using ELM (batch) + OnlineELM (incremental)

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { OnlineELM } from '../core/OnlineELM';

export class EncoderELM {
    public elm: ELM;
    private config: ELMConfig;

    // Online (OS-ELM / RLS) state
    private online?: OnlineELM;
    private onlineInputDim?: number;
    private onlineOutputDim?: number;

    constructor(config: ELMConfig) {
        if (typeof (config as any).hiddenUnits !== 'number') {
            throw new Error('EncoderELM requires config.hiddenUnits (number).');
        }
        if (!(config as any).activation) {
            throw new Error('EncoderELM requires config.activation.');
        }

        // Force text-encoder mode by default (safe even if NumericConfig is passed:
        // ELM will ignore tokenizer fields in numeric flows)
        this.config = {
            ...config,
            categories: (config as any).categories ?? [], // encoder has no labels
            useTokenizer: (config as any).useTokenizer ?? true,
            // keep charSet/maxLen if caller provided; otherwise ELM defaults will kick in
            log: {
                modelName: 'EncoderELM',
                verbose: config.log?.verbose ?? false,
                toFile: config.log?.toFile ?? false,
                level: (config.log as any)?.level ?? 'info',
            },
        } as ELMConfig;

        this.elm = new ELM(this.config);

        // Forward thresholds/file export if present
        if ((config as any).metrics) (this.elm as any).metrics = (config as any).metrics;
        if ((config as any).exportFileName) (this.elm as any).config.exportFileName = (config as any).exportFileName;
    }

    /** Batch training for string → dense vector mapping. */
    public train(inputStrings: string[], targetVectors: number[][]): void {
        if (!inputStrings?.length || !targetVectors?.length) {
            throw new Error('train: empty inputs');
        }
        if (inputStrings.length !== targetVectors.length) {
            throw new Error('train: inputStrings and targetVectors lengths differ');
        }

        const enc = (this.elm as any).encoder;
        if (!enc || typeof enc.encode !== 'function') {
            throw new Error('EncoderELM: underlying ELM has no encoder; set useTokenizer/maxLen/charSet in config.');
        }

        // X = normalized encoded text; Y = dense targets
        const X: number[][] = inputStrings.map(s => enc.normalize(enc.encode(s)));
        const Y: number[][] = targetVectors;

        // Closed-form solve via ELM
        // (ELM learns W,b randomly and solves β; Y can be any numeric outputDim)
        (this.elm as any).trainFromData(X, Y);
    }

    /** Encode a string into a dense feature vector using the trained model. */
    public encode(text: string): number[] {
        const enc = (this.elm as any).encoder;
        if (!enc || typeof enc.encode !== 'function') {
            throw new Error('encode: underlying ELM has no encoder');
        }
        const model = (this.elm as any).model;
        if (!model) throw new Error('EncoderELM model has not been trained yet.');

        const x = enc.normalize(enc.encode(text));      // 1 x D
        const { W, b, beta } = model as { W: number[][]; b: number[][]; beta: number[][] };

        // H = act( x W^T + b )
        const tempH = Matrix.multiply([x], Matrix.transpose(W));
        const act = Activations.get(((this.config as any).activation as Activation) ?? 'relu');
        const H = Activations.apply(
            tempH.map(row => row.map((v, j) => v + b[j][0])),
            act
        );

        // y = H β
        return Matrix.multiply(H, beta)[0];
    }

    /* ===================== Online / Incremental API ===================== */

    /**
     * Begin an online OS-ELM run for string→vector encoding.
     * Provide outputDim and either inputDim OR a sampleText we can encode to infer inputDim.
     */
    public beginOnline(opts: {
        outputDim: number;
        inputDim?: number;
        sampleText?: string;
        hiddenUnits?: number;
        ridgeLambda?: number;
        activation?: Activation;         // defaults to config.activation
        weightInit?: 'uniform' | 'xavier' | 'he';
        forgettingFactor?: number;       // ρ in (0,1], default 1
        seed?: number;
    }): void {
        const outputDim = opts.outputDim | 0;
        if (!(outputDim > 0)) throw new Error('beginOnline: outputDim must be > 0');

        // Derive inputDim if not provided
        let inputDim = opts.inputDim;
        if (inputDim == null) {
            const enc = (this.elm as any).encoder;
            if (!opts.sampleText || !enc) {
                throw new Error('beginOnline: provide inputDim or sampleText (and ensure encoder is available).');
            }
            inputDim = enc.normalize(enc.encode(opts.sampleText)).length;
        }

        const hiddenUnits = (opts.hiddenUnits ?? (this.config as any).hiddenUnits) | 0;
        if (!(hiddenUnits > 0)) throw new Error('beginOnline: hiddenUnits must be > 0');

        const activation = (opts.activation ?? (this.config as any).activation ?? 'relu') as Activation;

        // Build OnlineELM with our new config-style constructor
        this.online = new OnlineELM({
            inputDim: inputDim!,
            outputDim,
            hiddenUnits,
            activation,
            ridgeLambda: opts.ridgeLambda ?? 1e-2,
            weightInit: opts.weightInit ?? 'xavier',
            forgettingFactor: opts.forgettingFactor ?? 1.0,
            seed: opts.seed ?? 1337,
            log: { verbose: this.config.log?.verbose ?? false, modelName: 'EncoderELM-Online' },
        });

        this.onlineInputDim = inputDim!;
        this.onlineOutputDim = outputDim;
    }

    /**
     * Online partial fit with *pre-encoded* numeric vectors.
     * If not initialized, this call seeds the model via `init`, else it performs an `update`.
     */
    public partialTrainOnlineVectors(batch: Array<{ x: number[]; y: number[] }>): void {
        if (!this.online || this.onlineInputDim == null || this.onlineOutputDim == null) {
            throw new Error('partialTrainOnlineVectors: call beginOnline() first.');
        }
        if (!batch?.length) return;

        const D = this.onlineInputDim, O = this.onlineOutputDim;

        const X: number[][] = new Array(batch.length);
        const Y: number[][] = new Array(batch.length);

        for (let i = 0; i < batch.length; i++) {
            const { x, y } = batch[i];
            if (x.length !== D) throw new Error(`x length ${x.length} != inputDim ${D}`);
            if (y.length !== O) throw new Error(`y length ${y.length} != outputDim ${O}`);
            X[i] = x;
            Y[i] = y;
        }

        if (!(this.online as any).beta || !(this.online as any).P) {
            this.online.init(X, Y);
        } else {
            this.online.update(X, Y);
        }
    }

    /**
     * Online partial fit with raw texts and dense numeric targets.
     * Texts are encoded + normalized internally.
     */
    public partialTrainOnlineTexts(batch: Array<{ text: string; target: number[] }>): void {
        if (!this.online || this.onlineInputDim == null || this.onlineOutputDim == null) {
            throw new Error('partialTrainOnlineTexts: call beginOnline() first.');
        }
        if (!batch?.length) return;

        const enc = (this.elm as any).encoder;
        if (!enc) throw new Error('partialTrainOnlineTexts: encoder not available on underlying ELM');

        const D = this.onlineInputDim, O = this.onlineOutputDim;

        const X: number[][] = new Array(batch.length);
        const Y: number[][] = new Array(batch.length);

        for (let i = 0; i < batch.length; i++) {
            const { text, target } = batch[i];
            const x = enc.normalize(enc.encode(text));
            if (x.length !== D) throw new Error(`encoded text dim ${x.length} != inputDim ${D}`);
            if (target.length !== O) throw new Error(`target length ${target.length} != outputDim ${O}`);
            X[i] = x;
            Y[i] = target;
        }

        if (!(this.online as any).beta || !(this.online as any).P) {
            this.online.init(X, Y);
        } else {
            this.online.update(X, Y);
        }
    }

    /**
     * Finalize the online run by publishing learned weights into the standard ELM model.
     * After this, the normal encode() path works unchanged.
     */
    public endOnline(): void {
        if (!this.online) return;

        const W = (this.online as any).W as number[][];
        const b = (this.online as any).b as number[][];
        const beta = (this.online as any).beta as number[][];

        if (!W || !b || !beta) {
            throw new Error('endOnline: online model has no learned parameters (did you call init/fit/update?)');
        }

        (this.elm as any).model = { W, b, beta };

        // Clear online state
        this.online = undefined;
        this.onlineInputDim = undefined;
        this.onlineOutputDim = undefined;
    }

    /* ===================== I/O passthrough ===================== */

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
