// ELM.ts - Core ELM logic with TypeScript types (numeric & text modes)

import { Matrix, ensureRectNumber2D } from './Matrix';
import { Activations } from './Activations';
import {
    ELMConfig,
    SerializedELM,
    TrainResult,
    isTextConfig,
    normalizeConfig,
    deserializeTextBits,
} from './ELMConfig';
import { UniversalEncoder } from '../preprocessing/UniversalEncoder';
import { Augment } from '../utils/Augment';

/* =========================
 * Types
 * ========================= */

export interface ELMModel {
    W: number[][];    // hiddenUnits x inputDim
    b: number[][];    // hiddenUnits x 1
    beta: number[][]; // hiddenUnits x outDim
}

export interface PredictResult {
    label: string;
    prob: number;
}

export interface TopKResult {
    index: number;
    label: string;
    prob: number;
}

/* =========================
 * Small utils
 * ========================= */

const EPS = 1e-8;

// Seeded PRNG (xorshift-ish) for deterministic init
function makePRNG(seed = 123456789) {
    let s = seed | 0 || 1;
    return () => {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
        return ((s >>> 0) / 0xffffffff);
    };
}

function clampInt(x: number, lo: number, hi: number) {
    const xi = x | 0;
    return xi < lo ? lo : (xi > hi ? hi : xi);
}

function isOneHot2D(Y: any): Y is number[][] {
    return Array.isArray(Y) && Array.isArray(Y[0]) && Number.isFinite(Y[0][0]);
}

function maxLabel(y: number[]): number {
    let m = -Infinity;
    for (let i = 0; i < y.length; i++) {
        const v = y[i] | 0;
        if (v > m) m = v;
    }
    return m === -Infinity ? 0 : m;
}

/** One-hot (clamped) */
function toOneHotClamped(labels: number[], k: number): number[][] {
    const K = k | 0;
    const Y = new Array(labels.length);
    for (let i = 0; i < labels.length; i++) {
        const j = clampInt(labels[i], 0, K - 1);
        const row = new Array(K).fill(0);
        row[j] = 1;
        Y[i] = row;
    }
    return Y;
}

/** (HᵀH + λI)B = HᵀY solved via Cholesky */
function ridgeSolve(H: number[][], Y: number[][], lambda: number): number[][] {
    const Ht = Matrix.transpose(H);
    const A = Matrix.addRegularization(Matrix.multiply(Ht, H), lambda + 1e-10);
    const R = Matrix.multiply(Ht, Y);
    return Matrix.solveCholesky(A, R, 1e-10);
}

/* =========================
 * ELM class
 * ========================= */

export class ELM {
    public categories: string[];
    public hiddenUnits: number;
    public activation: string;

    // Text-mode fields
    public maxLen: number;
    public charSet: string;
    public useTokenizer: boolean;
    public tokenizerDelimiter?: RegExp;
    public encoder?: UniversalEncoder;

    // Common
    public model: ELMModel | null;
    public metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
        f1?: number;
        crossEntropy?: number;
        r2?: number;
    };
    public verbose: boolean;
    public savedModelJSON?: string;
    public config: ELMConfig;
    public modelName: string;
    public logToFile: boolean;
    public dropout: number;
    public ridgeLambda: number;

    // seedable RNG
    private rng: () => number;

    constructor(config: ELMConfig) {
        // Merge with mode-appropriate defaults
        const cfg = normalizeConfig(config);

        this.config = cfg;
        this.categories = cfg.categories;
        this.hiddenUnits = cfg.hiddenUnits;
        this.activation = cfg.activation ?? 'relu';
        this.useTokenizer = isTextConfig(cfg);
        this.maxLen = isTextConfig(cfg) ? cfg.maxLen : 0;
        this.charSet = isTextConfig(cfg) ? (cfg.charSet ?? 'abcdefghijklmnopqrstuvwxyz') : 'abcdefghijklmnopqrstuvwxyz';
        this.tokenizerDelimiter = isTextConfig(cfg) ? cfg.tokenizerDelimiter : undefined;

        this.metrics = cfg.metrics;
        this.verbose = cfg.log?.verbose ?? true;
        this.modelName = cfg.log?.modelName ?? 'Unnamed ELM Model';
        this.logToFile = cfg.log?.toFile ?? false;
        this.dropout = cfg.dropout ?? 0;
        this.ridgeLambda = Math.max(cfg.ridgeLambda ?? 1e-2, 1e-8);

        // Seeded RNG
        const seed = cfg.seed ?? 1337;
        this.rng = makePRNG(seed);

        // Create encoder only if tokenizer is enabled
        if (this.useTokenizer) {
            this.encoder = new UniversalEncoder({
                charSet: this.charSet,
                maxLen: this.maxLen,
                useTokenizer: this.useTokenizer,
                tokenizerDelimiter: this.tokenizerDelimiter,
                mode: this.useTokenizer ? 'token' : 'char'
            });
        }

        // Weights are allocated on first training call (inputDim known then)
        this.model = null;
    }

    /* ========= Encoder narrowing (Option A) ========= */

    private assertEncoder(): UniversalEncoder {
        if (!this.encoder) {
            throw new Error('Encoder is not initialized. Enable useTokenizer:true or construct an encoder.');
        }
        return this.encoder;
    }

    /* ========= initialization ========= */

    private xavierLimit(fanIn: number, fanOut: number) {
        return Math.sqrt(6 / (fanIn + fanOut));
    }

    private randomMatrix(rows: number, cols: number): number[][] {
        const weightInit = this.config.weightInit ?? 'uniform';
        if (weightInit === 'xavier') {
            const limit = this.xavierLimit(cols, rows);
            if (this.verbose) console.log(`✨ Xavier init with limit sqrt(6/(${cols}+${rows})) ≈ ${limit.toFixed(4)}`);
            return Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => (this.rng() * 2 - 1) * limit)
            );
        } else {
            if (this.verbose) console.log(`✨ Uniform init [-1,1] (seeded)`);
            return Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => (this.rng() * 2 - 1))
            );
        }
    }

    private buildHidden(X: number[][], W: number[][], b: number[][]): number[][] {
        const tempH = Matrix.multiply(X, Matrix.transpose(W)); // N x hidden
        const activationFn = Activations.get(this.activation);
        let H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            activationFn
        );

        if (this.dropout > 0) {
            const keepProb = 1 - this.dropout;
            for (let i = 0; i < H.length; i++) {
                for (let j = 0; j < H[0].length; j++) {
                    if (this.rng() < this.dropout) H[i][j] = 0;
                    else H[i][j] /= keepProb;
                }
            }
        }
        return H;
    }

    /* ========= public helpers ========= */

    public oneHot(n: number, index: number): number[] {
        return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
    }

    public setCategories(categories: string[]) {
        this.categories = categories;
    }

    public loadModelFromJSON(json: string): void {
        try {
            const parsed: SerializedELM = JSON.parse(json);
            const cfg = deserializeTextBits(parsed.config);
            // Rebuild instance config
            this.config = (cfg as unknown) as ELMConfig;
            this.categories = (cfg as any).categories ?? this.categories;
            this.hiddenUnits = (cfg as any).hiddenUnits ?? this.hiddenUnits;
            this.activation = (cfg as any).activation ?? this.activation;
            this.useTokenizer = (cfg as any).useTokenizer === true;
            this.maxLen = (cfg as any).maxLen ?? this.maxLen;
            this.charSet = (cfg as any).charSet ?? this.charSet;
            this.tokenizerDelimiter = (cfg as any).tokenizerDelimiter;

            if (this.useTokenizer) {
                this.encoder = new UniversalEncoder({
                    charSet: this.charSet,
                    maxLen: this.maxLen,
                    useTokenizer: this.useTokenizer,
                    tokenizerDelimiter: this.tokenizerDelimiter,
                    mode: this.useTokenizer ? 'token' : 'char'
                });
            } else {
                this.encoder = undefined;
            }

            // Restore weights
            const { W, b, B } = parsed as any;
            this.model = { W, b, beta: B };
            this.savedModelJSON = json;
            if (this.verbose) console.log(`✅ ${this.modelName} Model loaded from JSON`);
        } catch (e) {
            console.error(`❌ Failed to load ${this.modelName} model from JSON:`, e);
        }
    }

    /* ========= Numeric training tolerance ========= */

    /** Decide output dimension from config/categories/labels/one-hot */
    private resolveOutputDim(yOrY: number[] | number[][]): number {
        // Prefer explicit config
        const cfgOut = (this.config as any).outputDim as number | undefined;
        if (Number.isFinite(cfgOut) && (cfgOut as number) > 0) return (cfgOut as number) | 0;

        // Then categories length if present
        if (Array.isArray(this.categories) && this.categories.length > 0) return this.categories.length | 0;

        // Infer from data
        if (isOneHot2D(yOrY)) return ((yOrY[0] as number[]).length | 0) || 1;
        return (maxLabel(yOrY as number[]) + 1) | 0;
    }

    /** Coerce X, and turn labels→one-hot if needed. Always returns strict number[][] */
    private coerceXY(
        X: number[][],
        yOrY: number[] | number[][]
    ): { Xnum: number[][]; Ynum: number[][]; outDim: number } {
        const Xnum = ensureRectNumber2D(X, undefined, 'X');

        const outDim = this.resolveOutputDim(yOrY);
        let Ynum: number[][];

        if (isOneHot2D(yOrY)) {
            // Ensure rect with exact width outDim (pad/trunc to be safe)
            Ynum = ensureRectNumber2D(yOrY, outDim, 'Y(one-hot)');
        } else {
            // Labels → clamped one-hot
            Ynum = ensureRectNumber2D(toOneHotClamped(yOrY as number[], outDim), outDim, 'Y(labels→one-hot)');
        }

        // If categories length mismatches inferred outDim, adjust categories (non-breaking)
        if (!this.categories || this.categories.length !== outDim) {
            this.categories = Array.from({ length: outDim }, (_, i) => this.categories?.[i] ?? String(i));
        }

        return { Xnum, Ynum, outDim };
    }

    /* ========= Training on numeric vectors =========
     * y can be class indices OR one-hot.
     */
    public trainFromData(
        X: number[][],
        y: number[] | number[][],
        options?: {
            reuseWeights?: boolean;
            weights?: number[]; // per-sample weighting
        }
    ): TrainResult {
        if (!X?.length) throw new Error('trainFromData: X is empty');

        // Coerce & shape
        const { Xnum, Ynum, outDim } = this.coerceXY(X, y);
        const n = Xnum.length;
        const inputDim = Xnum[0].length;

        // init / reuse
        let W: number[][], b: number[][];
        const reuseWeights = options?.reuseWeights === true && this.model;
        if (reuseWeights && this.model) {
            W = this.model.W; b = this.model.b;
            if (this.verbose) console.log('🔄 Reusing existing weights/biases for training.');
        } else {
            W = this.randomMatrix(this.hiddenUnits, inputDim);
            b = this.randomMatrix(this.hiddenUnits, 1);
            if (this.verbose) console.log('✨ Initializing fresh weights/biases for training.');
        }

        // Hidden
        let H = this.buildHidden(Xnum, W, b);

        // Optional sample weights
        let Yw = Ynum;
        if (options?.weights) {
            const ww = options.weights;
            if (ww.length !== n) {
                throw new Error(`Weight array length ${ww.length} does not match sample count ${n}`);
            }
            H = H.map((row, i) => row.map(x => x * Math.sqrt(ww[i])));
            Yw = Ynum.map((row, i) => row.map(x => x * Math.sqrt(ww[i])));
        }

        // Solve ridge (stable)
        const beta = ridgeSolve(H, Yw, this.ridgeLambda);
        this.model = { W, b, beta };

        // Evaluate & maybe save
        const predictions = Matrix.multiply(H, beta);

        if (this.metrics) {
            const rmse = this.calculateRMSE(Ynum, predictions);
            const mae = this.calculateMAE(Ynum, predictions);
            const acc = this.calculateAccuracy(Ynum, predictions);
            const f1 = this.calculateF1Score(Ynum, predictions);
            const ce = this.calculateCrossEntropy(Ynum, predictions);
            const r2 = this.calculateR2Score(Ynum, predictions);

            const results: Record<string, number> = { rmse, mae, accuracy: acc, f1, crossEntropy: ce, r2 };
            let allPassed = true;

            if (this.metrics.rmse !== undefined && rmse > this.metrics.rmse) allPassed = false;
            if (this.metrics.mae !== undefined && mae > this.metrics.mae) allPassed = false;
            if (this.metrics.accuracy !== undefined && acc < this.metrics.accuracy) allPassed = false;
            if ((this.metrics as any).f1 !== undefined && f1 < (this.metrics as any).f1) allPassed = false;
            if ((this.metrics as any).crossEntropy !== undefined && ce > (this.metrics as any).crossEntropy) allPassed = false;
            if (this.metrics.r2 !== undefined && r2 < this.metrics.r2) allPassed = false;

            if (this.verbose) this.logMetrics(results);

            if (allPassed) {
                this.savedModelJSON = JSON.stringify({
                    config: this.serializeConfig(),
                    W, b, B: beta
                } as SerializedELM);
                if (this.verbose) console.log('✅ Model passed thresholds and was saved to JSON.');
                if (this.config.exportFileName) this.saveModelAsJSONFile(this.config.exportFileName);
            } else {
                if (this.verbose) console.log('❌ Model not saved: One or more thresholds not met.');
            }
        } else {
            // No metrics—always save
            this.savedModelJSON = JSON.stringify({
                config: this.serializeConfig(),
                W, b, B: beta
            } as SerializedELM);
            if (this.verbose) console.log('✅ Model trained with no metrics—saved by default.');
            if (this.config.exportFileName) this.saveModelAsJSONFile(this.config.exportFileName);
        }

        return { epochs: 1, metrics: undefined };
    }

    /* ========= Training from category strings (text mode) ========= */
    public train(
        augmentationOptions?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        },
        weights?: number[]
    ): TrainResult {
        if (!this.useTokenizer) {
            throw new Error('train(): text training requires useTokenizer:true');
        }
        const enc = this.assertEncoder();

        const X: number[][] = [];
        let Y: number[][] = [];

        this.categories.forEach((cat, i) => {
            const variants = Augment.generateVariants(cat, this.charSet, augmentationOptions);
            for (const variant of variants) {
                const vec = enc.normalize(enc.encode(variant));
                X.push(vec);
                Y.push(this.oneHot(this.categories.length, i));
            }
        });

        const inputDim = X[0].length;
        const W = this.randomMatrix(this.hiddenUnits, inputDim);
        const b = this.randomMatrix(this.hiddenUnits, 1);

        let H = this.buildHidden(X, W, b);

        if (weights) {
            if (weights.length !== H.length) {
                throw new Error(`Weight array length ${weights.length} does not match sample count ${H.length}`);
            }
            H = H.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
            Y = Y.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
        }

        const beta = ridgeSolve(H, Y, this.ridgeLambda);
        this.model = { W, b, beta };

        const predictions = Matrix.multiply(H, beta);

        if (this.metrics) {
            const rmse = this.calculateRMSE(Y, predictions);
            const mae = this.calculateMAE(Y, predictions);
            const acc = this.calculateAccuracy(Y, predictions);
            const f1 = this.calculateF1Score(Y, predictions);
            const ce = this.calculateCrossEntropy(Y, predictions);
            const r2 = this.calculateR2Score(Y, predictions);

            const results: Record<string, number> = { rmse, mae, accuracy: acc, f1, crossEntropy: ce, r2 };
            let allPassed = true;

            if (this.metrics.rmse !== undefined && rmse > this.metrics.rmse) allPassed = false;
            if (this.metrics.mae !== undefined && mae > this.metrics.mae) allPassed = false;
            if (this.metrics.accuracy !== undefined && acc < this.metrics.accuracy) allPassed = false;
            if ((this.metrics as any).f1 !== undefined && f1 < (this.metrics as any).f1) allPassed = false;
            if ((this.metrics as any).crossEntropy !== undefined && ce > (this.metrics as any).crossEntropy) allPassed = false;
            if (this.metrics.r2 !== undefined && r2 < this.metrics.r2) allPassed = false;

            if (this.verbose) this.logMetrics(results);

            if (allPassed) {
                this.savedModelJSON = JSON.stringify({
                    config: this.serializeConfig(),
                    W, b, B: beta
                } as SerializedELM);
                if (this.verbose) console.log('✅ Model passed thresholds and was saved to JSON.');
                if (this.config.exportFileName) this.saveModelAsJSONFile(this.config.exportFileName);
            } else {
                if (this.verbose) console.log('❌ Model not saved: One or more thresholds not met.');
            }
        } else {
            this.savedModelJSON = JSON.stringify({
                config: this.serializeConfig(),
                W, b, B: beta
            } as SerializedELM);
            if (this.verbose) console.log('✅ Model trained with no metrics—saved by default.');
            if (this.config.exportFileName) this.saveModelAsJSONFile(this.config.exportFileName);
        }

        return { epochs: 1, metrics: undefined };
    }

    /* ========= Prediction ========= */

    /** Text prediction (uses Option A narrowing) */
    public predict(text: string, topK: number = 5): PredictResult[] {
        if (!this.model) throw new Error('Model not trained.');
        if (!this.useTokenizer) {
            throw new Error('predict(text) requires useTokenizer:true');
        }
        const enc = this.assertEncoder();
        const vec = enc.normalize(enc.encode(text));

        const logits = this.predictLogitsFromVector(vec);
        const probs = Activations.softmax(logits);
        return probs
            .map((p, i) => ({ label: this.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

    /** Vector batch prediction (kept for back-compat) */
    public predictFromVector(inputVecRows: number[][], topK: number = 5): PredictResult[][] {
        if (!this.model) throw new Error('Model not trained.');
        return inputVecRows.map(vec => {
            const logits = this.predictLogitsFromVector(vec);
            const probs = Activations.softmax(logits);
            return probs
                .map((p, i) => ({ label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        });
    }

    /** Raw logits for a single numeric vector */
    public predictLogitsFromVector(vec: number[]): number[] {
        if (!this.model) throw new Error('Model not trained.');
        const { W, b, beta } = this.model;

        // Hidden
        const tempH = Matrix.multiply([vec], Matrix.transpose(W)); // 1 x hidden
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            activationFn
        ); // 1 x hidden

        // Output logits
        return Matrix.multiply(H, beta)[0]; // 1 x outDim → vec
    }

    /** Raw logits for a batch of numeric vectors */
    public predictLogitsFromVectors(X: number[][]): number[][] {
        if (!this.model) throw new Error('Model not trained.');
        const { W, b, beta } = this.model;
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            activationFn
        );
        return Matrix.multiply(H, beta);
    }

    /** Probability vector (softmax) for a single numeric vector */
    public predictProbaFromVector(vec: number[]): number[] {
        return Activations.softmax(this.predictLogitsFromVector(vec));
    }

    /** Probability matrix (softmax per row) for a batch of numeric vectors */
    public predictProbaFromVectors(X: number[][]): number[][] {
        return this.predictLogitsFromVectors(X).map(Activations.softmax);
    }

    /** Top-K results for a single numeric vector */
    public predictTopKFromVector(vec: number[], k = 5): TopKResult[] {
        const probs = this.predictProbaFromVector(vec);
        return probs
            .map((p, i) => ({ index: i, label: this.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, k);
    }

    /** Top-K results for a batch of numeric vectors */
    public predictTopKFromVectors(X: number[][], k = 5): TopKResult[][] {
        return this.predictProbaFromVectors(X).map(row =>
            row
                .map((p, i) => ({ index: i, label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k)
        );
    }

    /* ========= Metrics ========= */

    public calculateRMSE(Y: number[][], P: number[][]): number {
        const N = Y.length, C = Y[0].length;
        let sum = 0;
        for (let i = 0; i < N; i++) for (let j = 0; j < C; j++) {
            const d = Y[i][j] - P[i][j];
            sum += d * d;
        }
        return Math.sqrt(sum / (N * C));
    }

    public calculateMAE(Y: number[][], P: number[][]): number {
        const N = Y.length, C = Y[0].length;
        let sum = 0;
        for (let i = 0; i < N; i++) for (let j = 0; j < C; j++) {
            sum += Math.abs(Y[i][j] - P[i][j]);
        }
        return sum / (N * C);
    }

    public calculateAccuracy(Y: number[][], P: number[][]): number {
        let correct = 0;
        for (let i = 0; i < Y.length; i++) {
            const yMax = this.argmax(Y[i]);
            const pMax = this.argmax(P[i]);
            if (yMax === pMax) correct++;
        }
        return correct / Y.length;
    }

    public calculateF1Score(Y: number[][], P: number[][]): number {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < Y.length; i++) {
            const yIdx = this.argmax(Y[i]);
            const pIdx = this.argmax(P[i]);
            if (yIdx === pIdx) tp++;
            else { fp++; fn++; }
        }
        const precision = tp / (tp + fp || 1);
        const recall = tp / (tp + fn || 1);
        return 2 * (precision * recall) / (precision + recall || 1);
    }

    public calculateCrossEntropy(Y: number[][], P: number[][]): number {
        let loss = 0;
        for (let i = 0; i < Y.length; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                const pred = Math.min(Math.max(P[i][j], 1e-15), 1 - 1e-15);
                loss += -Y[i][j] * Math.log(pred);
            }
        }
        return loss / Y.length;
    }

    public calculateR2Score(Y: number[][], P: number[][]): number {
        const C = Y[0].length;
        const mean: number[] = new Array(C).fill(0);
        for (let i = 0; i < Y.length; i++) for (let j = 0; j < C; j++) mean[j] += Y[i][j];
        for (let j = 0; j < C; j++) mean[j] /= Y.length;

        let ssRes = 0, ssTot = 0;
        for (let i = 0; i < Y.length; i++) {
            for (let j = 0; j < C; j++) {
                ssRes += Math.pow(Y[i][j] - P[i][j], 2);
                ssTot += Math.pow(Y[i][j] - mean[j], 2);
            }
        }
        return 1 - ssRes / ssTot;
    }

    /* ========= Hidden layer / embeddings ========= */

    computeHiddenLayer(X: number[][]): number[][] {
        if (!this.model) throw new Error('Model not trained.');
        const WX = Matrix.multiply(X, Matrix.transpose(this.model.W));
        const WXb = WX.map(row => row.map((val, j) => val + this.model!.b[j][0]));
        const activationFn = Activations.get(this.activation);
        return WXb.map(row => row.map(activationFn));
    }

    getEmbedding(X: number[][]): number[][] {
        return this.computeHiddenLayer(X);
    }

    /* ========= Logging & export ========= */

    private logMetrics(results: Record<string, number>): void {
        const logLines: string[] = [`📋 ${this.modelName} — Metrics Summary:`];
        const push = (label: string, value: number, threshold: number | undefined, cmp: string) => {
            if (threshold !== undefined) logLines.push(`  ${label}: ${value.toFixed(4)} (threshold: ${cmp} ${threshold})`);
        };
        push('RMSE', results.rmse!, this.metrics?.rmse, '<=');
        push('MAE', results.mae!, this.metrics?.mae, '<=');
        push('Accuracy', results.accuracy!, this.metrics?.accuracy, '>=');
        push('F1 Score', results.f1!, (this.metrics as any)?.f1, '>=');
        push('Cross-Entropy', results.crossEntropy!, (this.metrics as any)?.crossEntropy, '<=');
        push('R² Score', results.r2!, this.metrics?.r2, '>=');

        if (this.verbose) console.log('\n' + logLines.join('\n'));

        if (this.logToFile) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const logFile = this.config.logFileName || `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_metrics_${timestamp}.txt`;

            const blob = new Blob([logLines.join('\n')], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = logFile;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    public saveModelAsJSONFile(filename?: string): void {
        if (!this.savedModelJSON) {
            if (this.verbose) console.warn('No model saved — did not meet metric thresholds.');
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fallback = `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.json`;
        const finalName = filename || this.config.exportFileName || fallback;

        const blob = new Blob([this.savedModelJSON], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = finalName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (this.verbose) console.log(`📦 Model exported as ${finalName}`);
    }

    private serializeConfig(): SerializedELM['config'] {
        const cfg = { ...(this.config as any) };
        // Remove non-serializable / volatile fields
        delete cfg.seed; delete cfg.log; delete cfg.encoder;

        // Serialize tokenizerDelimiter for JSON
        if (cfg.tokenizerDelimiter instanceof RegExp) {
            cfg.tokenizerDelimiter = cfg.tokenizerDelimiter.source;
        }
        return cfg;
    }

    private argmax(arr: number[]): number {
        let i = 0;
        for (let k = 1; k < arr.length; k++) if (arr[k] > arr[i]) i = k;
        return i;
    }
    public getEncoder() {
        return this.encoder;
    }
}
