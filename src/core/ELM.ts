// ELM.ts - Core ELM logic with TypeScript types

import { Matrix } from './Matrix';
import { Activations } from './Activations';
import { ELMConfig, defaultConfig } from './ELMConfig';
import { UniversalEncoder } from '../preprocessing/UniversalEncoder';
import { Augment } from '../utils/Augment';

export interface ELMModel {
    W: number[][];
    b: number[][];
    beta: number[][];
}

export interface PredictResult {
    label: string;
    prob: number;
}

export class ELM {
    public categories: string[];
    public hiddenUnits: number;
    public maxLen: number;
    public activation: string;
    public charSet: string;
    public useTokenizer: boolean;
    public tokenizerDelimiter?: RegExp;
    public encoder: UniversalEncoder;
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

    public inputWeights: Matrix;
    public biases: Matrix;

    constructor(config: ELMConfig & { charSet?: string; useTokenizer?: boolean; tokenizerDelimiter?: RegExp }) {
        const cfg = { ...defaultConfig, ...config };
        this.categories = cfg.categories;
        this.hiddenUnits = cfg.hiddenUnits;
        this.maxLen = cfg.maxLen;
        this.activation = cfg.activation;
        this.charSet = cfg.charSet ?? 'abcdefghijklmnopqrstuvwxyz';
        this.useTokenizer = cfg.useTokenizer ?? false;
        this.tokenizerDelimiter = cfg.tokenizerDelimiter;
        this.config = cfg;
        this.metrics = this.config.metrics;
        this.verbose = cfg.log?.verbose ?? true;
        this.modelName = cfg.log?.modelName ?? 'Unnamed ELM Model';
        this.logToFile = cfg.log?.toFile ?? false;
        this.dropout = cfg.dropout ?? 0;

        this.encoder = new UniversalEncoder({
            charSet: this.charSet,
            maxLen: this.maxLen,
            useTokenizer: this.useTokenizer,
            tokenizerDelimiter: this.tokenizerDelimiter,
            mode: this.useTokenizer ? 'token' : 'char'
        });

        this.inputWeights = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, cfg.maxLen));
        this.biases = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, 1));

        this.model = null;
    }

    public oneHot(n: number, index: number): number[] {
        return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
    }

    private pseudoInverse(H: number[][], lambda: number = 1e-3): number[][] {
        const Ht = Matrix.transpose(H);
        const HtH = Matrix.multiply(Ht, H);
        const HtH_reg = Matrix.addRegularization(HtH, lambda);
        const HtH_inv = Matrix.inverse(HtH_reg);
        return Matrix.multiply(HtH_inv, Ht);
    }

    private randomMatrix(rows: number, cols: number): number[][] {
        if (this.config.weightInit === "xavier") {
            if (this.verbose) console.log(`✨ Xavier init with limit sqrt(6/${rows}+${cols})`);
            const limit = Math.sqrt(6 / (rows + cols));
            return Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => Math.random() * 2 * limit - limit)
            );
        } else {
            if (this.verbose) console.log(`✨ Uniform init [-1,1]`);
            return Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => Math.random() * 2 - 1)
            );
        }
    }

    public setCategories(categories: string[]) {
        this.categories = categories;
    }

    public loadModelFromJSON(json: string): void {
        try {
            const parsed: ELMModel = JSON.parse(json);
            this.model = parsed;
            this.savedModelJSON = json;
            if (this.verbose) console.log(`✅ ${this.modelName} Model loaded from JSON`);
        } catch (e) {
            console.error(`❌ Failed to load ${this.modelName} model from JSON:`, e);
        }
    }

    public trainFromData(
        X: number[][],
        Y: number[][],
        options?: {
            reuseWeights?: boolean;
            weights?: number[];
        }
    ): void {
        const reuseWeights = options?.reuseWeights === true;

        let W: number[][], b: number[][];
        if (reuseWeights && this.model) {
            W = this.model.W;
            b = this.model.b;
            if (this.verbose) console.log("🔄 Reusing existing weights/biases for training.");
        } else {
            W = this.randomMatrix(this.hiddenUnits, X[0].length);
            b = this.randomMatrix(this.hiddenUnits, 1);
            if (this.verbose) console.log("✨ Initializing fresh weights/biases for training.");
        }

        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        let H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            activationFn
        );

        if (this.dropout > 0) {
            const keepProb = 1 - this.dropout;
            for (let i = 0; i < H.length; i++) {
                for (let j = 0; j < H[0].length; j++) {
                    if (Math.random() < this.dropout) {
                        H[i][j] = 0;
                    } else {
                        H[i][j] /= keepProb;
                    }
                }
            }
        }

        if (options?.weights) {
            const W_arr = options.weights;
            if (W_arr.length !== H.length) {
                throw new Error(`Weight array length ${W_arr.length} does not match sample count ${H.length}`);
            }
            // Scale each row by sqrt(weight)
            H = H.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
            Y = Y.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
        }

        const H_pinv = this.pseudoInverse(H);
        const beta = Matrix.multiply(H_pinv, Y);
        this.model = { W, b, beta };

        const predictions = Matrix.multiply(H, beta);

        if (this.metrics) {
            const rmse = this.calculateRMSE(Y, predictions);
            const mae = this.calculateMAE(Y, predictions);
            const acc = this.calculateAccuracy(Y, predictions);
            const f1 = this.calculateF1Score(Y, predictions);
            const ce = this.calculateCrossEntropy(Y, predictions);
            const r2 = this.calculateR2Score(Y, predictions);

            const results: Record<string, number> = {};
            let allPassed = true;

            if (this.metrics.rmse !== undefined) {
                results.rmse = rmse;
                if (rmse > this.metrics.rmse) allPassed = false;
            }
            if (this.metrics.mae !== undefined) {
                results.mae = mae;
                if (mae > this.metrics.mae) allPassed = false;
            }
            if (this.metrics.accuracy !== undefined) {
                results.accuracy = acc;
                if (acc < this.metrics.accuracy) allPassed = false;
            }
            if (this.metrics.f1 !== undefined) {
                results.f1 = f1;
                if (f1 < this.metrics.f1) allPassed = false;
            }
            if (this.metrics.crossEntropy !== undefined) {
                results.crossEntropy = ce;
                if (ce > this.metrics.crossEntropy) allPassed = false;
            }
            if (this.metrics.r2 !== undefined) {
                results.r2 = r2;
                if (r2 < this.metrics.r2) allPassed = false;
            }

            if (this.verbose) this.logMetrics(results);

            if (allPassed) {
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose) console.log("✅ Model passed thresholds and was saved to JSON.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            } else {
                if (this.verbose) console.log("❌ Model not saved: One or more thresholds not met.");
            }
        } else {
            // No metrics—always save the model
            this.savedModelJSON = JSON.stringify(this.model);
            if (this.verbose) console.log("✅ Model trained with no metrics—saved by default.");
            if (this.config.exportFileName) {
                this.saveModelAsJSONFile(this.config.exportFileName);
            }
        }
    }

    public train(
        augmentationOptions?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        },
        weights?: number[]
    ): void {
        const X: number[][] = [];
        let Y: number[][] = [];

        this.categories.forEach((cat, i) => {
            const variants = Augment.generateVariants(cat, this.charSet, augmentationOptions);
            for (const variant of variants) {
                const vec = this.encoder.normalize(this.encoder.encode(variant));
                X.push(vec);
                Y.push(this.oneHot(this.categories.length, i));
            }
        });

        const W = this.randomMatrix(this.hiddenUnits, X[0].length);
        const b = this.randomMatrix(this.hiddenUnits, 1);
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        let H = Activations.apply(
            tempH.map(row => row.map((val, j) => val + b[j][0])),
            activationFn
        );

        if (this.dropout > 0) {
            const keepProb = 1 - this.dropout;
            for (let i = 0; i < H.length; i++) {
                for (let j = 0; j < H[0].length; j++) {
                    if (Math.random() < this.dropout) {
                        H[i][j] = 0;
                    } else {
                        H[i][j] /= keepProb;
                    }
                }
            }
        }

        if (weights) {
            if (weights.length !== H.length) {
                throw new Error(`Weight array length ${weights.length} does not match sample count ${H.length}`);
            }
            // Scale each row of H and Y by sqrt(weight)
            H = H.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
            Y = Y.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
        }

        const H_pinv = this.pseudoInverse(H);
        const beta = Matrix.multiply(H_pinv, Y);
        this.model = { W, b, beta };

        const predictions = Matrix.multiply(H, beta);

        if (this.metrics) {
            const rmse = this.calculateRMSE(Y, predictions);
            const mae = this.calculateMAE(Y, predictions);
            const acc = this.calculateAccuracy(Y, predictions);
            const f1 = this.calculateF1Score(Y, predictions);
            const ce = this.calculateCrossEntropy(Y, predictions);
            const r2 = this.calculateR2Score(Y, predictions);

            const results: Record<string, number> = {};
            let allPassed = true;

            if (this.metrics.rmse !== undefined) {
                results.rmse = rmse;
                if (rmse > this.metrics.rmse) allPassed = false;
            }
            if (this.metrics.mae !== undefined) {
                results.mae = mae;
                if (mae > this.metrics.mae) allPassed = false;
            }
            if (this.metrics.accuracy !== undefined) {
                results.accuracy = acc;
                if (acc < this.metrics.accuracy) allPassed = false;
            }
            if (this.metrics.f1 !== undefined) {
                results.f1 = f1;
                if (f1 < this.metrics.f1) allPassed = false;
            }
            if (this.metrics.crossEntropy !== undefined) {
                results.crossEntropy = ce;
                if (ce > this.metrics.crossEntropy) allPassed = false;
            }
            if (this.metrics.r2 !== undefined) {
                results.r2 = r2;
                if (r2 < this.metrics.r2) allPassed = false;
            }

            if (this.verbose) {
                this.logMetrics(results);
            }

            if (allPassed) {
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose) console.log("✅ Model passed thresholds and was saved to JSON.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            } else {
                if (this.verbose) console.log("❌ Model not saved: One or more thresholds not met.");
            }
        } else {
            this.savedModelJSON = JSON.stringify(this.model);
            if (this.verbose) console.log("✅ Model trained with no metrics—saved by default.");
            if (this.config.exportFileName) {
                this.saveModelAsJSONFile(this.config.exportFileName);
            }
        }
    }

    private logMetrics(results: Record<string, number>): void {
        const logLines: string[] = [`📋 ${this.modelName} — Metrics Summary:`];
        const push = (label: string, value: number, threshold: number | undefined, cmp: string) => {
            if (threshold !== undefined) logLines.push(`  ${label}: ${value.toFixed(4)} (threshold: ${cmp} ${threshold})`);
        };
        push('RMSE', results.rmse!, this.metrics?.rmse, '<=');
        push('MAE', results.mae!, this.metrics?.mae, '<=');
        push('Accuracy', results.accuracy!, this.metrics?.accuracy, '>=');
        push('F1 Score', results.f1!, this.metrics?.f1, '>=');
        push('Cross-Entropy', results.crossEntropy!, this.metrics?.crossEntropy, '<=');
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
            if (this.verbose) console.warn("No model saved — did not meet metric thresholds.");
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fallback = `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.json`;
        const finalName = filename || this.config.exportFileName || fallback;

        const blob = new Blob([this.savedModelJSON], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = finalName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (this.verbose) console.log(`📦 Model exported as ${finalName}`);
    }

    public predict(text: string, topK: number = 5): PredictResult[] {
        if (!this.model) throw new Error("Model not trained.");

        const vec = this.encoder.normalize(this.encoder.encode(text));
        const { W, b, beta } = this.model;
        const tempH = Matrix.multiply([vec], Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const rawOutput = Matrix.multiply(H, beta)[0];
        const probs = Activations.softmax(rawOutput);

        return probs
            .map((p, i) => ({ label: this.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

    public predictFromVector(inputVec: number[][], topK: number = 5): PredictResult[][] {
        if (!this.model) throw new Error("Model not trained.");

        const { W, b, beta } = this.model;
        const tempH = Matrix.multiply(inputVec, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        return Matrix.multiply(H, beta).map(rawOutput => {
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        });
    }

    public calculateRMSE(Y: number[][], P: number[][]): number {
        const N = Y.length;
        let sum = 0;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                const diff = Y[i][j] - P[i][j];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum / (N * Y[0].length));
    }

    public calculateMAE(Y: number[][], P: number[][]): number {
        const N = Y.length;
        let sum = 0;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                sum += Math.abs(Y[i][j] - P[i][j]);
            }
        }
        return sum / (N * Y[0].length);
    }

    public calculateAccuracy(Y: number[][], P: number[][]): number {
        let correct = 0;
        for (let i = 0; i < Y.length; i++) {
            const yMax = Y[i].indexOf(Math.max(...Y[i]));
            const pMax = P[i].indexOf(Math.max(...P[i]));
            if (yMax === pMax) correct++;
        }
        return correct / Y.length;
    }

    public calculateF1Score(Y: number[][], P: number[][]): number {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < Y.length; i++) {
            const yIdx = Y[i].indexOf(1);
            const pIdx = P[i].indexOf(Math.max(...P[i]));
            if (yIdx === pIdx) tp++;
            else {
                fp++;
                fn++;
            }
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
        const Y_mean = Y[0].map((_, j) => Y.reduce((sum, y) => sum + y[j], 0) / Y.length);
        let ssRes = 0, ssTot = 0;
        for (let i = 0; i < Y.length; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                ssRes += Math.pow(Y[i][j] - P[i][j], 2);
                ssTot += Math.pow(Y[i][j] - Y_mean[j], 2);
            }
        }
        return 1 - ssRes / ssTot;
    }

    computeHiddenLayer(X: number[][]): number[][] {
        if (!this.model) throw new Error("Model not trained.");
        const WX = Matrix.multiply(X, Matrix.transpose(this.model.W));
        const WXb = WX.map(row => row.map((val, j) => val + this.model!.b[j][0]));
        const activationFn = Activations.get(this.activation);
        return WXb.map(row => row.map(activationFn));
    }

    getEmbedding(X: number[][]): number[][] {
        return this.computeHiddenLayer(X);
    }
}
