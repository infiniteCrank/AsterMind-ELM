// KernelELM.ts — Kernel Extreme Learning Machine (KELM)
// Uses Gram matrix K(X, X) and solves (K + λI) α = Y.
// Prediction: f(x) = K(x, X_train) α
//
// Works for multi-class (one-hot Y) classification or regression.
//
// Dependencies: Matrix, Activations (from your library)

import { Matrix } from './Matrix';
import { Activations } from './Activations';

export type KernelType = 'rbf' | 'linear' | 'poly';

export interface KernelSpec {
    type: KernelType;
    /** RBF: exp(-gamma * ||x - z||^2). If omitted, gamma = 1 / D. */
    gamma?: number;
    /** Poly: (gamma * x·z + coef0)^degree */
    degree?: number;
    coef0?: number;
}

export interface KernelELMConfig {
    /** Output dimension (classes for one-hot classification or dims for regression) */
    outputDim: number;

    /** Kernel parameters */
    kernel: KernelSpec;

    /** Ridge regularization λ (default 1e-2) */
    ridgeLambda?: number;

    /** Task type for predictProba convenience; 'classification' enables softmax helpers */
    task?: 'classification' | 'regression';

    /** Optional model name + verbosity */
    log?: { modelName?: string; verbose?: boolean };
}

export interface KernelELMJSON {
    config: KernelELMConfig & { __version: string };
    X: number[][];          // training inputs (N x D)  (careful: can be large)
    alpha: number[][];      // coefficients (N x K)
}

const EPS = 1e-12;

function l2sq(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function buildKernel(spec: KernelSpec, dim: number) {
    const type = spec.type;
    const gamma = spec.gamma ?? (type === 'rbf' || type === 'poly' ? 1 / Math.max(1, dim) : 1);
    const degree = spec.degree ?? 2;
    const coef0 = spec.coef0 ?? 1;

    if (type === 'rbf') {
        return (x: number[], z: number[]) => Math.exp(-gamma * l2sq(x, z));
    }
    if (type === 'linear') {
        return (x: number[], z: number[]) => dot(x, z);
    }
    // poly
    return (x: number[], z: number[]) => Math.pow(gamma * dot(x, z) + coef0, degree);
}

function softmaxRow(v: number[]): number[] {
    const m = Math.max(...v);
    const ex = v.map(x => Math.exp(x - m));
    const s = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map(e => e / s);
}

export class KernelELM {
    readonly cfg: Required<KernelELMConfig>;
    private kernel!: (x: number[], z: number[]) => number;

    private Xtrain: number[][] = [];   // (N x D)
    private alpha: number[][] = [];    // (N x K)

    private verbose: boolean;
    private name: string;

    constructor(config: KernelELMConfig) {
        // fill defaults
        this.cfg = {
            task: 'classification',
            ridgeLambda: config.ridgeLambda ?? 1e-2,
            ...config,
            log: { modelName: config.log?.modelName ?? 'KernelELM', verbose: config.log?.verbose ?? false },
        };
        this.verbose = this.cfg.log.verbose ?? false;
        this.name = this.cfg.log.modelName ?? 'KernelELM';
    }

    /** Train with full kernel (O(N^3) Cholesky). Y must be (N x K). */
    fit(X: number[][], Y: number[][]): void {
        if (!X?.length || !X[0]?.length) throw new Error('KernelELM.fit: empty X');
        if (!Y?.length || !Y[0]?.length) throw new Error('KernelELM.fit: empty Y');
        if (X.length !== Y.length) throw new Error(`KernelELM.fit: X rows ${X.length} != Y rows ${Y.length}`);
        if (Y[0].length !== this.cfg.outputDim) {
            throw new Error(`KernelELM.fit: Y dims ${Y[0].length} != outputDim ${this.cfg.outputDim}`);
        }

        const N = X.length, D = X[0].length, K = Y[0].length;
        this.kernel = buildKernel(this.cfg.kernel, D);

        // Gram matrix K (N x N)
        if (this.verbose) console.log(`🔧 [${this.name}] building Gram matrix: N=${N}, D=${D}`);
        const Kmat = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(N);
            Kmat[i] = row;
            row[i] = 1; // self-sim
            for (let j = i + 1; j < N; j++) {
                const v = this.kernel(X[i], X[j]);
                row[j] = v;
            }
        }
        // Fill lower triangle
        for (let i = 1; i < N; i++) {
            const row = Kmat[i];
            for (let j = 0; j < i; j++) row[j] = Kmat[j][i];
        }

        // (K + λI) α = Y  → α = (K + λI)^-1 Y
        if (this.verbose) console.log(`🧮 [${this.name}] solving (K + λI) α = Y, λ=${this.cfg.ridgeLambda}`);
        const Kreg = Matrix.addRegularization(Kmat, this.cfg.ridgeLambda + 1e-10);
        // Solve via Cholesky: A X = B
        const Alpha = Matrix.solveCholesky(Kreg, Y, 1e-12); // (N x K)

        this.Xtrain = X.map(r => r.slice());
        this.alpha = Alpha;

        if (this.verbose) console.log(`✅ [${this.name}] fit complete: stored X_train (${N}x${D}), alpha (${N}x${K})`);
    }

    /** Raw scores (logits) for a batch: (M x K) */
    predictLogitsFromVectors(X: number[][]): number[][] {
        if (!this.alpha.length) throw new Error('KernelELM.predict: model not fitted');
        const N = this.Xtrain.length;
        const M = X.length;
        // Build K(X, Xtrain): (M x N)
        const Kqx = new Array(M);
        for (let i = 0; i < M; i++) {
            const row = new Array(N);
            const xi = X[i];
            for (let j = 0; j < N; j++) {
                row[j] = this.kernel(xi, this.Xtrain[j]);
            }
            Kqx[i] = row;
        }
        // Scores = K(X, Xtrain) * alpha  → (M x K)
        return Matrix.multiply(Kqx, this.alpha);
    }

    /** Probabilities for classification (softmax row-wise). Returns (M x K). */
    predictProbaFromVectors(X: number[][]): number[][] {
        const logits = this.predictLogitsFromVectors(X);
        if (this.cfg.task === 'classification') {
            return logits.map(softmaxRow);
        }
        // For regression, just return raw scores
        return logits;
    }

    /** Top-K results for classification */
    predictTopKFromVectors(X: number[][], k = 5): Array<Array<{ index: number; prob: number }>> {
        const P = this.predictProbaFromVectors(X);
        return P.map(row =>
            row
                .map((p, i) => ({ index: i, prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k)
        );
    }

    /** Embedding for chaining: return kernel features Φ = K(X, X_train) (M x N). */
    getEmbedding(X: number[][]): number[][] {
        if (!this.alpha.length) throw new Error('KernelELM.getEmbedding: model not fitted');
        const N = this.Xtrain.length, M = X.length;
        const out = new Array(M);
        for (let i = 0; i < M; i++) {
            const row = new Array(N);
            for (let j = 0; j < N; j++) row[j] = this.kernel(X[i], this.Xtrain[j]);
            out[i] = row;
        }
        return out;
    }

    /** Export (careful: includes X_train; for large N consider Nyström landmarks) */
    toJSON(): KernelELMJSON {
        return {
            config: { ...this.cfg, __version: 'kelm-1.0.0' },
            X: this.Xtrain,
            alpha: this.alpha,
        };
    }

    /** Import */
    fromJSON(payload: string | KernelELMJSON) {
        const obj: KernelELMJSON = typeof payload === 'string' ? JSON.parse(payload) : payload;
        this.Xtrain = obj.X.map(r => r.slice());
        this.alpha = obj.alpha.map(r => r.slice());
        this.cfg.kernel = { ...obj.config.kernel };
        this.cfg.ridgeLambda = obj.config.ridgeLambda ?? this.cfg.ridgeLambda;
        this.cfg.task = obj.config.task ?? this.cfg.task;
        this.kernel = buildKernel(this.cfg.kernel, this.Xtrain[0]?.length ?? 1);
    }
}
