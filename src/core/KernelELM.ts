// KernelELM.ts — Kernel Extreme Learning Machine (Exact + Nyström) with registry
// Dependencies: Matrix (multiply, transpose, addRegularization, solveCholesky, identity, add)

import { Matrix } from './Matrix';

export type KernelType = 'rbf' | 'linear' | 'poly' | 'laplacian' | 'custom';

export interface KernelSpec {
    type: KernelType;
    /** RBF / Laplacian: gamma scale; default 1/D */
    gamma?: number;
    /** Poly: (gamma * x·z + coef0)^degree */
    degree?: number;
    coef0?: number;

    /** For type:'custom' — name of a kernel previously registered in KernelRegistry */
    name?: string;
}

export type KELMMode = 'exact' | 'nystrom';

export interface NystromOptions {
    /** # of landmarks (m). Required when strategy !== 'preset'. */
    m?: number;
    /** How to pick landmarks */
    strategy?: 'uniform' | 'kmeans++' | 'preset';
    /** Random seed for reproducible sampling */
    seed?: number;
    /** Provide your own landmarks (points or indices into X) */
    preset?: { points?: number[][]; indices?: number[] };
    /**
     * If true, apply Nyström whitening Φ = K_nm * K_mm^{-1/2}.
     * NOTE: requires eigen decomposition support — not implemented yet. Defaults false.
     */
    whiten?: boolean;
}

export interface KernelELMConfig {
    /** Output dimension (K) — #classes for one-hot classification or dims for regression */
    outputDim: number;
    /** Kernel parameters */
    kernel: KernelSpec;
    /** Regularization λ (ridge) */
    ridgeLambda?: number;
    /** Task type for predictProba helpers */
    task?: 'classification' | 'regression';
    /** Solver mode */
    mode?: KELMMode;
    /** Nyström options (used when mode='nystrom') */
    nystrom?: NystromOptions;
    /** Optional model name + verbosity */
    log?: { modelName?: string; verbose?: boolean };
}

export interface KernelELMJSON {
    config: KernelELMConfig & { __version: string };
    // exact mode payload:
    X?: number[][];
    alpha?: number[][];
    // nystrom mode payload:
    Z?: number[][];
    W?: number[][];
}

/* ================= Kernel registry ================= */

type KernelFn = (x: number[], z: number[]) => number;

export class KernelRegistry {
    private static map = new Map<string, KernelFn>();

    static register(name: string, fn: KernelFn) {
        if (!name || typeof fn !== 'function') throw new Error('KernelRegistry.register: invalid args');
        this.map.set(name, fn);
    }
    static has(name: string) { return this.map.has(name); }
    static get(name: string): KernelFn {
        const f = this.map.get(name);
        if (!f) throw new Error(`KernelRegistry: kernel "${name}" not found`);
        return f;
    }
}

/* ================ utils ================ */

const EPS = 1e-12;

function l2sq(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return s;
}
function l1(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
}
function dot(a: number[], b: number[]): number {
    let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
}
function softmaxRow(v: number[]): number[] {
    const m = Math.max(...v);
    const ex = v.map(x => Math.exp(x - m));
    const s = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map(e => e / s);
}
function makePRNG(seed = 123456789) {
    let s = seed | 0 || 1;
    return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0xffffffff; };
}

/* ================= kernels ================= */

function buildKernel(spec: KernelSpec, dim: number): KernelFn {
    const type = spec.type;
    if (type === 'custom') {
        if (!spec.name) throw new Error('custom kernel requires "name"');
        return KernelRegistry.get(spec.name);
    }
    if (type === 'linear') {
        return (x, z) => dot(x, z);
    }
    if (type === 'poly') {
        const gamma = spec.gamma ?? 1 / Math.max(1, dim);
        const degree = spec.degree ?? 2;
        const coef0 = spec.coef0 ?? 1;
        return (x, z) => Math.pow(gamma * dot(x, z) + coef0, degree);
    }
    if (type === 'laplacian') { // exp(-gamma * ||x-z||_1)
        const gamma = spec.gamma ?? 1 / Math.max(1, dim);
        return (x, z) => Math.exp(-gamma * l1(x, z));
    }
    // default: rbf
    const gamma = spec.gamma ?? 1 / Math.max(1, dim);
    return (x, z) => Math.exp(-gamma * l2sq(x, z));
}

/* ============== landmark selection (Nyström) ============== */

function pickUniform(X: number[][], m: number, seed = 1337): number[] {
    const prng = makePRNG(seed);
    const N = X.length;
    const idx = Array.from({ length: N }, (_, i) => i);
    // Fisher–Yates shuffle first m
    for (let i = 0; i < m; i++) {
        const j = i + Math.floor(prng() * (N - i));
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx.slice(0, m);
}

function pickKMeansPP(X: number[][], m: number, seed = 1337): number[] {
    const prng = makePRNG(seed);
    const N = X.length;
    if (m >= N) return Array.from({ length: N }, (_, i) => i);
    const centers: number[] = [];
    centers.push(Math.floor(prng() * N));
    const D2 = new Float64Array(N).fill(Infinity);

    while (centers.length < m) {
        // update distances
        const c = centers[centers.length - 1];
        for (let i = 0; i < N; i++) {
            const d2 = l2sq(X[i], X[c]);
            if (d2 < D2[i]) D2[i] = d2;
        }
        // sample next center proportional to D^2
        let sum = 0; for (let i = 0; i < N; i++) sum += D2[i];
        let r = prng() * (sum || 1);
        let next = 0;
        for (let i = 0; i < N; i++) { r -= D2[i]; if (r <= 0) { next = i; break; } }
        centers.push(next);
    }
    return centers;
}

/* ======================= KernelELM ======================= */

export class KernelELM {
    readonly cfg: Required<KernelELMConfig>;
    private kernel!: KernelFn;

    // exact mode
    private Xtrain: number[][] = [];
    private alpha: number[][] = [];

    // nystrom mode
    private Z: number[][] = []; // landmarks (m x D)
    private W: number[][] = []; // weights in landmark feature space (m x K)

    private verbose: boolean;
    private name: string;

    constructor(config: KernelELMConfig) {
        this.cfg = {
            task: config.task ?? 'classification',
            ridgeLambda: config.ridgeLambda ?? 1e-2,
            mode: config.mode ?? 'exact',
            nystrom: {
                m: config.nystrom?.m,
                strategy: config.nystrom?.strategy ?? 'uniform',
                seed: config.nystrom?.seed ?? 1337,
                preset: config.nystrom?.preset,
                whiten: config.nystrom?.whiten ?? false,
            },
            ...config,
            log: { modelName: config.log?.modelName ?? 'KernelELM', verbose: config.log?.verbose ?? false },
        };
        this.verbose = !!this.cfg.log.verbose;
        this.name = this.cfg.log.modelName ?? 'KernelELM';
    }

    /** Train (exact or Nyström) */
    fit(X: number[][], Y: number[][]): void {
        if (!X?.length || !X[0]?.length) throw new Error('KernelELM.fit: empty X');
        if (!Y?.length || !Y[0]?.length) throw new Error('KernelELM.fit: empty Y');
        if (X.length !== Y.length) throw new Error(`KernelELM.fit: X rows ${X.length} != Y rows ${Y.length}`);
        if (Y[0].length !== this.cfg.outputDim) {
            throw new Error(`KernelELM.fit: Y dims ${Y[0].length} != outputDim ${this.cfg.outputDim}`);
        }
        const N = X.length, D = X[0].length, K = Y[0].length;
        this.kernel = buildKernel(this.cfg.kernel, D);

        if (this.cfg.mode === 'exact') {
            // Gram K (N x N)
            if (this.verbose) console.log(`🔧 [${this.name}] exact Gram: N=${N}, D=${D}`);
            const Kmat = new Array(N);
            for (let i = 0; i < N; i++) {
                const row = new Array(N);
                Kmat[i] = row;
                row[i] = 1;
                for (let j = i + 1; j < N; j++) {
                    const v = this.kernel(X[i], X[j]);
                    row[j] = v;
                }
            }
            // lower triangle
            for (let i = 1; i < N; i++) {
                const row = Kmat[i];
                for (let j = 0; j < i; j++) row[j] = Kmat[j][i];
            }
            const A = Matrix.addRegularization(Kmat, this.cfg.ridgeLambda + 1e-10);
            const Alpha = Matrix.solveCholesky(A, Y, 1e-12); // (N x K)
            this.Xtrain = X.map(r => r.slice());
            this.alpha = Alpha;
            this.Z = []; this.W = [];
            if (this.verbose) console.log(`✅ [${this.name}] exact fit: stored X(${N}x${D}), alpha(${N}x${K})`);
            return;
        }

        // Nyström (feature-space ridge on K_nm)
        const ny = this.cfg.nystrom!;
        let idx: number[] | undefined;
        let Z: number[][];
        if (ny.strategy === 'preset' && (ny.preset?.points || ny.preset?.indices)) {
            if (ny.preset?.points) {
                Z = ny.preset.points.map(r => r.slice());
            } else {
                idx = ny.preset!.indices!;
                Z = idx.map(i => X[i]);
            }
        } else {
            const m = ny.m ?? Math.max(10, Math.min(300, Math.floor(Math.sqrt(N))));
            idx = ny.strategy === 'kmeans++'
                ? pickKMeansPP(X, m, ny.seed)
                : pickUniform(X, m, ny.seed);
            Z = idx.map(i => X[i]);
        }
        const m = Z.length;
        if (this.verbose) console.log(`🔹 [${this.name}] Nyström: m=${m}, strategy=${ny.strategy}`);

        // Build K_nm (N x m)
        const Knm = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(m);
            const xi = X[i];
            for (let j = 0; j < m; j++) row[j] = this.kernel(xi, Z[j]);
            Knm[i] = row;
        }

        // Optional whitening would use K_mm^{-1/2}; not implemented without eigens
        if (ny.whiten) {
            if (this.verbose) console.warn(`[${this.name}] Nyström whitening requested but not implemented — proceeding without whitening.`);
        }

        // Ridge regression in feature space (Φ = K_nm)
        // W = (Φᵀ Φ + λ I)^-1 Φᵀ Y  ⇒ (m x K)
        const PhiT = Matrix.transpose(Knm);              // (m x N)
        const G = Matrix.multiply(PhiT, Knm);            // (m x m)
        const Greg = Matrix.addRegularization(G, this.cfg.ridgeLambda + 1e-10);
        const R = Matrix.multiply(PhiT, Y);              // (m x K)
        const W = Matrix.solveCholesky(Greg, R, 1e-12);  // (m x K)

        this.Z = Z;
        this.W = W;
        this.Xtrain = []; this.alpha = [];
        if (this.verbose) console.log(`✅ [${this.name}] Nyström fit: Z(${m}x${D}), W(${m}x${K})`);
    }

    /* ================= prediction ================= */

    /** Raw logits for batch: (M x K) */
    predictLogitsFromVectors(X: number[][]): number[][] {
        if (this.cfg.mode === 'exact') {
            if (!this.alpha.length) throw new Error('predict: exact model not fitted');
            const N = this.Xtrain.length, M = X.length;
            const Kqx = new Array(M);
            for (let i = 0; i < M; i++) {
                const row = new Array(N);
                const xi = X[i];
                for (let j = 0; j < N; j++) row[j] = this.kernel(xi, this.Xtrain[j]);
                Kqx[i] = row;
            }
            return Matrix.multiply(Kqx, this.alpha);
        }
        // Nyström
        if (!this.Z.length || !this.W.length) throw new Error('predict: Nyström model not fitted');
        const M = X.length, m = this.Z.length;
        const Kxm = new Array(M);
        for (let i = 0; i < M; i++) {
            const row = new Array(m);
            const xi = X[i];
            for (let j = 0; j < m; j++) row[j] = this.kernel(xi, this.Z[j]);
            Kxm[i] = row;
        }
        return Matrix.multiply(Kxm, this.W); // (M x K)
    }

    /** Probabilities (softmax) for classification; raw scores for regression */
    predictProbaFromVectors(X: number[][]): number[][] {
        const logits = this.predictLogitsFromVectors(X);
        return this.cfg.task === 'classification' ? logits.map(softmaxRow) : logits;
    }

    /** Top-K for classification */
    predictTopKFromVectors(X: number[][], k = 5): Array<Array<{ index: number; prob: number }>> {
        const P = this.predictProbaFromVectors(X);
        return P.map(row =>
            row.map((p, i) => ({ index: i, prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k)
        );
    }

    /** Embedding for chaining:
     *  - exact: Φ = K(X, X_train)  (M x N)
     *  - nystrom: Φ = K(X, Z)      (M x m)
     */
    getEmbedding(X: number[][]): number[][] {
        if (this.cfg.mode === 'exact') {
            if (!this.alpha.length) throw new Error('getEmbedding: exact model not fitted');
            const N = this.Xtrain.length, M = X.length;
            const out = new Array(M);
            for (let i = 0; i < M; i++) {
                const row = new Array(N);
                for (let j = 0; j < N; j++) row[j] = this.kernel(X[i], this.Xtrain[j]);
                out[i] = row;
            }
            return out;
        }
        if (!this.Z.length) throw new Error('getEmbedding: Nyström model not fitted');
        const M = X.length, m = this.Z.length;
        const out = new Array(M);
        for (let i = 0; i < M; i++) {
            const row = new Array(m);
            for (let j = 0; j < m; j++) row[j] = this.kernel(X[i], this.Z[j]);
            out[i] = row;
        }
        return out;
    }

    /* ================= JSON I/O ================= */

    toJSON(): KernelELMJSON {
        const base = { config: { ...this.cfg, __version: 'kelm-2.0.0' } };
        if (this.cfg.mode === 'exact') {
            return { ...base, X: this.Xtrain, alpha: this.alpha };
        }
        return { ...base, Z: this.Z, W: this.W };
    }

    fromJSON(payload: string | KernelELMJSON) {
        const obj: KernelELMJSON = typeof payload === 'string' ? JSON.parse(payload) : payload;
        // restore shallow config pieces (keep runtime defaults where missing)
        this.cfg.kernel = { ...obj.config.kernel };
        this.cfg.ridgeLambda = obj.config.ridgeLambda ?? this.cfg.ridgeLambda;
        this.cfg.task = obj.config.task ?? this.cfg.task;
        this.cfg.mode = obj.config.mode ?? this.cfg.mode;
        this.cfg.nystrom = { ...this.cfg.nystrom, ...(obj.config.nystrom ?? {}) };
        // rebuild kernel from dim (infer from stored X/Z)
        if (obj.X && obj.alpha) {
            this.Xtrain = obj.X.map(r => r.slice());
            this.alpha = obj.alpha.map(r => r.slice());
            this.Z = []; this.W = [];
            const D = this.Xtrain[0]?.length ?? 1;
            this.kernel = buildKernel(this.cfg.kernel, D);
        } else if (obj.Z && obj.W) {
            this.Z = obj.Z.map(r => r.slice());
            this.W = obj.W.map(r => r.slice());
            this.Xtrain = []; this.alpha = [];
            const D = this.Z[0]?.length ?? 1;
            this.kernel = buildKernel(this.cfg.kernel, D);
        } else {
            throw new Error('KernelELM.fromJSON: invalid payload');
        }
    }
}
