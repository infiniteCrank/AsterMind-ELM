// KernelELM.ts — Kernel Extreme Learning Machine (Exact + Nyström + Whitening)
// Dependencies: Matrix (multiply, transpose, addRegularization, solveCholesky, identity, zeros)

import { Matrix } from './Matrix';

/* ====================== Types ====================== */

export type KernelType = 'rbf' | 'linear' | 'poly' | 'laplacian' | 'custom';

export interface KernelSpec {
    type: KernelType;
    /** RBF / Laplacian: gamma scale; default 1/D */
    gamma?: number;
    /** Poly: (gamma * x·z + coef0)^degree */
    degree?: number;
    coef0?: number;
    /** For custom kernels — lookup name in KernelRegistry */
    name?: string;
}

export type KELMMode = 'exact' | 'nystrom';

export interface NystromOptions {
    /** # landmarks (m). Required unless using preset. If omitted, defaults to ~sqrt(N). */
    m?: number;
    /** Landmark selection strategy */
    strategy?: 'uniform' | 'kmeans++' | 'preset';
    /** Random seed for sampling / kmeans++ */
    seed?: number;
    /** Provide explicit landmarks */
    preset?: { points?: number[][]; indices?: number[] };
    /** Apply whitening Φ = K_nm · K_mm^{-1/2} (symmetric). Default: false */
    whiten?: boolean;
    /** Jitter added to K_mm before eig-invsqrt. Default: 1e-10 */
    jitter?: number;
}

export interface KernelELMConfig {
    /** Output dimension K (classes for one-hot classification, or dims for regression) */
    outputDim: number;
    /** Kernel parameters */
    kernel: KernelSpec;
    /** Ridge regularization λ (default 1e-2) */
    ridgeLambda?: number;
    /** Convenience for classification helpers */
    task?: 'classification' | 'regression';
    /** Solver mode (default 'exact') */
    mode?: KELMMode;
    /** Nyström options (used when mode='nystrom') */
    nystrom?: NystromOptions;
    /** Logging */
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
    /** Symmetric whitener K_mm^{-1/2} (present when whiten=true) */
    R?: number[][];
}

/* ============== Resolved internal config ============== */

type TaskKind = 'classification' | 'regression';

interface ResolvedNystromOptions {
    m: number | undefined;
    strategy: 'uniform' | 'kmeans++' | 'preset';
    seed: number;
    preset?: { points?: number[][]; indices?: number[] };
    whiten: boolean;
    jitter: number;
}

interface ResolvedKernelELMConfig {
    outputDim: number;
    kernel: KernelSpec;
    ridgeLambda: number;
    task: TaskKind;
    mode: KELMMode;
    nystrom: ResolvedNystromOptions;
    log: { modelName: string; verbose: boolean };
}

/* ================== Kernel registry ================== */

type KernelFn = (x: number[], z: number[]) => number;

export class KernelRegistry {
    private static map = new Map<string, KernelFn>();
    static register(name: string, fn: KernelFn) {
        if (!name || typeof fn !== 'function') throw new Error('KernelRegistry.register: invalid args');
        this.map.set(name, fn);
    }
    static get(name: string): KernelFn {
        const f = this.map.get(name);
        if (!f) throw new Error(`KernelRegistry: kernel "${name}" not found`);
        return f;
    }
}

/* ======================= Utils ======================= */

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

function buildKernel(spec: KernelSpec, dim: number): KernelFn {
    switch (spec.type) {
        case 'custom':
            if (!spec.name) throw new Error('custom kernel requires "name"');
            return KernelRegistry.get(spec.name);
        case 'linear':
            return (x, z) => dot(x, z);
        case 'poly': {
            const gamma = spec.gamma ?? 1 / Math.max(1, dim);
            const degree = spec.degree ?? 2;
            const coef0 = spec.coef0 ?? 1;
            return (x, z) => Math.pow(gamma * dot(x, z) + coef0, degree);
        }
        case 'laplacian': {
            const gamma = spec.gamma ?? 1 / Math.max(1, dim);
            return (x, z) => Math.exp(-gamma * l1(x, z));
        }
        case 'rbf':
        default: {
            const gamma = spec.gamma ?? 1 / Math.max(1, dim);
            return (x, z) => Math.exp(-gamma * l2sq(x, z));
        }
    }
}

/* ============== Landmark selection (Nyström) ============== */

function pickUniform(X: number[][], m: number, seed = 1337): number[] {
    const prng = makePRNG(seed);
    const N = X.length;
    const idx = Array.from({ length: N }, (_, i) => i);
    // Fisher–Yates (only first m)
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
        const c = centers[centers.length - 1];
        for (let i = 0; i < N; i++) {
            const d2 = l2sq(X[i], X[c]);
            if (d2 < D2[i]) D2[i] = d2;
        }
        let sum = 0; for (let i = 0; i < N; i++) sum += D2[i];
        let r = prng() * (sum || 1);
        let next = 0;
        for (let i = 0; i < N; i++) { r -= D2[i]; if (r <= 0) { next = i; break; } }
        centers.push(next);
    }
    return centers;
}

/* ====================== KernelELM ====================== */

export class KernelELM {
    public readonly cfg: ResolvedKernelELMConfig;
    private kernel!: KernelFn;

    // exact mode params
    private Xtrain: number[][] = [];
    private alpha: number[][] = [];

    // nystrom params
    private Z: number[][] = [];   // landmarks (m x D)
    private W: number[][] = [];   // weights in feature space (m x K)
    private R: number[][] = [];   // symmetric whitener K_mm^{-1/2} (m x m) when whitening

    private readonly verbose: boolean;
    private readonly name: string;

    constructor(config: KernelELMConfig) {
        const resolved: ResolvedKernelELMConfig = {
            outputDim: config.outputDim,
            kernel: config.kernel,
            ridgeLambda: config.ridgeLambda ?? 1e-2,
            task: config.task ?? 'classification',
            mode: config.mode ?? 'exact',
            nystrom: {
                m: config.nystrom?.m,
                strategy: config.nystrom?.strategy ?? 'uniform',
                seed: config.nystrom?.seed ?? 1337,
                preset: config.nystrom?.preset,
                whiten: config.nystrom?.whiten ?? false,
                jitter: config.nystrom?.jitter ?? 1e-10,
            },
            log: {
                modelName: config.log?.modelName ?? 'KernelELM',
                verbose: config.log?.verbose ?? false,
            },
        };
        this.cfg = resolved;
        this.verbose = this.cfg.log.verbose;
        this.name = this.cfg.log.modelName;
    }

    /* ------------------- Train ------------------- */

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
                const row = new Array(N); Kmat[i] = row; row[i] = 1;
                for (let j = i + 1; j < N; j++) row[j] = this.kernel(X[i], X[j]);
            }
            for (let i = 1; i < N; i++) for (let j = 0; j < i; j++) Kmat[i][j] = Kmat[j][i];

            // (K + λI) α = Y
            const A = Matrix.addRegularization(Kmat, this.cfg.ridgeLambda + 1e-10);
            const Alpha = Matrix.solveCholesky(A, Y, 1e-12); // (N x K)

            this.Xtrain = X.map(r => r.slice());
            this.alpha = Alpha;
            this.Z = []; this.W = []; this.R = [];
            if (this.verbose) console.log(`✅ [${this.name}] exact fit complete: alpha(${N}x${K})`);
            return;
        }

        // ---------- Nyström ----------
        const ny = this.cfg.nystrom;
        let Z: number[][];
        if (ny.strategy === 'preset' && (ny.preset?.points || ny.preset?.indices)) {
            Z = ny.preset.points ? ny.preset.points.map(r => r.slice())
                : (ny.preset!.indices as number[]).map(i => X[i]);
        } else {
            const m = ny.m ?? Math.max(10, Math.min(300, Math.floor(Math.sqrt(N))));
            const idx = (ny.strategy === 'kmeans++') ? pickKMeansPP(X, m, ny.seed) : pickUniform(X, m, ny.seed);
            Z = idx.map(i => X[i]);
        }
        const m = Z.length;
        if (this.verbose) console.log(`🔹 [${this.name}] Nyström: m=${m}, strategy=${ny.strategy}, whiten=${ny.whiten ? 'on' : 'off'}`);

        // K_nm (N x m)
        const Knm = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(m), xi = X[i];
            for (let j = 0; j < m; j++) row[j] = this.kernel(xi, Z[j]);
            Knm[i] = row;
        }

        // Optional whitening with R = K_mm^{-1/2} (symmetric via eigen)
        let Phi = Knm;
        let R: number[][] = [];
        if (ny.whiten) {
            // K_mm (m x m)
            const Kmm = new Array(m);
            for (let i = 0; i < m; i++) {
                const row = new Array(m); Kmm[i] = row; row[i] = 1;
                for (let j = i + 1; j < m; j++) row[j] = this.kernel(Z[i], Z[j]);
            }
            for (let i = 1; i < m; i++) for (let j = 0; j < i; j++) Kmm[i][j] = Kmm[j][i];

            // R = K_mm^{-1/2} with jitter
            const KmmJ = Matrix.addRegularization(Kmm, ny.jitter);
            R = Matrix.invSqrtSym(KmmJ, ny.jitter);
            Phi = Matrix.multiply(Knm, R); // (N x m)
        }

        // Ridge in feature space: W = (Φᵀ Φ + λ I)^-1 Φᵀ Y   (m x K)
        const PhiT = Matrix.transpose(Phi);
        const G = Matrix.multiply(PhiT, Phi);                         // (m x m)
        const Greg = Matrix.addRegularization(G, this.cfg.ridgeLambda + 1e-10);
        const Rhs = Matrix.multiply(PhiT, Y);                          // (m x K)
        const W = Matrix.solveCholesky(Greg, Rhs, 1e-12);              // (m x K)

        this.Z = Z;
        this.W = W;
        this.R = R; // empty when whiten=false
        this.Xtrain = []; this.alpha = [];
        if (this.verbose) console.log(`✅ [${this.name}] Nyström fit complete: Z(${m}x${D}), W(${m}x${K})`);
    }

    /* --------------- Features / Predict --------------- */

    private featuresFor(X: number[][]): number[][] {
        if (this.cfg.mode === 'exact') {
            const N = this.Xtrain.length, M = X.length;
            const Kqx = new Array(M);
            for (let i = 0; i < M; i++) {
                const row = new Array(N), xi = X[i];
                for (let j = 0; j < N; j++) row[j] = this.kernel(xi, this.Xtrain[j]);
                Kqx[i] = row;
            }
            return Kqx;
        }
        // Nyström
        if (!this.Z.length) throw new Error('featuresFor: Nyström model not fitted');
        const M = X.length, m = this.Z.length;
        const Kxm = new Array(M);
        for (let i = 0; i < M; i++) {
            const row = new Array(m), xi = X[i];
            for (let j = 0; j < m; j++) row[j] = this.kernel(xi, this.Z[j]);
            Kxm[i] = row;
        }
        return this.R.length ? Matrix.multiply(Kxm, this.R) : Kxm;
    }

    /** Raw logits for batch (M x K) */
    predictLogitsFromVectors(X: number[][]): number[][] {
        const Phi = this.featuresFor(X);
        if (this.cfg.mode === 'exact') {
            if (!this.alpha.length) throw new Error('predict: exact model not fitted');
            return Matrix.multiply(Phi, this.alpha);
        }
        if (!this.W.length) throw new Error('predict: Nyström model not fitted');
        return Matrix.multiply(Phi, this.W);
    }

    /** Probabilities for classification; raw scores for regression */
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
     *  - nystrom: Φ = K(X, Z)      (M x m)  or K(X,Z)·R if whiten=true
     */
    getEmbedding(X: number[][]): number[][] {
        return this.featuresFor(X);
    }

    /* -------------------- JSON I/O -------------------- */

    toJSON(): KernelELMJSON {
        const base = { config: { ...this.cfg, __version: 'kelm-2.1.0' } as KernelELMJSON['config'] };
        if (this.cfg.mode === 'exact') {
            return { ...base, X: this.Xtrain, alpha: this.alpha };
        }
        return { ...base, Z: this.Z, W: this.W, R: this.R.length ? this.R : undefined };
    }

    fromJSON(payload: string | KernelELMJSON) {
        const obj: KernelELMJSON = typeof payload === 'string' ? JSON.parse(payload) : payload;

        // Merge config (keep current defaults where missing)
        this.cfg.kernel = { ...obj.config.kernel };
        this.cfg.ridgeLambda = obj.config.ridgeLambda ?? this.cfg.ridgeLambda;
        this.cfg.task = (obj.config.task ?? this.cfg.task) as TaskKind;
        this.cfg.mode = (obj.config.mode ?? this.cfg.mode) as KELMMode;
        this.cfg.nystrom = { ...this.cfg.nystrom, ...(obj.config.nystrom ?? {}) };

        // Restore params
        if (obj.X && obj.alpha) {
            this.Xtrain = obj.X.map(r => r.slice());
            this.alpha = obj.alpha.map(r => r.slice());
            this.Z = []; this.W = []; this.R = [];
            const D = this.Xtrain[0]?.length ?? 1;
            this.kernel = buildKernel(this.cfg.kernel, D);
            return;
        }
        if (obj.Z && obj.W) {
            this.Z = obj.Z.map(r => r.slice());
            this.W = obj.W.map(r => r.slice());
            this.R = obj.R ? obj.R.map(r => r.slice()) : [];
            this.Xtrain = []; this.alpha = [];
            const D = this.Z[0]?.length ?? 1;
            this.kernel = buildKernel(this.cfg.kernel, D);
            return;
        }
        throw new Error('KernelELM.fromJSON: invalid payload');
    }
}
