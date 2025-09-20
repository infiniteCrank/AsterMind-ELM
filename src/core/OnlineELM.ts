// OnlineELM.ts — Online / OS-ELM with RLS updates

import { Matrix } from './Matrix';
import { Activations } from './Activations';
import { WeightInit } from './ELMConfig'

export interface OnlineELMConfig {
    inputDim: number;
    outputDim: number;
    hiddenUnits: number;

    activation?: string;               // 'relu' | 'tanh' | 'linear' | 'gelu' | 'leakyrelu'
    ridgeLambda?: number;              // λ for ridge / P stability
    seed?: number;                     // deterministic init
    weightInit?: WeightInit;           // random feature init
    forgettingFactor?: number;         // ρ in (0,1]; 1=no forgetting

    log?: { verbose?: boolean; modelName?: string };
}

export interface OnlineELMModel {
    W: number[][];
    b: number[][];
    beta: number[][];
    P: number[][];
}

/* ========== utils ========== */
const EPS = 1e-10;

function makePRNG(seed = 123456789) {
    let s = seed | 0 || 1;
    return () => {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
        return ((s >>> 0) / 0xffffffff);
    };
}

/* ========== Online ELM (RLS) ========== */
export class OnlineELM {
    readonly inputDim: number;
    readonly outputDim: number;
    readonly hiddenUnits: number;

    public activation: string;         // <- mutable so loadFromJSON can adjust
    public ridgeLambda: number;        // <- mutable for loadFromJSON
    readonly weightInit: WeightInit;
    readonly forgettingFactor: number;

    readonly verbose: boolean;
    readonly modelName: string;

    private rng: () => number;
    private actFn: (x: number) => number; // cached activation fn

    // Parameters (null until init())
    public W: number[][];                // hidden x input
    public b: number[][];                // hidden x 1
    public beta: number[][] | null;      // hidden x output
    public P: number[][] | null;         // hidden x hidden

    constructor(cfg: OnlineELMConfig) {
        this.inputDim = cfg.inputDim | 0;
        this.outputDim = cfg.outputDim | 0;
        this.hiddenUnits = cfg.hiddenUnits | 0;
        if (this.inputDim <= 0 || this.outputDim <= 0 || this.hiddenUnits <= 0) {
            throw new Error(`OnlineELM: invalid dims (inputDim=${this.inputDim}, outputDim=${this.outputDim}, hidden=${this.hiddenUnits})`);
        }

        this.activation = cfg.activation ?? 'relu';
        this.ridgeLambda = Math.max(cfg.ridgeLambda ?? 1e-2, EPS);
        this.weightInit = cfg.weightInit ?? 'xavier';
        this.forgettingFactor = Math.max(Math.min(cfg.forgettingFactor ?? 1.0, 1.0), 1e-4);

        this.verbose = cfg.log?.verbose ?? false;
        this.modelName = cfg.log?.modelName ?? 'Online ELM';

        const seed = cfg.seed ?? 1337;
        this.rng = makePRNG(seed);
        this.actFn = Activations.get(this.activation);

        // Random features
        this.W = this.initW(this.hiddenUnits, this.inputDim);
        this.b = this.initB(this.hiddenUnits);

        // Not initialized yet — init() will set these
        this.beta = null;
        this.P = null;
    }

    /* ===== init helpers ===== */
    private xavierLimit(fanIn: number, fanOut: number) { return Math.sqrt(6 / (fanIn + fanOut)); }
    private heLimit(fanIn: number) { return Math.sqrt(6 / fanIn); }

    private initW(rows: number, cols: number): number[][] {
        let limit = 1;
        if (this.weightInit === 'xavier') {
            limit = this.xavierLimit(cols, rows);
            if (this.verbose) console.log(`✨ [${this.modelName}] Xavier W ~ U(±${limit.toFixed(4)})`);
        } else if (this.weightInit === 'he') {
            limit = this.heLimit(cols);
            if (this.verbose) console.log(`✨ [${this.modelName}] He W ~ U(±${limit.toFixed(4)})`);
        } else if (this.verbose) {
            console.log(`✨ [${this.modelName}] Uniform W ~ U(±1)`);
        }
        const rnd = () => (this.rng() * 2 - 1) * limit;
        return Array.from({ length: rows }, () => Array.from({ length: cols }, rnd));
    }

    private initB(rows: number): number[][] {
        const rnd = () => (this.rng() * 2 - 1) * 0.01;
        return Array.from({ length: rows }, () => [rnd()]);
    }

    private hidden(X: number[][]): number[][] {
        const tempH = Matrix.multiply(X, Matrix.transpose(this.W)); // (n x hidden)
        const f = this.actFn;
        return tempH.map(row => row.map((v, j) => f(v + this.b[j][0])));
    }

    /* ===== public API ===== */

    /** Initialize β and P from a batch (ridge): P0=(HᵀH+λI)^-1, β0=P0 HᵀY */
    public init(X0: number[][], Y0: number[][]): void {
        if (!X0?.length || !Y0?.length) throw new Error('init: empty X0 or Y0');
        if (X0.length !== Y0.length) throw new Error(`init: X0 rows ${X0.length} != Y0 rows ${Y0.length}`);
        if (X0[0].length !== this.inputDim) throw new Error(`init: X0 cols ${X0[0].length} != inputDim ${this.inputDim}`);
        if (Y0[0].length !== this.outputDim) throw new Error(`init: Y0 cols ${Y0[0].length} != outputDim ${this.outputDim}`);

        const H0 = this.hidden(X0);                  // (n x h)
        const Ht = Matrix.transpose(H0);             // (h x n)
        const A = Matrix.addRegularization(
            Matrix.multiply(Ht, H0),
            this.ridgeLambda + 1e-10
        );                                           // (h x h)
        const R = Matrix.multiply(Ht, Y0);           // (h x k)
        const P0 = Matrix.solveCholesky(A, Matrix.identity(this.hiddenUnits), 1e-10); // A^-1
        const B0 = Matrix.multiply(P0, R);           // (h x k)

        this.P = P0;
        this.beta = B0;

        if (this.verbose) console.log(`✅ [${this.modelName}] init: n=${X0.length}, hidden=${this.hiddenUnits}, out=${this.outputDim}`);
    }

    /** If not initialized, init(); otherwise RLS update. */
    public fit(X: number[][], Y: number[][]): void {
        if (!X?.length || !Y?.length) throw new Error('fit: empty X or Y');
        if (X.length !== Y.length) throw new Error(`fit: X rows ${X.length} != Y rows ${Y.length}`);
        if (!this.P || !this.beta) this.init(X, Y);
        else this.update(X, Y);
    }

    /**
     * RLS / OS-ELM update with forgetting ρ:
     *   S = I + HPHᵀ
     *   K = P Hᵀ S^-1
     *   β ← β + K (Y - Hβ)
     *   P ← (P - K H P) / ρ
     */
    public update(X: number[][], Y: number[][]): void {
        if (!X?.length || !Y?.length) throw new Error('update: empty X or Y');
        if (X.length !== Y.length) throw new Error(`update: X rows ${X.length} != Y rows ${Y.length}`);
        if (!this.P || !this.beta) throw new Error('update: model not initialized (call init() first)');

        const n = X.length;
        const H = this.hidden(X);                       // (n x h)
        const Ht = Matrix.transpose(H);                 // (h x n)

        const rho = this.forgettingFactor;
        let P = this.P;
        if (rho < 1.0) {
            // Equivalent to P <- P / ρ (more responsive to new data)
            P = P.map(row => row.map(v => v / rho));
        }

        // S = I + H P Hᵀ  (n x n, SPD)
        const HP = Matrix.multiply(H, P);               // (n x h)
        const HPHt = Matrix.multiply(HP, Ht);           // (n x n)
        const S = Matrix.add(HPHt, Matrix.identity(n));
        const S_inv = Matrix.solveCholesky(S, Matrix.identity(n), 1e-10);

        // K = P Hᵀ S^-1  (h x n)
        const PHt = Matrix.multiply(P, Ht);             // (h x n)
        const K = Matrix.multiply(PHt, S_inv);          // (h x n)

        // Innovation: (Y - Hβ)  (n x k)
        const Hbeta = Matrix.multiply(H, this.beta);
        const innov = Y.map((row, i) => row.map((yij, j) => yij - Hbeta[i][j]));

        // β ← β + K * innov
        const Delta = Matrix.multiply(K, innov);        // (h x k)
        this.beta = this.beta.map((row, i) => row.map((bij, j) => bij + Delta[i][j]));

        // P ← P - K H P
        const KH = Matrix.multiply(K, H);               // (h x h)
        const KHP = Matrix.multiply(KH, P);             // (h x h)
        this.P = P.map((row, i) => row.map((pij, j) => pij - KHP[i][j]));

        if (this.verbose) {
            const diagAvg = this.P.reduce((s, r, i) => s + r[i], 0) / this.P.length;
            console.log(`🔁 [${this.modelName}] update: n=${n}, avg diag(P)≈${diagAvg.toFixed(6)}`);
        }
    }

    /* ===== Prediction ===== */

    private logitsFromVectors(X: number[][]): number[][] {
        if (!this.beta) throw new Error('predict: model not initialized');
        const H = this.hidden(X);
        return Matrix.multiply(H, this.beta);
    }

    public predictLogitsFromVector(x: number[]): number[] {
        return this.logitsFromVectors([x])[0];
    }

    public predictLogitsFromVectors(X: number[][]): number[][] {
        return this.logitsFromVectors(X);
    }

    public predictProbaFromVector(x: number[]): number[] {
        return Activations.softmax(this.predictLogitsFromVector(x));
    }

    public predictProbaFromVectors(X: number[][]): number[][] {
        return this.predictLogitsFromVectors(X).map(Activations.softmax);
    }

    public predictTopKFromVector(x: number[], k = 5): Array<{ index: number; prob: number }> {
        const p = this.predictProbaFromVector(x);
        const kk = Math.max(1, Math.min(k, p.length));
        return p.map((prob, index) => ({ index, prob }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, kk);
    }

    public predictTopKFromVectors(X: number[][], k = 5): Array<Array<{ index: number; prob: number }>> {
        return this.predictProbaFromVectors(X).map(p => {
            const kk = Math.max(1, Math.min(k, p.length));
            return p.map((prob, index) => ({ index, prob }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, kk);
        });
    }

    /* ===== Serialization ===== */

    public toJSON(includeP = false): OnlineELMJSON {
        if (!this.beta || !this.P) throw new Error('toJSON: model not initialized');
        const cfg = {
            hiddenUnits: this.hiddenUnits,
            inputDim: this.inputDim,
            outputDim: this.outputDim,
            activation: this.activation,
            ridgeLambda: this.ridgeLambda,
            weightInit: this.weightInit,
            forgettingFactor: this.forgettingFactor,
            __version: 'online-elm-1.0.0',
        };
        const o: OnlineELMJSON = { W: this.W, b: this.b, B: this.beta, config: cfg };
        if (includeP) o.P = this.P;
        return o;
    }

    public loadFromJSON(json: string | OnlineELMJSON): void {
        const parsed = typeof json === 'string' ? JSON.parse(json) : json;
        const { W, b, B, P, config } = parsed;
        if (!W || !b || !B) throw new Error('loadFromJSON: missing W/b/B');

        if (W.length !== this.hiddenUnits || W[0].length !== this.inputDim) {
            throw new Error(`loadFromJSON: mismatched W shape (${W.length}x${W[0].length})`);
        }
        if (b.length !== this.hiddenUnits || b[0].length !== 1) {
            throw new Error(`loadFromJSON: mismatched b shape (${b.length}x${b[0].length})`);
        }
        if (B.length !== this.hiddenUnits || B[0].length !== this.outputDim) {
            throw new Error(`loadFromJSON: mismatched B shape (${B.length}x${B[0].length})`);
        }

        this.W = W;
        this.b = b;
        this.beta = B;
        this.P = P ?? null;

        if (config?.activation) {
            this.activation = config.activation;
            this.actFn = Activations.get(this.activation);   // refresh cache
        }
        if (config?.ridgeLambda) this.ridgeLambda = config.ridgeLambda;

        if (this.verbose) console.log(`✅ [${this.modelName}] model loaded (v=${config?.__version ?? 'n/a'})`);
    }
}

export interface OnlineELMJSON {
    W: number[][];
    b: number[][];
    B: number[][];
    P?: number[][];
    config: {
        hiddenUnits: number;
        inputDim: number;
        outputDim: number;
        activation: string;
        ridgeLambda: number;
        weightInit: WeightInit;
        forgettingFactor: number;
        __version: string;
    };
}
