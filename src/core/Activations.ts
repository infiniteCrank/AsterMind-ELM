// Activations.ts - Common activation functions (with derivatives)

export type ActivationName =
    | 'relu'
    | 'leakyrelu' | 'leaky-relu'
    | 'sigmoid'
    | 'tanh'
    | 'linear' | 'identity' | 'none'
    | 'gelu';

export class Activations {
    /* ========= Forward ========= */

    /** Rectified Linear Unit */
    static relu(x: number): number {
        return x > 0 ? x : 0;
    }

    /** Leaky ReLU with configurable slope for x<0 (default 0.01) */
    static leakyRelu(x: number, alpha = 0.01): number {
        return x >= 0 ? x : alpha * x;
    }

    /** Logistic sigmoid */
    static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    /** Hyperbolic tangent */
    static tanh(x: number): number {
        return Math.tanh(x);
    }

    /** Linear / identity activation */
    static linear(x: number): number {
        return x;
    }

    /**
     * GELU (Gaussian Error Linear Unit), tanh approximation.
     * 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
     */
    static gelu(x: number): number {
        const k = Math.sqrt(2 / Math.PI);
        const u = k * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(u));
    }

    /**
     * Softmax with numerical stability and optional temperature.
     * @param arr logits
     * @param temperature >0; higher = flatter distribution
     */
    static softmax(arr: number[], temperature = 1): number[] {
        const t = Math.max(temperature, 1e-12);
        let max = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i] / t;
            if (v > max) max = v;
        }
        const exps = new Array(arr.length);
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            const e = Math.exp(arr[i] / t - max);
            exps[i] = e;
            sum += e;
        }
        const denom = sum || 1e-12;
        for (let i = 0; i < exps.length; i++) exps[i] = exps[i] / denom;
        return exps;
    }

    /* ========= Derivatives (elementwise) ========= */

    /** d/dx ReLU */
    static dRelu(x: number): number {
        // subgradient at 0 -> 0
        return x > 0 ? 1 : 0;
    }

    /** d/dx LeakyReLU */
    static dLeakyRelu(x: number, alpha = 0.01): number {
        return x >= 0 ? 1 : alpha;
    }

    /** d/dx Sigmoid = s(x)*(1-s(x)) */
    static dSigmoid(x: number): number {
        const s = Activations.sigmoid(x);
        return s * (1 - s);
    }

    /** d/dx tanh = 1 - tanh(x)^2 */
    static dTanh(x: number): number {
        const t = Math.tanh(x);
        return 1 - t * t;
    }

    /** d/dx Linear = 1 */
    static dLinear(_: number): number {
        return 1;
    }

    /**
     * d/dx GELU (tanh approximation)
     * 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2) * du/dx
     * where u = k*(x + 0.044715 x^3), du/dx = k*(1 + 0.134145 x^2), k = sqrt(2/pi)
     */
    static dGelu(x: number): number {
        const k = Math.sqrt(2 / Math.PI);
        const x2 = x * x;
        const u = k * (x + 0.044715 * x * x2);
        const t = Math.tanh(u);
        const sech2 = 1 - t * t;
        const du = k * (1 + 0.134145 * x2);
        return 0.5 * (1 + t) + 0.5 * x * sech2 * du;
    }

    /* ========= Apply helpers ========= */

    /** Apply an elementwise activation across a 2D matrix, returning a new matrix. */
    static apply(matrix: number[][], fn: (x: number) => number): number[][] {
        const out: number[][] = new Array(matrix.length);
        for (let i = 0; i < matrix.length; i++) {
            const row = matrix[i];
            const r: number[] = new Array(row.length);
            for (let j = 0; j < row.length; j++) r[j] = fn(row[j]);
            out[i] = r;
        }
        return out;
    }

    /** Apply an elementwise derivative across a 2D matrix, returning a new matrix. */
    static applyDerivative(matrix: number[][], dfn: (x: number) => number): number[][] {
        const out: number[][] = new Array(matrix.length);
        for (let i = 0; i < matrix.length; i++) {
            const row = matrix[i];
            const r: number[] = new Array(row.length);
            for (let j = 0; j < row.length; j++) r[j] = dfn(row[j]);
            out[i] = r;
        }
        return out;
    }

    /* ========= Getters ========= */

    /**
     * Get an activation function by name. Case-insensitive.
     * For leaky ReLU, you can pass { alpha } to override the negative slope.
     */
    static get(name: string, opts?: { alpha?: number }): (x: number) => number {
        const key = name.toLowerCase() as ActivationName;
        switch (key) {
            case 'relu': return this.relu;
            case 'leakyrelu':
            case 'leaky-relu': {
                const alpha = opts?.alpha ?? 0.01;
                return (x: number) => this.leakyRelu(x, alpha);
            }
            case 'sigmoid': return this.sigmoid;
            case 'tanh': return this.tanh;
            case 'linear':
            case 'identity':
            case 'none': return this.linear;
            case 'gelu': return this.gelu;
            default:
                throw new Error(`Unknown activation: ${name}`);
        }
    }

    /** Get derivative function by name (mirrors get). */
    static getDerivative(name: string, opts?: { alpha?: number }): (x: number) => number {
        const key = name.toLowerCase() as ActivationName;
        switch (key) {
            case 'relu': return this.dRelu;
            case 'leakyrelu':
            case 'leaky-relu': {
                const alpha = opts?.alpha ?? 0.01;
                return (x: number) => this.dLeakyRelu(x, alpha);
            }
            case 'sigmoid': return this.dSigmoid;
            case 'tanh': return this.dTanh;
            case 'linear':
            case 'identity':
            case 'none': return this.dLinear;
            case 'gelu': return this.dGelu;
            default:
                throw new Error(`Unknown activation derivative: ${name}`);
        }
    }

    /** Get both forward and derivative together. */
    static getPair(name: string, opts?: { alpha?: number }): {
        f: (x: number) => number;
        df: (x: number) => number;
    } {
        return { f: this.get(name, opts), df: this.getDerivative(name, opts) };
    }

    /* ========= Optional: Softmax Jacobian (for research/tools) ========= */

    /**
     * Given softmax probabilities p, returns the Jacobian J = diag(p) - p p^T
     * (Useful for analysis; not typically needed for ELM.)
     */
    static softmaxJacobian(p: number[]): number[][] {
        const n = p.length;
        const J = new Array(n);
        for (let i = 0; i < n; i++) {
            const row = new Array(n);
            for (let j = 0; j < n; j++) {
                row[j] = (i === j ? p[i] : 0) - p[i] * p[j];
            }
            J[i] = row;
        }
        return J;
    }
}
