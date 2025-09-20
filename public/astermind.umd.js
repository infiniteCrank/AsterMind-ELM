(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.astermind = {}));
})(this, (function (exports) { 'use strict';

    // Activations.ts - Common activation functions (with derivatives)
    class Activations {
        /* ========= Forward ========= */
        /** Rectified Linear Unit */
        static relu(x) {
            return x > 0 ? x : 0;
        }
        /** Leaky ReLU with configurable slope for x<0 (default 0.01) */
        static leakyRelu(x, alpha = 0.01) {
            return x >= 0 ? x : alpha * x;
        }
        /** Logistic sigmoid */
        static sigmoid(x) {
            return 1 / (1 + Math.exp(-x));
        }
        /** Hyperbolic tangent */
        static tanh(x) {
            return Math.tanh(x);
        }
        /** Linear / identity activation */
        static linear(x) {
            return x;
        }
        /**
         * GELU (Gaussian Error Linear Unit), tanh approximation.
         * 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
         */
        static gelu(x) {
            const k = Math.sqrt(2 / Math.PI);
            const u = k * (x + 0.044715 * x * x * x);
            return 0.5 * x * (1 + Math.tanh(u));
        }
        /**
         * Softmax with numerical stability and optional temperature.
         * @param arr logits
         * @param temperature >0; higher = flatter distribution
         */
        static softmax(arr, temperature = 1) {
            const t = Math.max(temperature, 1e-12);
            let max = -Infinity;
            for (let i = 0; i < arr.length; i++) {
                const v = arr[i] / t;
                if (v > max)
                    max = v;
            }
            const exps = new Array(arr.length);
            let sum = 0;
            for (let i = 0; i < arr.length; i++) {
                const e = Math.exp(arr[i] / t - max);
                exps[i] = e;
                sum += e;
            }
            const denom = sum || 1e-12;
            for (let i = 0; i < exps.length; i++)
                exps[i] = exps[i] / denom;
            return exps;
        }
        /* ========= Derivatives (elementwise) ========= */
        /** d/dx ReLU */
        static dRelu(x) {
            // subgradient at 0 -> 0
            return x > 0 ? 1 : 0;
        }
        /** d/dx LeakyReLU */
        static dLeakyRelu(x, alpha = 0.01) {
            return x >= 0 ? 1 : alpha;
        }
        /** d/dx Sigmoid = s(x)*(1-s(x)) */
        static dSigmoid(x) {
            const s = Activations.sigmoid(x);
            return s * (1 - s);
        }
        /** d/dx tanh = 1 - tanh(x)^2 */
        static dTanh(x) {
            const t = Math.tanh(x);
            return 1 - t * t;
        }
        /** d/dx Linear = 1 */
        static dLinear(_) {
            return 1;
        }
        /**
         * d/dx GELU (tanh approximation)
         * 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2) * du/dx
         * where u = k*(x + 0.044715 x^3), du/dx = k*(1 + 0.134145 x^2), k = sqrt(2/pi)
         */
        static dGelu(x) {
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
        static apply(matrix, fn) {
            const out = new Array(matrix.length);
            for (let i = 0; i < matrix.length; i++) {
                const row = matrix[i];
                const r = new Array(row.length);
                for (let j = 0; j < row.length; j++)
                    r[j] = fn(row[j]);
                out[i] = r;
            }
            return out;
        }
        /** Apply an elementwise derivative across a 2D matrix, returning a new matrix. */
        static applyDerivative(matrix, dfn) {
            const out = new Array(matrix.length);
            for (let i = 0; i < matrix.length; i++) {
                const row = matrix[i];
                const r = new Array(row.length);
                for (let j = 0; j < row.length; j++)
                    r[j] = dfn(row[j]);
                out[i] = r;
            }
            return out;
        }
        /* ========= Getters ========= */
        /**
         * Get an activation function by name. Case-insensitive.
         * For leaky ReLU, you can pass { alpha } to override the negative slope.
         */
        static get(name, opts) {
            var _a;
            const key = name.toLowerCase();
            switch (key) {
                case 'relu': return this.relu;
                case 'leakyrelu':
                case 'leaky-relu': {
                    const alpha = (_a = opts === null || opts === void 0 ? void 0 : opts.alpha) !== null && _a !== void 0 ? _a : 0.01;
                    return (x) => this.leakyRelu(x, alpha);
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
        static getDerivative(name, opts) {
            var _a;
            const key = name.toLowerCase();
            switch (key) {
                case 'relu': return this.dRelu;
                case 'leakyrelu':
                case 'leaky-relu': {
                    const alpha = (_a = opts === null || opts === void 0 ? void 0 : opts.alpha) !== null && _a !== void 0 ? _a : 0.01;
                    return (x) => this.dLeakyRelu(x, alpha);
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
        static getPair(name, opts) {
            return { f: this.get(name, opts), df: this.getDerivative(name, opts) };
        }
        /* ========= Optional: Softmax Jacobian (for research/tools) ========= */
        /**
         * Given softmax probabilities p, returns the Jacobian J = diag(p) - p p^T
         * (Useful for analysis; not typically needed for ELM.)
         */
        static softmaxJacobian(p) {
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

    // Matrix.ts — tolerant, safe helpers with dimension checks and stable ops
    class DimError extends Error {
        constructor(msg) {
            super(msg);
            this.name = 'DimError';
        }
    }
    const EPS$4 = 1e-12;
    /* ===================== Array-like coercion helpers ===================== */
    // ✅ Narrow to ArrayLike<number> so numeric indexing is allowed
    function isArrayLikeRow(row) {
        return row != null && typeof row.length === 'number';
    }
    /**
     * Coerce any 2D array-like into a strict rectangular number[][]
     * - If width is not provided, infer from the first row's length
     * - Pads/truncates to width
     * - Non-finite values become 0
     */
    function ensureRectNumber2D(M, width, name = 'matrix') {
        if (!M || typeof M.length !== 'number') {
            throw new DimError(`${name} must be a non-empty 2D array`);
        }
        const rows = Array.from(M);
        if (rows.length === 0)
            throw new DimError(`${name} is empty`);
        const first = rows[0];
        if (!isArrayLikeRow(first))
            throw new DimError(`${name} row 0 missing/invalid`);
        const C = ((width !== null && width !== void 0 ? width : first.length) | 0);
        if (C <= 0)
            throw new DimError(`${name} has zero width`);
        const out = new Array(rows.length);
        for (let r = 0; r < rows.length; r++) {
            const src = rows[r];
            const rr = new Array(C);
            if (isArrayLikeRow(src)) {
                const sr = src; // ✅ typed
                for (let c = 0; c < C; c++) {
                    const v = sr[c];
                    rr[c] = Number.isFinite(v) ? Number(v) : 0;
                }
            }
            else {
                for (let c = 0; c < C; c++)
                    rr[c] = 0;
            }
            out[r] = rr;
        }
        return out;
    }
    /**
     * Relaxed rectangularity check:
     * - Accepts any array-like rows (typed arrays included)
     * - Verifies consistent width and finite numbers
     */
    function assertRect(A, name = 'matrix') {
        if (!A || typeof A.length !== 'number') {
            throw new DimError(`${name} must be a non-empty 2D array`);
        }
        const rows = A.length | 0;
        if (rows <= 0)
            throw new DimError(`${name} must be a non-empty 2D array`);
        const first = A[0];
        if (!isArrayLikeRow(first))
            throw new DimError(`${name} row 0 missing/invalid`);
        const C = first.length | 0;
        if (C <= 0)
            throw new DimError(`${name} must have positive column count`);
        for (let r = 0; r < rows; r++) {
            const rowAny = A[r];
            if (!isArrayLikeRow(rowAny)) {
                throw new DimError(`${name} row ${r} invalid`);
            }
            const row = rowAny; // ✅ typed
            if ((row.length | 0) !== C) {
                throw new DimError(`${name} has ragged rows: row 0 = ${C} cols, row ${r} = ${row.length} cols`);
            }
            for (let c = 0; c < C; c++) {
                const v = row[c];
                if (!Number.isFinite(v)) {
                    throw new DimError(`${name} row ${r}, col ${c} is not finite: ${v}`);
                }
            }
        }
    }
    function assertMulDims(A, B) {
        assertRect(A, 'A');
        assertRect(B, 'B');
        const nA = A[0].length;
        const mB = B.length;
        if (nA !== mB) {
            throw new DimError(`matmul dims mismatch: A(${A.length}x${nA}) * B(${mB}x${B[0].length})`);
        }
    }
    function isSquare(A) {
        return isArrayLikeRow(A === null || A === void 0 ? void 0 : A[0]) && (A.length === (A[0].length | 0));
    }
    function isSymmetric(A, tol = 1e-10) {
        if (!isSquare(A))
            return false;
        const n = A.length;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(A[i][j] - A[j][i]) > tol)
                    return false;
            }
        }
        return true;
    }
    /* ============================== Matrix ============================== */
    class Matrix {
        /* ========= constructors / basics ========= */
        static shape(A) {
            assertRect(A, 'A');
            return [A.length, A[0].length];
        }
        static clone(A) {
            assertRect(A, 'A');
            return ensureRectNumber2D(A, A[0].length, 'A(clone)');
        }
        static zeros(rows, cols) {
            const out = new Array(rows);
            for (let i = 0; i < rows; i++)
                out[i] = new Array(cols).fill(0);
            return out;
        }
        static identity(n) {
            const I = Matrix.zeros(n, n);
            for (let i = 0; i < n; i++)
                I[i][i] = 1;
            return I;
        }
        static transpose(A) {
            assertRect(A, 'A');
            const m = A.length, n = A[0].length;
            const T = Matrix.zeros(n, m);
            for (let i = 0; i < m; i++) {
                const Ai = A[i];
                for (let j = 0; j < n; j++)
                    T[j][i] = Number(Ai[j]);
            }
            return T;
        }
        /* ========= algebra ========= */
        static add(A, B) {
            A = ensureRectNumber2D(A, undefined, 'A');
            B = ensureRectNumber2D(B, undefined, 'B');
            assertRect(A, 'A');
            assertRect(B, 'B');
            if (A.length !== B.length || A[0].length !== B[0].length) {
                throw new DimError(`add dims mismatch: A(${A.length}x${A[0].length}) vs B(${B.length}x${B[0].length})`);
            }
            const m = A.length, n = A[0].length;
            const C = Matrix.zeros(m, n);
            for (let i = 0; i < m; i++) {
                const Ai = A[i], Bi = B[i], Ci = C[i];
                for (let j = 0; j < n; j++)
                    Ci[j] = Ai[j] + Bi[j];
            }
            return C;
        }
        /** Adds lambda to the diagonal (ridge regularization) */
        static addRegularization(A, lambda = 1e-6) {
            A = ensureRectNumber2D(A, undefined, 'A');
            assertRect(A, 'A');
            if (!isSquare(A)) {
                throw new DimError(`addRegularization expects square matrix, got ${A.length}x${A[0].length}`);
            }
            const C = Matrix.clone(A);
            for (let i = 0; i < C.length; i++)
                C[i][i] += lambda;
            return C;
        }
        static multiply(A, B) {
            A = ensureRectNumber2D(A, undefined, 'A');
            B = ensureRectNumber2D(B, undefined, 'B');
            assertMulDims(A, B);
            const m = A.length, n = B.length, p = B[0].length;
            const C = Matrix.zeros(m, p);
            for (let i = 0; i < m; i++) {
                const Ai = A[i];
                for (let k = 0; k < n; k++) {
                    const aik = Number(Ai[k]);
                    const Bk = B[k];
                    for (let j = 0; j < p; j++)
                        C[i][j] += aik * Number(Bk[j]);
                }
            }
            return C;
        }
        static multiplyVec(A, v) {
            A = ensureRectNumber2D(A, undefined, 'A');
            assertRect(A, 'A');
            if (!v || typeof v.length !== 'number') {
                throw new DimError(`matvec expects vector 'v' with length ${A[0].length}`);
            }
            if (A[0].length !== v.length) {
                throw new DimError(`matvec dims mismatch: A cols ${A[0].length} vs v len ${v.length}`);
            }
            const m = A.length, n = v.length;
            const out = new Array(m).fill(0);
            for (let i = 0; i < m; i++) {
                const Ai = A[i];
                let s = 0;
                for (let j = 0; j < n; j++)
                    s += Number(Ai[j]) * Number(v[j]);
                out[i] = s;
            }
            return out;
        }
        /* ========= decompositions / solve ========= */
        static cholesky(A, jitter = 0) {
            A = ensureRectNumber2D(A, undefined, 'A');
            assertRect(A, 'A');
            if (!isSquare(A))
                throw new DimError(`cholesky expects square matrix, got ${A.length}x${A[0].length}`);
            const n = A.length;
            const L = Matrix.zeros(n, n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j <= i; j++) {
                    let sum = A[i][j];
                    for (let k = 0; k < j; k++)
                        sum -= L[i][k] * L[j][k];
                    if (i === j) {
                        const v = sum + jitter;
                        L[i][j] = Math.sqrt(Math.max(v, EPS$4));
                    }
                    else {
                        L[i][j] = sum / L[j][j];
                    }
                }
            }
            return L;
        }
        static solveCholesky(A, B, jitter = 1e-10) {
            A = ensureRectNumber2D(A, undefined, 'A');
            B = ensureRectNumber2D(B, undefined, 'B');
            assertRect(A, 'A');
            assertRect(B, 'B');
            if (!isSquare(A) || A.length !== B.length) {
                throw new DimError(`solveCholesky dims: A(${A.length}x${A[0].length}) vs B(${B.length}x${B[0].length})`);
            }
            const n = A.length, k = B[0].length;
            const L = Matrix.cholesky(A, jitter);
            // Solve L Z = B (forward)
            const Z = Matrix.zeros(n, k);
            for (let i = 0; i < n; i++) {
                for (let c = 0; c < k; c++) {
                    let s = B[i][c];
                    for (let p = 0; p < i; p++)
                        s -= L[i][p] * Z[p][c];
                    Z[i][c] = s / L[i][i];
                }
            }
            // Solve L^T X = Z (backward)
            const X = Matrix.zeros(n, k);
            for (let i = n - 1; i >= 0; i--) {
                for (let c = 0; c < k; c++) {
                    let s = Z[i][c];
                    for (let p = i + 1; p < n; p++)
                        s -= L[p][i] * X[p][c];
                    X[i][c] = s / L[i][i];
                }
            }
            return X;
        }
        static inverse(A) {
            A = ensureRectNumber2D(A, undefined, 'A');
            assertRect(A, 'A');
            if (!isSquare(A))
                throw new DimError(`inverse expects square matrix, got ${A.length}x${A[0].length}`);
            const n = A.length;
            const M = Matrix.clone(A);
            const I = Matrix.identity(n);
            // Augment [M | I]
            const aug = new Array(n);
            for (let i = 0; i < n; i++)
                aug[i] = M[i].concat(I[i]);
            const cols = 2 * n;
            for (let p = 0; p < n; p++) {
                // Pivot
                let maxRow = p, maxVal = Math.abs(aug[p][p]);
                for (let r = p + 1; r < n; r++) {
                    const v = Math.abs(aug[r][p]);
                    if (v > maxVal) {
                        maxVal = v;
                        maxRow = r;
                    }
                }
                if (maxVal < EPS$4)
                    throw new Error('Matrix is singular or ill-conditioned');
                if (maxRow !== p) {
                    const tmp = aug[p];
                    aug[p] = aug[maxRow];
                    aug[maxRow] = tmp;
                }
                // Normalize pivot row
                const piv = aug[p][p];
                const invPiv = 1 / piv;
                for (let c = 0; c < cols; c++)
                    aug[p][c] *= invPiv;
                // Eliminate other rows
                for (let r = 0; r < n; r++) {
                    if (r === p)
                        continue;
                    const f = aug[r][p];
                    if (Math.abs(f) < EPS$4)
                        continue;
                    for (let c = 0; c < cols; c++)
                        aug[r][c] -= f * aug[p][c];
                }
            }
            // Extract right half as inverse
            const inv = Matrix.zeros(n, n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++)
                    inv[i][j] = aug[i][n + j];
            }
            return inv;
        }
        /* ========= helpers ========= */
        static inverseSPDOrFallback(A) {
            if (isSymmetric(A)) {
                try {
                    return Matrix.solveCholesky(A, Matrix.identity(A.length), 1e-10);
                }
                catch (_a) {
                    // fall through
                }
            }
            return Matrix.inverse(A);
        }
        /* ========= Symmetric Eigen (Jacobi) & Inverse Square Root ========= */
        static assertSquare(A, ctx = 'Matrix') {
            assertRect(A, ctx);
            if (!isSquare(A)) {
                throw new DimError(`${ctx}: expected square matrix, got ${A.length}x${A[0].length}`);
            }
        }
        static eigSym(A, maxIter = 64, tol = 1e-12) {
            A = ensureRectNumber2D(A, undefined, 'eigSym/A');
            Matrix.assertSquare(A, 'eigSym');
            const n = A.length;
            const B = Matrix.clone(A);
            let V = Matrix.identity(n);
            const abs = Math.abs;
            const offdiagNorm = () => {
                let s = 0;
                for (let i = 0; i < n; i++) {
                    for (let j = i + 1; j < n; j++) {
                        const v = B[i][j];
                        s += v * v;
                    }
                }
                return Math.sqrt(s);
            };
            for (let it = 0; it < maxIter; it++) {
                if (offdiagNorm() <= tol)
                    break;
                let p = 0, q = 1, max = 0;
                for (let i = 0; i < n; i++) {
                    for (let j = i + 1; j < n; j++) {
                        const v = abs(B[i][j]);
                        if (v > max) {
                            max = v;
                            p = i;
                            q = j;
                        }
                    }
                }
                if (max <= tol)
                    break;
                const app = B[p][p], aqq = B[q][q], apq = B[p][q];
                const tau = (aqq - app) / (2 * apq);
                const t = Math.sign(tau) / (abs(tau) + Math.sqrt(1 + tau * tau));
                const c = 1 / Math.sqrt(1 + t * t);
                const s = t * c;
                const Bpp = c * c * app - 2 * s * c * apq + s * s * aqq;
                const Bqq = s * s * app + 2 * s * c * apq + c * c * aqq;
                B[p][p] = Bpp;
                B[q][q] = Bqq;
                B[p][q] = B[q][p] = 0;
                for (let k = 0; k < n; k++) {
                    if (k === p || k === q)
                        continue;
                    const aip = B[k][p], aiq = B[k][q];
                    const new_kp = c * aip - s * aiq;
                    const new_kq = s * aip + c * aiq;
                    B[k][p] = B[p][k] = new_kp;
                    B[k][q] = B[q][k] = new_kq;
                }
                for (let k = 0; k < n; k++) {
                    const vip = V[k][p], viq = V[k][q];
                    V[k][p] = c * vip - s * viq;
                    V[k][q] = s * vip + c * viq;
                }
            }
            const vals = new Array(n);
            for (let i = 0; i < n; i++)
                vals[i] = B[i][i];
            const order = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
            const values = order.map(i => vals[i]);
            const vectors = Matrix.zeros(n, n);
            for (let r = 0; r < n; r++) {
                for (let c = 0; c < n; c++)
                    vectors[r][c] = V[r][order[c]];
            }
            return { values, vectors };
        }
        static invSqrtSym(A, eps = 1e-10) {
            A = ensureRectNumber2D(A, undefined, 'invSqrtSym/A');
            Matrix.assertSquare(A, 'invSqrtSym');
            const { values, vectors: U } = Matrix.eigSym(A);
            const n = values.length;
            const Dm12 = Matrix.zeros(n, n);
            for (let i = 0; i < n; i++) {
                const lam = Math.max(values[i], eps);
                Dm12[i][i] = 1 / Math.sqrt(lam);
            }
            const UD = Matrix.multiply(U, Dm12);
            return Matrix.multiply(UD, Matrix.transpose(U));
        }
    }

    // ELMConfig.ts - Configuration interfaces, defaults, helpers for ELM-based models
    /* =========== Defaults =========== */
    const defaultBase = {
        hiddenUnits: 50,
        activation: 'relu',
        ridgeLambda: 1e-2,
        weightInit: 'xavier',
        seed: 1337,
        dropout: 0,
        log: { verbose: true, toFile: false, modelName: 'Unnamed ELM Model', level: 'info' },
    };
    const defaultNumericConfig = Object.assign(Object.assign({}, defaultBase), { useTokenizer: false });
    const defaultTextConfig = Object.assign(Object.assign({}, defaultBase), { useTokenizer: true, maxLen: 30, charSet: 'abcdefghijklmnopqrstuvwxyz', tokenizerDelimiter: /\s+/ });
    /* =========== Type guards =========== */
    function isTextConfig(cfg) {
        return cfg.useTokenizer === true;
    }
    function isNumericConfig(cfg) {
        return cfg.useTokenizer !== true;
    }
    /* =========== Helpers =========== */
    /**
     * Normalize a user config with sensible defaults depending on mode.
     * (Keeps the original structural type, only fills in missing optional fields.)
     */
    function normalizeConfig(cfg) {
        var _a, _b, _c, _d;
        if (isTextConfig(cfg)) {
            const merged = Object.assign(Object.assign(Object.assign({}, defaultTextConfig), cfg), { log: Object.assign(Object.assign({}, ((_a = defaultBase.log) !== null && _a !== void 0 ? _a : {})), ((_b = cfg.log) !== null && _b !== void 0 ? _b : {})) });
            return merged;
        }
        else {
            const merged = Object.assign(Object.assign(Object.assign({}, defaultNumericConfig), cfg), { log: Object.assign(Object.assign({}, ((_c = defaultBase.log) !== null && _c !== void 0 ? _c : {})), ((_d = cfg.log) !== null && _d !== void 0 ? _d : {})) });
            return merged;
        }
    }
    /**
     * Rehydrate text-specific fields from a JSON-safe config
     * (e.g., convert tokenizerDelimiter source string → RegExp).
     */
    function deserializeTextBits(config) {
        var _a, _b, _c, _d;
        // If useTokenizer not true, assume numeric config
        if (config.useTokenizer !== true) {
            const nc = Object.assign(Object.assign(Object.assign({}, defaultNumericConfig), config), { log: Object.assign(Object.assign({}, ((_a = defaultBase.log) !== null && _a !== void 0 ? _a : {})), ((_b = config.log) !== null && _b !== void 0 ? _b : {})) });
            return nc;
        }
        // Text config: coerce delimiter
        const tDelim = config.tokenizerDelimiter;
        let delimiter = undefined;
        if (tDelim instanceof RegExp) {
            delimiter = tDelim;
        }
        else if (typeof tDelim === 'string' && tDelim.length > 0) {
            delimiter = new RegExp(tDelim);
        }
        else {
            delimiter = defaultTextConfig.tokenizerDelimiter;
        }
        const tc = Object.assign(Object.assign(Object.assign({}, defaultTextConfig), config), { tokenizerDelimiter: delimiter, log: Object.assign(Object.assign({}, ((_c = defaultBase.log) !== null && _c !== void 0 ? _c : {})), ((_d = config.log) !== null && _d !== void 0 ? _d : {})), useTokenizer: true });
        return tc;
    }

    class Tokenizer {
        constructor(customDelimiter) {
            this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
        }
        tokenize(text) {
            if (typeof text !== 'string') {
                console.warn('[Tokenizer] Expected a string, got:', typeof text, text);
                try {
                    text = String(text !== null && text !== void 0 ? text : '');
                }
                catch (_a) {
                    return [];
                }
            }
            return text
                .trim()
                .toLowerCase()
                .split(this.delimiter)
                .filter(Boolean);
        }
        ngrams(tokens, n) {
            if (n <= 0 || tokens.length < n)
                return [];
            const result = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                result.push(tokens.slice(i, i + n).join(' '));
            }
            return result;
        }
    }

    // TextEncoder.ts - Text preprocessing and one-hot encoding for ELM
    const defaultTextEncoderConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false
    };
    class TextEncoder {
        constructor(config = {}) {
            const cfg = Object.assign(Object.assign({}, defaultTextEncoderConfig), config);
            this.charSet = cfg.charSet;
            this.charSize = cfg.charSet.length;
            this.maxLen = cfg.maxLen;
            this.useTokenizer = cfg.useTokenizer;
            if (this.useTokenizer) {
                this.tokenizer = new Tokenizer(config.tokenizerDelimiter);
            }
        }
        charToOneHot(c) {
            const index = this.charSet.indexOf(c.toLowerCase());
            const vec = Array(this.charSize).fill(0);
            if (index !== -1)
                vec[index] = 1;
            return vec;
        }
        textToVector(text) {
            let cleaned;
            if (this.useTokenizer && this.tokenizer) {
                const tokens = this.tokenizer.tokenize(text).join('');
                cleaned = tokens.slice(0, this.maxLen).padEnd(this.maxLen, ' ');
            }
            else {
                cleaned = text.toLowerCase().replace(new RegExp(`[^${this.charSet}]`, 'g'), '').padEnd(this.maxLen, ' ').slice(0, this.maxLen);
            }
            const vec = [];
            for (let i = 0; i < cleaned.length; i++) {
                vec.push(...this.charToOneHot(cleaned[i]));
            }
            return vec;
        }
        normalizeVector(v) {
            const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
            return norm > 0 ? v.map(x => x / norm) : v;
        }
        getVectorSize() {
            return this.charSize * this.maxLen;
        }
        getCharSet() {
            return this.charSet;
        }
        getMaxLen() {
            return this.maxLen;
        }
    }

    // UniversalEncoder.ts - Automatically selects appropriate encoder (char or token based)
    const defaultUniversalConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false,
        mode: 'char'
    };
    class UniversalEncoder {
        constructor(config = {}) {
            const merged = Object.assign(Object.assign({}, defaultUniversalConfig), config);
            const useTokenizer = merged.mode === 'token';
            this.encoder = new TextEncoder({
                charSet: merged.charSet,
                maxLen: merged.maxLen,
                useTokenizer,
                tokenizerDelimiter: config.tokenizerDelimiter
            });
        }
        encode(text) {
            return this.encoder.textToVector(text);
        }
        normalize(v) {
            return this.encoder.normalizeVector(v);
        }
        getVectorSize() {
            return this.encoder.getVectorSize();
        }
    }

    // Augment.ts - Basic augmentation utilities for category training examples
    class Augment {
        static addSuffix(text, suffixes) {
            return suffixes.map(suffix => `${text} ${suffix}`);
        }
        static addPrefix(text, prefixes) {
            return prefixes.map(prefix => `${prefix} ${text}`);
        }
        static addNoise(text, charSet, noiseRate = 0.1) {
            const chars = text.split('');
            for (let i = 0; i < chars.length; i++) {
                if (Math.random() < noiseRate) {
                    const randomChar = charSet[Math.floor(Math.random() * charSet.length)];
                    chars[i] = randomChar;
                }
            }
            return chars.join('');
        }
        static mix(text, mixins) {
            return mixins.map(m => `${text} ${m}`);
        }
        static generateVariants(text, charSet, options) {
            const variants = [text];
            if (options === null || options === void 0 ? void 0 : options.suffixes) {
                variants.push(...this.addSuffix(text, options.suffixes));
            }
            if (options === null || options === void 0 ? void 0 : options.prefixes) {
                variants.push(...this.addPrefix(text, options.prefixes));
            }
            if (options === null || options === void 0 ? void 0 : options.includeNoise) {
                variants.push(this.addNoise(text, charSet));
            }
            return variants;
        }
    }

    // ELM.ts - Core ELM logic with TypeScript types (numeric & text modes)
    // Seeded PRNG (xorshift-ish) for deterministic init
    function makePRNG$2(seed = 123456789) {
        let s = seed | 0 || 1;
        return () => {
            s ^= s << 13;
            s ^= s >>> 17;
            s ^= s << 5;
            return ((s >>> 0) / 0xffffffff);
        };
    }
    function clampInt(x, lo, hi) {
        const xi = x | 0;
        return xi < lo ? lo : (xi > hi ? hi : xi);
    }
    function isOneHot2D(Y) {
        return Array.isArray(Y) && Array.isArray(Y[0]) && Number.isFinite(Y[0][0]);
    }
    function maxLabel(y) {
        let m = -Infinity;
        for (let i = 0; i < y.length; i++) {
            const v = y[i] | 0;
            if (v > m)
                m = v;
        }
        return m === -Infinity ? 0 : m;
    }
    /** One-hot (clamped) */
    function toOneHotClamped(labels, k) {
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
    function ridgeSolve(H, Y, lambda) {
        const Ht = Matrix.transpose(H);
        const A = Matrix.addRegularization(Matrix.multiply(Ht, H), lambda + 1e-10);
        const R = Matrix.multiply(Ht, Y);
        return Matrix.solveCholesky(A, R, 1e-10);
    }
    /* =========================
     * ELM class
     * ========================= */
    class ELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l;
            // Merge with mode-appropriate defaults
            const cfg = normalizeConfig(config);
            this.config = cfg;
            this.categories = cfg.categories;
            this.hiddenUnits = cfg.hiddenUnits;
            this.activation = (_a = cfg.activation) !== null && _a !== void 0 ? _a : 'relu';
            this.useTokenizer = isTextConfig(cfg);
            this.maxLen = isTextConfig(cfg) ? cfg.maxLen : 0;
            this.charSet = isTextConfig(cfg) ? ((_b = cfg.charSet) !== null && _b !== void 0 ? _b : 'abcdefghijklmnopqrstuvwxyz') : 'abcdefghijklmnopqrstuvwxyz';
            this.tokenizerDelimiter = isTextConfig(cfg) ? cfg.tokenizerDelimiter : undefined;
            this.metrics = cfg.metrics;
            this.verbose = (_d = (_c = cfg.log) === null || _c === void 0 ? void 0 : _c.verbose) !== null && _d !== void 0 ? _d : true;
            this.modelName = (_f = (_e = cfg.log) === null || _e === void 0 ? void 0 : _e.modelName) !== null && _f !== void 0 ? _f : 'Unnamed ELM Model';
            this.logToFile = (_h = (_g = cfg.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false;
            this.dropout = (_j = cfg.dropout) !== null && _j !== void 0 ? _j : 0;
            this.ridgeLambda = Math.max((_k = cfg.ridgeLambda) !== null && _k !== void 0 ? _k : 1e-2, 1e-8);
            // Seeded RNG
            const seed = (_l = cfg.seed) !== null && _l !== void 0 ? _l : 1337;
            this.rng = makePRNG$2(seed);
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
        assertEncoder() {
            if (!this.encoder) {
                throw new Error('Encoder is not initialized. Enable useTokenizer:true or construct an encoder.');
            }
            return this.encoder;
        }
        /* ========= initialization ========= */
        xavierLimit(fanIn, fanOut) {
            return Math.sqrt(6 / (fanIn + fanOut));
        }
        randomMatrix(rows, cols) {
            var _a;
            const weightInit = (_a = this.config.weightInit) !== null && _a !== void 0 ? _a : 'uniform';
            if (weightInit === 'xavier') {
                const limit = this.xavierLimit(cols, rows);
                if (this.verbose)
                    console.log(`✨ Xavier init with limit sqrt(6/(${cols}+${rows})) ≈ ${limit.toFixed(4)}`);
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (this.rng() * 2 - 1) * limit));
            }
            else {
                if (this.verbose)
                    console.log(`✨ Uniform init [-1,1] (seeded)`);
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (this.rng() * 2 - 1)));
            }
        }
        buildHidden(X, W, b) {
            const tempH = Matrix.multiply(X, Matrix.transpose(W)); // N x hidden
            const activationFn = Activations.get(this.activation);
            let H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            if (this.dropout > 0) {
                const keepProb = 1 - this.dropout;
                for (let i = 0; i < H.length; i++) {
                    for (let j = 0; j < H[0].length; j++) {
                        if (this.rng() < this.dropout)
                            H[i][j] = 0;
                        else
                            H[i][j] /= keepProb;
                    }
                }
            }
            return H;
        }
        /* ========= public helpers ========= */
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
        setCategories(categories) {
            this.categories = categories;
        }
        loadModelFromJSON(json) {
            var _a, _b, _c, _d, _e;
            try {
                const parsed = JSON.parse(json);
                const cfg = deserializeTextBits(parsed.config);
                // Rebuild instance config
                this.config = cfg;
                this.categories = (_a = cfg.categories) !== null && _a !== void 0 ? _a : this.categories;
                this.hiddenUnits = (_b = cfg.hiddenUnits) !== null && _b !== void 0 ? _b : this.hiddenUnits;
                this.activation = (_c = cfg.activation) !== null && _c !== void 0 ? _c : this.activation;
                this.useTokenizer = cfg.useTokenizer === true;
                this.maxLen = (_d = cfg.maxLen) !== null && _d !== void 0 ? _d : this.maxLen;
                this.charSet = (_e = cfg.charSet) !== null && _e !== void 0 ? _e : this.charSet;
                this.tokenizerDelimiter = cfg.tokenizerDelimiter;
                if (this.useTokenizer) {
                    this.encoder = new UniversalEncoder({
                        charSet: this.charSet,
                        maxLen: this.maxLen,
                        useTokenizer: this.useTokenizer,
                        tokenizerDelimiter: this.tokenizerDelimiter,
                        mode: this.useTokenizer ? 'token' : 'char'
                    });
                }
                else {
                    this.encoder = undefined;
                }
                // Restore weights
                const { W, b, B } = parsed;
                this.model = { W, b, beta: B };
                this.savedModelJSON = json;
                if (this.verbose)
                    console.log(`✅ ${this.modelName} Model loaded from JSON`);
            }
            catch (e) {
                console.error(`❌ Failed to load ${this.modelName} model from JSON:`, e);
            }
        }
        /* ========= Numeric training tolerance ========= */
        /** Decide output dimension from config/categories/labels/one-hot */
        resolveOutputDim(yOrY) {
            // Prefer explicit config
            const cfgOut = this.config.outputDim;
            if (Number.isFinite(cfgOut) && cfgOut > 0)
                return cfgOut | 0;
            // Then categories length if present
            if (Array.isArray(this.categories) && this.categories.length > 0)
                return this.categories.length | 0;
            // Infer from data
            if (isOneHot2D(yOrY))
                return (yOrY[0].length | 0) || 1;
            return (maxLabel(yOrY) + 1) | 0;
        }
        /** Coerce X, and turn labels→one-hot if needed. Always returns strict number[][] */
        coerceXY(X, yOrY) {
            const Xnum = ensureRectNumber2D(X, undefined, 'X');
            const outDim = this.resolveOutputDim(yOrY);
            let Ynum;
            if (isOneHot2D(yOrY)) {
                // Ensure rect with exact width outDim (pad/trunc to be safe)
                Ynum = ensureRectNumber2D(yOrY, outDim, 'Y(one-hot)');
            }
            else {
                // Labels → clamped one-hot
                Ynum = ensureRectNumber2D(toOneHotClamped(yOrY, outDim), outDim, 'Y(labels→one-hot)');
            }
            // If categories length mismatches inferred outDim, adjust categories (non-breaking)
            if (!this.categories || this.categories.length !== outDim) {
                this.categories = Array.from({ length: outDim }, (_, i) => { var _a, _b; return (_b = (_a = this.categories) === null || _a === void 0 ? void 0 : _a[i]) !== null && _b !== void 0 ? _b : String(i); });
            }
            return { Xnum, Ynum, outDim };
        }
        /* ========= Training on numeric vectors =========
         * y can be class indices OR one-hot.
         */
        trainFromData(X, y, options) {
            if (!(X === null || X === void 0 ? void 0 : X.length))
                throw new Error('trainFromData: X is empty');
            // Coerce & shape
            const { Xnum, Ynum, outDim } = this.coerceXY(X, y);
            const n = Xnum.length;
            const inputDim = Xnum[0].length;
            // init / reuse
            let W, b;
            const reuseWeights = (options === null || options === void 0 ? void 0 : options.reuseWeights) === true && this.model;
            if (reuseWeights && this.model) {
                W = this.model.W;
                b = this.model.b;
                if (this.verbose)
                    console.log('🔄 Reusing existing weights/biases for training.');
            }
            else {
                W = this.randomMatrix(this.hiddenUnits, inputDim);
                b = this.randomMatrix(this.hiddenUnits, 1);
                if (this.verbose)
                    console.log('✨ Initializing fresh weights/biases for training.');
            }
            // Hidden
            let H = this.buildHidden(Xnum, W, b);
            // Optional sample weights
            let Yw = Ynum;
            if (options === null || options === void 0 ? void 0 : options.weights) {
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
                const results = { rmse, mae, accuracy: acc, f1, crossEntropy: ce, r2 };
                let allPassed = true;
                if (this.metrics.rmse !== undefined && rmse > this.metrics.rmse)
                    allPassed = false;
                if (this.metrics.mae !== undefined && mae > this.metrics.mae)
                    allPassed = false;
                if (this.metrics.accuracy !== undefined && acc < this.metrics.accuracy)
                    allPassed = false;
                if (this.metrics.f1 !== undefined && f1 < this.metrics.f1)
                    allPassed = false;
                if (this.metrics.crossEntropy !== undefined && ce > this.metrics.crossEntropy)
                    allPassed = false;
                if (this.metrics.r2 !== undefined && r2 < this.metrics.r2)
                    allPassed = false;
                if (this.verbose)
                    this.logMetrics(results);
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify({
                        config: this.serializeConfig(),
                        W, b, B: beta
                    });
                    if (this.verbose)
                        console.log('✅ Model passed thresholds and was saved to JSON.');
                    if (this.config.exportFileName)
                        this.saveModelAsJSONFile(this.config.exportFileName);
                }
                else {
                    if (this.verbose)
                        console.log('❌ Model not saved: One or more thresholds not met.');
                }
            }
            else {
                // No metrics—always save
                this.savedModelJSON = JSON.stringify({
                    config: this.serializeConfig(),
                    W, b, B: beta
                });
                if (this.verbose)
                    console.log('✅ Model trained with no metrics—saved by default.');
                if (this.config.exportFileName)
                    this.saveModelAsJSONFile(this.config.exportFileName);
            }
            return { epochs: 1, metrics: undefined };
        }
        /* ========= Training from category strings (text mode) ========= */
        train(augmentationOptions, weights) {
            if (!this.useTokenizer) {
                throw new Error('train(): text training requires useTokenizer:true');
            }
            const enc = this.assertEncoder();
            const X = [];
            let Y = [];
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
                const results = { rmse, mae, accuracy: acc, f1, crossEntropy: ce, r2 };
                let allPassed = true;
                if (this.metrics.rmse !== undefined && rmse > this.metrics.rmse)
                    allPassed = false;
                if (this.metrics.mae !== undefined && mae > this.metrics.mae)
                    allPassed = false;
                if (this.metrics.accuracy !== undefined && acc < this.metrics.accuracy)
                    allPassed = false;
                if (this.metrics.f1 !== undefined && f1 < this.metrics.f1)
                    allPassed = false;
                if (this.metrics.crossEntropy !== undefined && ce > this.metrics.crossEntropy)
                    allPassed = false;
                if (this.metrics.r2 !== undefined && r2 < this.metrics.r2)
                    allPassed = false;
                if (this.verbose)
                    this.logMetrics(results);
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify({
                        config: this.serializeConfig(),
                        W, b, B: beta
                    });
                    if (this.verbose)
                        console.log('✅ Model passed thresholds and was saved to JSON.');
                    if (this.config.exportFileName)
                        this.saveModelAsJSONFile(this.config.exportFileName);
                }
                else {
                    if (this.verbose)
                        console.log('❌ Model not saved: One or more thresholds not met.');
                }
            }
            else {
                this.savedModelJSON = JSON.stringify({
                    config: this.serializeConfig(),
                    W, b, B: beta
                });
                if (this.verbose)
                    console.log('✅ Model trained with no metrics—saved by default.');
                if (this.config.exportFileName)
                    this.saveModelAsJSONFile(this.config.exportFileName);
            }
            return { epochs: 1, metrics: undefined };
        }
        /* ========= Prediction ========= */
        /** Text prediction (uses Option A narrowing) */
        predict(text, topK = 5) {
            if (!this.model)
                throw new Error('Model not trained.');
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
        predictFromVector(inputVecRows, topK = 5) {
            if (!this.model)
                throw new Error('Model not trained.');
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
        predictLogitsFromVector(vec) {
            if (!this.model)
                throw new Error('Model not trained.');
            const { W, b, beta } = this.model;
            // Hidden
            const tempH = Matrix.multiply([vec], Matrix.transpose(W)); // 1 x hidden
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn); // 1 x hidden
            // Output logits
            return Matrix.multiply(H, beta)[0]; // 1 x outDim → vec
        }
        /** Raw logits for a batch of numeric vectors */
        predictLogitsFromVectors(X) {
            if (!this.model)
                throw new Error('Model not trained.');
            const { W, b, beta } = this.model;
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            return Matrix.multiply(H, beta);
        }
        /** Probability vector (softmax) for a single numeric vector */
        predictProbaFromVector(vec) {
            return Activations.softmax(this.predictLogitsFromVector(vec));
        }
        /** Probability matrix (softmax per row) for a batch of numeric vectors */
        predictProbaFromVectors(X) {
            return this.predictLogitsFromVectors(X).map(Activations.softmax);
        }
        /** Top-K results for a single numeric vector */
        predictTopKFromVector(vec, k = 5) {
            const probs = this.predictProbaFromVector(vec);
            return probs
                .map((p, i) => ({ index: i, label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k);
        }
        /** Top-K results for a batch of numeric vectors */
        predictTopKFromVectors(X, k = 5) {
            return this.predictProbaFromVectors(X).map(row => row
                .map((p, i) => ({ index: i, label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k));
        }
        /* ========= Metrics ========= */
        calculateRMSE(Y, P) {
            const N = Y.length, C = Y[0].length;
            let sum = 0;
            for (let i = 0; i < N; i++)
                for (let j = 0; j < C; j++) {
                    const d = Y[i][j] - P[i][j];
                    sum += d * d;
                }
            return Math.sqrt(sum / (N * C));
        }
        calculateMAE(Y, P) {
            const N = Y.length, C = Y[0].length;
            let sum = 0;
            for (let i = 0; i < N; i++)
                for (let j = 0; j < C; j++) {
                    sum += Math.abs(Y[i][j] - P[i][j]);
                }
            return sum / (N * C);
        }
        calculateAccuracy(Y, P) {
            let correct = 0;
            for (let i = 0; i < Y.length; i++) {
                const yMax = this.argmax(Y[i]);
                const pMax = this.argmax(P[i]);
                if (yMax === pMax)
                    correct++;
            }
            return correct / Y.length;
        }
        calculateF1Score(Y, P) {
            let tp = 0, fp = 0, fn = 0;
            for (let i = 0; i < Y.length; i++) {
                const yIdx = this.argmax(Y[i]);
                const pIdx = this.argmax(P[i]);
                if (yIdx === pIdx)
                    tp++;
                else {
                    fp++;
                    fn++;
                }
            }
            const precision = tp / (tp + fp || 1);
            const recall = tp / (tp + fn || 1);
            return 2 * (precision * recall) / (precision + recall || 1);
        }
        calculateCrossEntropy(Y, P) {
            let loss = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    const pred = Math.min(Math.max(P[i][j], 1e-15), 1 - 1e-15);
                    loss += -Y[i][j] * Math.log(pred);
                }
            }
            return loss / Y.length;
        }
        calculateR2Score(Y, P) {
            const C = Y[0].length;
            const mean = new Array(C).fill(0);
            for (let i = 0; i < Y.length; i++)
                for (let j = 0; j < C; j++)
                    mean[j] += Y[i][j];
            for (let j = 0; j < C; j++)
                mean[j] /= Y.length;
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
        computeHiddenLayer(X) {
            if (!this.model)
                throw new Error('Model not trained.');
            const WX = Matrix.multiply(X, Matrix.transpose(this.model.W));
            const WXb = WX.map(row => row.map((val, j) => val + this.model.b[j][0]));
            const activationFn = Activations.get(this.activation);
            return WXb.map(row => row.map(activationFn));
        }
        getEmbedding(X) {
            return this.computeHiddenLayer(X);
        }
        /* ========= Logging & export ========= */
        logMetrics(results) {
            var _a, _b, _c, _d, _e, _f;
            const logLines = [`📋 ${this.modelName} — Metrics Summary:`];
            const push = (label, value, threshold, cmp) => {
                if (threshold !== undefined)
                    logLines.push(`  ${label}: ${value.toFixed(4)} (threshold: ${cmp} ${threshold})`);
            };
            push('RMSE', results.rmse, (_a = this.metrics) === null || _a === void 0 ? void 0 : _a.rmse, '<=');
            push('MAE', results.mae, (_b = this.metrics) === null || _b === void 0 ? void 0 : _b.mae, '<=');
            push('Accuracy', results.accuracy, (_c = this.metrics) === null || _c === void 0 ? void 0 : _c.accuracy, '>=');
            push('F1 Score', results.f1, (_d = this.metrics) === null || _d === void 0 ? void 0 : _d.f1, '>=');
            push('Cross-Entropy', results.crossEntropy, (_e = this.metrics) === null || _e === void 0 ? void 0 : _e.crossEntropy, '<=');
            push('R² Score', results.r2, (_f = this.metrics) === null || _f === void 0 ? void 0 : _f.r2, '>=');
            if (this.verbose)
                console.log('\n' + logLines.join('\n'));
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
        saveModelAsJSONFile(filename) {
            if (!this.savedModelJSON) {
                if (this.verbose)
                    console.warn('No model saved — did not meet metric thresholds.');
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
            if (this.verbose)
                console.log(`📦 Model exported as ${finalName}`);
        }
        serializeConfig() {
            const cfg = Object.assign({}, this.config);
            // Remove non-serializable / volatile fields
            delete cfg.seed;
            delete cfg.log;
            delete cfg.encoder;
            // Serialize tokenizerDelimiter for JSON
            if (cfg.tokenizerDelimiter instanceof RegExp) {
                cfg.tokenizerDelimiter = cfg.tokenizerDelimiter.source;
            }
            return cfg;
        }
        argmax(arr) {
            let i = 0;
            for (let k = 1; k < arr.length; k++)
                if (arr[k] > arr[i])
                    i = k;
            return i;
        }
        getEncoder() {
            return this.encoder;
        }
    }

    // ELMChain.ts — simple encoder pipeline with checks, normalization, and profiling
    function l2NormalizeRows$1(M) {
        return M.map(row => {
            let s = 0;
            for (let i = 0; i < row.length; i++)
                s += row[i] * row[i];
            const n = Math.sqrt(s) || 1;
            const inv = 1 / n;
            return row.map(v => v * inv);
        });
    }
    function asBatch(x) {
        return Array.isArray(x[0]) ? x : [x];
    }
    function fromBatch(y, originalWasVector) {
        var _a;
        return originalWasVector ? ((_a = y[0]) !== null && _a !== void 0 ? _a : []) : y;
    }
    class ELMChain {
        constructor(encoders = [], opts) {
            var _a, _b, _c, _d, _e;
            this.lastDims = []; // input dim -> stage dims (for summary)
            this.encoders = [...encoders];
            this.opts = {
                normalizeEach: (_a = opts === null || opts === void 0 ? void 0 : opts.normalizeEach) !== null && _a !== void 0 ? _a : false,
                normalizeFinal: (_b = opts === null || opts === void 0 ? void 0 : opts.normalizeFinal) !== null && _b !== void 0 ? _b : false,
                validate: (_c = opts === null || opts === void 0 ? void 0 : opts.validate) !== null && _c !== void 0 ? _c : true,
                strict: (_d = opts === null || opts === void 0 ? void 0 : opts.strict) !== null && _d !== void 0 ? _d : true,
                name: (_e = opts === null || opts === void 0 ? void 0 : opts.name) !== null && _e !== void 0 ? _e : 'ELMChain',
            };
        }
        /** Add encoder at end */
        add(encoder) {
            this.encoders.push(encoder);
        }
        /** Insert encoder at position (0..length) */
        insertAt(index, encoder) {
            if (index < 0 || index > this.encoders.length)
                throw new Error('insertAt: index out of range');
            this.encoders.splice(index, 0, encoder);
        }
        /** Remove encoder at index; returns removed or undefined */
        removeAt(index) {
            if (index < 0 || index >= this.encoders.length)
                return undefined;
            return this.encoders.splice(index, 1)[0];
        }
        /** Remove all encoders */
        clear() {
            this.encoders.length = 0;
            this.lastDims.length = 0;
        }
        /** Number of stages */
        length() {
            return this.encoders.length;
        }
        /** Human-friendly overview (dims are filled after the first successful run) */
        summary() {
            const lines = [];
            lines.push(`📦 ${this.opts.name} — ${this.encoders.length} stage(s)`);
            this.encoders.forEach((enc, i) => {
                var _a, _b, _c;
                const nm = (_a = enc.name) !== null && _a !== void 0 ? _a : `Encoder#${i}`;
                const dimIn = (_b = this.lastDims[i]) !== null && _b !== void 0 ? _b : '?';
                const dimOut = (_c = this.lastDims[i + 1]) !== null && _c !== void 0 ? _c : '?';
                lines.push(`  ${i}: ${nm}    ${dimIn} → ${dimOut}`);
            });
            return lines.join('\n');
        }
        getEmbedding(input) {
            var _a, _b;
            const wasVector = !Array.isArray(input[0]);
            const X0 = asBatch(input);
            if (this.opts.validate) {
                if (!X0.length || !((_a = X0[0]) === null || _a === void 0 ? void 0 : _a.length))
                    throw new Error('ELMChain.getEmbedding: empty input');
            }
            let X = X0;
            this.lastDims = [X0[0].length];
            for (let i = 0; i < this.encoders.length; i++) {
                const enc = this.encoders[i];
                try {
                    if (this.opts.validate) {
                        // Ensure rows consistent
                        const d = X[0].length;
                        for (let r = 1; r < X.length; r++) {
                            if (X[r].length !== d)
                                throw new Error(`Stage ${i} input row ${r} has dim ${X[r].length} != ${d}`);
                        }
                    }
                    let Y = enc.getEmbedding(X);
                    if (this.opts.validate) {
                        if (!Y.length || !((_b = Y[0]) === null || _b === void 0 ? void 0 : _b.length)) {
                            throw new Error(`Stage ${i} produced empty output`);
                        }
                    }
                    if (this.opts.normalizeEach) {
                        Y = l2NormalizeRows$1(Y);
                    }
                    // Record dims for summary
                    this.lastDims[i + 1] = Y[0].length;
                    X = Y;
                }
                catch (err) {
                    if (this.opts.strict)
                        throw err;
                    // Non-strict: return what we have so far
                    return fromBatch(X, wasVector);
                }
            }
            if (this.opts.normalizeFinal && !this.opts.normalizeEach) {
                X = l2NormalizeRows$1(X);
            }
            return fromBatch(X, wasVector);
        }
        /**
         * Run once to collect per-stage timings (ms) and final dims.
         * Returns { timings, dims } where dims[i] is input dim to stage i,
         * dims[i+1] is that stage’s output dim.
         */
        profile(input) {
            !Array.isArray(input[0]);
            let X = asBatch(input);
            const timings = [];
            const dims = [X[0].length];
            for (let i = 0; i < this.encoders.length; i++) {
                const t0 = performance.now();
                X = this.encoders[i].getEmbedding(X);
                const t1 = performance.now();
                timings.push(t1 - t0);
                dims[i + 1] = X[0].length;
            }
            // Don’t mutate options; just return diagnostics
            return { timings, dims };
        }
    }

    // ELMAdapter.ts — unify ELM / OnlineELM as EncoderLike for ELMChain
    function assertNonEmptyBatch(X, where) {
        if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0) {
            throw new Error(`${where}: expected non-empty (N x D) batch`);
        }
    }
    function matmulXWtAddB(X, // (N x D)
    W, // (H x D)
    b // (H x 1)
    ) {
        var _a, _b, _c, _d, _e;
        const N = X.length, D = X[0].length, H = W.length;
        // quick shape sanity
        if (((_a = W[0]) === null || _a === void 0 ? void 0 : _a.length) !== D)
            throw new Error(`matmulXWtAddB: W is ${W.length}x${(_b = W[0]) === null || _b === void 0 ? void 0 : _b.length}, expected Hx${D}`);
        if (b.length !== H || ((_d = (_c = b[0]) === null || _c === void 0 ? void 0 : _c.length) !== null && _d !== void 0 ? _d : 0) !== 1)
            throw new Error(`matmulXWtAddB: b is ${b.length}x${(_e = b[0]) === null || _e === void 0 ? void 0 : _e.length}, expected Hx1`);
        const out = new Array(N);
        for (let n = 0; n < N; n++) {
            const xn = X[n];
            const row = new Array(H);
            for (let h = 0; h < H; h++) {
                const wh = W[h];
                let s = b[h][0] || 0;
                // unrolled dot
                for (let d = 0; d < D; d++)
                    s += xn[d] * wh[d];
                row[h] = s;
            }
            out[n] = row;
        }
        return out;
    }
    class ELMAdapter {
        constructor(target) {
            var _a, _b;
            this.target = target;
            this.mode = target.type === 'online' ? ((_a = target.mode) !== null && _a !== void 0 ? _a : 'hidden') : 'hidden';
            this.name = (_b = target.name) !== null && _b !== void 0 ? _b : (target.type === 'elm' ? 'ELM' : `OnlineELM(${this.mode})`);
        }
        /** Return embeddings for a batch (N x D) -> (N x H/L) */
        getEmbedding(X) {
            var _a, _b, _c, _d;
            assertNonEmptyBatch(X, `${this.name}.getEmbedding`);
            if (this.target.type === 'elm') {
                const m = this.target.model;
                // ELM already exposes getEmbedding()
                if (typeof m.getEmbedding !== 'function') {
                    throw new Error(`${this.name}: underlying ELM lacks getEmbedding(X)`);
                }
                try {
                    return m.getEmbedding(X);
                }
                catch (err) {
                    // Helpful hint if model wasn’t trained
                    if (m.model == null) {
                        throw new Error(`${this.name}: model not trained/initialized (call train/trainFromData or load model).`);
                    }
                    throw err;
                }
            }
            // OnlineELM path
            const o = this.target.model;
            // Guard dims early
            const D = X[0].length;
            if (!Array.isArray(o.W) || ((_a = o.W[0]) === null || _a === void 0 ? void 0 : _a.length) !== D) {
                throw new Error(`${this.name}: input dim ${D} does not match model.W columns ${(_d = (_c = (_b = o.W) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.length) !== null && _d !== void 0 ? _d : 'n/a'}`);
            }
            if (this.mode === 'logits') {
                // Use public logits as an “embedding”
                try {
                    return o.predictLogitsFromVectors(X);
                }
                catch (err) {
                    if (o.beta == null) {
                        throw new Error(`${this.name}: model not initialized (call init()/fit() before logits mode).`);
                    }
                    throw err;
                }
            }
            // mode === 'hidden' → compute hidden activations: act(X Wᵀ + b)
            const W = o.W;
            const BIAS = o.b;
            const actName = o.activation;
            const act = Activations.get((actName !== null && actName !== void 0 ? actName : 'relu').toLowerCase());
            const Hpre = matmulXWtAddB(X, W, BIAS);
            // apply activation in-place
            for (let n = 0; n < Hpre.length; n++) {
                const row = Hpre[n];
                for (let j = 0; j < row.length; j++)
                    row[j] = act(row[j]);
            }
            return Hpre;
        }
    }
    /* -------- convenience helpers -------- */
    function wrapELM(model, name) {
        return new ELMAdapter({ type: 'elm', model, name });
    }
    function wrapOnlineELM(model, opts) {
        return new ELMAdapter({ type: 'online', model, name: opts === null || opts === void 0 ? void 0 : opts.name, mode: opts === null || opts === void 0 ? void 0 : opts.mode });
    }

    // DeepELM.ts — stacked ELM autoencoders + top ELM classifier
    class DeepELM {
        constructor(cfg) {
            this.aeLayers = [];
            this.chain = null;
            this.clf = null;
            this.cfg = Object.assign({ clfHiddenUnits: 0, clfActivation: 'linear', clfWeightInit: 'xavier', normalizeEach: false, normalizeFinal: true }, cfg);
        }
        /** Layer-wise unsupervised training with Y=X (autoencoder). Returns transformed X_L. */
        fitAutoencoders(X) {
            var _a, _b, _c, _d;
            let cur = X;
            this.aeLayers = [];
            for (let i = 0; i < this.cfg.layers.length; i++) {
                const spec = this.cfg.layers[i];
                // Minimal ELM config for numeric mode—categories aren’t used by trainFromData:
                const elm = new ELM({
                    categories: ['ae'], // placeholder (unused in trainFromData)
                    hiddenUnits: spec.hiddenUnits,
                    activation: (_a = spec.activation) !== null && _a !== void 0 ? _a : 'relu',
                    weightInit: (_b = spec.weightInit) !== null && _b !== void 0 ? _b : 'xavier',
                    dropout: (_c = spec.dropout) !== null && _c !== void 0 ? _c : 0,
                    log: { modelName: (_d = spec.name) !== null && _d !== void 0 ? _d : `AE#${i + 1}`, verbose: false },
                });
                // Autoencoder: targets are the inputs
                elm.trainFromData(cur, cur);
                this.aeLayers.push(elm);
                // Forward to next layer using hidden activations
                cur = elm.getEmbedding(cur);
                if (this.cfg.normalizeEach) {
                    cur = l2NormalizeRows(cur);
                }
            }
            // Build chain for fast forward passes
            this.chain = new ELMChain(this.aeLayers.map((m, i) => {
                const a = wrapELM(m, m['modelName'] || `AE#${i + 1}`);
                return a;
            }), {
                normalizeEach: !!this.cfg.normalizeEach,
                normalizeFinal: !!this.cfg.normalizeFinal,
                name: 'DeepELM-Chain',
            });
            return this.transform(X);
        }
        /** Supervised training of a top classifier ELM on last-layer features. */
        fitClassifier(X, yOneHot) {
            var _a, _b;
            if (!this.chain)
                throw new Error('fitClassifier: call fitAutoencoders() first');
            const Z = this.chain.getEmbedding(X);
            // If clfHiddenUnits === 0, we mimic a “linear readout” by using a very small hidden layer with linear activation.
            const hidden = Math.max(1, this.cfg.clfHiddenUnits || 1);
            this.clf = new ELM({
                categories: Array.from({ length: this.cfg.numClasses }, (_, i) => String(i)),
                hiddenUnits: hidden,
                activation: (_a = this.cfg.clfActivation) !== null && _a !== void 0 ? _a : 'linear',
                weightInit: (_b = this.cfg.clfWeightInit) !== null && _b !== void 0 ? _b : 'xavier',
                log: { modelName: 'DeepELM-Classifier', verbose: false },
            });
            this.clf.trainFromData(Z, yOneHot);
        }
        /** One-shot convenience: train AEs then classifier. */
        fit(X, yOneHot) {
            this.fitAutoencoders(X);
            this.fitClassifier(X, yOneHot);
        }
        /** Forward through stacked AEs (no classifier). */
        transform(X) {
            if (!this.chain)
                throw new Error('transform: model not fitted');
            const Z = this.chain.getEmbedding(X);
            return Z;
        }
        /** Classifier probabilities (softmax) for a batch. */
        predictProba(X) {
            if (!this.clf)
                throw new Error('predictProba: classifier not fitted');
            // Reuse existing ELM method on batch:
            const Z = this.transform(X);
            const res = this.clf.predictFromVector(Z, this.cfg.numClasses);
            // predictFromVector returns topK lists; convert back into dense probs when possible
            // If you’d rather have dense probs, expose a new method on ELM to return raw softmax scores for a batch.
            return topKListToDense(res, this.cfg.numClasses);
        }
        /** Utility: export all models for persistence. */
        toJSON() {
            var _a;
            return {
                cfg: this.cfg,
                layers: this.aeLayers.map(m => { var _a; return (_a = m.savedModelJSON) !== null && _a !== void 0 ? _a : JSON.stringify(m.model); }),
                clf: this.clf ? ((_a = this.clf.savedModelJSON) !== null && _a !== void 0 ? _a : JSON.stringify(this.clf.model)) : null,
                __version: 'deep-elm-1.0.0',
            };
        }
        /** Utility: load from exported payload. */
        fromJSON(payload) {
            const { cfg, layers, clf } = payload !== null && payload !== void 0 ? payload : {};
            if (!Array.isArray(layers))
                throw new Error('fromJSON: invalid payload');
            this.cfg = Object.assign(Object.assign({}, this.cfg), cfg);
            this.aeLayers = layers.map((j, i) => {
                const m = new ELM({ categories: ['ae'], hiddenUnits: 1 });
                m.loadModelFromJSON(j);
                return m;
            });
            this.chain = new ELMChain(this.aeLayers.map((m, i) => wrapELM(m, `AE#${i + 1}`)), {
                normalizeEach: !!this.cfg.normalizeEach,
                normalizeFinal: !!this.cfg.normalizeFinal,
                name: 'DeepELM-Chain',
            });
            if (clf) {
                const c = new ELM({ categories: Array.from({ length: this.cfg.numClasses }, (_, i) => String(i)), hiddenUnits: 1 });
                c.loadModelFromJSON(clf);
                this.clf = c;
            }
        }
    }
    /* ---------- helpers ---------- */
    function l2NormalizeRows(M) {
        return M.map(r => {
            let s = 0;
            for (let i = 0; i < r.length; i++)
                s += r[i] * r[i];
            const inv = 1 / (Math.sqrt(s) || 1);
            return r.map(v => v * inv);
        });
    }
    function topKListToDense(list, K) {
        // Convert the ELM.predictFromVector top-K output back to dense [N x K] probs if needed.
        // (If your ELM exposes a dense “predictProbaFromVectors” for the batch, prefer that.)
        return list.map(row => {
            const out = new Array(K).fill(0);
            for (const { label, prob } of row) {
                const idx = Number(label);
                if (Number.isFinite(idx) && idx >= 0 && idx < K)
                    out[idx] = prob;
            }
            return out;
        });
    }

    /******************************************************************************
    Copyright (c) Microsoft Corporation.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
    OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.
    ***************************************************************************** */
    /* global Reflect, Promise, SuppressedError, Symbol, Iterator */


    function __awaiter(thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    typeof SuppressedError === "function" ? SuppressedError : function (error, suppressed, message) {
        var e = new Error(message);
        return e.name = "SuppressedError", e.error = error, e.suppressed = suppressed, e;
    };

    // OnlineELM.ts — Online / OS-ELM with RLS updates
    /* ========== utils ========== */
    const EPS$3 = 1e-10;
    function makePRNG$1(seed = 123456789) {
        let s = seed | 0 || 1;
        return () => {
            s ^= s << 13;
            s ^= s >>> 17;
            s ^= s << 5;
            return ((s >>> 0) / 0xffffffff);
        };
    }
    /* ========== Online ELM (RLS) ========== */
    class OnlineELM {
        constructor(cfg) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j;
            this.inputDim = cfg.inputDim | 0;
            this.outputDim = cfg.outputDim | 0;
            this.hiddenUnits = cfg.hiddenUnits | 0;
            if (this.inputDim <= 0 || this.outputDim <= 0 || this.hiddenUnits <= 0) {
                throw new Error(`OnlineELM: invalid dims (inputDim=${this.inputDim}, outputDim=${this.outputDim}, hidden=${this.hiddenUnits})`);
            }
            this.activation = (_a = cfg.activation) !== null && _a !== void 0 ? _a : 'relu';
            this.ridgeLambda = Math.max((_b = cfg.ridgeLambda) !== null && _b !== void 0 ? _b : 1e-2, EPS$3);
            this.weightInit = (_c = cfg.weightInit) !== null && _c !== void 0 ? _c : 'xavier';
            this.forgettingFactor = Math.max(Math.min((_d = cfg.forgettingFactor) !== null && _d !== void 0 ? _d : 1.0, 1.0), 1e-4);
            this.verbose = (_f = (_e = cfg.log) === null || _e === void 0 ? void 0 : _e.verbose) !== null && _f !== void 0 ? _f : false;
            this.modelName = (_h = (_g = cfg.log) === null || _g === void 0 ? void 0 : _g.modelName) !== null && _h !== void 0 ? _h : 'Online ELM';
            const seed = (_j = cfg.seed) !== null && _j !== void 0 ? _j : 1337;
            this.rng = makePRNG$1(seed);
            this.actFn = Activations.get(this.activation);
            // Random features
            this.W = this.initW(this.hiddenUnits, this.inputDim);
            this.b = this.initB(this.hiddenUnits);
            // Not initialized yet — init() will set these
            this.beta = null;
            this.P = null;
        }
        /* ===== init helpers ===== */
        xavierLimit(fanIn, fanOut) { return Math.sqrt(6 / (fanIn + fanOut)); }
        heLimit(fanIn) { return Math.sqrt(6 / fanIn); }
        initW(rows, cols) {
            let limit = 1;
            if (this.weightInit === 'xavier') {
                limit = this.xavierLimit(cols, rows);
                if (this.verbose)
                    console.log(`✨ [${this.modelName}] Xavier W ~ U(±${limit.toFixed(4)})`);
            }
            else if (this.weightInit === 'he') {
                limit = this.heLimit(cols);
                if (this.verbose)
                    console.log(`✨ [${this.modelName}] He W ~ U(±${limit.toFixed(4)})`);
            }
            else if (this.verbose) {
                console.log(`✨ [${this.modelName}] Uniform W ~ U(±1)`);
            }
            const rnd = () => (this.rng() * 2 - 1) * limit;
            return Array.from({ length: rows }, () => Array.from({ length: cols }, rnd));
        }
        initB(rows) {
            const rnd = () => (this.rng() * 2 - 1) * 0.01;
            return Array.from({ length: rows }, () => [rnd()]);
        }
        hidden(X) {
            const tempH = Matrix.multiply(X, Matrix.transpose(this.W)); // (n x hidden)
            const f = this.actFn;
            return tempH.map(row => row.map((v, j) => f(v + this.b[j][0])));
        }
        /* ===== public API ===== */
        /** Initialize β and P from a batch (ridge): P0=(HᵀH+λI)^-1, β0=P0 HᵀY */
        init(X0, Y0) {
            if (!(X0 === null || X0 === void 0 ? void 0 : X0.length) || !(Y0 === null || Y0 === void 0 ? void 0 : Y0.length))
                throw new Error('init: empty X0 or Y0');
            if (X0.length !== Y0.length)
                throw new Error(`init: X0 rows ${X0.length} != Y0 rows ${Y0.length}`);
            if (X0[0].length !== this.inputDim)
                throw new Error(`init: X0 cols ${X0[0].length} != inputDim ${this.inputDim}`);
            if (Y0[0].length !== this.outputDim)
                throw new Error(`init: Y0 cols ${Y0[0].length} != outputDim ${this.outputDim}`);
            const H0 = this.hidden(X0); // (n x h)
            const Ht = Matrix.transpose(H0); // (h x n)
            const A = Matrix.addRegularization(Matrix.multiply(Ht, H0), this.ridgeLambda + 1e-10); // (h x h)
            const R = Matrix.multiply(Ht, Y0); // (h x k)
            const P0 = Matrix.solveCholesky(A, Matrix.identity(this.hiddenUnits), 1e-10); // A^-1
            const B0 = Matrix.multiply(P0, R); // (h x k)
            this.P = P0;
            this.beta = B0;
            if (this.verbose)
                console.log(`✅ [${this.modelName}] init: n=${X0.length}, hidden=${this.hiddenUnits}, out=${this.outputDim}`);
        }
        /** If not initialized, init(); otherwise RLS update. */
        fit(X, Y) {
            if (!(X === null || X === void 0 ? void 0 : X.length) || !(Y === null || Y === void 0 ? void 0 : Y.length))
                throw new Error('fit: empty X or Y');
            if (X.length !== Y.length)
                throw new Error(`fit: X rows ${X.length} != Y rows ${Y.length}`);
            if (!this.P || !this.beta)
                this.init(X, Y);
            else
                this.update(X, Y);
        }
        /**
         * RLS / OS-ELM update with forgetting ρ:
         *   S = I + HPHᵀ
         *   K = P Hᵀ S^-1
         *   β ← β + K (Y - Hβ)
         *   P ← (P - K H P) / ρ
         */
        update(X, Y) {
            if (!(X === null || X === void 0 ? void 0 : X.length) || !(Y === null || Y === void 0 ? void 0 : Y.length))
                throw new Error('update: empty X or Y');
            if (X.length !== Y.length)
                throw new Error(`update: X rows ${X.length} != Y rows ${Y.length}`);
            if (!this.P || !this.beta)
                throw new Error('update: model not initialized (call init() first)');
            const n = X.length;
            const H = this.hidden(X); // (n x h)
            const Ht = Matrix.transpose(H); // (h x n)
            const rho = this.forgettingFactor;
            let P = this.P;
            if (rho < 1.0) {
                // Equivalent to P <- P / ρ (more responsive to new data)
                P = P.map(row => row.map(v => v / rho));
            }
            // S = I + H P Hᵀ  (n x n, SPD)
            const HP = Matrix.multiply(H, P); // (n x h)
            const HPHt = Matrix.multiply(HP, Ht); // (n x n)
            const S = Matrix.add(HPHt, Matrix.identity(n));
            const S_inv = Matrix.solveCholesky(S, Matrix.identity(n), 1e-10);
            // K = P Hᵀ S^-1  (h x n)
            const PHt = Matrix.multiply(P, Ht); // (h x n)
            const K = Matrix.multiply(PHt, S_inv); // (h x n)
            // Innovation: (Y - Hβ)  (n x k)
            const Hbeta = Matrix.multiply(H, this.beta);
            const innov = Y.map((row, i) => row.map((yij, j) => yij - Hbeta[i][j]));
            // β ← β + K * innov
            const Delta = Matrix.multiply(K, innov); // (h x k)
            this.beta = this.beta.map((row, i) => row.map((bij, j) => bij + Delta[i][j]));
            // P ← P - K H P
            const KH = Matrix.multiply(K, H); // (h x h)
            const KHP = Matrix.multiply(KH, P); // (h x h)
            this.P = P.map((row, i) => row.map((pij, j) => pij - KHP[i][j]));
            if (this.verbose) {
                const diagAvg = this.P.reduce((s, r, i) => s + r[i], 0) / this.P.length;
                console.log(`🔁 [${this.modelName}] update: n=${n}, avg diag(P)≈${diagAvg.toFixed(6)}`);
            }
        }
        /* ===== Prediction ===== */
        logitsFromVectors(X) {
            if (!this.beta)
                throw new Error('predict: model not initialized');
            const H = this.hidden(X);
            return Matrix.multiply(H, this.beta);
        }
        predictLogitsFromVector(x) {
            return this.logitsFromVectors([x])[0];
        }
        predictLogitsFromVectors(X) {
            return this.logitsFromVectors(X);
        }
        predictProbaFromVector(x) {
            return Activations.softmax(this.predictLogitsFromVector(x));
        }
        predictProbaFromVectors(X) {
            return this.predictLogitsFromVectors(X).map(Activations.softmax);
        }
        predictTopKFromVector(x, k = 5) {
            const p = this.predictProbaFromVector(x);
            const kk = Math.max(1, Math.min(k, p.length));
            return p.map((prob, index) => ({ index, prob }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, kk);
        }
        predictTopKFromVectors(X, k = 5) {
            return this.predictProbaFromVectors(X).map(p => {
                const kk = Math.max(1, Math.min(k, p.length));
                return p.map((prob, index) => ({ index, prob }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, kk);
            });
        }
        /* ===== Serialization ===== */
        toJSON(includeP = false) {
            if (!this.beta || !this.P)
                throw new Error('toJSON: model not initialized');
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
            const o = { W: this.W, b: this.b, B: this.beta, config: cfg };
            if (includeP)
                o.P = this.P;
            return o;
        }
        loadFromJSON(json) {
            var _a;
            const parsed = typeof json === 'string' ? JSON.parse(json) : json;
            const { W, b, B, P, config } = parsed;
            if (!W || !b || !B)
                throw new Error('loadFromJSON: missing W/b/B');
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
            this.P = P !== null && P !== void 0 ? P : null;
            if (config === null || config === void 0 ? void 0 : config.activation) {
                this.activation = config.activation;
                this.actFn = Activations.get(this.activation); // refresh cache
            }
            if (config === null || config === void 0 ? void 0 : config.ridgeLambda)
                this.ridgeLambda = config.ridgeLambda;
            if (this.verbose)
                console.log(`✅ [${this.modelName}] model loaded (v=${(_a = config === null || config === void 0 ? void 0 : config.__version) !== null && _a !== void 0 ? _a : 'n/a'})`);
        }
    }

    // src/workers/elm-worker.ts
    /// <reference lib="webworker" />
    let kind = 'none';
    let elm = null;
    let oelm = null;
    const ok = (id, result) => ({ id, ok: true, result });
    const err = (id, error) => {
        var _a;
        return ({
            id,
            ok: false,
            error: String((_a = error === null || error === void 0 ? void 0 : error.message) !== null && _a !== void 0 ? _a : error),
        });
    };
    const post = (m) => self.postMessage(m);
    const postProgress = (p) => post(p);
    // -------- helpers --------
    function coerce2D(M, width, name = 'matrix') {
        return ensureRectNumber2D(M, width, name);
    }
    function hasXY(p) {
        return p && p.X != null;
    }
    function isNumber2D(x) {
        return Array.isArray(x) && Array.isArray(x[0]) && typeof x[0][0] === 'number';
    }
    function isNumber1D(x) {
        return Array.isArray(x) && (x.length === 0 || typeof x[0] === 'number');
    }
    function lower(s) { return String(s !== null && s !== void 0 ? s : '').toLowerCase().trim(); }
    // Generic “train” router: supports aliases + both model kinds.
    function routeTrain(id, action, payload) {
        return __awaiter(this, void 0, void 0, function* () {
            const a = lower(action);
            if (kind === 'elm') {
                if (!elm)
                    throw new Error('ELM not initialized');
                // Two modes:
                // 1) Numeric: payload has X (and Y or y). Use trainFromData (labels or one-hot allowed).
                // 2) Text: no X provided → use train(augmentationOptions, weights).
                if (hasXY(payload)) {
                    const { X, Y, y, options } = payload !== null && payload !== void 0 ? payload : {};
                    // Let ELM.ts handle labels vs one-hot; we still rectangularize X if possible
                    const Xrect = coerce2D(X, undefined, 'X');
                    const targets = isNumber2D(Y) ? Y : (isNumber1D(y) ? y : Y !== null && Y !== void 0 ? Y : y);
                    elm.trainFromData(Xrect, targets, options);
                    postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                    post(ok(id, true));
                    return;
                }
                else {
                    const { augmentationOptions, weights } = payload !== null && payload !== void 0 ? payload : {};
                    elm.train(augmentationOptions, weights);
                    postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                    post(ok(id, true));
                    return;
                }
            }
            if (kind === 'online') {
                if (!oelm)
                    throw new Error('OnlineELM not initialized');
                const { X, Y } = payload !== null && payload !== void 0 ? payload : {};
                const Xrect = coerce2D(X, undefined, 'X');
                const Yrect = coerce2D(Y, undefined, 'Y');
                // Map aliases to OnlineELM APIs
                if (a === 'fit' || a === 'train' || a === 'trainfromdata') {
                    oelm.fit(Xrect, Yrect);
                }
                else if (a === 'update') {
                    oelm.update(Xrect, Yrect);
                }
                else {
                    // default to fit when ambiguous
                    oelm.fit(Xrect, Yrect);
                }
                postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                post(ok(id, true));
                return;
            }
            throw new Error('No model initialized');
        });
    }
    self.onmessage = (ev) => __awaiter(void 0, void 0, void 0, function* () {
        var _a, _b;
        const { id, action, payload } = ev.data || {};
        try {
            switch (action) {
                // ---------- lifecycle ----------
                case 'init':
                case 'initELM': {
                    elm = new ELM(payload);
                    oelm = null;
                    kind = 'elm';
                    post(ok(id, { kind }));
                    break;
                }
                case 'initOnlineELM':
                case 'initOnline': {
                    oelm = new OnlineELM(payload);
                    elm = null;
                    kind = 'online';
                    post(ok(id, { kind }));
                    break;
                }
                case 'dispose': {
                    elm = null;
                    oelm = null;
                    kind = 'none';
                    post(ok(id, { kind }));
                    break;
                }
                case 'getKind': {
                    post(ok(id, { kind }));
                    break;
                }
                case 'setVerbose': {
                    const v = !!(payload === null || payload === void 0 ? void 0 : payload.verbose);
                    if (kind === 'elm' && elm)
                        elm.verbose = v;
                    if (kind === 'online' && oelm)
                        oelm.verbose = v;
                    post(ok(id));
                    break;
                }
                // ---------- tolerant generic actions (aliases) ----------
                case 'train':
                case 'fit':
                case 'update':
                case 'trainFromData': {
                    yield routeTrain(id, action, payload);
                    break;
                }
                case 'predict':
                case 'predictLogits':
                case 'predictlogits': {
                    if (kind === 'elm') {
                        if (!elm)
                            throw new Error('ELM not initialized');
                        // Two predict styles:
                        // - text: { text, topK }
                        // - numeric: { X } returns logits matrix
                        if ((payload === null || payload === void 0 ? void 0 : payload.text) != null) {
                            const { text, topK } = payload;
                            const r = elm.predict(String(text), topK !== null && topK !== void 0 ? topK : 5);
                            post(ok(id, r));
                        }
                        else {
                            const { X } = payload !== null && payload !== void 0 ? payload : {};
                            const Xrect = coerce2D(X, undefined, 'X');
                            const r = elm.predictLogitsFromVectors(Xrect);
                            post(ok(id, r));
                        }
                        break;
                    }
                    if (kind === 'online') {
                        if (!oelm)
                            throw new Error('OnlineELM not initialized');
                        const { X } = payload !== null && payload !== void 0 ? payload : {};
                        const Xrect = coerce2D(X, undefined, 'X');
                        const r = oelm.predictLogitsFromVectors(Xrect);
                        post(ok(id, r));
                        break;
                    }
                    throw new Error('No model initialized');
                }
                // ---------- explicit ELM routes (back-compat) ----------
                case 'elm.train': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const { augmentationOptions, weights } = payload !== null && payload !== void 0 ? payload : {};
                    const onProgress = (phase, pct) => postProgress({ id, type: 'progress', phase, pct });
                    elm.train(augmentationOptions, weights);
                    postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                    post(ok(id));
                    break;
                }
                case 'elm.trainFromData': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const { X, Y, y, options } = payload !== null && payload !== void 0 ? payload : {};
                    const Xrect = coerce2D(X, undefined, 'X');
                    // Let ELM handle labels vs one-hot
                    const targets = isNumber2D(Y) ? Y : (isNumber1D(y) ? y : Y !== null && Y !== void 0 ? Y : y);
                    elm.trainFromData(Xrect, targets, options);
                    postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                    post(ok(id));
                    break;
                }
                case 'elm.predict': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const { text, topK } = payload !== null && payload !== void 0 ? payload : {};
                    const r = elm.predict(String(text), topK !== null && topK !== void 0 ? topK : 5);
                    post(ok(id, r));
                    break;
                }
                case 'elm.predictFromVector': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const { X, topK } = payload !== null && payload !== void 0 ? payload : {};
                    const r = elm.predictFromVector(coerce2D(X, undefined, 'X'), topK !== null && topK !== void 0 ? topK : 5);
                    post(ok(id, r));
                    break;
                }
                case 'elm.getEmbedding': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const { X } = payload !== null && payload !== void 0 ? payload : {};
                    const r = elm.getEmbedding(coerce2D(X, undefined, 'X'));
                    post(ok(id, r));
                    break;
                }
                case 'elm.toJSON': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    const json = (_a = elm.savedModelJSON) !== null && _a !== void 0 ? _a : JSON.stringify(elm.model);
                    post(ok(id, json));
                    break;
                }
                case 'elm.loadJSON': {
                    if (!elm)
                        throw new Error('ELM not initialized');
                    elm.loadModelFromJSON(String((_b = payload === null || payload === void 0 ? void 0 : payload.json) !== null && _b !== void 0 ? _b : ''));
                    post(ok(id));
                    break;
                }
                // ---------- explicit OnlineELM routes (back-compat) ----------
                case 'oelm.init': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    const { X0, Y0 } = payload !== null && payload !== void 0 ? payload : {};
                    oelm.init(coerce2D(X0, undefined, 'X0'), coerce2D(Y0, undefined, 'Y0'));
                    post(ok(id));
                    break;
                }
                case 'oelm.fit': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    const { X, Y } = payload !== null && payload !== void 0 ? payload : {};
                    oelm.fit(coerce2D(X, undefined, 'X'), coerce2D(Y, undefined, 'Y'));
                    post(ok(id));
                    break;
                }
                case 'oelm.update': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    const { X, Y } = payload !== null && payload !== void 0 ? payload : {};
                    oelm.update(coerce2D(X, undefined, 'X'), coerce2D(Y, undefined, 'Y'));
                    post(ok(id));
                    break;
                }
                case 'oelm.logits': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    const { X } = payload !== null && payload !== void 0 ? payload : {};
                    const r = oelm.predictLogitsFromVectors(coerce2D(X, undefined, 'X'));
                    post(ok(id, r));
                    break;
                }
                case 'oelm.toJSON': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    const json = oelm.toJSON(true);
                    post(ok(id, json));
                    break;
                }
                case 'oelm.loadJSON': {
                    if (!oelm)
                        throw new Error('OnlineELM not initialized');
                    oelm.loadFromJSON(payload === null || payload === void 0 ? void 0 : payload.json);
                    post(ok(id));
                    break;
                }
                default: {
                    // Graceful unknown → alias try
                    const a = lower(action);
                    if (a === 'initelm' || a === 'init') {
                        elm = new ELM(payload);
                        oelm = null;
                        kind = 'elm';
                        post(ok(id, { kind }));
                        break;
                    }
                    if (a === 'initonline' || a === 'initonlineelm') {
                        oelm = new OnlineELM(payload);
                        elm = null;
                        kind = 'online';
                        post(ok(id, { kind }));
                        break;
                    }
                    post(err(id, `Unknown action: ${action}`));
                }
            }
        }
        catch (e) {
            post(err(id, e));
        }
    });

    // src/workers/ELMWorkerClient.ts
    class ELMWorkerClient {
        constructor(worker) {
            this.pending = new Map();
            this.worker = worker;
            this.worker.onmessage = (ev) => {
                var _a;
                const msg = ev.data;
                // Progress event
                if ((msg === null || msg === void 0 ? void 0 : msg.type) === 'progress' && (msg === null || msg === void 0 ? void 0 : msg.id)) {
                    const pend = this.pending.get(msg.id);
                    (_a = pend === null || pend === void 0 ? void 0 : pend.onProgress) === null || _a === void 0 ? void 0 : _a.call(pend, msg);
                    return;
                }
                // RPC response
                const id = msg === null || msg === void 0 ? void 0 : msg.id;
                if (!id)
                    return;
                const pend = this.pending.get(id);
                if (!pend)
                    return;
                this.pending.delete(id);
                if (msg.ok)
                    pend.resolve(msg.result);
                else
                    pend.reject(new Error(msg.error));
            };
        }
        call(action, payload, onProgress) {
            const id = Math.random().toString(36).slice(2);
            return new Promise((resolve, reject) => {
                this.pending.set(id, { resolve, reject, onProgress });
                this.worker.postMessage({ id, action, payload });
            });
        }
        // -------- lifecycle --------
        getKind() { return this.call('getKind'); }
        dispose() { return this.call('dispose'); }
        setVerbose(verbose) { return this.call('setVerbose', { verbose }); }
        // -------- ELM --------
        initELM(config) { return this.call('initELM', config); }
        elmTrain(opts, onProgress) {
            return this.call('elm.train', opts, onProgress);
        }
        elmTrainFromData(X, Y, options, onProgress) {
            return this.call('elm.trainFromData', { X, Y, options }, onProgress);
        }
        elmPredict(text, topK = 5) { return this.call('elm.predict', { text, topK }); }
        elmPredictFromVector(X, topK = 5) { return this.call('elm.predictFromVector', { X, topK }); }
        elmGetEmbedding(X) { return this.call('elm.getEmbedding', { X }); }
        elmToJSON() { return this.call('elm.toJSON'); }
        elmLoadJSON(json) { return this.call('elm.loadJSON', { json }); }
        // -------- OnlineELM --------
        initOnlineELM(config) { return this.call('initOnlineELM', config); }
        oelmInit(X0, Y0) { return this.call('oelm.init', { X0, Y0 }); }
        oelmFit(X, Y) { return this.call('oelm.fit', { X, Y }); }
        oelmUpdate(X, Y) { return this.call('oelm.update', { X, Y }); }
        oelmLogits(X) { return this.call('oelm.logits', { X }); }
        oelmToJSON() { return this.call('oelm.toJSON'); }
        oelmLoadJSON(json) { return this.call('oelm.loadJSON', { json }); }
    }

    // EmbeddingStore.ts — Powerful in-memory vector store with fast KNN, thresholds, and JSON I/O
    const EPS$2 = 1e-12;
    /* ================= math utils ================= */
    function l2Norm$1(v) {
        let s = 0;
        for (let i = 0; i < v.length; i++)
            s += v[i] * v[i];
        return Math.sqrt(s);
    }
    function l1Dist(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++)
            s += Math.abs(a[i] - b[i]);
        return s;
    }
    function dot$2(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++)
            s += a[i] * b[i];
        return s;
    }
    function normalizeToUnit(v) {
        const out = new Float32Array(v.length);
        const n = l2Norm$1(v);
        if (n < EPS$2)
            return out; // zero vector → stay zero; cosine with zero returns 0
        const inv = 1 / n;
        for (let i = 0; i < v.length; i++)
            out[i] = v[i] * inv;
        return out;
    }
    /** Quickselect (nth_element) on-place for top-k largest by score. Returns cutoff value index. */
    function quickselectTopK(arr, k, scoreOf) {
        if (k <= 0 || k >= arr.length)
            return arr.length - 1;
        let left = 0, right = arr.length - 1;
        const target = k - 1; // 0-based index of kth largest after partition
        function swap(i, j) {
            const t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
        function partition(l, r, pivotIdx) {
            const pivotScore = scoreOf(arr[pivotIdx]);
            swap(pivotIdx, r);
            let store = l;
            for (let i = l; i < r; i++) {
                if (scoreOf(arr[i]) > pivotScore) { // ">" for largest-first
                    swap(store, i);
                    store++;
                }
            }
            swap(store, r);
            return store;
        }
        while (true) {
            const pivotIdx = Math.floor((left + right) / 2);
            const idx = partition(left, right, pivotIdx);
            if (idx === target)
                return idx;
            if (target < idx)
                right = idx - 1;
            else
                left = idx + 1;
        }
    }
    /* ================= store ================= */
    class EmbeddingStore {
        constructor(dim, opts) {
            var _a, _b;
            // Data
            this.ids = [];
            this.metas = [];
            this.vecs = []; // if storeUnit=true -> unit vectors; else raw vectors
            // Index
            this.idToIdx = new Map();
            if (!Number.isFinite(dim) || dim <= 0)
                throw new Error(`EmbeddingStore: invalid dim=${dim}`);
            this.dim = dim | 0;
            this.storeUnit = (_a = opts === null || opts === void 0 ? void 0 : opts.storeUnit) !== null && _a !== void 0 ? _a : true;
            this.alsoStoreRaw = (_b = opts === null || opts === void 0 ? void 0 : opts.alsoStoreRaw) !== null && _b !== void 0 ? _b : this.storeUnit; // default: if normalizing, also keep raw so Euclidean is valid
            if ((opts === null || opts === void 0 ? void 0 : opts.capacity) !== undefined) {
                if (!Number.isFinite(opts.capacity) || opts.capacity <= 0)
                    throw new Error(`capacity must be > 0`);
                this.capacity = Math.floor(opts.capacity);
            }
            if (this.alsoStoreRaw) {
                this.rawVecs = [];
                this.rawNorms = new Float32Array(0);
            }
            if (!this.storeUnit) {
                // storing raw in vecs → maintain norms for fast cosine
                this.norms = new Float32Array(0);
            }
        }
        /* ========== basic ops ========== */
        size() { return this.ids.length; }
        dimension() { return this.dim; }
        isUnitStored() { return this.storeUnit; }
        keepsRaw() { return !!this.rawVecs; }
        getCapacity() { return this.capacity; }
        setCapacity(capacity) {
            if (capacity === undefined) {
                this.capacity = undefined;
                return;
            }
            if (!Number.isFinite(capacity) || capacity <= 0)
                throw new Error(`capacity must be > 0`);
            this.capacity = Math.floor(capacity);
            this.enforceCapacity();
        }
        clear() {
            this.ids = [];
            this.vecs = [];
            this.metas = [];
            this.idToIdx.clear();
            if (this.rawVecs)
                this.rawVecs = [];
            if (this.norms)
                this.norms = new Float32Array(0);
            if (this.rawNorms)
                this.rawNorms = new Float32Array(0);
        }
        has(id) { return this.idToIdx.has(id); }
        get(id) {
            const idx = this.idToIdx.get(id);
            if (idx === undefined)
                return undefined;
            return {
                id,
                vec: this.vecs[idx],
                raw: this.rawVecs ? this.rawVecs[idx] : undefined,
                meta: this.metas[idx],
            };
        }
        /** Remove by id. Returns true if removed. */
        remove(id) {
            const idx = this.idToIdx.get(id);
            if (idx === undefined)
                return false;
            // capture id, splice arrays
            this.ids.splice(idx, 1);
            this.vecs.splice(idx, 1);
            this.metas.splice(idx, 1);
            if (this.rawVecs)
                this.rawVecs.splice(idx, 1);
            if (this.norms)
                this.norms = this.removeFromNorms(this.norms, idx);
            if (this.rawNorms)
                this.rawNorms = this.removeFromNorms(this.rawNorms, idx);
            this.idToIdx.delete(id);
            this.rebuildIndex(idx);
            return true;
        }
        /** Add or replace an item by id. Returns true if added, false if replaced. */
        upsert(item) {
            var _a;
            const { id, vec, meta } = item;
            if (!id)
                throw new Error('upsert: id is required');
            if (!vec || vec.length !== this.dim) {
                throw new Error(`upsert: vector dim ${(_a = vec === null || vec === void 0 ? void 0 : vec.length) !== null && _a !== void 0 ? _a : 'n/a'} != store dim ${this.dim}`);
            }
            const raw = new Float32Array(vec);
            const unit = this.storeUnit ? normalizeToUnit(raw) : raw;
            const idx = this.idToIdx.get(id);
            if (idx !== undefined) {
                // replace in place
                this.vecs[idx] = unit;
                this.metas[idx] = meta;
                if (this.rawVecs)
                    this.rawVecs[idx] = raw;
                if (this.norms && !this.storeUnit)
                    this.norms[idx] = l2Norm$1(raw);
                if (this.rawNorms && this.rawVecs)
                    this.rawNorms[idx] = l2Norm$1(raw);
                return false;
            }
            else {
                this.ids.push(id);
                this.vecs.push(unit);
                this.metas.push(meta);
                if (this.rawVecs)
                    this.rawVecs.push(raw);
                if (this.norms && !this.storeUnit) {
                    // append norm
                    const n = l2Norm$1(raw);
                    const newNorms = new Float32Array(this.ids.length);
                    newNorms.set(this.norms, 0);
                    newNorms[this.ids.length - 1] = n;
                    this.norms = newNorms;
                }
                if (this.rawNorms && this.rawVecs) {
                    const n = l2Norm$1(raw);
                    const newNorms = new Float32Array(this.ids.length);
                    newNorms.set(this.rawNorms, 0);
                    newNorms[this.ids.length - 1] = n;
                    this.rawNorms = newNorms;
                }
                this.idToIdx.set(id, this.ids.length - 1);
                this.enforceCapacity();
                return true;
            }
        }
        add(item) {
            const added = this.upsert(item);
            if (!added)
                throw new Error(`add: id "${item.id}" already exists (use upsert instead)`);
        }
        addAll(items, allowUpsert = true) {
            for (const it of items) {
                if (allowUpsert)
                    this.upsert(it);
                else
                    this.add(it);
            }
        }
        /** Merge another store (same dim & normalization strategy) into this one. */
        merge(other, allowOverwrite = true) {
            var _a;
            if (other.dimension() !== this.dim)
                throw new Error('merge: dimension mismatch');
            if (other.isUnitStored() !== this.storeUnit)
                throw new Error('merge: normalized flag mismatch');
            if (other.keepsRaw() !== this.keepsRaw())
                throw new Error('merge: raw retention mismatch');
            for (let i = 0; i < other.ids.length; i++) {
                const id = other.ids[i];
                const vec = other.vecs[i];
                const raw = (_a = other.rawVecs) === null || _a === void 0 ? void 0 : _a[i];
                const meta = other.metas[i];
                if (!allowOverwrite && this.has(id))
                    continue;
                // Use upsert path, but avoid double-normalizing when both stores have unit vectors:
                this.upsert({ id, vec, meta });
                if (this.rawVecs && raw)
                    this.rawVecs[this.idToIdx.get(id)] = new Float32Array(raw);
            }
        }
        /* ========== querying ========== */
        /** Top-K KNN query. For L2/L1 we return NEGATIVE distance so higher is better. */
        query(queryVec, k = 10, opts) {
            var _a, _b, _c, _d, _e, _f;
            if (queryVec.length !== this.dim) {
                throw new Error(`query: vector dim ${queryVec.length} != store dim ${this.dim}`);
            }
            const metric = (_a = opts === null || opts === void 0 ? void 0 : opts.metric) !== null && _a !== void 0 ? _a : 'cosine';
            const filter = opts === null || opts === void 0 ? void 0 : opts.filter;
            const returnVectors = (_b = opts === null || opts === void 0 ? void 0 : opts.returnVectors) !== null && _b !== void 0 ? _b : false;
            const minScore = opts === null || opts === void 0 ? void 0 : opts.minScore;
            const maxDistance = opts === null || opts === void 0 ? void 0 : opts.maxDistance;
            const restrictSet = (opts === null || opts === void 0 ? void 0 : opts.restrictToIds) ? new Set(opts.restrictToIds) : undefined;
            let q;
            let qNorm = 0;
            if (metric === 'cosine') {
                // cosine → normalize query; stored data either unit (fast) or raw (use cached norms)
                q = normalizeToUnit(queryVec);
            }
            else if (metric === 'dot') {
                q = new Float32Array(queryVec);
                qNorm = l2Norm$1(q); // only used for potential future scoring transforms
            }
            else {
                // L2/L1 use RAW query
                q = new Float32Array(queryVec);
                qNorm = l2Norm$1(q);
            }
            const hits = [];
            const N = this.vecs.length;
            // helpers
            const pushHit = (i, score) => {
                if (restrictSet && !restrictSet.has(this.ids[i]))
                    return;
                if (filter && !filter(this.metas[i], this.ids[i]))
                    return;
                // Apply thresholds
                if (metric === 'euclidean' || metric === 'manhattan') {
                    const dist = -score; // score is negative distance
                    if (maxDistance !== undefined && dist > maxDistance)
                        return;
                }
                else {
                    if (minScore !== undefined && score < minScore)
                        return;
                }
                hits.push(returnVectors
                    ? { id: this.ids[i], score, index: i, meta: this.metas[i], vec: this.vecs[i] }
                    : { id: this.ids[i], score, index: i, meta: this.metas[i] });
            };
            if (metric === 'cosine') {
                if (this.storeUnit) {
                    // both unit → score = dot
                    for (let i = 0; i < N; i++) {
                        const s = dot$2(q, this.vecs[i]);
                        pushHit(i, s);
                    }
                }
                else {
                    // stored raw in vecs → use cached norms (if available) for cos = dot / (||q||*||v||)
                    if (!this.norms || this.norms.length !== N) {
                        // build norms on-demand once
                        this.norms = new Float32Array(N);
                        for (let i = 0; i < N; i++)
                            this.norms[i] = l2Norm$1(this.vecs[i]);
                    }
                    const qn = l2Norm$1(q) || 1; // guard
                    for (let i = 0; i < N; i++) {
                        const dn = this.norms[i] || 1;
                        const s = dn < EPS$2 ? 0 : dot$2(q, this.vecs[i]) / (qn * dn);
                        pushHit(i, s);
                    }
                }
            }
            else if (metric === 'dot') {
                for (let i = 0; i < N; i++) {
                    const s = dot$2(q, this.storeUnit ? this.vecs[i] : this.vecs[i]); // same storage
                    pushHit(i, s);
                }
            }
            else if (metric === 'euclidean') {
                // Need RAW vectors
                const base = (_c = this.rawVecs) !== null && _c !== void 0 ? _c : (!this.storeUnit ? this.vecs : null);
                if (!base)
                    throw new Error('euclidean query requires raw vectors; create store with alsoStoreRaw=true or storeUnit=false');
                // Use fast formula: ||q - v|| = sqrt(||q||^2 + ||v||^2 - 2 q·v)
                const vNorms = this.rawVecs ? ((_d = this.rawNorms) !== null && _d !== void 0 ? _d : this.buildRawNorms()) :
                    (_e = this.norms) !== null && _e !== void 0 ? _e : this.buildNorms();
                const q2 = qNorm * qNorm;
                for (let i = 0; i < N; i++) {
                    const d2 = Math.max(q2 + vNorms[i] * vNorms[i] - 2 * dot$2(q, base[i]), 0);
                    const dist = Math.sqrt(d2);
                    pushHit(i, -dist); // NEGATIVE distance so higher is better
                }
            }
            else { // 'manhattan'
                const base = (_f = this.rawVecs) !== null && _f !== void 0 ? _f : (!this.storeUnit ? this.vecs : null);
                if (!base)
                    throw new Error('manhattan query requires raw vectors; create store with alsoStoreRaw=true or storeUnit=false');
                for (let i = 0; i < N; i++) {
                    const dist = l1Dist(q, base[i]);
                    pushHit(i, -dist); // NEGATIVE distance
                }
            }
            if (hits.length === 0)
                return [];
            const kk = Math.max(1, Math.min(k, hits.length));
            // Use quickselect to avoid full O(n log n) sort
            quickselectTopK(hits, kk, (h) => h.score);
            // Now sort just the top-K region for stable ordering
            hits
                .slice(0, kk)
                .sort((a, b) => b.score - a.score)
                .forEach((h, i) => (hits[i] = h));
            return hits.slice(0, kk);
        }
        /** Batch query helper. Returns array of results aligned to input queries. */
        queryBatch(queries, k = 10, opts) {
            return queries.map(q => this.query(q, k, opts));
        }
        /** Convenience: query by id */
        queryById(id, k = 10, opts) {
            var _a;
            const rec = this.get(id);
            if (!rec)
                return [];
            const use = ((opts === null || opts === void 0 ? void 0 : opts.metric) === 'euclidean' || (opts === null || opts === void 0 ? void 0 : opts.metric) === 'manhattan')
                ? ((_a = rec.raw) !== null && _a !== void 0 ? _a : rec.vec) // prefer raw for distance
                : rec.vec;
            return this.query(use, k, opts);
        }
        /* ========== export / import ========== */
        toJSON() {
            const includeRaw = !!this.rawVecs;
            return {
                dim: this.dim,
                normalized: this.storeUnit,
                alsoStoredRaw: includeRaw,
                capacity: this.capacity,
                items: this.ids.map((id, i) => ({
                    id,
                    vec: Array.from(this.vecs[i]),
                    raw: includeRaw ? Array.from(this.rawVecs[i]) : undefined,
                    meta: this.metas[i],
                })),
                __version: 'embedding-store-2.0.0',
            };
        }
        static fromJSON(obj) {
            var _a, _b;
            const parsed = typeof obj === 'string' ? JSON.parse(obj) : obj;
            if (!parsed || !parsed.dim || !Array.isArray(parsed.items)) {
                throw new Error('EmbeddingStore.fromJSON: invalid payload');
            }
            const store = new EmbeddingStore(parsed.dim, {
                storeUnit: parsed.normalized,
                capacity: parsed.capacity,
                alsoStoreRaw: (_a = parsed.alsoStoredRaw) !== null && _a !== void 0 ? _a : false,
            });
            for (const it of parsed.items) {
                if (!it || typeof it.id !== 'string' || !Array.isArray(it.vec))
                    continue;
                if (it.vec.length !== parsed.dim) {
                    throw new Error(`fromJSON: vector dim ${it.vec.length} != dim ${parsed.dim} for id ${it.id}`);
                }
                // Use public API to keep norms consistent
                store.upsert({ id: it.id, vec: (_b = it.raw) !== null && _b !== void 0 ? _b : it.vec, meta: it.meta });
                // If payload includes both vec and raw, ensure both sides are *exactly* respected
                if (store.storeUnit && store.rawVecs && it.raw) {
                    const idx = store.idToIdx.get(it.id);
                    store.rawVecs[idx] = new Float32Array(it.raw);
                    if (store.rawNorms) {
                        const newNorms = new Float32Array(store.size());
                        newNorms.set(store.rawNorms, 0);
                        newNorms[idx] = l2Norm$1(store.rawVecs[idx]);
                        store.rawNorms = newNorms;
                    }
                }
                else if (!store.storeUnit && it.vec) ;
            }
            return store;
        }
        /* ========== diagnostics / utils ========== */
        /** Estimate memory footprint in bytes (arrays only; metadata excluded). */
        memoryUsageBytes() {
            const f32 = 4;
            let bytes = 0;
            for (const v of this.vecs)
                bytes += v.length * f32;
            if (this.rawVecs)
                for (const v of this.rawVecs)
                    bytes += v.length * f32;
            if (this.norms)
                bytes += this.norms.length * f32;
            if (this.rawNorms)
                bytes += this.rawNorms.length * f32;
            // ids + metas are JS objects; not included
            return bytes;
        }
        /** Re-normalize all vectors in-place (useful if you bulk-updated raw storage). */
        reNormalizeAll() {
            if (!this.storeUnit)
                return; // nothing to do
            for (let i = 0; i < this.vecs.length; i++) {
                const raw = this.rawVecs ? this.rawVecs[i] : this.vecs[i];
                this.vecs[i] = normalizeToUnit(raw);
            }
        }
        /** Iterate over all items */
        *entries() {
            var _a;
            for (let i = 0; i < this.ids.length; i++) {
                yield { id: this.ids[i], vec: this.vecs[i], raw: (_a = this.rawVecs) === null || _a === void 0 ? void 0 : _a[i], meta: this.metas[i] };
            }
        }
        /* ========== internals ========== */
        removeFromNorms(src, removeIdx) {
            const out = new Float32Array(src.length - 1);
            if (removeIdx > 0)
                out.set(src.subarray(0, removeIdx), 0);
            if (removeIdx < src.length - 1)
                out.set(src.subarray(removeIdx + 1), removeIdx);
            return out;
        }
        /** After a splice at 'start', rebuild id→index for shifted tail */
        rebuildIndex(start = 0) {
            if (start <= 0) {
                this.idToIdx.clear();
                for (let i = 0; i < this.ids.length; i++)
                    this.idToIdx.set(this.ids[i], i);
                return;
            }
            for (let i = start; i < this.ids.length; i++)
                this.idToIdx.set(this.ids[i], i);
        }
        /** Enforce capacity by evicting oldest items (front of arrays) */
        enforceCapacity() {
            if (this.capacity === undefined)
                return;
            while (this.ids.length > this.capacity) {
                const removedId = this.ids[0];
                // shift( ) is O(n); for very large stores consider a circular buffer
                this.ids.shift();
                this.vecs.shift();
                this.metas.shift();
                if (this.rawVecs)
                    this.rawVecs.shift();
                if (this.norms)
                    this.norms = this.removeFromNorms(this.norms, 0);
                if (this.rawNorms)
                    this.rawNorms = this.removeFromNorms(this.rawNorms, 0);
                this.idToIdx.delete(removedId);
                // rebuild full index (ids shifted)
                this.idToIdx.clear();
                for (let i = 0; i < this.ids.length; i++)
                    this.idToIdx.set(this.ids[i], i);
            }
        }
        buildNorms() {
            const out = new Float32Array(this.vecs.length);
            for (let i = 0; i < this.vecs.length; i++)
                out[i] = l2Norm$1(this.vecs[i]);
            this.norms = out;
            return out;
        }
        buildRawNorms() {
            if (!this.rawVecs)
                throw new Error('no raw vectors to build norms for');
            const out = new Float32Array(this.rawVecs.length);
            for (let i = 0; i < this.rawVecs.length; i++)
                out[i] = l2Norm$1(this.rawVecs[i]);
            this.rawNorms = out;
            return out;
        }
    }

    const EPS$1 = 1e-12;
    /* ---------- math helpers ---------- */
    function l2Norm(v) {
        let s = 0;
        for (let i = 0; i < v.length; i++)
            s += v[i] * v[i];
        return Math.sqrt(s);
    }
    function normalize(v) {
        const out = new Float32Array(v.length);
        const n = l2Norm(v);
        if (n < EPS$1)
            return out; // keep zeros; cosine with zero gives 0
        const inv = 1 / n;
        for (let i = 0; i < v.length; i++)
            out[i] = v[i] * inv;
        return out;
    }
    function dot$1(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++)
            s += a[i] * b[i];
        return s;
    }
    /* ---------- main evaluation ---------- */
    function evaluateEnsembleRetrieval(queries, reference, chains, k, options) {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q;
        const metric = (_a = options === null || options === void 0 ? void 0 : options.metric) !== null && _a !== void 0 ? _a : "cosine";
        const aggregate = (_b = options === null || options === void 0 ? void 0 : options.aggregate) !== null && _b !== void 0 ? _b : "mean";
        const weights = options === null || options === void 0 ? void 0 : options.weights;
        const topK = Math.max(1, (_d = (_c = options === null || options === void 0 ? void 0 : options.k) !== null && _c !== void 0 ? _c : k) !== null && _d !== void 0 ? _d : 5);
        const ignoreUnlabeled = (_e = options === null || options === void 0 ? void 0 : options.ignoreUnlabeledQueries) !== null && _e !== void 0 ? _e : true;
        const reportPerLabel = (_f = options === null || options === void 0 ? void 0 : options.reportPerLabel) !== null && _f !== void 0 ? _f : false;
        const returnRankings = (_g = options === null || options === void 0 ? void 0 : options.returnRankings) !== null && _g !== void 0 ? _g : false;
        const logEvery = Math.max(1, (_h = options === null || options === void 0 ? void 0 : options.logEvery) !== null && _h !== void 0 ? _h : 10);
        if (chains.length === 0) {
            throw new Error("evaluateEnsembleRetrieval: 'chains' must be non-empty.");
        }
        if (aggregate === "weighted") {
            if (!weights || weights.length !== chains.length) {
                throw new Error(`aggregate='weighted' requires weights.length === chains.length`);
            }
            // normalize weights to sum=1 for interpretability
            const sumW = weights.reduce((s, w) => s + w, 0) || 1;
            for (let i = 0; i < weights.length; i++)
                weights[i] = weights[i] / sumW;
        }
        console.log("🔹 Precomputing embeddings...");
        // Pull raw embeddings from each chain
        const chainQueryEmb = [];
        const chainRefEmb = [];
        for (let c = 0; c < chains.length; c++) {
            const qMat = chains[c].getEmbedding(queries.map(q => {
                const v = q.embedding;
                if (!v || v.length === 0)
                    throw new Error(`Query ${c} has empty embedding`);
                return Array.from(v);
            }));
            const rMat = chains[c].getEmbedding(reference.map(r => {
                const v = r.embedding;
                if (!v || v.length === 0)
                    throw new Error(`Reference has empty embedding`);
                return Array.from(v);
            }));
            // Validate dims & normalize if cosine
            const qArr = qMat.map(row => Float32Array.from(row));
            const rArr = rMat.map(row => Float32Array.from(row));
            if (metric === "cosine") {
                chainQueryEmb.push(qArr.map(normalize));
                chainRefEmb.push(rArr.map(normalize));
            }
            else {
                chainQueryEmb.push(qArr);
                chainRefEmb.push(rArr);
            }
            // Basic safety: check dimensions match across Q/R for this chain
            const dimQ = (_k = (_j = qArr[0]) === null || _j === void 0 ? void 0 : _j.length) !== null && _k !== void 0 ? _k : 0;
            const dimR = (_m = (_l = rArr[0]) === null || _l === void 0 ? void 0 : _l.length) !== null && _m !== void 0 ? _m : 0;
            if (dimQ === 0 || dimR === 0 || dimQ !== dimR) {
                throw new Error(`Chain ${c}: query/ref embedding dims mismatch (${dimQ} vs ${dimR})`);
            }
        }
        console.log("✅ Precomputation complete. Starting retrieval evaluation...");
        let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
        let used = 0;
        const perLabelRaw = {};
        const rankings = [];
        for (let i = 0; i < queries.length; i++) {
            if (i % logEvery === 0)
                console.log(`🔍 Query ${i + 1}/${queries.length}`);
            const correctLabel = ((_o = queries[i].metadata.label) !== null && _o !== void 0 ? _o : "").toString();
            if (!correctLabel && ignoreUnlabeled) {
                continue; // skip this query entirely
            }
            // Accumulate ensemble scores per reference
            // We keep (label, score) per reference j
            const scores = new Array(reference.length);
            for (let j = 0; j < reference.length; j++) {
                let sAgg;
                if (aggregate === "max") {
                    // Take max across chains
                    let sMax = -Infinity;
                    for (let c = 0; c < chains.length; c++) {
                        const q = chainQueryEmb[c][i];
                        const r = chainRefEmb[c][j];
                        const s = metric === "cosine" || metric === "dot" ? dot$1(q, r) : dot$1(q, r); // only cosine/dot supported
                        if (s > sMax)
                            sMax = s;
                    }
                    sAgg = sMax;
                }
                else if (aggregate === "sum") {
                    let sSum = 0;
                    for (let c = 0; c < chains.length; c++) {
                        const q = chainQueryEmb[c][i];
                        const r = chainRefEmb[c][j];
                        sSum += (metric === "cosine" || metric === "dot") ? dot$1(q, r) : dot$1(q, r);
                    }
                    sAgg = sSum;
                }
                else if (aggregate === "weighted") {
                    let sW = 0;
                    for (let c = 0; c < chains.length; c++) {
                        const q = chainQueryEmb[c][i];
                        const r = chainRefEmb[c][j];
                        sW += ((metric === "cosine" || metric === "dot") ? dot$1(q, r) : dot$1(q, r)) * weights[c];
                    }
                    sAgg = sW;
                }
                else { // "mean"
                    let sSum = 0;
                    for (let c = 0; c < chains.length; c++) {
                        const q = chainQueryEmb[c][i];
                        const r = chainRefEmb[c][j];
                        sSum += (metric === "cosine" || metric === "dot") ? dot$1(q, r) : dot$1(q, r);
                    }
                    sAgg = sSum / chains.length;
                }
                scores[j] = {
                    label: ((_p = reference[j].metadata.label) !== null && _p !== void 0 ? _p : "").toString(),
                    score: sAgg
                };
            }
            // Sort by score desc
            scores.sort((a, b) => b.score - a.score);
            const rankedLabels = scores.map(s => s.label);
            // Update metrics
            const r1 = rankedLabels[0] === correctLabel ? 1 : 0;
            const rK = rankedLabels.slice(0, topK).includes(correctLabel) ? 1 : 0;
            const rank = rankedLabels.indexOf(correctLabel);
            const rr = rank === -1 ? 0 : 1 / (rank + 1);
            hitsAt1 += r1;
            hitsAtK += rK;
            reciprocalRanks += rr;
            used++;
            if (reportPerLabel) {
                const bucket = (_q = perLabelRaw[correctLabel]) !== null && _q !== void 0 ? _q : (perLabelRaw[correctLabel] = { count: 0, hitsAt1: 0, hitsAtK: 0, mrrSum: 0 });
                bucket.count++;
                bucket.hitsAt1 += r1;
                bucket.hitsAtK += rK;
                bucket.mrrSum += rr;
            }
            if (returnRankings) {
                rankings.push({
                    queryIndex: i,
                    queryId: queries[i].id,
                    label: correctLabel,
                    topK: scores.slice(0, topK),
                    correctRank: rank
                });
            }
        }
        const denom = used || 1;
        const result = {
            usedQueries: used,
            recallAt1: hitsAt1 / denom,
            recallAtK: hitsAtK / denom,
            mrr: reciprocalRanks / denom
        };
        if (reportPerLabel) {
            const out = {};
            for (const [label, s] of Object.entries(perLabelRaw)) {
                out[label] = {
                    support: s.count,
                    recallAt1: s.hitsAt1 / (s.count || 1),
                    recallAtK: s.hitsAtK / (s.count || 1),
                    mrr: s.mrrSum / (s.count || 1)
                };
            }
            result.perLabel = out;
        }
        if (returnRankings)
            result.rankings = rankings;
        return result;
    }

    // Evaluation.ts — Classification & Regression metrics (no deps)
    const EPS = 1e-12;
    /* =========================
     * Helpers
     * ========================= */
    function isOneHot(Y) {
        return Array.isArray(Y[0]);
    }
    function argmax(a) {
        let i = 0;
        for (let k = 1; k < a.length; k++)
            if (a[k] > a[i])
                i = k;
        return i;
    }
    function toIndexLabels(yTrue, yPred, numClasses) {
        let yTrueIdx;
        let yPredIdx;
        if (isOneHot(yTrue))
            yTrueIdx = yTrue.map(argmax);
        else
            yTrueIdx = yTrue;
        if (isOneHot(yPred))
            yPredIdx = yPred.map(argmax);
        else
            yPredIdx = yPred;
        const C = 1 + Math.max(Math.max(...yTrueIdx), Math.max(...yPredIdx));
        return { yTrueIdx, yPredIdx, C };
    }
    /* =========================
     * Confusion matrix
     * ========================= */
    function confusionMatrixFromIndices(yTrueIdx, yPredIdx, C) {
        if (yTrueIdx.length !== yPredIdx.length) {
            throw new Error(`confusionMatrix: length mismatch (${yTrueIdx.length} vs ${yPredIdx.length})`);
        }
        const classes = C !== null && C !== void 0 ? C : 1 + Math.max(Math.max(...yTrueIdx), Math.max(...yPredIdx));
        const M = Array.from({ length: classes }, () => new Array(classes).fill(0));
        for (let i = 0; i < yTrueIdx.length; i++) {
            const r = yTrueIdx[i] | 0;
            const c = yPredIdx[i] | 0;
            if (r >= 0 && r < classes && c >= 0 && c < classes)
                M[r][c]++;
        }
        return M;
    }
    /* =========================
     * Per-class metrics
     * ========================= */
    function perClassFromCM(M, labels) {
        var _a;
        const C = M.length;
        const totals = new Array(C).fill(0);
        const colTotals = new Array(C).fill(0);
        let N = 0;
        for (let i = 0; i < C; i++) {
            let rsum = 0;
            for (let j = 0; j < C; j++) {
                rsum += M[i][j];
                colTotals[j] += M[i][j];
                N += M[i][j];
            }
            totals[i] = rsum;
        }
        const perClass = [];
        for (let k = 0; k < C; k++) {
            const tp = M[k][k];
            const fp = colTotals[k] - tp;
            const fn = totals[k] - tp;
            const tn = N - tp - fp - fn;
            const precision = tp / (tp + fp + EPS);
            const recall = tp / (tp + fn + EPS);
            const f1 = (2 * precision * recall) / (precision + recall + EPS);
            perClass.push({
                label: (_a = labels === null || labels === void 0 ? void 0 : labels[k]) !== null && _a !== void 0 ? _a : k,
                support: totals[k],
                tp, fp, fn, tn,
                precision, recall, f1
            });
        }
        return perClass;
    }
    /* =========================
     * Averages
     * ========================= */
    function averagesFromPerClass(per, accuracy) {
        const C = per.length;
        let sumP = 0, sumR = 0, sumF = 0;
        let sumWP = 0, sumWR = 0, sumWF = 0, total = 0;
        let tp = 0, fp = 0, fn = 0; // for micro
        for (const c of per) {
            sumP += c.precision;
            sumR += c.recall;
            sumF += c.f1;
            sumWP += c.precision * c.support;
            sumWR += c.recall * c.support;
            sumWF += c.f1 * c.support;
            total += c.support;
            tp += c.tp;
            fp += c.fp;
            fn += c.fn;
        }
        const microP = tp / (tp + fp + EPS);
        const microR = tp / (tp + fn + EPS);
        const microF = (2 * microP * microR) / (microP + microR + EPS);
        return {
            accuracy,
            macroPrecision: sumP / C,
            macroRecall: sumR / C,
            macroF1: sumF / C,
            microPrecision: microP,
            microRecall: microR,
            microF1: microF,
            weightedPrecision: sumWP / (total + EPS),
            weightedRecall: sumWR / (total + EPS),
            weightedF1: sumWF / (total + EPS)
        };
    }
    /* =========================
     * Log loss / cross-entropy
     * ========================= */
    function logLoss(yTrue, yPredProba) {
        if (!isOneHot(yTrue) || !isOneHot(yPredProba)) {
            throw new Error('logLoss expects one-hot ground truth and probability matrix (N x C).');
        }
        const Y = yTrue;
        const P = yPredProba;
        if (Y.length !== P.length)
            throw new Error('logLoss: length mismatch');
        const N = Y.length;
        let sum = 0;
        for (let i = 0; i < N; i++) {
            const yi = Y[i];
            const pi = P[i];
            if (yi.length !== pi.length)
                throw new Error('logLoss: class count mismatch');
            for (let j = 0; j < yi.length; j++) {
                if (yi[j] > 0) {
                    const p = Math.min(Math.max(pi[j], EPS), 1 - EPS);
                    sum += -Math.log(p);
                }
            }
        }
        return sum / N;
    }
    /* =========================
     * Top-K accuracy
     * ========================= */
    function topKAccuracy(yTrueIdx, yPredProba, k = 5) {
        const N = yTrueIdx.length;
        let correct = 0;
        for (let i = 0; i < N; i++) {
            const probs = yPredProba[i];
            const idx = probs.map((p, j) => j).sort((a, b) => probs[b] - probs[a]).slice(0, Math.max(1, Math.min(k, probs.length)));
            if (idx.includes(yTrueIdx[i]))
                correct++;
        }
        return correct / N;
    }
    function pairSortByScore(yTrue01, yScore) {
        const pairs = yScore.map((s, i) => [s, yTrue01[i]]);
        pairs.sort((a, b) => b[0] - a[0]);
        return pairs;
    }
    function binaryROC(yTrue01, yScore) {
        if (yTrue01.length !== yScore.length)
            throw new Error('binaryROC: length mismatch');
        const pairs = pairSortByScore(yTrue01, yScore);
        const P = yTrue01.reduce((s, v) => s + (v ? 1 : 0), 0);
        const N = yTrue01.length - P;
        let tp = 0, fp = 0;
        const tpr = [0], fpr = [0], thr = [Infinity];
        for (let i = 0; i < pairs.length; i++) {
            const [score, y] = pairs[i];
            if (y === 1)
                tp++;
            else
                fp++;
            tpr.push(tp / (P + EPS));
            fpr.push(fp / (N + EPS));
            thr.push(score);
        }
        tpr.push(1);
        fpr.push(1);
        thr.push(-Infinity);
        // Trapezoidal AUC
        let auc = 0;
        for (let i = 1; i < tpr.length; i++) {
            const dx = fpr[i] - fpr[i - 1];
            const yAvg = (tpr[i] + tpr[i - 1]) / 2;
            auc += dx * yAvg;
        }
        return { thresholds: thr, tpr, fpr, auc };
    }
    function binaryPR(yTrue01, yScore) {
        if (yTrue01.length !== yScore.length)
            throw new Error('binaryPR: length mismatch');
        const pairs = pairSortByScore(yTrue01, yScore);
        const P = yTrue01.reduce((s, v) => s + (v ? 1 : 0), 0);
        let tp = 0, fp = 0;
        const precision = [], recall = [], thr = [];
        // Add starting point
        precision.push(P > 0 ? P / (P + 0) : 1);
        recall.push(0);
        thr.push(Infinity);
        for (let i = 0; i < pairs.length; i++) {
            const [score, y] = pairs[i];
            if (y === 1)
                tp++;
            else
                fp++;
            const prec = tp / (tp + fp + EPS);
            const rec = tp / (P + EPS);
            precision.push(prec);
            recall.push(rec);
            thr.push(score);
        }
        // AUPRC via trapezoid over recall axis
        let auc = 0;
        for (let i = 1; i < precision.length; i++) {
            const dx = recall[i] - recall[i - 1];
            const yAvg = (precision[i] + precision[i - 1]) / 2;
            auc += dx * yAvg;
        }
        return { thresholds: thr, precision, recall, auc };
    }
    /* =========================
     * Main: evaluate classification
     * ========================= */
    /**
     * Evaluate multi-class classification.
     * - yTrue can be indices (N) or one-hot (N x C)
     * - yPred can be indices (N) or probabilities (N x C)
     * - If yPred are probabilities, we also compute logLoss and optional topK.
     */
    function evaluateClassification(yTrue, yPred, opts) {
        const labels = opts === null || opts === void 0 ? void 0 : opts.labels;
        const { yTrueIdx, yPredIdx, C } = toIndexLabels(yTrue, yPred);
        const M = confusionMatrixFromIndices(yTrueIdx, yPredIdx, C);
        const per = perClassFromCM(M, labels);
        const correct = yTrueIdx.reduce((s, yt, i) => s + (yt === yPredIdx[i] ? 1 : 0), 0);
        const accuracy = correct / yTrueIdx.length;
        const averages = averagesFromPerClass(per, accuracy);
        // Optional extras if we have probabilities
        if (isOneHot(yTrue) && isOneHot(yPred)) {
            try {
                averages.logLoss = logLoss(yTrue, yPred);
                if ((opts === null || opts === void 0 ? void 0 : opts.topK) && opts.topK > 1) {
                    averages.topKAccuracy = topKAccuracy(yTrueIdx, yPred, opts.topK);
                }
            }
            catch ( /* ignore extras if shapes disagree */_a) { /* ignore extras if shapes disagree */ }
        }
        return { confusionMatrix: M, perClass: per, averages };
    }
    /* =========================
     * Regression
     * ========================= */
    function evaluateRegression(yTrue, yPred) {
        const Y = Array.isArray(yTrue[0]) ? yTrue : yTrue.map(v => [v]);
        const P = Array.isArray(yPred[0]) ? yPred : yPred.map(v => [v]);
        if (Y.length !== P.length)
            throw new Error('evaluateRegression: length mismatch');
        const N = Y.length;
        const D = Y[0].length;
        const perOutput = [];
        let sumMSE = 0, sumMAE = 0, sumR2 = 0;
        for (let d = 0; d < D; d++) {
            let mse = 0, mae = 0;
            // mean of Y[:,d]
            let mean = 0;
            for (let i = 0; i < N; i++)
                mean += Y[i][d];
            mean /= N;
            let ssTot = 0, ssRes = 0;
            for (let i = 0; i < N; i++) {
                const y = Y[i][d], p = P[i][d];
                const e = y - p;
                mse += e * e;
                mae += Math.abs(e);
                ssRes += e * e;
                const dy = y - mean;
                ssTot += dy * dy;
            }
            mse /= N;
            const rmse = Math.sqrt(mse);
            mae /= N;
            const r2 = 1 - (ssRes / (ssTot + EPS));
            perOutput.push({ index: d, mse, rmse, mae, r2 });
            sumMSE += mse;
            sumMAE += mae;
            sumR2 += r2;
        }
        const mse = sumMSE / D;
        const rmse = Math.sqrt(mse);
        const mae = sumMAE / D;
        const r2 = sumR2 / D;
        return { perOutput, mse, rmse, mae, r2 };
    }
    /* =========================
     * Pretty report (optional)
     * ========================= */
    function formatClassificationReport(rep) {
        const lines = [];
        lines.push('Class\tSupport\tPrecision\tRecall\tF1');
        for (const c of rep.perClass) {
            lines.push(`${c.label}\t${c.support}\t${c.precision.toFixed(4)}\t${c.recall.toFixed(4)}\t${c.f1.toFixed(4)}`);
        }
        const a = rep.averages;
        lines.push('');
        lines.push(`Accuracy:\t${a.accuracy.toFixed(4)}`);
        lines.push(`Macro P/R/F1:\t${a.macroPrecision.toFixed(4)}\t${a.macroRecall.toFixed(4)}\t${a.macroF1.toFixed(4)}`);
        lines.push(`Micro P/R/F1:\t${a.microPrecision.toFixed(4)}\t${a.microRecall.toFixed(4)}\t${a.microF1.toFixed(4)}`);
        lines.push(`Weighted P/R/F1:\t${a.weightedPrecision.toFixed(4)}\t${a.weightedRecall.toFixed(4)}\t${a.weightedF1.toFixed(4)}`);
        if (a.logLoss !== undefined)
            lines.push(`LogLoss:\t${a.logLoss.toFixed(6)}`);
        if (a.topKAccuracy !== undefined)
            lines.push(`TopK Acc:\t${a.topKAccuracy.toFixed(4)}`);
        return lines.join('\n');
    }

    // KernelELM.ts — Kernel Extreme Learning Machine (Exact + Nyström + Whitening)
    // Dependencies: Matrix (multiply, transpose, addRegularization, solveCholesky, identity, zeros)
    class KernelRegistry {
        static register(name, fn) {
            if (!name || typeof fn !== 'function')
                throw new Error('KernelRegistry.register: invalid args');
            this.map.set(name, fn);
        }
        static get(name) {
            const f = this.map.get(name);
            if (!f)
                throw new Error(`KernelRegistry: kernel "${name}" not found`);
            return f;
        }
    }
    KernelRegistry.map = new Map();
    function l2sq(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
    function l1(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++)
            s += Math.abs(a[i] - b[i]);
        return s;
    }
    function dot(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++)
            s += a[i] * b[i];
        return s;
    }
    function softmaxRow(v) {
        const m = Math.max(...v);
        const ex = v.map(x => Math.exp(x - m));
        const s = ex.reduce((a, b) => a + b, 0) || 1;
        return ex.map(e => e / s);
    }
    function makePRNG(seed = 123456789) {
        let s = seed | 0 || 1;
        return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0xffffffff; };
    }
    function buildKernel(spec, dim) {
        var _a, _b, _c, _d, _e;
        switch (spec.type) {
            case 'custom':
                if (!spec.name)
                    throw new Error('custom kernel requires "name"');
                return KernelRegistry.get(spec.name);
            case 'linear':
                return (x, z) => dot(x, z);
            case 'poly': {
                const gamma = (_a = spec.gamma) !== null && _a !== void 0 ? _a : 1 / Math.max(1, dim);
                const degree = (_b = spec.degree) !== null && _b !== void 0 ? _b : 2;
                const coef0 = (_c = spec.coef0) !== null && _c !== void 0 ? _c : 1;
                return (x, z) => Math.pow(gamma * dot(x, z) + coef0, degree);
            }
            case 'laplacian': {
                const gamma = (_d = spec.gamma) !== null && _d !== void 0 ? _d : 1 / Math.max(1, dim);
                return (x, z) => Math.exp(-gamma * l1(x, z));
            }
            case 'rbf':
            default: {
                const gamma = (_e = spec.gamma) !== null && _e !== void 0 ? _e : 1 / Math.max(1, dim);
                return (x, z) => Math.exp(-gamma * l2sq(x, z));
            }
        }
    }
    /* ============== Landmark selection (Nyström) ============== */
    function pickUniform(X, m, seed = 1337) {
        const prng = makePRNG(seed);
        const N = X.length;
        const idx = Array.from({ length: N }, (_, i) => i);
        // Fisher–Yates (only first m)
        for (let i = 0; i < m; i++) {
            const j = i + Math.floor(prng() * (N - i));
            const t = idx[i];
            idx[i] = idx[j];
            idx[j] = t;
        }
        return idx.slice(0, m);
    }
    function pickKMeansPP(X, m, seed = 1337) {
        const prng = makePRNG(seed);
        const N = X.length;
        if (m >= N)
            return Array.from({ length: N }, (_, i) => i);
        const centers = [];
        centers.push(Math.floor(prng() * N));
        const D2 = new Float64Array(N).fill(Infinity);
        while (centers.length < m) {
            const c = centers[centers.length - 1];
            for (let i = 0; i < N; i++) {
                const d2 = l2sq(X[i], X[c]);
                if (d2 < D2[i])
                    D2[i] = d2;
            }
            let sum = 0;
            for (let i = 0; i < N; i++)
                sum += D2[i];
            let r = prng() * (sum || 1);
            let next = 0;
            for (let i = 0; i < N; i++) {
                r -= D2[i];
                if (r <= 0) {
                    next = i;
                    break;
                }
            }
            centers.push(next);
        }
        return centers;
    }
    /* ====================== KernelELM ====================== */
    class KernelELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s;
            // exact mode params
            this.Xtrain = [];
            this.alpha = [];
            // nystrom params
            this.Z = []; // landmarks (m x D)
            this.W = []; // weights in feature space (m x K)
            this.R = []; // symmetric whitener K_mm^{-1/2} (m x m) when whitening
            const resolved = {
                outputDim: config.outputDim,
                kernel: config.kernel,
                ridgeLambda: (_a = config.ridgeLambda) !== null && _a !== void 0 ? _a : 1e-2,
                task: (_b = config.task) !== null && _b !== void 0 ? _b : 'classification',
                mode: (_c = config.mode) !== null && _c !== void 0 ? _c : 'exact',
                nystrom: {
                    m: (_d = config.nystrom) === null || _d === void 0 ? void 0 : _d.m,
                    strategy: (_f = (_e = config.nystrom) === null || _e === void 0 ? void 0 : _e.strategy) !== null && _f !== void 0 ? _f : 'uniform',
                    seed: (_h = (_g = config.nystrom) === null || _g === void 0 ? void 0 : _g.seed) !== null && _h !== void 0 ? _h : 1337,
                    preset: (_j = config.nystrom) === null || _j === void 0 ? void 0 : _j.preset,
                    whiten: (_l = (_k = config.nystrom) === null || _k === void 0 ? void 0 : _k.whiten) !== null && _l !== void 0 ? _l : false,
                    jitter: (_o = (_m = config.nystrom) === null || _m === void 0 ? void 0 : _m.jitter) !== null && _o !== void 0 ? _o : 1e-10,
                },
                log: {
                    modelName: (_q = (_p = config.log) === null || _p === void 0 ? void 0 : _p.modelName) !== null && _q !== void 0 ? _q : 'KernelELM',
                    verbose: (_s = (_r = config.log) === null || _r === void 0 ? void 0 : _r.verbose) !== null && _s !== void 0 ? _s : false,
                },
            };
            this.cfg = resolved;
            this.verbose = this.cfg.log.verbose;
            this.name = this.cfg.log.modelName;
        }
        /* ------------------- Train ------------------- */
        fit(X, Y) {
            var _a, _b, _c, _d, _e;
            if (!(X === null || X === void 0 ? void 0 : X.length) || !((_a = X[0]) === null || _a === void 0 ? void 0 : _a.length))
                throw new Error('KernelELM.fit: empty X');
            if (!(Y === null || Y === void 0 ? void 0 : Y.length) || !((_b = Y[0]) === null || _b === void 0 ? void 0 : _b.length))
                throw new Error('KernelELM.fit: empty Y');
            if (X.length !== Y.length)
                throw new Error(`KernelELM.fit: X rows ${X.length} != Y rows ${Y.length}`);
            if (Y[0].length !== this.cfg.outputDim) {
                throw new Error(`KernelELM.fit: Y dims ${Y[0].length} != outputDim ${this.cfg.outputDim}`);
            }
            const N = X.length, D = X[0].length, K = Y[0].length;
            this.kernel = buildKernel(this.cfg.kernel, D);
            if (this.cfg.mode === 'exact') {
                // Gram K (N x N)
                if (this.verbose)
                    console.log(`🔧 [${this.name}] exact Gram: N=${N}, D=${D}`);
                const Kmat = new Array(N);
                for (let i = 0; i < N; i++) {
                    const row = new Array(N);
                    Kmat[i] = row;
                    row[i] = 1;
                    for (let j = i + 1; j < N; j++)
                        row[j] = this.kernel(X[i], X[j]);
                }
                for (let i = 1; i < N; i++)
                    for (let j = 0; j < i; j++)
                        Kmat[i][j] = Kmat[j][i];
                // (K + λI) α = Y
                const A = Matrix.addRegularization(Kmat, this.cfg.ridgeLambda + 1e-10);
                const Alpha = Matrix.solveCholesky(A, Y, 1e-12); // (N x K)
                this.Xtrain = X.map(r => r.slice());
                this.alpha = Alpha;
                this.Z = [];
                this.W = [];
                this.R = [];
                if (this.verbose)
                    console.log(`✅ [${this.name}] exact fit complete: alpha(${N}x${K})`);
                return;
            }
            // ---------- Nyström ----------
            const ny = this.cfg.nystrom;
            let Z;
            if (ny.strategy === 'preset' && (((_c = ny.preset) === null || _c === void 0 ? void 0 : _c.points) || ((_d = ny.preset) === null || _d === void 0 ? void 0 : _d.indices))) {
                Z = ny.preset.points ? ny.preset.points.map(r => r.slice())
                    : ny.preset.indices.map(i => X[i]);
            }
            else {
                const m = (_e = ny.m) !== null && _e !== void 0 ? _e : Math.max(10, Math.min(300, Math.floor(Math.sqrt(N))));
                const idx = (ny.strategy === 'kmeans++') ? pickKMeansPP(X, m, ny.seed) : pickUniform(X, m, ny.seed);
                Z = idx.map(i => X[i]);
            }
            const m = Z.length;
            if (this.verbose)
                console.log(`🔹 [${this.name}] Nyström: m=${m}, strategy=${ny.strategy}, whiten=${ny.whiten ? 'on' : 'off'}`);
            // K_nm (N x m)
            const Knm = new Array(N);
            for (let i = 0; i < N; i++) {
                const row = new Array(m), xi = X[i];
                for (let j = 0; j < m; j++)
                    row[j] = this.kernel(xi, Z[j]);
                Knm[i] = row;
            }
            // Optional whitening with R = K_mm^{-1/2} (symmetric via eigen)
            let Phi = Knm;
            let R = [];
            if (ny.whiten) {
                // K_mm (m x m)
                const Kmm = new Array(m);
                for (let i = 0; i < m; i++) {
                    const row = new Array(m);
                    Kmm[i] = row;
                    row[i] = 1;
                    for (let j = i + 1; j < m; j++)
                        row[j] = this.kernel(Z[i], Z[j]);
                }
                for (let i = 1; i < m; i++)
                    for (let j = 0; j < i; j++)
                        Kmm[i][j] = Kmm[j][i];
                // R = K_mm^{-1/2} with jitter
                const KmmJ = Matrix.addRegularization(Kmm, ny.jitter);
                R = Matrix.invSqrtSym(KmmJ, ny.jitter);
                Phi = Matrix.multiply(Knm, R); // (N x m)
            }
            // Ridge in feature space: W = (Φᵀ Φ + λ I)^-1 Φᵀ Y   (m x K)
            const PhiT = Matrix.transpose(Phi);
            const G = Matrix.multiply(PhiT, Phi); // (m x m)
            const Greg = Matrix.addRegularization(G, this.cfg.ridgeLambda + 1e-10);
            const Rhs = Matrix.multiply(PhiT, Y); // (m x K)
            const W = Matrix.solveCholesky(Greg, Rhs, 1e-12); // (m x K)
            this.Z = Z;
            this.W = W;
            this.R = R; // empty when whiten=false
            this.Xtrain = [];
            this.alpha = [];
            if (this.verbose)
                console.log(`✅ [${this.name}] Nyström fit complete: Z(${m}x${D}), W(${m}x${K})`);
        }
        /* --------------- Features / Predict --------------- */
        featuresFor(X) {
            if (this.cfg.mode === 'exact') {
                const N = this.Xtrain.length, M = X.length;
                const Kqx = new Array(M);
                for (let i = 0; i < M; i++) {
                    const row = new Array(N), xi = X[i];
                    for (let j = 0; j < N; j++)
                        row[j] = this.kernel(xi, this.Xtrain[j]);
                    Kqx[i] = row;
                }
                return Kqx;
            }
            // Nyström
            if (!this.Z.length)
                throw new Error('featuresFor: Nyström model not fitted');
            const M = X.length, m = this.Z.length;
            const Kxm = new Array(M);
            for (let i = 0; i < M; i++) {
                const row = new Array(m), xi = X[i];
                for (let j = 0; j < m; j++)
                    row[j] = this.kernel(xi, this.Z[j]);
                Kxm[i] = row;
            }
            return this.R.length ? Matrix.multiply(Kxm, this.R) : Kxm;
        }
        /** Raw logits for batch (M x K) */
        predictLogitsFromVectors(X) {
            const Phi = this.featuresFor(X);
            if (this.cfg.mode === 'exact') {
                if (!this.alpha.length)
                    throw new Error('predict: exact model not fitted');
                return Matrix.multiply(Phi, this.alpha);
            }
            if (!this.W.length)
                throw new Error('predict: Nyström model not fitted');
            return Matrix.multiply(Phi, this.W);
        }
        /** Probabilities for classification; raw scores for regression */
        predictProbaFromVectors(X) {
            const logits = this.predictLogitsFromVectors(X);
            return this.cfg.task === 'classification' ? logits.map(softmaxRow) : logits;
        }
        /** Top-K for classification */
        predictTopKFromVectors(X, k = 5) {
            const P = this.predictProbaFromVectors(X);
            return P.map(row => row.map((p, i) => ({ index: i, prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, k));
        }
        /** Embedding for chaining:
         *  - exact: Φ = K(X, X_train)  (M x N)
         *  - nystrom: Φ = K(X, Z)      (M x m)  or K(X,Z)·R if whiten=true
         */
        getEmbedding(X) {
            return this.featuresFor(X);
        }
        /* -------------------- JSON I/O -------------------- */
        toJSON() {
            const base = { config: Object.assign(Object.assign({}, this.cfg), { __version: 'kelm-2.1.0' }) };
            if (this.cfg.mode === 'exact') {
                return Object.assign(Object.assign({}, base), { X: this.Xtrain, alpha: this.alpha });
            }
            return Object.assign(Object.assign({}, base), { Z: this.Z, W: this.W, R: this.R.length ? this.R : undefined });
        }
        fromJSON(payload) {
            var _a, _b, _c, _d, _e, _f, _g, _h;
            const obj = typeof payload === 'string' ? JSON.parse(payload) : payload;
            // Merge config (keep current defaults where missing)
            this.cfg.kernel = Object.assign({}, obj.config.kernel);
            this.cfg.ridgeLambda = (_a = obj.config.ridgeLambda) !== null && _a !== void 0 ? _a : this.cfg.ridgeLambda;
            this.cfg.task = ((_b = obj.config.task) !== null && _b !== void 0 ? _b : this.cfg.task);
            this.cfg.mode = ((_c = obj.config.mode) !== null && _c !== void 0 ? _c : this.cfg.mode);
            this.cfg.nystrom = Object.assign(Object.assign({}, this.cfg.nystrom), ((_d = obj.config.nystrom) !== null && _d !== void 0 ? _d : {}));
            // Restore params
            if (obj.X && obj.alpha) {
                this.Xtrain = obj.X.map(r => r.slice());
                this.alpha = obj.alpha.map(r => r.slice());
                this.Z = [];
                this.W = [];
                this.R = [];
                const D = (_f = (_e = this.Xtrain[0]) === null || _e === void 0 ? void 0 : _e.length) !== null && _f !== void 0 ? _f : 1;
                this.kernel = buildKernel(this.cfg.kernel, D);
                return;
            }
            if (obj.Z && obj.W) {
                this.Z = obj.Z.map(r => r.slice());
                this.W = obj.W.map(r => r.slice());
                this.R = obj.R ? obj.R.map(r => r.slice()) : [];
                this.Xtrain = [];
                this.alpha = [];
                const D = (_h = (_g = this.Z[0]) === null || _g === void 0 ? void 0 : _g.length) !== null && _h !== void 0 ? _h : 1;
                this.kernel = buildKernel(this.cfg.kernel, D);
                return;
            }
            throw new Error('KernelELM.fromJSON: invalid payload');
        }
    }

    class TFIDF {
        constructor(corpusDocs) {
            this.termFrequency = {};
            this.inverseDocFreq = {};
            this.wordsInDoc = [];
            this.processedWords = [];
            this.scores = {};
            this.corpus = "";
            this.corpus = corpusDocs.join(" ");
            const wordsFinal = [];
            const re = /[^a-zA-Z0-9]+/g;
            corpusDocs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                tokens.forEach(word => {
                    const cleaned = word.replace(re, " ");
                    wordsFinal.push(...cleaned.split(/\s+/).filter(Boolean));
                });
            });
            this.wordsInDoc = wordsFinal;
            this.processedWords = TFIDF.processWords(wordsFinal);
            // Compute term frequency
            this.processedWords.forEach(token => {
                this.termFrequency[token] = (this.termFrequency[token] || 0) + 1;
            });
            // Compute inverse document frequency
            for (const term in this.termFrequency) {
                const count = TFIDF.countDocsContainingTerm(corpusDocs, term);
                this.inverseDocFreq[term] = Math.log(corpusDocs.length / (1 + count));
            }
        }
        static countDocsContainingTerm(corpusDocs, term) {
            return corpusDocs.reduce((acc, doc) => (doc.includes(term) ? acc + 1 : acc), 0);
        }
        static processWords(words) {
            const filtered = TFIDF.removeStopWordsAndStem(words).map(w => TFIDF.lemmatize(w));
            const bigrams = TFIDF.generateNGrams(filtered, 2);
            const trigrams = TFIDF.generateNGrams(filtered, 3);
            return [...filtered, ...bigrams, ...trigrams];
        }
        static removeStopWordsAndStem(words) {
            const stopWords = new Set([
                "a", "and", "the", "is", "to", "of", "in", "it", "that", "you",
                "this", "for", "on", "are", "with", "as", "be", "by", "at", "from",
                "or", "an", "but", "not", "we"
            ]);
            return words.filter(w => !stopWords.has(w)).map(w => TFIDF.advancedStem(w));
        }
        static advancedStem(word) {
            const programmingKeywords = new Set([
                "func", "package", "import", "interface", "go",
                "goroutine", "channel", "select", "struct",
                "map", "slice", "var", "const", "type",
                "defer", "fallthrough"
            ]);
            if (programmingKeywords.has(word))
                return word;
            const suffixes = ["es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"];
            for (const suffix of suffixes) {
                if (word.endsWith(suffix)) {
                    if (suffix === "es" && word.length > 2 && word[word.length - 3] === "i") {
                        return word.slice(0, -2);
                    }
                    return word.slice(0, -suffix.length);
                }
            }
            return word;
        }
        static lemmatize(word) {
            const rules = {
                execute: "execute",
                running: "run",
                returns: "return",
                defined: "define",
                compiles: "compile",
                calls: "call",
                creating: "create",
                invoke: "invoke",
                declares: "declare",
                references: "reference",
                implements: "implement",
                utilizes: "utilize",
                tests: "test",
                loops: "loop",
                deletes: "delete",
                functions: "function"
            };
            if (rules[word])
                return rules[word];
            if (word.endsWith("ing"))
                return word.slice(0, -3);
            if (word.endsWith("ed"))
                return word.slice(0, -2);
            return word;
        }
        static generateNGrams(tokens, n) {
            if (tokens.length < n)
                return [];
            const ngrams = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                ngrams.push(tokens.slice(i, i + n).join(" "));
            }
            return ngrams;
        }
        calculateScores() {
            const totalWords = this.processedWords.length;
            const scores = {};
            this.processedWords.forEach(token => {
                const tf = this.termFrequency[token] || 0;
                scores[token] = (tf / totalWords) * (this.inverseDocFreq[token] || 0);
            });
            this.scores = scores;
            return scores;
        }
        extractKeywords(topN) {
            const entries = Object.entries(this.scores).sort((a, b) => b[1] - a[1]);
            return Object.fromEntries(entries.slice(0, topN));
        }
        processedWordsIndex(word) {
            return this.processedWords.indexOf(word);
        }
    }
    class TFIDFVectorizer {
        constructor(docs, maxVocabSize = 2000) {
            this.docTexts = docs;
            this.tfidf = new TFIDF(docs);
            // Collect all unique terms with frequencies
            const termFreq = {};
            docs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
                const processed = TFIDF.processWords(cleaned);
                processed.forEach(t => {
                    termFreq[t] = (termFreq[t] || 0) + 1;
                });
            });
            // Sort terms by frequency descending
            const sortedTerms = Object.entries(termFreq)
                .sort((a, b) => b[1] - a[1])
                .slice(0, maxVocabSize)
                .map(([term]) => term);
            this.vocabulary = sortedTerms;
            console.log(`✅ TFIDFVectorizer vocabulary capped at: ${this.vocabulary.length} terms.`);
        }
        /**
         * Returns the dense TFIDF vector for a given document text.
         */
        vectorize(doc) {
            const tokens = doc.split(/\s+/);
            const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
            const processed = TFIDF.processWords(cleaned);
            // Compute term frequency in this document
            const termFreq = {};
            processed.forEach(token => {
                termFreq[token] = (termFreq[token] || 0) + 1;
            });
            const totalTerms = processed.length;
            return this.vocabulary.map(term => {
                const tf = totalTerms > 0 ? (termFreq[term] || 0) / totalTerms : 0;
                const idf = this.tfidf.inverseDocFreq[term] || 0;
                return tf * idf;
            });
        }
        /**
         * Returns vectors for all original training docs.
         */
        vectorizeAll() {
            return this.docTexts.map(doc => this.vectorize(doc));
        }
        /**
         * Optional L2 normalization utility.
         */
        static l2normalize(vec) {
            const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
            return norm === 0 ? vec : vec.map(x => x / norm);
        }
    }

    class KNN {
        /**
         * Compute cosine similarity between two numeric vectors.
         */
        static cosineSimilarity(vec1, vec2) {
            let dot = 0, norm1 = 0, norm2 = 0;
            for (let i = 0; i < vec1.length; i++) {
                dot += vec1[i] * vec2[i];
                norm1 += vec1[i] * vec1[i];
                norm2 += vec2[i] * vec2[i];
            }
            if (norm1 === 0 || norm2 === 0)
                return 0;
            return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        }
        /**
         * Compute Euclidean distance between two numeric vectors.
         */
        static euclideanDistance(vec1, vec2) {
            let sum = 0;
            for (let i = 0; i < vec1.length; i++) {
                const diff = vec1[i] - vec2[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
        /**
         * Find k nearest neighbors.
         * @param queryVec - Query vector
         * @param dataset - Dataset to search
         * @param k - Number of neighbors
         * @param topX - Number of top results to return
         * @param metric - Similarity metric
         */
        static find(queryVec, dataset, k = 5, topX = 3, metric = "cosine") {
            const similarities = dataset.map((item, idx) => {
                let score;
                if (metric === "cosine") {
                    score = this.cosineSimilarity(queryVec, item.vector);
                }
                else {
                    // For Euclidean, invert distance so higher = closer
                    const dist = this.euclideanDistance(queryVec, item.vector);
                    score = -dist;
                }
                return { index: idx, score };
            });
            similarities.sort((a, b) => b.score - a.score);
            const labelWeights = {};
            for (let i = 0; i < Math.min(k, similarities.length); i++) {
                const label = dataset[similarities[i].index].label;
                const weight = similarities[i].score;
                labelWeights[label] = (labelWeights[label] || 0) + weight;
            }
            const weightedLabels = Object.entries(labelWeights)
                .map(([label, weight]) => ({ label, weight }))
                .sort((a, b) => b.weight - a.weight);
            return weightedLabels.slice(0, topX);
        }
    }

    // BindUI.ts - Utility to bind ELM model to HTML inputs and outputs
    function bindAutocompleteUI({ model, inputElement, outputElement, topK = 5 }) {
        inputElement.addEventListener('input', () => {
            const typed = inputElement.value.trim();
            if (typed.length === 0) {
                outputElement.innerHTML = '<em>Start typing...</em>';
                return;
            }
            try {
                const results = model.predict(typed, topK);
                outputElement.innerHTML = results.map(r => `
                <div><strong>${r.label}</strong>: ${(r.prob * 100).toFixed(1)}%</div>
            `).join('');
            }
            catch (e) {
                const message = e instanceof Error ? e.message : 'Unknown error';
                outputElement.innerHTML = `<span style="color: red;">Error: ${message}</span>`;
            }
        });
    }

    // Presets.ts — Reusable configuration presets for ELM (updated for new ELMConfig union)
    /**
     * NOTE:
     * - These are TEXT presets (token-mode). They set `useTokenizer: true`.
     * - If you need char-level, create an inline config where `useTokenizer: false`
     *   and pass it directly to ELM (numeric presets generally need an explicit inputSize).
     */
    /** English token-level preset */
    const EnglishTokenPreset = {
        useTokenizer: true,
        maxLen: 20,
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/};

    // ✅ AutoComplete.ts — ELM | KernelELM (Nyström+whiten) | OnlineELM
    // Fixes:
    //  • Avoids union narrowing on EnglishTokenPreset by shimming preset fields (no ExtendedELMConfig maxLen error)
    //  • activation typed as Activation (not string)
    //  • Removed non-existent "task" option in trainFromData()
    /** Safe accessor for preset fields (avoids type errors on ExtendedELMConfig) */
    const PRESET = (() => {
        var _a, _b, _c, _d;
        const p = EnglishTokenPreset;
        return {
            maxLen: (_a = p === null || p === void 0 ? void 0 : p.maxLen) !== null && _a !== void 0 ? _a : 30,
            charSet: (_b = p === null || p === void 0 ? void 0 : p.charSet) !== null && _b !== void 0 ? _b : 'abcdefghijklmnopqrstuvwxyz',
            useTokenizer: (_c = p === null || p === void 0 ? void 0 : p.useTokenizer) !== null && _c !== void 0 ? _c : true,
            tokenizerDelimiter: (_d = p === null || p === void 0 ? void 0 : p.tokenizerDelimiter) !== null && _d !== void 0 ? _d : /\s+/
        };
    })();
    function oneHot(idx, n) {
        const v = new Array(n).fill(0);
        if (idx >= 0 && idx < n)
            v[idx] = 1;
        return v;
    }
    function sortTopK(labels, probs, k) {
        return probs
            .map((p, i) => ({ label: labels[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, k);
    }
    class AutoComplete {
        constructor(pairs, options) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _1, _2, _3, _4, _5;
            this.trainPairs = pairs;
            this.activation = (_a = options.activation) !== null && _a !== void 0 ? _a : 'relu';
            this.engine = (_b = options.engine) !== null && _b !== void 0 ? _b : 'elm';
            this.topKDefault = (_c = options.topK) !== null && _c !== void 0 ? _c : 5;
            // Labels
            this.categories = Array.from(new Set(pairs.map(p => p.label)));
            // Text → numeric encoder (Kernel/Online need numeric; ELM can also consume numeric directly)
            this.encoder = new UniversalEncoder({
                charSet: PRESET.charSet,
                maxLen: PRESET.maxLen,
                useTokenizer: PRESET.useTokenizer,
                tokenizerDelimiter: PRESET.tokenizerDelimiter,
                mode: (PRESET.useTokenizer ? 'token' : 'char'),
            });
            const hiddenUnits = (_d = options.hiddenUnits) !== null && _d !== void 0 ? _d : 128;
            const ridgeLambda = (_e = options.ridgeLambda) !== null && _e !== void 0 ? _e : 1e-2;
            const weightInit = (_f = options.weightInit) !== null && _f !== void 0 ? _f : 'xavier';
            const verbose = (_g = options.verbose) !== null && _g !== void 0 ? _g : false;
            if (this.engine === 'kernel') {
                const D = this.encoder.getVectorSize();
                const ktype = (_j = (_h = options.kernel) === null || _h === void 0 ? void 0 : _h.type) !== null && _j !== void 0 ? _j : 'rbf';
                const kernel = ktype === 'poly'
                    ? { type: 'poly', gamma: (_l = (_k = options.kernel) === null || _k === void 0 ? void 0 : _k.gamma) !== null && _l !== void 0 ? _l : (1 / Math.max(1, D)), degree: (_o = (_m = options.kernel) === null || _m === void 0 ? void 0 : _m.degree) !== null && _o !== void 0 ? _o : 2, coef0: (_q = (_p = options.kernel) === null || _p === void 0 ? void 0 : _p.coef0) !== null && _q !== void 0 ? _q : 1 }
                    : ktype === 'linear'
                        ? { type: 'linear' }
                        : ktype === 'laplacian'
                            ? { type: 'laplacian', gamma: (_s = (_r = options.kernel) === null || _r === void 0 ? void 0 : _r.gamma) !== null && _s !== void 0 ? _s : (1 / Math.max(1, D)) }
                            : { type: 'rbf', gamma: (_u = (_t = options.kernel) === null || _t === void 0 ? void 0 : _t.gamma) !== null && _u !== void 0 ? _u : (1 / Math.max(1, D)) };
                this.model = new KernelELM({
                    outputDim: this.categories.length,
                    kernel,
                    ridgeLambda,
                    task: 'classification',
                    mode: 'nystrom',
                    nystrom: {
                        m: (_v = options.kernel) === null || _v === void 0 ? void 0 : _v.m,
                        strategy: (_x = (_w = options.kernel) === null || _w === void 0 ? void 0 : _w.strategy) !== null && _x !== void 0 ? _x : 'uniform',
                        seed: (_z = (_y = options.kernel) === null || _y === void 0 ? void 0 : _y.seed) !== null && _z !== void 0 ? _z : 1337,
                        preset: (_0 = options.kernel) === null || _0 === void 0 ? void 0 : _0.preset,
                        whiten: (_2 = (_1 = options.kernel) === null || _1 === void 0 ? void 0 : _1.whiten) !== null && _2 !== void 0 ? _2 : true,
                        jitter: (_4 = (_3 = options.kernel) === null || _3 === void 0 ? void 0 : _3.jitter) !== null && _4 !== void 0 ? _4 : 1e-10,
                    },
                    log: { modelName: 'AutoComplete-KELM', verbose }
                });
            }
            else if (this.engine === 'online') {
                const inputDim = this.encoder.getVectorSize();
                this.model = new OnlineELM({
                    inputDim,
                    outputDim: this.categories.length,
                    hiddenUnits,
                    activation: this.activation,
                    ridgeLambda,
                    weightInit: (_5 = weightInit) !== null && _5 !== void 0 ? _5 : 'he',
                    forgettingFactor: 0.997,
                    log: { modelName: 'AutoComplete-OnlineELM', verbose }
                });
            }
            else {
                // Classic ELM — use TextConfig branch explicitly
                this.model = new ELM({
                    categories: this.categories,
                    hiddenUnits,
                    activation: this.activation,
                    ridgeLambda,
                    weightInit: weightInit === 'he' ? 'xavier' : weightInit, // map 'he' to 'xavier' if needed
                    // Text branch fields:
                    useTokenizer: true,
                    maxLen: PRESET.maxLen,
                    charSet: PRESET.charSet,
                    tokenizerDelimiter: PRESET.tokenizerDelimiter,
                    // Logging / export
                    metrics: options.metrics,
                    log: { modelName: 'AutoComplete', verbose },
                    exportFileName: options.exportFileName
                });
            }
            // Bind UI to a small adapter that calls our predict()
            bindAutocompleteUI({
                model: {
                    predict: (text, k = this.topKDefault) => this.predict(text, k)
                },
                inputElement: options.inputElement,
                outputElement: options.outputElement,
                topK: options.topK
            });
        }
        /* ============= Training ============= */
        train() {
            // Build numeric X/Y
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = this.encoder.normalize(this.encoder.encode(input));
                const idx = this.categories.indexOf(label);
                if (idx === -1)
                    continue;
                X.push(vec);
                Y.push(oneHot(idx, this.categories.length));
            }
            if (this.engine === 'kernel') {
                this.model.fit(X, Y);
                return;
            }
            if (this.engine === 'online') {
                this.model.init(X, Y); // then .update() for new batches
                return;
            }
            // Classic ELM — options: { reuseWeights?, weights? }; do NOT pass "task"
            this.model.trainFromData(X, Y);
        }
        /* ============= Prediction ============= */
        predict(input, topN = 1) {
            const k = Math.max(1, topN);
            if (this.engine === 'elm') {
                const out = this.model.predict(input, k);
                return out.map(p => ({ completion: p.label, prob: p.prob }));
            }
            const x = this.encoder.normalize(this.encoder.encode(input));
            if (this.engine === 'kernel') {
                const probs = this.model.predictProbaFromVectors([x])[0];
                return sortTopK(this.categories, probs, k).map(p => ({ completion: p.label, prob: p.prob }));
            }
            const probs = this.model.predictProbaFromVector(x);
            return sortTopK(this.categories, probs, k).map(p => ({ completion: p.label, prob: p.prob }));
        }
        /* ============= Persistence ============= */
        getModel() { return this.model; }
        loadModelFromJSON(json) {
            if (this.model.fromJSON) {
                this.model.fromJSON(json);
            }
            else if (this.model.loadModelFromJSON) {
                this.model.loadModelFromJSON(json);
            }
            else if (this.model.loadFromJSON) {
                this.model.loadFromJSON(json);
            }
            else {
                console.warn('No compatible load method found on model.');
            }
        }
        saveModelAsJSONFile(filename = 'model.json') {
            let payload;
            if (this.model.toJSON) {
                payload = this.model.toJSON(true); // OnlineELM supports includeP; KernelELM ignores extra arg
            }
            else if (this.model.savedModelJSON) {
                payload = this.model.savedModelJSON;
            }
            else {
                console.warn('No compatible toJSON/savedModelJSON on model; skipping export.');
                return;
            }
            const blob = new Blob([typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        /* ============= Evaluation helpers ============= */
        top1Accuracy(pairs) {
            var _a;
            let correct = 0;
            for (const { input, label } of pairs) {
                const [pred] = this.predict(input, 1);
                if (((_a = pred === null || pred === void 0 ? void 0 : pred.completion) === null || _a === void 0 ? void 0 : _a.toLowerCase().trim()) === label.toLowerCase().trim())
                    correct++;
            }
            return correct / Math.max(1, pairs.length);
        }
        crossEntropy(pairs) {
            var _a;
            let total = 0;
            for (const { input, label } of pairs) {
                const preds = this.predict(input, this.categories.length);
                const match = preds.find(p => p.completion.toLowerCase().trim() === label.toLowerCase().trim());
                const prob = (_a = match === null || match === void 0 ? void 0 : match.prob) !== null && _a !== void 0 ? _a : 1e-12;
                total += -Math.log(prob);
            }
            return total / Math.max(1, pairs.length);
        }
        /** Internal CE via W/b/β (only for classic ELM); others fall back to external CE. */
        internalCrossEntropy(verbose = false) {
            if (!(this.model instanceof ELM)) {
                const ce = this.crossEntropy(this.trainPairs);
                if (verbose)
                    console.log(`📏 Internal CE not applicable to ${this.engine}; external CE: ${ce.toFixed(4)}`);
                return ce;
            }
            const elm = this.model;
            const { model, categories } = elm;
            if (!model) {
                if (verbose)
                    console.warn('⚠️ Cannot compute internal cross-entropy: model not trained.');
                return Infinity;
            }
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = this.encoder.normalize(this.encoder.encode(input));
                const idx = categories.indexOf(label);
                if (idx === -1)
                    continue;
                X.push(vec);
                Y.push(oneHot(idx, categories.length));
            }
            const { W, b, beta } = model; // W: hidden x in, b: hidden x 1, beta: hidden x out
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const act = Activations.get(this.activation);
            const H = tempH.map(row => row.map((v, j) => act(v + b[j][0])));
            const logits = Matrix.multiply(H, beta);
            const probs = logits.map(row => Activations.softmax(row));
            let total = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    if (Y[i][j] === 1) {
                        const p = Math.min(Math.max(probs[i][j], 1e-15), 1 - 1e-15);
                        total += -Math.log(p);
                    }
                }
            }
            const ce = total / Math.max(1, Y.length);
            if (verbose)
                console.log(`📏 Internal Cross-Entropy (ELM W/b/β): ${ce.toFixed(4)}`);
            return ce;
        }
    }

    // CharacterLangEncoderELM.ts — robust char/token text encoder on top of ELM
    // Upgrades:
    //  • Safe preset extraction (no union-type errors on maxLen/charSet)
    //  • Proper (inputs, labels) training via trainFromData()
    //  • Hidden-layer embeddings via elm.getEmbedding() (with matrix fallback)
    //  • Batch encode(), JSON I/O passthrough, gentle logging
    //  • Activation typed, no reliance on private fields
    // If you have a preset (optional). Otherwise remove this import.
    // import { EnglishTokenPreset } from '../config/Presets';
    class CharacterLangEncoderELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
            // Make sure we have the basics
            if (!config.hiddenUnits) {
                throw new Error('CharacterLangEncoderELM requires hiddenUnits');
            }
            // Activation defaults to 'relu' if not provided
            this.activation = (_a = config.activation) !== null && _a !== void 0 ? _a : 'relu';
            // Safely coerce into a *text* config (avoid NumericConfig branch)
            // We do not assume a preset exists; provide conservative defaults.
            const textMaxLen = (_b = config === null || config === void 0 ? void 0 : config.maxLen) !== null && _b !== void 0 ? _b : 64;
            const textCharSet = (_c = config === null || config === void 0 ? void 0 : config.charSet) !== null && _c !== void 0 ? _c : 'abcdefghijklmnopqrstuvwxyz';
            const textTokDelim = (_d = config === null || config === void 0 ? void 0 : config.tokenizerDelimiter) !== null && _d !== void 0 ? _d : /\s+/;
            // Merge into a TEXT-leaning config object.
            // NOTE: We keep categories if provided, but we will override them in train() from labels.
            this.config = Object.assign(Object.assign({}, config), { 
                // Force text branch:
                useTokenizer: true, maxLen: textMaxLen, charSet: textCharSet, tokenizerDelimiter: textTokDelim, activation: this.activation, 
                // Make logging robust:
                log: {
                    modelName: 'CharacterLangEncoderELM',
                    verbose: (_f = (_e = config.log) === null || _e === void 0 ? void 0 : _e.verbose) !== null && _f !== void 0 ? _f : false,
                    toFile: (_h = (_g = config.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false,
                    level: (_k = (_j = config.log) === null || _j === void 0 ? void 0 : _j.level) !== null && _k !== void 0 ? _k : 'info',
                } }); // cast to any to avoid union friction
            this.elm = new ELM(this.config);
            // Forward thresholds/export if present
            if (config.metrics) {
                this.elm.metrics = config.metrics;
            }
            if (this.config.exportFileName) {
                this.elm.config.exportFileName = this.config.exportFileName;
            }
        }
        /**
         * Train on parallel arrays: inputs (strings) + labels (strings).
         * We:
         *  • dedupe labels → categories
         *  • encode inputs with the ELM’s text encoder
         *  • one-hot the labels
         *  • call trainFromData(X, Y)
         */
        train(inputStrings, labels) {
            var _a, _b, _c, _d;
            if (!(inputStrings === null || inputStrings === void 0 ? void 0 : inputStrings.length) || !(labels === null || labels === void 0 ? void 0 : labels.length) || inputStrings.length !== labels.length) {
                throw new Error('train() expects equal-length inputStrings and labels');
            }
            // Build categories from labels
            const categories = Array.from(new Set(labels));
            this.elm.setCategories(categories);
            // Get the encoder (support getEncoder() or .encoder)
            const enc = (_c = (_b = (_a = this.elm).getEncoder) === null || _b === void 0 ? void 0 : _b.call(_a)) !== null && _c !== void 0 ? _c : this.elm.encoder;
            if (!(enc === null || enc === void 0 ? void 0 : enc.encode) || !(enc === null || enc === void 0 ? void 0 : enc.normalize)) {
                throw new Error('ELM text encoder is not available. Ensure useTokenizer/maxLen/charSet are set.');
            }
            const X = [];
            const Y = [];
            for (let i = 0; i < inputStrings.length; i++) {
                const x = enc.normalize(enc.encode(String((_d = inputStrings[i]) !== null && _d !== void 0 ? _d : '')));
                X.push(x);
                const li = categories.indexOf(labels[i]);
                const y = new Array(categories.length).fill(0);
                if (li >= 0)
                    y[li] = 1;
                Y.push(y);
            }
            // Classic ELM closed-form training
            this.elm.trainFromData(X, Y);
        }
        /**
         * Returns a dense embedding for one string.
         * Uses ELM.getEmbedding() if available; otherwise computes H = act(XW^T + b).
         * By design this returns the *hidden* feature (length = hiddenUnits).
         */
        encode(text) {
            var _a, _b, _c;
            // Get encoder
            const enc = (_c = (_b = (_a = this.elm).getEncoder) === null || _b === void 0 ? void 0 : _b.call(_a)) !== null && _c !== void 0 ? _c : this.elm.encoder;
            if (!(enc === null || enc === void 0 ? void 0 : enc.encode) || !(enc === null || enc === void 0 ? void 0 : enc.normalize)) {
                throw new Error('ELM text encoder is not available. Train or configure text settings first.');
            }
            const x = enc.normalize(enc.encode(String(text !== null && text !== void 0 ? text : '')));
            // Prefer official embedding API if present
            if (typeof this.elm.getEmbedding === 'function') {
                const E = this.elm.getEmbedding([x]);
                if (Array.isArray(E) && Array.isArray(E[0]))
                    return E[0];
            }
            // Fallback: compute hidden act via model params (W,b)
            const model = this.elm.model;
            if (!model)
                throw new Error('Model not trained.');
            const { W, b } = model; // W: hidden x in, b: hidden x 1
            const tempH = Matrix.multiply([x], Matrix.transpose(W)); // (1 x hidden)
            const act = Activations.get(this.activation);
            const H = tempH.map(row => row.map((v, j) => act(v + b[j][0]))); // (1 x hidden)
            // Return hidden vector
            return H[0];
        }
        /** Batch encoding convenience */
        encodeBatch(texts) {
            return texts.map(t => this.encode(t));
        }
        /** Load/save passthroughs */
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // FeatureCombinerELM.ts — combine encoder vectors + metadata, train numeric ELM
    class FeatureCombinerELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h;
            this.categories = [];
            const hidden = config.hiddenUnits;
            const act = config.activation;
            if (typeof hidden !== 'number') {
                throw new Error('FeatureCombinerELM requires config.hiddenUnits (number)');
            }
            if (!act) {
                throw new Error('FeatureCombinerELM requires config.activation');
            }
            // Force numeric mode (tokenizer off). Provide a safe inputSize placeholder;
            // ELM's trainFromData learns actual dims from X at train-time.
            this.config = Object.assign(Object.assign({}, config), { categories: (_a = config.categories) !== null && _a !== void 0 ? _a : [], useTokenizer: false, inputSize: (_b = config.inputSize) !== null && _b !== void 0 ? _b : 1, log: {
                    modelName: 'FeatureCombinerELM',
                    verbose: (_d = (_c = config.log) === null || _c === void 0 ? void 0 : _c.verbose) !== null && _d !== void 0 ? _d : false,
                    toFile: (_f = (_e = config.log) === null || _e === void 0 ? void 0 : _e.toFile) !== null && _f !== void 0 ? _f : false,
                    // @ts-ignore optional level passthrough
                    level: (_h = (_g = config.log) === null || _g === void 0 ? void 0 : _g.level) !== null && _h !== void 0 ? _h : 'info',
                } });
            this.elm = new ELM(this.config);
            // Optional thresholds/export passthrough
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        /** Concatenate encoder vector + metadata vector */
        static combineFeatures(encodedVec, meta) {
            // Fast path avoids spread copies in tight loops
            const out = new Array(encodedVec.length + meta.length);
            let i = 0;
            for (; i < encodedVec.length; i++)
                out[i] = encodedVec[i];
            for (let j = 0; j < meta.length; j++)
                out[i + j] = meta[j];
            return out;
        }
        /** Convenience for batch combination */
        static combineBatch(encoded, metas) {
            if (encoded.length !== metas.length) {
                throw new Error(`combineBatch: encoded length ${encoded.length} != metas length ${metas.length}`);
            }
            const X = new Array(encoded.length);
            for (let i = 0; i < encoded.length; i++) {
                X[i] = FeatureCombinerELM.combineFeatures(encoded[i], metas[i]);
            }
            return X;
        }
        /** Train from encoder vectors + metadata + labels (classification) */
        train(encoded, metas, labels) {
            if (!(encoded === null || encoded === void 0 ? void 0 : encoded.length) || !(metas === null || metas === void 0 ? void 0 : metas.length) || !(labels === null || labels === void 0 ? void 0 : labels.length)) {
                throw new Error('train: empty encoded/metas/labels');
            }
            if (encoded.length !== metas.length || encoded.length !== labels.length) {
                throw new Error('train: lengths must match (encoded, metas, labels)');
            }
            const X = FeatureCombinerELM.combineBatch(encoded, metas);
            this.categories = Array.from(new Set(labels));
            this.elm.setCategories(this.categories);
            const Y = labels.map((lab) => {
                const idx = this.categories.indexOf(lab);
                const row = new Array(this.categories.length).fill(0);
                if (idx >= 0)
                    row[idx] = 1;
                return row;
            });
            // Closed-form solve via ELM; no private internals needed
            this.elm.trainFromData(X, Y);
        }
        /** Predict top-K labels from a single (vec, meta) pair */
        predict(encodedVec, meta, topK = 1) {
            const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
            const batches = this.elm.predictFromVector(input, topK);
            return batches[0];
        }
        /** Predict the single best label + prob */
        predictLabel(encodedVec, meta) {
            const [top] = this.predict(encodedVec, meta, 1);
            return top;
        }
        /** Get hidden embedding for (vec, meta) pair (useful for chaining) */
        getEmbedding(encodedVec, meta) {
            const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
            const H = this.elm.getEmbedding(input);
            return H[0];
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // ConfidenceClassifierELM.ts — numeric confidence classifier on top of ELM
    // Upgrades:
    //  • Numeric-only pipeline (no tokenizer)
    //  • Proper trainFromData(X, Y) with one-hot labels
    //  • Vector-safe prediction (predictFromVector)
    //  • Score helpers, batch APIs, and simple evaluation
    //  • Robust logging + safe handling of ELMConfig union
    /**
     * ConfidenceClassifierELM is a lightweight wrapper that classifies whether
     * an upstream model’s prediction is "low" or "high" confidence based on
     * (embedding, metadata) numeric features.
     */
    class ConfidenceClassifierELM {
        constructor(baseConfig, opts = {}) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l;
            this.baseConfig = baseConfig;
            this.categories = (_a = opts.categories) !== null && _a !== void 0 ? _a : ['low', 'high'];
            this.activation = (_b = opts.activation) !== null && _b !== void 0 ? _b : ((_c = baseConfig.activation) !== null && _c !== void 0 ? _c : 'relu');
            // We force a numeric ELM config. Many ELM builds don’t require inputSize
            // at construction because trainFromData(X,Y) uses X[0].length to size W.
            // We still pass useTokenizer=false and categories to be explicit.
            const cfg = Object.assign(Object.assign({}, this.baseConfig), { useTokenizer: false, categories: this.categories, activation: this.activation, log: {
                    modelName: 'ConfidenceClassifierELM',
                    verbose: (_f = (_e = (_d = baseConfig.log) === null || _d === void 0 ? void 0 : _d.verbose) !== null && _e !== void 0 ? _e : opts.verbose) !== null && _f !== void 0 ? _f : false,
                    toFile: (_h = (_g = baseConfig.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false,
                    level: (_k = (_j = baseConfig.log) === null || _j === void 0 ? void 0 : _j.level) !== null && _k !== void 0 ? _k : 'info',
                }, 
                // Optional passthroughs:
                exportFileName: (_l = opts.exportFileName) !== null && _l !== void 0 ? _l : this.baseConfig.exportFileName });
            this.elm = new ELM(cfg);
            // Forward thresholds if present
            if (this.baseConfig.metrics) {
                this.elm.metrics = this.baseConfig.metrics;
            }
        }
        /** One-hot helper */
        oneHot(n, idx) {
            const v = new Array(n).fill(0);
            if (idx >= 0 && idx < n)
                v[idx] = 1;
            return v;
        }
        /**
         * Train from numeric (vector, meta) → combined features + labels.
         * `vectors[i]` and `metas[i]` must be aligned with `labels[i]`.
         */
        train(vectors, metas, labels) {
            if (!(vectors === null || vectors === void 0 ? void 0 : vectors.length) || !(metas === null || metas === void 0 ? void 0 : metas.length) || !(labels === null || labels === void 0 ? void 0 : labels.length)) {
                throw new Error('train: empty inputs');
            }
            if (vectors.length !== metas.length || vectors.length !== labels.length) {
                throw new Error('train: vectors, metas, labels must have same length');
            }
            // Ensure categories include all observed labels (keeps order of existing categories first)
            const uniq = Array.from(new Set(labels));
            const merged = Array.from(new Set([...this.categories, ...uniq]));
            this.categories = merged;
            this.elm.setCategories(this.categories);
            // Build X, Y
            const X = new Array(vectors.length);
            const Y = new Array(vectors.length);
            for (let i = 0; i < vectors.length; i++) {
                const x = FeatureCombinerELM.combineFeatures(vectors[i], metas[i]); // numeric feature vector
                X[i] = x;
                const li = this.categories.indexOf(labels[i]);
                Y[i] = this.oneHot(this.categories.length, li);
            }
            // Closed-form ELM training
            this.elm.trainFromData(X, Y);
        }
        /** Predict full distribution for a single (vec, meta). */
        predict(vec, meta, topK = 2) {
            var _a, _b;
            const x = FeatureCombinerELM.combineFeatures(vec, meta);
            // Prefer vector-safe API; most Astermind builds expose predictFromVector([x], topK)
            const fn = this.elm.predictFromVector;
            if (typeof fn === 'function') {
                const out = fn.call(this.elm, [x], topK); // PredictResult[][]
                return Array.isArray(out) && Array.isArray(out[0]) ? out[0] : (out !== null && out !== void 0 ? out : []);
            }
            // Fallback to predict() if it supports numeric vectors (some builds do)
            const maybe = (_b = (_a = this.elm).predict) === null || _b === void 0 ? void 0 : _b.call(_a, x, topK);
            if (Array.isArray(maybe))
                return maybe;
            throw new Error('No vector-safe predict available on underlying ELM.');
        }
        /** Probability the label is "high" (or the second category by default). */
        predictScore(vec, meta, positive = 'high') {
            var _a;
            const dist = this.predict(vec, meta, this.categories.length);
            const hit = dist.find(d => d.label === positive);
            return (_a = hit === null || hit === void 0 ? void 0 : hit.prob) !== null && _a !== void 0 ? _a : 0;
        }
        /** Predicted top-1 label. */
        predictLabel(vec, meta) {
            var _a, _b;
            const dist = this.predict(vec, meta, 1);
            return (_b = (_a = dist[0]) === null || _a === void 0 ? void 0 : _a.label) !== null && _b !== void 0 ? _b : this.categories[0];
        }
        /** Batch prediction (distributions). */
        predictBatch(vectors, metas, topK = 2) {
            if (vectors.length !== metas.length) {
                throw new Error('predictBatch: vectors and metas must have same length');
            }
            return vectors.map((v, i) => this.predict(v, metas[i], topK));
        }
        /* ============ Simple evaluation helpers ============ */
        /** Compute accuracy and confusion counts for a labeled set. */
        evaluate(vectors, metas, labels) {
            if (vectors.length !== metas.length || vectors.length !== labels.length) {
                throw new Error('evaluate: inputs must have same length');
            }
            const confusion = {};
            for (const a of this.categories) {
                confusion[a] = {};
                for (const b of this.categories)
                    confusion[a][b] = 0;
            }
            let correct = 0;
            for (let i = 0; i < vectors.length; i++) {
                const pred = this.predictLabel(vectors[i], metas[i]);
                const gold = labels[i];
                if (pred === gold)
                    correct++;
                if (!confusion[gold])
                    confusion[gold] = {};
                if (confusion[gold][pred] === undefined)
                    confusion[gold][pred] = 0;
                confusion[gold][pred]++;
            }
            return { accuracy: correct / labels.length, confusion };
        }
        /* ============ I/O passthroughs ============ */
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
        /** Access underlying ELM if needed */
        getELM() {
            return this.elm;
        }
        /** Current category ordering used by the model */
        getCategories() {
            return this.categories.slice();
        }
    }

    // EncoderELM.ts — string→vector encoder using ELM (batch) + OnlineELM (incremental)
    class EncoderELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h;
            if (typeof config.hiddenUnits !== 'number') {
                throw new Error('EncoderELM requires config.hiddenUnits (number).');
            }
            if (!config.activation) {
                throw new Error('EncoderELM requires config.activation.');
            }
            // Force text-encoder mode by default (safe even if NumericConfig is passed:
            // ELM will ignore tokenizer fields in numeric flows)
            this.config = Object.assign(Object.assign({}, config), { categories: (_a = config.categories) !== null && _a !== void 0 ? _a : [], useTokenizer: (_b = config.useTokenizer) !== null && _b !== void 0 ? _b : true, 
                // keep charSet/maxLen if caller provided; otherwise ELM defaults will kick in
                log: {
                    modelName: 'EncoderELM',
                    verbose: (_d = (_c = config.log) === null || _c === void 0 ? void 0 : _c.verbose) !== null && _d !== void 0 ? _d : false,
                    toFile: (_f = (_e = config.log) === null || _e === void 0 ? void 0 : _e.toFile) !== null && _f !== void 0 ? _f : false,
                    level: (_h = (_g = config.log) === null || _g === void 0 ? void 0 : _g.level) !== null && _h !== void 0 ? _h : 'info',
                } });
            this.elm = new ELM(this.config);
            // Forward thresholds/file export if present
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        /** Batch training for string → dense vector mapping. */
        train(inputStrings, targetVectors) {
            if (!(inputStrings === null || inputStrings === void 0 ? void 0 : inputStrings.length) || !(targetVectors === null || targetVectors === void 0 ? void 0 : targetVectors.length)) {
                throw new Error('train: empty inputs');
            }
            if (inputStrings.length !== targetVectors.length) {
                throw new Error('train: inputStrings and targetVectors lengths differ');
            }
            const enc = this.elm.encoder;
            if (!enc || typeof enc.encode !== 'function') {
                throw new Error('EncoderELM: underlying ELM has no encoder; set useTokenizer/maxLen/charSet in config.');
            }
            // X = normalized encoded text; Y = dense targets
            const X = inputStrings.map(s => enc.normalize(enc.encode(s)));
            const Y = targetVectors;
            // Closed-form solve via ELM
            // (ELM learns W,b randomly and solves β; Y can be any numeric outputDim)
            this.elm.trainFromData(X, Y);
        }
        /** Encode a string into a dense feature vector using the trained model. */
        encode(text) {
            var _a;
            const enc = this.elm.encoder;
            if (!enc || typeof enc.encode !== 'function') {
                throw new Error('encode: underlying ELM has no encoder');
            }
            const model = this.elm.model;
            if (!model)
                throw new Error('EncoderELM model has not been trained yet.');
            const x = enc.normalize(enc.encode(text)); // 1 x D
            const { W, b, beta } = model;
            // H = act( x W^T + b )
            const tempH = Matrix.multiply([x], Matrix.transpose(W));
            const act = Activations.get((_a = this.config.activation) !== null && _a !== void 0 ? _a : 'relu');
            const H = Activations.apply(tempH.map(row => row.map((v, j) => v + b[j][0])), act);
            // y = H β
            return Matrix.multiply(H, beta)[0];
        }
        /* ===================== Online / Incremental API ===================== */
        /**
         * Begin an online OS-ELM run for string→vector encoding.
         * Provide outputDim and either inputDim OR a sampleText we can encode to infer inputDim.
         */
        beginOnline(opts) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j;
            const outputDim = opts.outputDim | 0;
            if (!(outputDim > 0))
                throw new Error('beginOnline: outputDim must be > 0');
            // Derive inputDim if not provided
            let inputDim = opts.inputDim;
            if (inputDim == null) {
                const enc = this.elm.encoder;
                if (!opts.sampleText || !enc) {
                    throw new Error('beginOnline: provide inputDim or sampleText (and ensure encoder is available).');
                }
                inputDim = enc.normalize(enc.encode(opts.sampleText)).length;
            }
            const hiddenUnits = ((_a = opts.hiddenUnits) !== null && _a !== void 0 ? _a : this.config.hiddenUnits) | 0;
            if (!(hiddenUnits > 0))
                throw new Error('beginOnline: hiddenUnits must be > 0');
            const activation = ((_c = (_b = opts.activation) !== null && _b !== void 0 ? _b : this.config.activation) !== null && _c !== void 0 ? _c : 'relu');
            // Build OnlineELM with our new config-style constructor
            this.online = new OnlineELM({
                inputDim: inputDim,
                outputDim,
                hiddenUnits,
                activation,
                ridgeLambda: (_d = opts.ridgeLambda) !== null && _d !== void 0 ? _d : 1e-2,
                weightInit: (_e = opts.weightInit) !== null && _e !== void 0 ? _e : 'xavier',
                forgettingFactor: (_f = opts.forgettingFactor) !== null && _f !== void 0 ? _f : 1.0,
                seed: (_g = opts.seed) !== null && _g !== void 0 ? _g : 1337,
                log: { verbose: (_j = (_h = this.config.log) === null || _h === void 0 ? void 0 : _h.verbose) !== null && _j !== void 0 ? _j : false, modelName: 'EncoderELM-Online' },
            });
            this.onlineInputDim = inputDim;
            this.onlineOutputDim = outputDim;
        }
        /**
         * Online partial fit with *pre-encoded* numeric vectors.
         * If not initialized, this call seeds the model via `init`, else it performs an `update`.
         */
        partialTrainOnlineVectors(batch) {
            if (!this.online || this.onlineInputDim == null || this.onlineOutputDim == null) {
                throw new Error('partialTrainOnlineVectors: call beginOnline() first.');
            }
            if (!(batch === null || batch === void 0 ? void 0 : batch.length))
                return;
            const D = this.onlineInputDim, O = this.onlineOutputDim;
            const X = new Array(batch.length);
            const Y = new Array(batch.length);
            for (let i = 0; i < batch.length; i++) {
                const { x, y } = batch[i];
                if (x.length !== D)
                    throw new Error(`x length ${x.length} != inputDim ${D}`);
                if (y.length !== O)
                    throw new Error(`y length ${y.length} != outputDim ${O}`);
                X[i] = x;
                Y[i] = y;
            }
            if (!this.online.beta || !this.online.P) {
                this.online.init(X, Y);
            }
            else {
                this.online.update(X, Y);
            }
        }
        /**
         * Online partial fit with raw texts and dense numeric targets.
         * Texts are encoded + normalized internally.
         */
        partialTrainOnlineTexts(batch) {
            if (!this.online || this.onlineInputDim == null || this.onlineOutputDim == null) {
                throw new Error('partialTrainOnlineTexts: call beginOnline() first.');
            }
            if (!(batch === null || batch === void 0 ? void 0 : batch.length))
                return;
            const enc = this.elm.encoder;
            if (!enc)
                throw new Error('partialTrainOnlineTexts: encoder not available on underlying ELM');
            const D = this.onlineInputDim, O = this.onlineOutputDim;
            const X = new Array(batch.length);
            const Y = new Array(batch.length);
            for (let i = 0; i < batch.length; i++) {
                const { text, target } = batch[i];
                const x = enc.normalize(enc.encode(text));
                if (x.length !== D)
                    throw new Error(`encoded text dim ${x.length} != inputDim ${D}`);
                if (target.length !== O)
                    throw new Error(`target length ${target.length} != outputDim ${O}`);
                X[i] = x;
                Y[i] = target;
            }
            if (!this.online.beta || !this.online.P) {
                this.online.init(X, Y);
            }
            else {
                this.online.update(X, Y);
            }
        }
        /**
         * Finalize the online run by publishing learned weights into the standard ELM model.
         * After this, the normal encode() path works unchanged.
         */
        endOnline() {
            if (!this.online)
                return;
            const W = this.online.W;
            const b = this.online.b;
            const beta = this.online.beta;
            if (!W || !b || !beta) {
                throw new Error('endOnline: online model has no learned parameters (did you call init/fit/update?)');
            }
            this.elm.model = { W, b, beta };
            // Clear online state
            this.online = undefined;
            this.onlineInputDim = undefined;
            this.onlineOutputDim = undefined;
        }
        /* ===================== I/O passthrough ===================== */
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // intentClassifier.ts — ELM-based intent classification (text → label)
    class IntentClassifier {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g;
            this.categories = [];
            // Basic guardrails (common footguns)
            const hidden = config.hiddenUnits;
            const act = config.activation;
            if (typeof hidden !== 'number') {
                throw new Error('IntentClassifier requires config.hiddenUnits (number)');
            }
            if (!act) {
                throw new Error('IntentClassifier requires config.activation');
            }
            // Force TEXT mode (tokenizer on). We set categories during train().
            this.config = Object.assign(Object.assign({}, config), { categories: (_a = config.categories) !== null && _a !== void 0 ? _a : [], useTokenizer: true, log: {
                    modelName: 'IntentClassifier',
                    verbose: (_c = (_b = config.log) === null || _b === void 0 ? void 0 : _b.verbose) !== null && _c !== void 0 ? _c : false,
                    toFile: (_e = (_d = config.log) === null || _d === void 0 ? void 0 : _d.toFile) !== null && _e !== void 0 ? _e : false,
                    // @ts-ignore: optional passthrough
                    level: (_g = (_f = config.log) === null || _f === void 0 ? void 0 : _f.level) !== null && _g !== void 0 ? _g : 'info',
                } });
            this.model = new ELM(this.config);
            // Optional thresholds/export passthrough
            if (config.metrics)
                this.model.metrics = config.metrics;
            if (config.exportFileName)
                this.model.config.exportFileName = config.exportFileName;
        }
        /* ==================== Training ==================== */
        /**
         * Train from (text, label) pairs using closed-form ELM solve.
         * Uses the ELM's UniversalEncoder (token mode).
         */
        train(textLabelPairs, augmentation) {
            var _a, _b, _c, _d, _e;
            if (!(textLabelPairs === null || textLabelPairs === void 0 ? void 0 : textLabelPairs.length))
                throw new Error('train: empty training data');
            // Build label set
            this.categories = Array.from(new Set(textLabelPairs.map(p => p.label)));
            this.model.setCategories(this.categories);
            // Prepare encoder
            const enc = (_c = (_b = (_a = this.model).getEncoder) === null || _b === void 0 ? void 0 : _b.call(_a)) !== null && _c !== void 0 ? _c : this.model.encoder;
            if (!enc)
                throw new Error('IntentClassifier: encoder unavailable on ELM instance.');
            // Inline augmentation (prefix/suffix/noise) — lightweight so we avoid importing Augment here
            const charSet = (augmentation === null || augmentation === void 0 ? void 0 : augmentation.charSet) ||
                enc.charSet ||
                'abcdefghijklmnopqrstuvwxyz';
            const makeNoisy = (s, rate) => {
                var _a, _b;
                if (rate === void 0) { rate = (_a = augmentation === null || augmentation === void 0 ? void 0 : augmentation.noiseRate) !== null && _a !== void 0 ? _a : 0.05; }
                if (!(augmentation === null || augmentation === void 0 ? void 0 : augmentation.includeNoise) || rate <= 0)
                    return [s];
                const arr = s.split('');
                for (let i = 0; i < arr.length; i++) {
                    if (Math.random() < rate) {
                        const r = Math.floor(Math.random() * charSet.length);
                        arr[i] = (_b = charSet[r]) !== null && _b !== void 0 ? _b : arr[i];
                    }
                }
                return [s, arr.join('')];
            };
            const expanded = [];
            for (const p of textLabelPairs) {
                const base = [p.text];
                const withPrefixes = ((_d = augmentation === null || augmentation === void 0 ? void 0 : augmentation.prefixes) !== null && _d !== void 0 ? _d : []).map(px => `${px}${p.text}`);
                const withSuffixes = ((_e = augmentation === null || augmentation === void 0 ? void 0 : augmentation.suffixes) !== null && _e !== void 0 ? _e : []).map(sx => `${p.text}${sx}`);
                const candidates = [...base, ...withPrefixes, ...withSuffixes];
                for (const c of candidates) {
                    for (const v of makeNoisy(c)) {
                        expanded.push({ text: v, label: p.label });
                    }
                }
            }
            // Encode + one-hot
            const X = new Array(expanded.length);
            const Y = new Array(expanded.length);
            for (let i = 0; i < expanded.length; i++) {
                const { text, label } = expanded[i];
                const vec = enc.normalize(enc.encode(text));
                X[i] = vec;
                const row = new Array(this.categories.length).fill(0);
                const li = this.categories.indexOf(label);
                if (li >= 0)
                    row[li] = 1;
                Y[i] = row;
            }
            // Closed-form ELM training
            this.model.trainFromData(X, Y);
        }
        /* ==================== Inference ==================== */
        /** Top-K predictions with an optional probability threshold */
        predict(text, topK = 1, threshold = 0) {
            const res = this.model.predict(text, Math.max(1, topK));
            return threshold > 0 ? res.filter(r => r.prob >= threshold) : res;
        }
        /** Batched predict */
        predictBatch(texts, topK = 1, threshold = 0) {
            return texts.map(t => this.predict(t, topK, threshold));
        }
        /** Convenience: best label + prob (or undefined if below threshold) */
        predictLabel(text, threshold = 0) {
            const [top] = this.predict(text, 1, threshold);
            return top;
        }
        /* ==================== Model I/O ==================== */
        loadModelFromJSON(json) {
            this.model.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.model.saveModelAsJSONFile(filename);
        }
    }

    // IO.ts - Import/export utilities for labeled training data
    class IO {
        static importJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data))
                    throw new Error('Invalid format');
                return data.filter(item => typeof item.text === 'string' && typeof item.label === 'string');
            }
            catch (err) {
                console.error('Failed to parse training data JSON:', err);
                return [];
            }
        }
        static exportJSON(pairs) {
            return JSON.stringify(pairs, null, 2);
        }
        static importDelimited(text, delimiter = ',', hasHeader = true) {
            var _a, _b, _c, _d;
            const lines = text.trim().split('\n');
            const examples = [];
            const headers = hasHeader
                ? lines[0].split(delimiter).map(h => h.trim().toLowerCase())
                : lines[0].split(delimiter).length === 1
                    ? ['label']
                    : ['text', 'label'];
            const startIndex = hasHeader ? 1 : 0;
            for (let i = startIndex; i < lines.length; i++) {
                const parts = lines[i].split(delimiter);
                if (parts.length === 1) {
                    examples.push({ text: parts[0].trim(), label: parts[0].trim() });
                }
                else {
                    const textIdx = headers.indexOf('text');
                    const labelIdx = headers.indexOf('label');
                    const text = textIdx !== -1 ? (_a = parts[textIdx]) === null || _a === void 0 ? void 0 : _a.trim() : (_b = parts[0]) === null || _b === void 0 ? void 0 : _b.trim();
                    const label = labelIdx !== -1 ? (_c = parts[labelIdx]) === null || _c === void 0 ? void 0 : _c.trim() : (_d = parts[1]) === null || _d === void 0 ? void 0 : _d.trim();
                    if (text && label) {
                        examples.push({ text, label });
                    }
                }
            }
            return examples;
        }
        static exportDelimited(pairs, delimiter = ',', includeHeader = true) {
            const header = includeHeader ? `text${delimiter}label\n` : '';
            const rows = pairs.map(p => `${p.text.replace(new RegExp(delimiter, 'g'), '')}${delimiter}${p.label.replace(new RegExp(delimiter, 'g'), '')}`);
            return header + rows.join('\n');
        }
        static importCSV(csv, hasHeader = true) {
            return this.importDelimited(csv, ',', hasHeader);
        }
        static exportCSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, ',', includeHeader);
        }
        static importTSV(tsv, hasHeader = true) {
            return this.importDelimited(tsv, '\t', hasHeader);
        }
        static exportTSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, '\t', includeHeader);
        }
        static inferSchemaFromCSV(csv) {
            var _a;
            const lines = csv.trim().split('\n');
            if (lines.length === 0)
                return { fields: [] };
            const header = lines[0].split(',').map(h => h.trim().toLowerCase());
            const row = ((_a = lines[1]) === null || _a === void 0 ? void 0 : _a.split(',')) || [];
            const fields = header.map((name, i) => {
                var _a;
                const sample = (_a = row[i]) === null || _a === void 0 ? void 0 : _a.trim();
                let type = 'unknown';
                if (!sample)
                    type = 'unknown';
                else if (!isNaN(Number(sample)))
                    type = 'number';
                else if (sample === 'true' || sample === 'false')
                    type = 'boolean';
                else
                    type = 'string';
                return { name, type };
            });
            const suggestedMapping = {
                text: header.find(h => h.includes('text') || h.includes('utterance') || h.includes('input')) || header[0],
                label: header.find(h => h.includes('label') || h.includes('intent') || h.includes('tag')) || header[1] || header[0],
            };
            return { fields, suggestedMapping };
        }
        static inferSchemaFromJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data) || data.length === 0 || typeof data[0] !== 'object')
                    return { fields: [] };
                const keys = Object.keys(data[0]);
                const fields = keys.map(key => {
                    const val = data[0][key];
                    let type = 'unknown';
                    if (typeof val === 'string')
                        type = 'string';
                    else if (typeof val === 'number')
                        type = 'number';
                    else if (typeof val === 'boolean')
                        type = 'boolean';
                    return { name: key.toLowerCase(), type };
                });
                const suggestedMapping = {
                    text: keys.find(k => k.toLowerCase().includes('text') || k.toLowerCase().includes('utterance') || k.toLowerCase().includes('input')) || keys[0],
                    label: keys.find(k => k.toLowerCase().includes('label') || k.toLowerCase().includes('intent') || k.toLowerCase().includes('tag')) || keys[1] || keys[0],
                };
                return { fields, suggestedMapping };
            }
            catch (err) {
                console.error('Failed to infer schema from JSON:', err);
                return { fields: [] };
            }
        }
    }

    // LanguageClassifier.ts — upgraded for new ELM/OnlineELM APIs (with requireEncoder guard)
    class LanguageClassifier {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f;
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: 'LanguageClassifier',
                    verbose: (_b = (_a = config.log) === null || _a === void 0 ? void 0 : _a.verbose) !== null && _b !== void 0 ? _b : false,
                    toFile: (_d = (_c = config.log) === null || _c === void 0 ? void 0 : _c.toFile) !== null && _d !== void 0 ? _d : false,
                    level: (_f = (_e = config.log) === null || _e === void 0 ? void 0 : _e.level) !== null && _f !== void 0 ? _f : 'info',
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        /* ============== tiny helper to guarantee an encoder ============== */
        requireEncoder() {
            const enc = this.elm.encoder;
            if (!enc) {
                throw new Error('LanguageClassifier: encoder unavailable. Use text mode (useTokenizer=true with maxLen/charSet) ' +
                    'or pass a UniversalEncoder in the ELM config.');
            }
            return enc;
        }
        /* ================= I/O helpers ================= */
        loadTrainingData(raw, format = 'json') {
            switch (format) {
                case 'csv': return IO.importCSV(raw);
                case 'tsv': return IO.importTSV(raw);
                case 'json':
                default: return IO.importJSON(raw);
            }
        }
        /* ================= Supervised training ================= */
        /** Train from labeled text examples (uses internal encoder). */
        train(data) {
            if (!(data === null || data === void 0 ? void 0 : data.length))
                throw new Error('LanguageClassifier.train: empty dataset');
            const enc = this.requireEncoder();
            const categories = Array.from(new Set(data.map(d => d.label)));
            this.elm.setCategories(categories);
            const X = [];
            const Y = [];
            for (const { text, label } of data) {
                const x = enc.normalize(enc.encode(text));
                const yi = categories.indexOf(label);
                if (yi < 0)
                    continue;
                X.push(x);
                Y.push(this.elm.oneHot(categories.length, yi));
            }
            this.elm.trainFromData(X, Y);
        }
        /** Predict from raw text (uses internal encoder). */
        predict(text, topK = 3) {
            // let ELM handle encode→predict (works in text mode)
            return this.elm.predict(text, topK);
        }
        /** Train using already-encoded numeric vectors (no text encoder). */
        trainVectors(data) {
            var _a;
            if (!(data === null || data === void 0 ? void 0 : data.length))
                throw new Error('LanguageClassifier.trainVectors: empty dataset');
            const categories = Array.from(new Set(data.map(d => d.label)));
            this.elm.setCategories(categories);
            const X = data.map(d => d.vector);
            const Y = data.map(d => this.elm.oneHot(categories.length, categories.indexOf(d.label)));
            if (typeof this.elm.trainFromData === 'function') {
                this.elm.trainFromData(X, Y);
                return;
            }
            // Fallback closed-form (compat)
            const hidden = this.config.hiddenUnits;
            const W = this.elm.randomMatrix(hidden, X[0].length);
            const b = this.elm.randomMatrix(hidden, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const act = Activations.get((_a = this.config.activation) !== null && _a !== void 0 ? _a : 'relu');
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), act);
            const Hpinv = this.elm.pseudoInverse(H);
            const beta = Matrix.multiply(Hpinv, Y);
            this.elm.model = { W, b, beta };
        }
        /** Predict from an already-encoded vector (no text encoder). */
        predictFromVector(vec, topK = 1) {
            const out = this.elm.predictFromVector([vec], topK);
            return out[0];
        }
        /* ================= Online (incremental) API ================= */
        beginOnline(opts) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j;
            const cats = opts.categories.slice();
            const D = opts.inputDim | 0;
            if (!cats.length)
                throw new Error('beginOnline: categories must be non-empty');
            if (D <= 0)
                throw new Error('beginOnline: inputDim must be > 0');
            const H = ((_a = opts.hiddenUnits) !== null && _a !== void 0 ? _a : this.config.hiddenUnits) | 0;
            if (H <= 0)
                throw new Error('beginOnline: hiddenUnits must be > 0');
            const activation = (_c = (_b = opts.activation) !== null && _b !== void 0 ? _b : this.config.activation) !== null && _c !== void 0 ? _c : 'relu';
            const ridgeLambda = Math.max((_d = opts.lambda) !== null && _d !== void 0 ? _d : 1e-2, 1e-12);
            this.onlineMdl = new OnlineELM({
                inputDim: D,
                outputDim: cats.length,
                hiddenUnits: H,
                activation,
                ridgeLambda,
                seed: (_e = opts.seed) !== null && _e !== void 0 ? _e : 1337,
                weightInit: (_f = opts.weightInit) !== null && _f !== void 0 ? _f : 'xavier',
                forgettingFactor: (_g = opts.forgettingFactor) !== null && _g !== void 0 ? _g : 1.0,
                log: { verbose: (_j = (_h = this.config.log) === null || _h === void 0 ? void 0 : _h.verbose) !== null && _j !== void 0 ? _j : false, modelName: 'LanguageClassifier/Online' },
            });
            this.onlineCats = cats;
            this.onlineInputDim = D;
        }
        partialTrainVectorsOnline(batch) {
            if (!this.onlineMdl || !this.onlineCats || !this.onlineInputDim) {
                throw new Error('Call beginOnline() before partialTrainVectorsOnline().');
            }
            if (!batch.length)
                return;
            const D = this.onlineInputDim;
            const O = this.onlineCats.length;
            const X = new Array(batch.length);
            const Y = new Array(batch.length);
            for (let i = 0; i < batch.length; i++) {
                const { vector, label } = batch[i];
                if (vector.length !== D)
                    throw new Error(`vector dim ${vector.length} != inputDim ${D}`);
                X[i] = vector.slice();
                const y = new Array(O).fill(0);
                const li = this.onlineCats.indexOf(label);
                if (li < 0)
                    throw new Error(`Unknown label "${label}" for this online run.`);
                y[li] = 1;
                Y[i] = y;
            }
            if (this.onlineMdl.beta && this.onlineMdl.P) {
                this.onlineMdl.update(X, Y);
            }
            else {
                this.onlineMdl.init(X, Y);
            }
        }
        endOnline() {
            if (!this.onlineMdl || !this.onlineCats)
                return;
            const W = this.onlineMdl.W;
            const b = this.onlineMdl.b;
            const B = this.onlineMdl.beta;
            if (!W || !b || !B)
                throw new Error('endOnline: online model is not initialized.');
            this.elm.setCategories(this.onlineCats);
            this.elm.model = { W, b, beta: B };
            this.onlineMdl = undefined;
            this.onlineCats = undefined;
            this.onlineInputDim = undefined;
        }
        /* ================= Persistence ================= */
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // RefinerELM.ts — numeric “refinement” classifier on top of arbitrary feature vectors
    class RefinerELM {
        constructor(opts) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
            if (!Number.isFinite(opts.inputSize) || opts.inputSize <= 0) {
                throw new Error('RefinerELM: opts.inputSize must be a positive number.');
            }
            if (!Number.isFinite(opts.hiddenUnits) || opts.hiddenUnits <= 0) {
                throw new Error('RefinerELM: opts.hiddenUnits must be a positive number.');
            }
            // Build a *numeric* ELM config (no text fields here)
            const numericConfig = {
                // numeric discriminator:
                useTokenizer: false,
                inputSize: opts.inputSize,
                // required for ELM
                categories: (_a = opts.categories) !== null && _a !== void 0 ? _a : [],
                // base config
                hiddenUnits: opts.hiddenUnits,
                activation: (_b = opts.activation) !== null && _b !== void 0 ? _b : 'relu',
                ridgeLambda: opts.ridgeLambda,
                dropout: opts.dropout,
                weightInit: opts.weightInit,
                // misc
                exportFileName: opts.exportFileName,
                log: {
                    modelName: (_d = (_c = opts.log) === null || _c === void 0 ? void 0 : _c.modelName) !== null && _d !== void 0 ? _d : 'RefinerELM',
                    verbose: (_f = (_e = opts.log) === null || _e === void 0 ? void 0 : _e.verbose) !== null && _f !== void 0 ? _f : false,
                    toFile: (_h = (_g = opts.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false,
                    level: (_k = (_j = opts.log) === null || _j === void 0 ? void 0 : _j.level) !== null && _k !== void 0 ? _k : 'info',
                },
            };
            this.elm = new ELM(numericConfig);
            // Set metric thresholds on the instance (not inside the config)
            if (opts.metrics) {
                this.elm.metrics = opts.metrics;
            }
        }
        /** Train from feature vectors + string labels. */
        train(inputs, labels, opts) {
            var _a;
            if (!(inputs === null || inputs === void 0 ? void 0 : inputs.length) || !(labels === null || labels === void 0 ? void 0 : labels.length) || inputs.length !== labels.length) {
                throw new Error('RefinerELM.train: inputs/labels must be non-empty and aligned.');
            }
            // Allow overriding categories at train time
            const categories = (_a = opts === null || opts === void 0 ? void 0 : opts.categories) !== null && _a !== void 0 ? _a : Array.from(new Set(labels));
            this.elm.setCategories(categories);
            const Y = labels.map((label) => this.elm.oneHot(categories.length, categories.indexOf(label)));
            // Public training path; no 'task' key here
            const options = {};
            if ((opts === null || opts === void 0 ? void 0 : opts.reuseWeights) !== undefined)
                options.reuseWeights = opts.reuseWeights;
            if (opts === null || opts === void 0 ? void 0 : opts.sampleWeights)
                options.weights = opts.sampleWeights;
            this.elm.trainFromData(inputs, Y, options);
        }
        /** Full probability vector aligned to `this.elm.categories`. */
        predictProbaFromVector(vec) {
            // Use the vector-safe path provided by the core ELM
            const out = this.elm.predictFromVector([vec], /*topK*/ this.elm.categories.length);
            // predictFromVector returns Array<PredictResult[]>, i.e., topK sorted.
            // We want a dense prob vector in category order, so map from topK back:
            const probs = new Array(this.elm.categories.length).fill(0);
            if (out && out[0]) {
                for (const { label, prob } of out[0]) {
                    const idx = this.elm.categories.indexOf(label);
                    if (idx >= 0)
                        probs[idx] = prob;
                }
            }
            return probs;
        }
        /** Top-K predictions ({label, prob}) for a single vector. */
        predict(vec, topK = 1) {
            const [res] = this.elm.predictFromVector([vec], topK);
            return res;
        }
        /** Batch top-K predictions for an array of vectors. */
        predictBatch(vectors, topK = 1) {
            return this.elm.predictFromVector(vectors, topK);
        }
        /** Hidden-layer embedding(s) — useful for chaining. */
        embed(vec) {
            return this.elm.getEmbedding([vec])[0];
        }
        embedBatch(vectors) {
            return this.elm.getEmbedding(vectors);
        }
        /** Persistence passthroughs */
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // VotingClassifierELM.ts — meta-classifier that learns to combine multiple ELMs' predictions
    class VotingClassifierELM {
        // Keep constructor shape compatible with your existing calls
        constructor(baseConfig) {
            this.baseConfig = baseConfig;
            this.modelWeights = [];
            this.usesConfidence = false;
            this.categories = baseConfig.categories || ['English', 'French', 'Spanish'];
        }
        setModelWeights(weights) {
            this.modelWeights = weights.slice();
        }
        calibrateWeights(predictionLists, trueLabels) {
            var _a, _b;
            const numModels = predictionLists.length;
            const numExamples = trueLabels.length;
            const accuracies = new Array(numModels).fill(0);
            for (let m = 0; m < numModels; m++) {
                let correct = 0;
                for (let i = 0; i < numExamples; i++) {
                    if (predictionLists[m][i] === trueLabels[i])
                        correct++;
                }
                accuracies[m] = correct / Math.max(1, numExamples);
            }
            const total = accuracies.reduce((s, a) => s + a, 0) || 1;
            this.modelWeights = accuracies.map(a => a / total);
            if ((_b = (_a = this.baseConfig) === null || _a === void 0 ? void 0 : _a.log) === null || _b === void 0 ? void 0 : _b.verbose) {
                console.log('🔧 Calibrated model weights:', this.modelWeights);
            }
        }
        /** Train meta-classifier on model predictions (+ optional confidences) and true labels. */
        train(predictionLists, // shape: [numModels][numExamples]
        confidenceLists, trueLabels) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q;
            if (!Array.isArray(predictionLists) || predictionLists.length === 0 || !trueLabels) {
                throw new Error('VotingClassifierELM.train: invalid inputs');
            }
            const numModels = predictionLists.length;
            const numExamples = predictionLists[0].length;
            for (const list of predictionLists) {
                if (list.length !== numExamples)
                    throw new Error('Prediction list lengths must match');
            }
            this.usesConfidence = Array.isArray(confidenceLists);
            if (this.usesConfidence) {
                if (confidenceLists.length !== numModels)
                    throw new Error('Confidence list count != numModels');
                for (const list of confidenceLists) {
                    if (list.length !== numExamples)
                        throw new Error('Confidence list length mismatch');
                }
            }
            if (!this.modelWeights.length || this.modelWeights.length !== numModels) {
                this.calibrateWeights(predictionLists, trueLabels);
            }
            // Categories (target space) => from true labels
            this.categories = Array.from(new Set(trueLabels));
            const C = this.categories.length;
            // Compute numeric input size for the meta-ELM:
            // per-model features = one-hot over C + (optional) 1 confidence
            const perModel = C + (this.usesConfidence ? 1 : 0);
            this.inputSize = numModels * perModel;
            // Build X, Y
            const X = new Array(numExamples);
            for (let i = 0; i < numExamples; i++) {
                let row = [];
                for (let m = 0; m < numModels; m++) {
                    const predLabel = predictionLists[m][i];
                    if (predLabel == null)
                        throw new Error(`Invalid label at predictionLists[${m}][${i}]`);
                    const w = (_a = this.modelWeights[m]) !== null && _a !== void 0 ? _a : 1;
                    // one-hot over final categories (C)
                    const idx = this.categories.indexOf(predLabel);
                    const oh = new Array(C).fill(0);
                    if (idx >= 0)
                        oh[idx] = 1;
                    row = row.concat(oh.map(x => x * w));
                    if (this.usesConfidence) {
                        const conf = confidenceLists[m][i];
                        const norm = Math.max(0, Math.min(1, Number(conf) || 0));
                        row.push(norm * w);
                    }
                }
                X[i] = row;
            }
            const Y = trueLabels.map(lbl => {
                const idx = this.categories.indexOf(lbl);
                const oh = new Array(C).fill(0);
                if (idx >= 0)
                    oh[idx] = 1;
                return oh;
            });
            // Construct numeric ELM config now that we know inputSize
            const cfg = {
                useTokenizer: false, // numeric mode
                inputSize: this.inputSize,
                categories: this.categories,
                hiddenUnits: (_b = this.baseConfig.hiddenUnits) !== null && _b !== void 0 ? _b : 64,
                activation: (_c = this.baseConfig.activation) !== null && _c !== void 0 ? _c : 'relu',
                ridgeLambda: this.baseConfig.ridgeLambda,
                dropout: this.baseConfig.dropout,
                weightInit: this.baseConfig.weightInit,
                exportFileName: this.baseConfig.exportFileName,
                log: {
                    modelName: (_f = (_e = (_d = this.baseConfig) === null || _d === void 0 ? void 0 : _d.log) === null || _e === void 0 ? void 0 : _e.modelName) !== null && _f !== void 0 ? _f : 'VotingClassifierELM',
                    verbose: (_j = (_h = (_g = this.baseConfig) === null || _g === void 0 ? void 0 : _g.log) === null || _h === void 0 ? void 0 : _h.verbose) !== null && _j !== void 0 ? _j : false,
                    toFile: (_m = (_l = (_k = this.baseConfig) === null || _k === void 0 ? void 0 : _k.log) === null || _l === void 0 ? void 0 : _l.toFile) !== null && _m !== void 0 ? _m : false,
                    level: (_q = (_p = (_o = this.baseConfig) === null || _o === void 0 ? void 0 : _o.log) === null || _p === void 0 ? void 0 : _p.level) !== null && _q !== void 0 ? _q : 'info',
                },
            };
            // Create (or recreate) the inner ELM with correct dims
            this.elm = new ELM(cfg);
            // Forward optional metrics gate
            if (this.baseConfig.metrics) {
                this.elm.metrics = this.baseConfig.metrics;
            }
            // Train numerically
            this.elm.trainFromData(X, Y);
        }
        /** Predict final label from a single stacked set of model labels (+ optional confidences). */
        predict(labels, confidences, topK = 1) {
            var _a;
            if (!this.elm)
                throw new Error('VotingClassifierELM: call train() before predict().');
            if (!(labels === null || labels === void 0 ? void 0 : labels.length))
                throw new Error('VotingClassifierELM.predict: empty labels');
            const C = this.categories.length;
            const numModels = labels.length;
            // Build numeric input row consistent with training
            let row = [];
            for (let m = 0; m < numModels; m++) {
                const w = (_a = this.modelWeights[m]) !== null && _a !== void 0 ? _a : 1;
                const idx = this.categories.indexOf(labels[m]);
                const oh = new Array(C).fill(0);
                if (idx >= 0)
                    oh[idx] = 1;
                row = row.concat(oh.map(x => x * w));
                if (this.usesConfidence) {
                    const norm = Math.max(0, Math.min(1, Number(confidences === null || confidences === void 0 ? void 0 : confidences[m]) || 0));
                    row.push(norm * w);
                }
            }
            const [res] = this.elm.predictFromVector([row], topK);
            return res;
        }
        loadModelFromJSON(json) {
            var _a, _b, _c, _d, _e;
            if (!this.elm)
                this.elm = new ELM({
                    // minimal placeholder; will be overwritten by fromJSON content
                    useTokenizer: false,
                    inputSize: 1,
                    categories: ['_tmp'],
                    hiddenUnits: 1,
                    activation: 'relu',
                    log: { modelName: 'VotingClassifierELM' },
                });
            this.elm.loadModelFromJSON(json);
            // Try to recover categories & inputSize from loaded model
            this.categories = (_a = this.elm.categories) !== null && _a !== void 0 ? _a : this.categories;
            this.inputSize = (_e = ((_d = (_c = (_b = this.elm.model) === null || _b === void 0 ? void 0 : _b.W) === null || _c === void 0 ? void 0 : _c[0]) === null || _d === void 0 ? void 0 : _d.length)) !== null && _e !== void 0 ? _e : this.inputSize;
        }
        saveModelAsJSONFile(filename) {
            if (!this.elm)
                throw new Error('VotingClassifierELM: no model to save.');
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    exports.Activations = Activations;
    exports.Augment = Augment;
    exports.AutoComplete = AutoComplete;
    exports.CharacterLangEncoderELM = CharacterLangEncoderELM;
    exports.ConfidenceClassifierELM = ConfidenceClassifierELM;
    exports.DeepELM = DeepELM;
    exports.DimError = DimError;
    exports.ELM = ELM;
    exports.ELMAdapter = ELMAdapter;
    exports.ELMChain = ELMChain;
    exports.ELMWorkerClient = ELMWorkerClient;
    exports.EmbeddingStore = EmbeddingStore;
    exports.EncoderELM = EncoderELM;
    exports.FeatureCombinerELM = FeatureCombinerELM;
    exports.IO = IO;
    exports.IntentClassifier = IntentClassifier;
    exports.KNN = KNN;
    exports.KernelELM = KernelELM;
    exports.KernelRegistry = KernelRegistry;
    exports.LanguageClassifier = LanguageClassifier;
    exports.Matrix = Matrix;
    exports.OnlineELM = OnlineELM;
    exports.RefinerELM = RefinerELM;
    exports.TFIDF = TFIDF;
    exports.TFIDFVectorizer = TFIDFVectorizer;
    exports.TextEncoder = TextEncoder;
    exports.Tokenizer = Tokenizer;
    exports.UniversalEncoder = UniversalEncoder;
    exports.VotingClassifierELM = VotingClassifierELM;
    exports.assertRect = assertRect;
    exports.binaryPR = binaryPR;
    exports.binaryROC = binaryROC;
    exports.bindAutocompleteUI = bindAutocompleteUI;
    exports.confusionMatrixFromIndices = confusionMatrixFromIndices;
    exports.defaultNumericConfig = defaultNumericConfig;
    exports.defaultTextConfig = defaultTextConfig;
    exports.deserializeTextBits = deserializeTextBits;
    exports.ensureRectNumber2D = ensureRectNumber2D;
    exports.evaluateClassification = evaluateClassification;
    exports.evaluateEnsembleRetrieval = evaluateEnsembleRetrieval;
    exports.evaluateRegression = evaluateRegression;
    exports.formatClassificationReport = formatClassificationReport;
    exports.isNumericConfig = isNumericConfig;
    exports.isTextConfig = isTextConfig;
    exports.logLoss = logLoss;
    exports.normalizeConfig = normalizeConfig;
    exports.topKAccuracy = topKAccuracy;
    exports.wrapELM = wrapELM;
    exports.wrapOnlineELM = wrapOnlineELM;

}));
//# sourceMappingURL=astermind.umd.js.map
