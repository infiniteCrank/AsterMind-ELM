// Matrix.ts — tolerant, safe helpers with dimension checks and stable ops

export class DimError extends Error {
    constructor(msg: string) {
        super(msg);
        this.name = 'DimError';
    }
}

const EPS = 1e-12;

/* ===================== Array-like coercion helpers ===================== */

// ✅ Narrow to ArrayLike<number> so numeric indexing is allowed
function isArrayLikeRow(row: any): row is ArrayLike<number> {
    return row != null && typeof row.length === 'number';
}

/**
 * Coerce any 2D array-like into a strict rectangular number[][]
 * - If width is not provided, infer from the first row's length
 * - Pads/truncates to width
 * - Non-finite values become 0
 */
export function ensureRectNumber2D(
    M: any,
    width?: number,
    name = 'matrix'
): number[][] {
    if (!M || typeof M.length !== 'number') {
        throw new DimError(`${name} must be a non-empty 2D array`);
    }
    const rows = Array.from(M as any[]);
    if (rows.length === 0) throw new DimError(`${name} is empty`);

    const first = rows[0];
    if (!isArrayLikeRow(first)) throw new DimError(`${name} row 0 missing/invalid`);
    const C = ((width ?? first.length) | 0);
    if (C <= 0) throw new DimError(`${name} has zero width`);

    const out: number[][] = new Array(rows.length);
    for (let r = 0; r < rows.length; r++) {
        const src = rows[r];
        const rr = new Array(C);
        if (isArrayLikeRow(src)) {
            const sr: ArrayLike<number> = src; // ✅ typed
            for (let c = 0; c < C; c++) {
                const v = sr[c];
                rr[c] = Number.isFinite(v) ? Number(v) : 0;
            }
        } else {
            for (let c = 0; c < C; c++) rr[c] = 0;
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
export function assertRect(A: any, name = 'matrix') {
    if (!A || typeof A.length !== 'number') {
        throw new DimError(`${name} must be a non-empty 2D array`);
    }
    const rows = A.length | 0;
    if (rows <= 0) throw new DimError(`${name} must be a non-empty 2D array`);

    const first = A[0];
    if (!isArrayLikeRow(first)) throw new DimError(`${name} row 0 missing/invalid`);
    const C = first.length | 0;
    if (C <= 0) throw new DimError(`${name} must have positive column count`);

    for (let r = 0; r < rows; r++) {
        const rowAny = A[r];
        if (!isArrayLikeRow(rowAny)) {
            throw new DimError(`${name} row ${r} invalid`);
        }
        const row: ArrayLike<number> = rowAny; // ✅ typed
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

function assertMulDims(A: any, B: any) {
    assertRect(A, 'A'); assertRect(B, 'B');
    const nA = A[0].length;
    const mB = B.length;
    if (nA !== mB) {
        throw new DimError(`matmul dims mismatch: A(${A.length}x${nA}) * B(${mB}x${B[0].length})`);
    }
}

function isSquare(A: any) {
    return isArrayLikeRow(A?.[0]) && (A.length === (A[0].length | 0));
}

function isSymmetric(A: any, tol = 1e-10) {
    if (!isSquare(A)) return false;
    const n = A.length;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(A[i][j] - A[j][i]) > tol) return false;
        }
    }
    return true;
}

/* ============================== Matrix ============================== */

export class Matrix {
    /* ========= constructors / basics ========= */

    static shape(A: any): [number, number] {
        assertRect(A, 'A');
        return [A.length, A[0].length];
    }

    static clone(A: any): number[][] {
        assertRect(A, 'A');
        return ensureRectNumber2D(A, A[0].length, 'A(clone)');
    }

    static zeros(rows: number, cols: number): number[][] {
        const out = new Array(rows);
        for (let i = 0; i < rows; i++) out[i] = new Array(cols).fill(0);
        return out;
    }

    static identity(n: number): number[][] {
        const I = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) I[i][i] = 1;
        return I;
    }

    static transpose(A: any): number[][] {
        assertRect(A, 'A');
        const m = A.length, n = A[0].length;
        const T = Matrix.zeros(n, m);
        for (let i = 0; i < m; i++) {
            const Ai: ArrayLike<number> = A[i];
            for (let j = 0; j < n; j++) T[j][i] = Number(Ai[j]);
        }
        return T;
    }

    /* ========= algebra ========= */

    static add(A: any, B: any): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        B = ensureRectNumber2D(B, undefined, 'B');
        assertRect(A, 'A'); assertRect(B, 'B');

        if (A.length !== B.length || A[0].length !== B[0].length) {
            throw new DimError(`add dims mismatch: A(${A.length}x${A[0].length}) vs B(${B.length}x${B[0].length})`);
        }
        const m = A.length, n = A[0].length;
        const C = Matrix.zeros(m, n);
        for (let i = 0; i < m; i++) {
            const Ai = A[i], Bi = B[i], Ci = C[i];
            for (let j = 0; j < n; j++) Ci[j] = Ai[j] + Bi[j];
        }
        return C;
    }

    /** Adds lambda to the diagonal (ridge regularization) */
    static addRegularization(A: any, lambda = 1e-6): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        assertRect(A, 'A');
        if (!isSquare(A)) {
            throw new DimError(`addRegularization expects square matrix, got ${A.length}x${A[0].length}`);
        }
        const C = Matrix.clone(A);
        for (let i = 0; i < C.length; i++) C[i][i] += lambda;
        return C;
    }

    static multiply(A: any, B: any): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        B = ensureRectNumber2D(B, undefined, 'B');
        assertMulDims(A, B);

        const m = A.length, n = B.length, p = B[0].length;
        const C = Matrix.zeros(m, p);

        for (let i = 0; i < m; i++) {
            const Ai: ArrayLike<number> = A[i];
            for (let k = 0; k < n; k++) {
                const aik = Number(Ai[k]);
                const Bk: ArrayLike<number> = B[k];
                for (let j = 0; j < p; j++) C[i][j] += aik * Number(Bk[j]);
            }
        }
        return C;
    }

    static multiplyVec(A: any, v: any): number[] {
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
            const Ai: ArrayLike<number> = A[i];
            let s = 0;
            for (let j = 0; j < n; j++) s += Number(Ai[j]) * Number(v[j]);
            out[i] = s;
        }
        return out;
    }

    /* ========= decompositions / solve ========= */

    static cholesky(A: any, jitter = 0): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        assertRect(A, 'A');
        if (!isSquare(A)) throw new DimError(`cholesky expects square matrix, got ${A.length}x${A[0].length}`);

        const n = A.length;
        const L = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let sum = A[i][j];
                for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
                if (i === j) {
                    const v = sum + jitter;
                    L[i][j] = Math.sqrt(Math.max(v, EPS));
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
        return L;
    }

    static solveCholesky(A: any, B: any, jitter = 1e-10): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        B = ensureRectNumber2D(B, undefined, 'B');
        assertRect(A, 'A'); assertRect(B, 'B');

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
                for (let p = 0; p < i; p++) s -= L[i][p] * Z[p][c];
                Z[i][c] = s / L[i][i];
            }
        }
        // Solve L^T X = Z (backward)
        const X = Matrix.zeros(n, k);
        for (let i = n - 1; i >= 0; i--) {
            for (let c = 0; c < k; c++) {
                let s = Z[i][c];
                for (let p = i + 1; p < n; p++) s -= L[p][i] * X[p][c];
                X[i][c] = s / L[i][i];
            }
        }
        return X;
    }

    static inverse(A: any): number[][] {
        A = ensureRectNumber2D(A, undefined, 'A');
        assertRect(A, 'A');
        if (!isSquare(A)) throw new DimError(`inverse expects square matrix, got ${A.length}x${A[0].length}`);

        const n = A.length;
        const M = Matrix.clone(A);
        const I = Matrix.identity(n);

        // Augment [M | I]
        const aug = new Array(n);
        for (let i = 0; i < n; i++) aug[i] = M[i].concat(I[i]);

        const cols = 2 * n;

        for (let p = 0; p < n; p++) {
            // Pivot
            let maxRow = p, maxVal = Math.abs(aug[p][p]);
            for (let r = p + 1; r < n; r++) {
                const v = Math.abs(aug[r][p]);
                if (v > maxVal) { maxVal = v; maxRow = r; }
            }
            if (maxVal < EPS) throw new Error('Matrix is singular or ill-conditioned');

            if (maxRow !== p) {
                const tmp = aug[p]; aug[p] = aug[maxRow]; aug[maxRow] = tmp;
            }

            // Normalize pivot row
            const piv = aug[p][p];
            const invPiv = 1 / piv;
            for (let c = 0; c < cols; c++) aug[p][c] *= invPiv;

            // Eliminate other rows
            for (let r = 0; r < n; r++) {
                if (r === p) continue;
                const f = aug[r][p];
                if (Math.abs(f) < EPS) continue;
                for (let c = 0; c < cols; c++) aug[r][c] -= f * aug[p][c];
            }
        }

        // Extract right half as inverse
        const inv = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) inv[i][j] = aug[i][n + j];
        }
        return inv;
    }

    /* ========= helpers ========= */

    static inverseSPDOrFallback(A: any): number[][] {
        if (isSymmetric(A)) {
            try {
                return Matrix.solveCholesky(A, Matrix.identity(A.length), 1e-10);
            } catch {
                // fall through
            }
        }
        return Matrix.inverse(A);
    }

    /* ========= Symmetric Eigen (Jacobi) & Inverse Square Root ========= */

    private static assertSquare(A: any, ctx = 'Matrix'): void {
        assertRect(A, ctx);
        if (!isSquare(A)) {
            throw new DimError(`${ctx}: expected square matrix, got ${A.length}x${A[0].length}`);
        }
    }

    static eigSym(
        A: any,
        maxIter = 64,
        tol = 1e-12
    ): { values: number[]; vectors: number[][] } {
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
            if (offdiagNorm() <= tol) break;

            let p = 0, q = 1, max = 0;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const v = abs(B[i][j]);
                    if (v > max) { max = v; p = i; q = j; }
                }
            }
            if (max <= tol) break;

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
                if (k === p || k === q) continue;
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
        for (let i = 0; i < n; i++) vals[i] = B[i][i];

        const order = vals.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
        const values = order.map(i => vals[i]);
        const vectors = Matrix.zeros(n, n);
        for (let r = 0; r < n; r++) {
            for (let c = 0; c < n; c++) vectors[r][c] = V[r][order[c]];
        }

        return { values, vectors };
    }

    static invSqrtSym(A: any, eps = 1e-10): number[][] {
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
