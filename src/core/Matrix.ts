// Matrix.ts — safe, fast-ish helpers with dimension checks and stable ops

export class DimError extends Error {
    constructor(msg: string) {
        super(msg);
        this.name = 'DimError';
    }
}

const EPS = 1e-12;

function assertRect(A: number[][], name = 'matrix') {
    if (!A || A.length === 0 || !Array.isArray(A[0])) {
        throw new DimError(`${name} must be a non-empty 2D array`);
    }
    const n = A[0].length;
    for (let i = 1; i < A.length; i++) {
        if (A[i].length !== n) {
            throw new DimError(`${name} has ragged rows: row 0 = ${n} cols, row ${i} = ${A[i].length} cols`);
        }
    }
}

function assertMulDims(A: number[][], B: number[][]) {
    assertRect(A, 'A'); assertRect(B, 'B');
    const nA = A[0].length;
    const mB = B.length;
    if (nA !== mB) {
        throw new DimError(`matmul dims mismatch: A(${A.length}x${nA}) * B(${mB}x${B[0].length})`);
    }
}

function isSquare(A: number[][]) {
    return A.length === (A[0]?.length ?? -1);
}

function isSymmetric(A: number[][], tol = 1e-10) {
    if (!isSquare(A)) return false;
    const n = A.length;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(A[i][j] - A[j][i]) > tol) return false;
        }
    }
    return true;
}

export class Matrix {
    /* ========= constructors / basics ========= */

    static shape(A: number[][]): [number, number] {
        assertRect(A, 'A');
        return [A.length, A[0].length];
    }

    static clone(A: number[][]): number[][] {
        assertRect(A, 'A');
        return A.map(r => r.slice());
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

    static transpose(A: number[][]): number[][] {
        assertRect(A, 'A');
        const m = A.length, n = A[0].length;
        const T = Matrix.zeros(n, m);
        for (let i = 0; i < m; i++) {
            const Ai = A[i];
            for (let j = 0; j < n; j++) T[j][i] = Ai[j];
        }
        return T;
    }

    /* ========= algebra ========= */

    static add(A: number[][], B: number[][]): number[][] {
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
    static addRegularization(A: number[][], lambda = 1e-6): number[][] {
        assertRect(A, 'A');
        if (!isSquare(A)) {
            throw new DimError(`addRegularization expects square matrix, got ${A.length}x${A[0].length}`);
        }
        const C = Matrix.clone(A);
        for (let i = 0; i < C.length; i++) C[i][i] += lambda;
        return C;
    }

    static multiply(A: number[][], B: number[][]): number[][] {
        assertMulDims(A, B);
        const m = A.length, n = B.length, p = B[0].length;
        const C = Matrix.zeros(m, p);

        for (let i = 0; i < m; i++) {
            const Ai = A[i];
            for (let k = 0; k < n; k++) {
                const aik = Ai[k];
                const Bk = B[k];
                for (let j = 0; j < p; j++) C[i][j] += aik * Bk[j];
            }
        }
        return C;
    }

    static multiplyVec(A: number[][], v: number[]): number[] {
        assertRect(A, 'A');
        if (A[0].length !== v.length) {
            throw new DimError(`matvec dims mismatch: A cols ${A[0].length} vs v len ${v.length}`);
        }
        const m = A.length, n = v.length;
        const out = new Array(m).fill(0);
        for (let i = 0; i < m; i++) {
            let s = 0, Ai = A[i];
            for (let j = 0; j < n; j++) s += Ai[j] * v[j];
            out[i] = s;
        }
        return out;
    }

    /* ========= decompositions / solve ========= */

    /** Cholesky decomposition for SPD matrices. Returns lower-triangular L with A = L L^T */
    static cholesky(A: number[][], jitter = 0): number[][] {
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

    /** Solve A X = B using Cholesky (A must be SPD). Returns X. */
    static solveCholesky(A: number[][], B: number[][], jitter = 1e-10): number[][] {
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
                X[i][i] = X[i][i]; // keep explicit
                X[i][c] = s / L[i][i];
            }
        }
        return X;
    }

    /** Inverse via Gauss-Jordan with partial pivoting (fallback when A not SPD). */
    static inverse(A: number[][]): number[][] {
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

    /** Try to invert SPD via Cholesky; fallback to generic inverse */
    static inverseSPDOrFallback(A: number[][]): number[][] {
        if (isSymmetric(A)) {
            try {
                // Solve A X = I via Cholesky
                return Matrix.solveCholesky(A, Matrix.identity(A.length), 1e-10);
            } catch {
                // fall through
            }
        }
        return Matrix.inverse(A);
    }

    /* ========= Symmetric Eigen (Jacobi) & Inverse Square Root ========= */

    /** Internal: require square matrix */
    private static assertSquare(A: number[][], ctx = 'Matrix'): void {
        assertRect(A, ctx);
        if (!isSquare(A)) {
            throw new DimError(`${ctx}: expected square matrix, got ${A.length}x${A[0].length}`);
        }
    }

    /**
     * Symmetric eigendecomposition using Jacobi rotations.
     * Returns eigenvalues (ascending) and eigenvectors (columns of V).
     * Complexity O(n^3). Works well for SPD / symmetric matrices up to ~1000x1000.
     */
    static eigSym(
        A: number[][],
        maxIter = 64,
        tol = 1e-12
    ): { values: number[]; vectors: number[][] } {
        Matrix.assertSquare(A, 'eigSym');
        const n = A.length;

        // Copy A to B; initialize V = I
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

            // Find largest off-diagonal |B[p][q]|
            let p = 0, q = 1, max = 0;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const v = abs(B[i][j]);
                    if (v > max) { max = v; p = i; q = j; }
                }
            }
            if (max <= tol) break;

            const app = B[p][p], aqq = B[q][q], apq = B[p][q];

            // Rotation params
            const tau = (aqq - app) / (2 * apq);
            const t = Math.sign(tau) / (abs(tau) + Math.sqrt(1 + tau * tau));
            const c = 1 / Math.sqrt(1 + t * t);
            const s = t * c;

            // Update B (symmetric)
            const Bpp = c * c * app - 2 * s * c * apq + s * s * aqq;
            const Bqq = s * s * app + 2 * s * c * apq + c * c * aqq;
            B[p][p] = Bpp;
            B[q][q] = Bqq;
            B[p][q] = B[q][p] = 0;

            for (let k = 0; k < n; k++) {
                if (k === p || k === q) continue;
                const aip = B[k][p], aiq = B[k][q];
                // Keep symmetric
                const new_kp = c * aip - s * aiq;
                const new_kq = s * aip + c * aiq;
                B[k][p] = B[p][k] = new_kp;
                B[k][q] = B[q][k] = new_kq;
            }

            // Update eigenvectors V = V * J
            for (let k = 0; k < n; k++) {
                const vip = V[k][p], viq = V[k][q];
                V[k][p] = c * vip - s * viq;
                V[k][q] = s * vip + c * viq;
            }
        }

        // Diagonal of B holds eigenvalues
        const vals = new Array(n);
        for (let i = 0; i < n; i++) vals[i] = B[i][i];

        // Sort ascending and permute V columns
        const order = vals.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
        const values = order.map(i => vals[i]);
        const vectors = Matrix.zeros(n, n);
        for (let r = 0; r < n; r++) {
            for (let c = 0; c < n; c++) vectors[r][c] = V[r][order[c]];
        }

        return { values, vectors };
    }

    /**
     * Symmetric inverse square root via eigendecomposition:
     * returns U * diag(1/sqrt(max(λ, eps))) * U^T
     * For SPD K_mm, this yields a symmetric K_mm^{-1/2}.
     */
    static invSqrtSym(A: number[][], eps = 1e-10): number[][] {
        Matrix.assertSquare(A, 'invSqrtSym');
        const { values, vectors: U } = Matrix.eigSym(A);
        const n = values.length;

        // D^{-1/2}
        const Dm12 = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) {
            const lam = Math.max(values[i], eps);
            Dm12[i][i] = 1 / Math.sqrt(lam);
        }

        // U * D^{-1/2} * U^T
        const UD = Matrix.multiply(U, Dm12);
        return Matrix.multiply(UD, Matrix.transpose(U));
    }
}
