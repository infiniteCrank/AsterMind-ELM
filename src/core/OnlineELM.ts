// OnlineELM.ts
export type Activation = (x: number) => number;

export class OnlineELM {
    inputDim: number;
    hiddenUnits: number;
    outputDim: number;
    lambda: number;
    W: Float64Array;       // input->hidden weights [hiddenUnits * inputDim]
    b: Float64Array;       // hidden biases [hiddenUnits]
    beta: Float64Array;    // hidden->output weights [hiddenUnits * outputDim]
    P: Float64Array;       // inverse gram approx [(hiddenUnits*outputDim)?? see note below]
    act: Activation;

    // NOTE: We keep P in hidden-space only: (hiddenUnits x hiddenUnits).
    // Then apply it blockwise for multi-output targets during updates.
    constructor(inputDim: number, hiddenUnits: number, outputDim: number, act: Activation, lambda = 1e-3) {
        this.inputDim = inputDim;
        this.hiddenUnits = hiddenUnits;
        this.outputDim = outputDim;
        this.lambda = lambda;
        this.act = act;

        this.W = new Float64Array(hiddenUnits * inputDim);
        this.b = new Float64Array(hiddenUnits);
        // Kaiming-ish init for W; small bias
        const scale = Math.sqrt(2 / inputDim);
        for (let i = 0; i < this.W.length; i++) this.W[i] = (Math.random() * 2 - 1) * scale;
        for (let i = 0; i < hiddenUnits; i++) this.b[i] = (Math.random() * 2 - 1) * 0.01;

        // beta: [hiddenUnits x outputDim]
        this.beta = new Float64Array(hiddenUnits * outputDim);

        // P starts as (1/λ) I in hidden space
        this.P = new Float64Array(hiddenUnits * hiddenUnits);
        for (let i = 0; i < hiddenUnits; i++) this.P[i * hiddenUnits + i] = 1 / lambda;
    }

    // Compute hidden activations H for a batch X (rows = batch, cols = inputDim)
    // Returns Float64Array [batchSize x hiddenUnits]
    hiddenBatch(X: Float64Array, batchSize: number): Float64Array {
        const H = new Float64Array(batchSize * this.hiddenUnits);
        for (let r = 0; r < batchSize; r++) {
            const baseX = r * this.inputDim;
            for (let h = 0; h < this.hiddenUnits; h++) {
                let sum = this.b[h];
                const baseW = h * this.inputDim;
                for (let c = 0; c < this.inputDim; c++) {
                    sum += this.W[baseW + c] * X[baseX + c];
                }
                H[r * this.hiddenUnits + h] = this.act(sum);
            }
        }
        return H;
    }

    // OS-ELM partial fit on a chunk:
    // X: [batchSize x inputDim], T: [batchSize x outputDim]
    partialFit(X: Float64Array, T: Float64Array, batchSize: number) {
        const H = this.hiddenBatch(X, batchSize);            // [B x H]
        // Compute common terms in hidden space once: S = I + H P H^T  => [B x B]
        const HPHt = symm_BxB_from_HPHt(H, this.P, this.hiddenUnits, batchSize); // BxB
        addIdentityInPlace(HPHt, batchSize);                 // S
        const S_inv = invSymmetric(HPHt, batchSize);         // [B x B]

        // K = P H^T S^{-1}    => [H x B]
        const PHt = mul_P_Ht(this.P, H, this.hiddenUnits, batchSize);  // [H x B]
        const K = mul(false, PHt, S_inv, this.hiddenUnits, batchSize, batchSize); // [H x B]

        // Compute residual (T - H beta)  => [B x O]
        const HB = mul(false, H, this.beta, batchSize, this.hiddenUnits, this.outputDim); // [B x O]
        for (let i = 0; i < HB.length; i++) HB[i] = T[i] - HB[i];

        // beta += K * (T - H beta)   => [H x O]
        const delta = mul(false, K, HB, this.hiddenUnits, batchSize, this.outputDim);
        for (let i = 0; i < this.beta.length; i++) this.beta[i] += delta[i];

        // P -= K H P  => KHP: [H x H]
        const KH = mul(false, K, H, this.hiddenUnits, batchSize, this.hiddenUnits);
        const KHP = mul(false, KH, this.P, this.hiddenUnits, this.hiddenUnits, this.hiddenUnits);
        for (let i = 0; i < this.P.length; i++) this.P[i] -= KHP[i];

        // help GC
        // (arrays go out of scope)
    }

    // Predict for a single feature vector x (length = inputDim)
    predictOne(x: Float64Array): Float64Array {
        const h = new Float64Array(this.hiddenUnits);
        for (let j = 0; j < this.hiddenUnits; j++) {
            let s = this.b[j];
            const baseW = j * this.inputDim;
            for (let c = 0; c < this.inputDim; c++) s += this.W[baseW + c] * x[c];
            h[j] = this.act(s);
        }
        // y = h^T beta
        const y = new Float64Array(this.outputDim);
        for (let o = 0; o < this.outputDim; o++) {
            let s = 0;
            for (let j = 0; j < this.hiddenUnits; j++) s += h[j] * this.beta[j * this.outputDim + o];
            y[o] = s;
        }
        return y;
    }

    toJSON() {
        return JSON.stringify({
            inputDim: this.inputDim,
            hiddenUnits: this.hiddenUnits,
            outputDim: this.outputDim,
            lambda: this.lambda,
            W: Array.from(this.W),
            b: Array.from(this.b),
            beta: Array.from(this.beta),
            P: Array.from(this.P)
        });
    }

    static fromJSON(json: string, act: Activation): OnlineELM {
        const o = JSON.parse(json);
        const mdl = new OnlineELM(o.inputDim, o.hiddenUnits, o.outputDim, act, o.lambda);
        mdl.W.set(o.W);
        mdl.b.set(o.b);
        mdl.beta.set(o.beta);
        mdl.P.set(o.P);
        return mdl;
    }
}

/* ---------- tiny linear algebra helpers (hidden-space) ---------- */
// Build S = I + H P H^T  (B x B), where H: [B x H], P: [H x H]
function symm_BxB_from_HPHt(H: Float64Array, P: Float64Array, Hdim: number, B: number): Float64Array {
    // tmp = H P  => [B x H]
    const tmp = mul(false, H, P, B, Hdim, Hdim);
    // S = tmp H^T => [B x B]
    const S = mul(false, tmp, H, B, Hdim, B, true); // treat second as transposed
    return S;
}

// Generic multiply: A [m x k] * B [k x n] => [m x n]
// If B_is_transposed is true, interpret B as [n x k] transposed
function mul(A_t: boolean, A: Float64Array, B: Float64Array, m: number, k: number, n: number, B_is_transposed = false): Float64Array {
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let t = 0; t < k; t++) {
                const a = A[i * k + t];
                const b = B_is_transposed ? B[j * k + t] : B[t * n + j];
                s += a * b;
            }
            out[i * n + j] = s;
        }
    }
    return out;
}

function addIdentityInPlace(M: Float64Array, n: number) {
    for (let i = 0; i < n; i++) M[i * n + i] += 1;
}

// Small symmetric positive-definite inverse (Cholesky)
// n = batch size (keep small, e.g. 64-1024)
function invSymmetric(S: Float64Array, n: number): Float64Array {
    // Cholesky decomposition S = L L^T
    const L = S.slice();
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = L[i * n + j];
            for (let k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
            if (i === j) {
                L[i * n + j] = Math.sqrt(Math.max(sum, 1e-12));
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
        for (let j = i + 1; j < n; j++) L[i * n + j] = 0;
    }
    // Solve L Y = I, then L^T X = Y => X = S^{-1}
    const inv = new Float64Array(n * n);
    // initialize inv to identity for RHS
    for (let i = 0; i < n; i++) inv[i * n + i] = 1;
    // forward solve Y: overwrite inv with Y
    for (let col = 0; col < n; col++) {
        for (let i = 0; i < n; i++) {
            let sum = inv[i * n + col];
            for (let k = 0; k < i; k++) sum -= L[i * n + k] * inv[k * n + col];
            inv[i * n + col] = sum / L[i * n + i];
        }
        // back solve X
        for (let i = n - 1; i >= 0; i--) {
            let sum = inv[i * n + col];
            for (let k = i + 1; k < n; k++) sum -= L[k * n + i] * inv[k * n + col];
            inv[i * n + col] = sum / L[i * n + i];
        }
    }
    return inv;
}

// PH^T = P * H^T  where P: [H x H], H: [B x H]  => [H x B]
function mul_P_Ht(P: Float64Array, H: Float64Array, Hdim: number, B: number): Float64Array {
    const out = new Float64Array(Hdim * B);
    for (let i = 0; i < Hdim; i++) {
        for (let j = 0; j < B; j++) {
            let s = 0;
            for (let k = 0; k < Hdim; k++) s += P[i * Hdim + k] * H[j * Hdim + k];
            out[i * B + j] = s;
        }
    }
    return out;
}
