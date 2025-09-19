// ELMChain.ts — simple encoder pipeline with checks, normalization, and profiling

export interface EncoderLike {
    /** Return hidden/embedding for a batch of vectors: (N x Din) -> (N x Dout) */
    getEmbedding(X: number[][]): number[][];
    /** Optional name for debugging / summary */
    name?: string;
}

export interface ChainOptions {
    /** L2-normalize rows after each encoder (default: false) */
    normalizeEach?: boolean;
    /** L2-normalize rows at the end (default: false) */
    normalizeFinal?: boolean;
    /** Check that each stage receives non-empty batch and consistent dims (default: true) */
    validate?: boolean;
    /** Throw if any encoder throws (default: true). If false, bubbles partial outputs. */
    strict?: boolean;
    /** Optional chain name used in logs & summary */
    name?: string;
}

function l2NormalizeRows(M: number[][]): number[][] {
    return M.map(row => {
        let s = 0;
        for (let i = 0; i < row.length; i++) s += row[i] * row[i];
        const n = Math.sqrt(s) || 1;
        const inv = 1 / n;
        return row.map(v => v * inv);
    });
}

function asBatch(x: number[] | number[][]): number[][] {
    return Array.isArray(x[0]) ? (x as number[][]) : [x as number[]];
}
function fromBatch(y: number[][], originalWasVector: boolean): number[] | number[][] {
    return originalWasVector ? (y[0] ?? []) : y;
}

export class ELMChain {
    private encoders: EncoderLike[];
    private opts: Required<ChainOptions>;
    private lastDims: number[] = []; // input dim -> stage dims (for summary)

    constructor(encoders: EncoderLike[] = [], opts?: ChainOptions) {
        this.encoders = [...encoders];
        this.opts = {
            normalizeEach: opts?.normalizeEach ?? false,
            normalizeFinal: opts?.normalizeFinal ?? false,
            validate: opts?.validate ?? true,
            strict: opts?.strict ?? true,
            name: opts?.name ?? 'ELMChain',
        };
    }

    /** Add encoder at end */
    add(encoder: EncoderLike): void {
        this.encoders.push(encoder);
    }

    /** Insert encoder at position (0..length) */
    insertAt(index: number, encoder: EncoderLike): void {
        if (index < 0 || index > this.encoders.length) throw new Error('insertAt: index out of range');
        this.encoders.splice(index, 0, encoder);
    }

    /** Remove encoder at index; returns removed or undefined */
    removeAt(index: number): EncoderLike | undefined {
        if (index < 0 || index >= this.encoders.length) return undefined;
        return this.encoders.splice(index, 1)[0];
    }

    /** Remove all encoders */
    clear(): void {
        this.encoders.length = 0;
        this.lastDims.length = 0;
    }

    /** Number of stages */
    length(): number {
        return this.encoders.length;
    }

    /** Human-friendly overview (dims are filled after the first successful run) */
    summary(): string {
        const lines: string[] = [];
        lines.push(`📦 ${this.opts.name} — ${this.encoders.length} stage(s)`);
        this.encoders.forEach((enc, i) => {
            const nm = enc.name ?? `Encoder#${i}`;
            const dimIn = this.lastDims[i] ?? '?';
            const dimOut = this.lastDims[i + 1] ?? '?';
            lines.push(`  ${i}: ${nm}    ${dimIn} → ${dimOut}`);
        });
        return lines.join('\n');
    }

    /**
     * Compute embeddings.
     * Overloads allow a single vector or a batch.
     */
    getEmbedding(input: number[]): number[];
    getEmbedding(input: number[][]): number[][];
    getEmbedding(input: number[] | number[][]): number[] | number[][] {
        const wasVector = !Array.isArray(input[0]);
        const X0 = asBatch(input);

        if (this.opts.validate) {
            if (!X0.length || !X0[0]?.length) throw new Error('ELMChain.getEmbedding: empty input');
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
                        if (X[r].length !== d) throw new Error(`Stage ${i} input row ${r} has dim ${X[r].length} != ${d}`);
                    }
                }

                let Y = enc.getEmbedding(X);

                if (this.opts.validate) {
                    if (!Y.length || !Y[0]?.length) {
                        throw new Error(`Stage ${i} produced empty output`);
                    }
                }

                if (this.opts.normalizeEach) {
                    Y = l2NormalizeRows(Y);
                }

                // Record dims for summary
                this.lastDims[i + 1] = Y[0].length;

                X = Y;
            } catch (err) {
                if (this.opts.strict) throw err;
                // Non-strict: return what we have so far
                return fromBatch(X, wasVector);
            }
        }

        if (this.opts.normalizeFinal && !this.opts.normalizeEach) {
            X = l2NormalizeRows(X);
        }

        return fromBatch(X, wasVector);
    }

    /**
     * Run once to collect per-stage timings (ms) and final dims.
     * Returns { timings, dims } where dims[i] is input dim to stage i,
     * dims[i+1] is that stage’s output dim.
     */
    profile(input: number[] | number[][]): { timings: number[]; dims: number[] } {
        const wasVector = !Array.isArray(input[0]);
        let X = asBatch(input);
        const timings: number[] = [];
        const dims: number[] = [X[0].length];

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
