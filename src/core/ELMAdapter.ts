// ELMAdapter.ts — unify ELM / OnlineELM as EncoderLike for ELMChain

import type { EncoderLike } from './ELMChain';
import { ELM } from './ELM';
import { OnlineELM } from './OnlineELM';
import { Activations } from './Activations';

type OnlineMode = 'hidden' | 'logits';

export type AdapterTarget =
    | { type: 'elm'; model: ELM; name?: string }
    | { type: 'online'; model: OnlineELM; name?: string; mode?: OnlineMode };

function assertNonEmptyBatch(X: number[][], where: string) {
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0) {
        throw new Error(`${where}: expected non-empty (N x D) batch`);
    }
}

function matmulXWtAddB(
    X: number[][],        // (N x D)
    W: number[][],        // (H x D)
    b: number[][]         // (H x 1)
): number[][] {         // -> (N x H)
    const N = X.length, D = X[0].length, H = W.length;
    // quick shape sanity
    if (W[0]?.length !== D) throw new Error(`matmulXWtAddB: W is ${W.length}x${W[0]?.length}, expected Hx${D}`);
    if (b.length !== H || (b[0]?.length ?? 0) !== 1) throw new Error(`matmulXWtAddB: b is ${b.length}x${b[0]?.length}, expected Hx1`);

    const out = new Array(N);
    for (let n = 0; n < N; n++) {
        const xn = X[n];
        const row = new Array(H);
        for (let h = 0; h < H; h++) {
            const wh = W[h];
            let s = b[h][0] || 0;
            // unrolled dot
            for (let d = 0; d < D; d++) s += xn[d] * wh[d];
            row[h] = s;
        }
        out[n] = row;
    }
    return out;
}

export class ELMAdapter implements EncoderLike {
    public readonly name: string;
    private readonly target: AdapterTarget;
    private readonly mode: OnlineMode;

    constructor(target: AdapterTarget) {
        this.target = target;
        this.mode = target.type === 'online' ? (target.mode ?? 'hidden') : 'hidden';
        this.name = target.name ?? (target.type === 'elm' ? 'ELM' : `OnlineELM(${this.mode})`);
    }

    /** Return embeddings for a batch (N x D) -> (N x H/L) */
    getEmbedding(X: number[][]): number[][] {
        assertNonEmptyBatch(X, `${this.name}.getEmbedding`);

        if (this.target.type === 'elm') {
            const m = this.target.model as ELM;
            // ELM already exposes getEmbedding()
            if (typeof m.getEmbedding !== 'function') {
                throw new Error(`${this.name}: underlying ELM lacks getEmbedding(X)`);
            }
            try {
                return m.getEmbedding(X);
            } catch (err: any) {
                // Helpful hint if model wasn’t trained
                if ((m as any).model == null) {
                    throw new Error(`${this.name}: model not trained/initialized (call train/trainFromData or load model).`);
                }
                throw err;
            }
        }

        // OnlineELM path
        const o = this.target.model as OnlineELM;

        // Guard dims early
        const D = X[0].length;
        if (!Array.isArray((o as any).W) || (o as any).W[0]?.length !== D) {
            throw new Error(`${this.name}: input dim ${D} does not match model.W columns ${(o as any).W?.[0]?.length ?? 'n/a'}`);
        }

        if (this.mode === 'logits') {
            // Use public logits as an “embedding”
            try {
                return o.predictLogitsFromVectors(X);
            } catch (err: any) {
                if ((o as any).beta == null) {
                    throw new Error(`${this.name}: model not initialized (call init()/fit() before logits mode).`);
                }
                throw err;
            }
        }

        // mode === 'hidden' → compute hidden activations: act(X Wᵀ + b)
        const W = (o as any).W as number[][];
        const BIAS = (o as any).b as number[][];
        const actName = (o as any).activation as string | undefined;
        const act = Activations.get((actName ?? 'relu').toLowerCase());
        const Hpre = matmulXWtAddB(X, W, BIAS);
        // apply activation in-place
        for (let n = 0; n < Hpre.length; n++) {
            const row = Hpre[n];
            for (let j = 0; j < row.length; j++) row[j] = act(row[j]);
        }
        return Hpre;
    }
}

/* -------- convenience helpers -------- */

export function wrapELM(model: ELM, name?: string): ELMAdapter {
    return new ELMAdapter({ type: 'elm', model, name });
}

export function wrapOnlineELM(model: OnlineELM, opts?: { name?: string; mode?: OnlineMode }): ELMAdapter {
    return new ELMAdapter({ type: 'online', model, name: opts?.name, mode: opts?.mode });
}
