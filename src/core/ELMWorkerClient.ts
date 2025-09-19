// src/workers/ELMWorkerClient.ts

export type WorkerProgress = {
    id: string;
    type: 'progress';
    phase: string;
    pct: number; // 0..1
    note?: string;
};

type RpcOk = { id: string; ok: true; result?: any };
type RpcErr = { id: string; ok: false; error: string };
type RpcMsg = WorkerProgress | RpcOk | RpcErr;

type Pending = {
    resolve: (v: any) => void;
    reject: (e: any) => void;
    onProgress?: (p: WorkerProgress) => void;
};

export class ELMWorkerClient {
    private worker: Worker;
    private pending = new Map<string, Pending>();

    constructor(worker: Worker) {
        this.worker = worker;
        this.worker.onmessage = (ev: MessageEvent<RpcMsg>) => {
            const msg = ev.data;

            // Progress event
            if ((msg as any)?.type === 'progress' && (msg as any)?.id) {
                const pend = this.pending.get((msg as any).id);
                pend?.onProgress?.(msg as WorkerProgress);
                return;
            }

            // RPC response
            const id = (msg as any)?.id;
            if (!id) return;
            const pend = this.pending.get(id);
            if (!pend) return;
            this.pending.delete(id);

            if ((msg as any).ok) pend.resolve((msg as RpcOk).result);
            else pend.reject(new Error((msg as RpcErr).error));
        };
    }

    private call<T = any>(
        action: string,
        payload?: any,
        onProgress?: (p: WorkerProgress) => void
    ): Promise<T> {
        const id = Math.random().toString(36).slice(2);
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve, reject, onProgress });
            this.worker.postMessage({ id, action, payload });
        });
    }

    // -------- lifecycle --------
    getKind() { return this.call<{ kind: 'none' | 'elm' | 'online' }>('getKind'); }
    dispose() { return this.call('dispose'); }
    setVerbose(verbose: boolean) { return this.call('setVerbose', { verbose }); }

    // -------- ELM --------
    initELM(config: any) { return this.call('initELM', config); }
    elmTrain(opts?: { augmentationOptions?: any; weights?: number[] }, onProgress?: (p: WorkerProgress) => void) {
        return this.call('elm.train', opts, onProgress);
    }
    elmTrainFromData(X: number[][], Y: number[][], options?: any, onProgress?: (p: WorkerProgress) => void) {
        return this.call('elm.trainFromData', { X, Y, options }, onProgress);
    }
    elmPredict(text: string, topK = 5) { return this.call('elm.predict', { text, topK }); }
    elmPredictFromVector(X: number[][], topK = 5) { return this.call('elm.predictFromVector', { X, topK }); }
    elmGetEmbedding(X: number[][]) { return this.call<number[][]>('elm.getEmbedding', { X }); }
    elmToJSON() { return this.call<string>('elm.toJSON'); }
    elmLoadJSON(json: string) { return this.call('elm.loadJSON', { json }); }

    // -------- OnlineELM --------
    initOnlineELM(config: any) { return this.call('initOnlineELM', config); }
    oelmInit(X0: number[][], Y0: number[][]) { return this.call('oelm.init', { X0, Y0 }); }
    oelmFit(X: number[][], Y: number[][]) { return this.call('oelm.fit', { X, Y }); }
    oelmUpdate(X: number[][], Y: number[][]) { return this.call('oelm.update', { X, Y }); }
    oelmLogits(X: number[][]) { return this.call<number[][]>('oelm.logits', { X }); }
    oelmToJSON() { return this.call('oelm.toJSON'); }
    oelmLoadJSON(json: any) { return this.call('oelm.loadJSON', { json }); }
}
