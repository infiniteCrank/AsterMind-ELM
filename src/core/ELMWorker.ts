// src/workers/elm-worker.ts
/// <reference lib="webworker" />

// Mark this file as a module (prevents TS from treating it as a script).
export { };

// Give 'self' the right type in this context.
declare const self: DedicatedWorkerGlobalScope;

import type { ELMConfig, TrainOptions } from './ELMConfig';

// ------------------------
// IMPORTS (choose ONE style)
// ------------------------

// A) Recommended (tsconfig: moduleResolution = "Bundler")
import { ELM } from './ELM';
import { OnlineELM, type OnlineELMJSON } from './OnlineELM';

// B) If you use "NodeNext" resolution, comment A out and uncomment B:
// import { ELM } from '../ELM.js';
// import { OnlineELM, type OnlineELMJSON } from '../OnlineELM.js';

// ------------------------

type ModelKind = 'none' | 'elm' | 'online';

type Req<Action extends string = string, P = any> = {
    id: string;
    action: Action;
    payload?: P;
};

type Res =
    | { id: string; ok: true; result?: any }
    | { id: string; ok: false; error: string };

type Progress = {
    id: string;
    type: 'progress';
    phase: 'encode' | 'formH' | 'solve' | 'done' | string;
    pct: number; // 0..1
    note?: string;
};

let kind: ModelKind = 'none';
let elm: ELM | null = null;
let oelm: OnlineELM | null = null;

const ok = (id: string, result?: any): Res => ({ id, ok: true, result });
const err = (id: string, error: unknown): Res => ({
    id,
    ok: false,
    error: String((error as any)?.message ?? error),
});

const post = (m: any) => self.postMessage(m);
const postProgress = (p: Progress) => post(p);

const as2D = (X: any): number[][] => X as number[][];

self.onmessage = async (ev: MessageEvent<Req>) => {
    const { id, action, payload } = ev.data || {};
    try {
        switch (action) {
            // ---------- lifecycle ----------
            case 'initELM': {
                elm = new ELM(payload as ELMConfig);
                oelm = null;
                kind = 'elm';
                post(ok(id, { kind }));
                break;
            }
            case 'initOnlineELM': {
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
                const v = !!payload?.verbose;
                if (kind === 'elm' && elm) (elm as any).verbose = v;
                if (kind === 'online' && oelm) (oelm as any).verbose = v;
                post(ok(id));
                break;
            }

            // ---------- ELM ----------
            case 'elm.train': {
                if (!elm) throw new Error('ELM not initialized');
                const { augmentationOptions, weights } = payload ?? {};
                const onProgress: TrainOptions['onProgress'] = (phase, pct) =>
                    postProgress({ id, type: 'progress', phase, pct });
                // train() in your ELM is synchronous; we just proxy progress events.
                elm.train(augmentationOptions, weights);
                postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                post(ok(id));
                break;
            }
            case 'elm.trainFromData': {
                if (!elm) throw new Error('ELM not initialized');
                const { X, Y, options } = payload ?? {};
                const onProgress: TrainOptions['onProgress'] = (phase, pct) =>
                    postProgress({ id, type: 'progress', phase, pct });
                elm.trainFromData(as2D(X), as2D(Y), options);
                postProgress({ id, type: 'progress', phase: 'done', pct: 1 });
                post(ok(id));
                break;
            }
            case 'elm.predict': {
                if (!elm) throw new Error('ELM not initialized');
                const { text, topK } = payload ?? {};
                const r = elm.predict(String(text), topK ?? 5);
                post(ok(id, r));
                break;
            }
            case 'elm.predictFromVector': {
                if (!elm) throw new Error('ELM not initialized');
                const { X, topK } = payload ?? {};
                const r = elm.predictFromVector(as2D(X), topK ?? 5);
                post(ok(id, r));
                break;
            }
            case 'elm.getEmbedding': {
                if (!elm) throw new Error('ELM not initialized');
                const { X } = payload ?? {};
                const r = elm.getEmbedding(as2D(X));
                post(ok(id, r));
                break;
            }
            case 'elm.toJSON': {
                if (!elm) throw new Error('ELM not initialized');
                const json = (elm as any).savedModelJSON ?? JSON.stringify((elm as any).model);
                post(ok(id, json));
                break;
            }
            case 'elm.loadJSON': {
                if (!elm) throw new Error('ELM not initialized');
                elm.loadModelFromJSON(String(payload?.json ?? ''));
                post(ok(id));
                break;
            }

            // ---------- OnlineELM ----------
            case 'oelm.init': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                const { X0, Y0 } = payload ?? {};
                oelm.init(as2D(X0), as2D(Y0));
                post(ok(id));
                break;
            }
            case 'oelm.fit': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                const { X, Y } = payload ?? {};
                oelm.fit(as2D(X), as2D(Y));
                post(ok(id));
                break;
            }
            case 'oelm.update': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                const { X, Y } = payload ?? {};
                oelm.update(as2D(X), as2D(Y));
                post(ok(id));
                break;
            }
            case 'oelm.logits': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                const { X } = payload ?? {};
                const r = oelm.predictLogitsFromVectors(as2D(X));
                post(ok(id, r));
                break;
            }
            case 'oelm.toJSON': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                const json: OnlineELMJSON = oelm.toJSON(true);
                post(ok(id, json));
                break;
            }
            case 'oelm.loadJSON': {
                if (!oelm) throw new Error('OnlineELM not initialized');
                oelm.loadFromJSON(payload?.json);
                post(ok(id));
                break;
            }

            default:
                post(err(id, `Unknown action: ${action}`));
        }
    } catch (e) {
        post(err(id, e));
    }
};
