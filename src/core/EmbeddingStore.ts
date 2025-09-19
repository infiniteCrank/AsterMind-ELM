// EmbeddingStore.ts — Powerful in-memory vector store with fast KNN, thresholds, and JSON I/O

export type Metric = 'cosine' | 'dot' | 'euclidean' | 'manhattan';

export interface EmbeddingItem<M extends Record<string, any> = Record<string, any>> {
    id: string;
    vec: number[] | Float32Array;
    meta?: M;
}

export interface QueryOptions<M extends Record<string, any> = Record<string, any>> {
    /** Similarity/distance metric (default: 'cosine') */
    metric?: Metric;
    /** Return only hits that pass this predicate */
    filter?: (meta: M | undefined, id: string) => boolean;
    /** Include vectors in results (default: false) */
    returnVectors?: boolean;
    /** For cosine/dot: keep only hits with score >= minScore */
    minScore?: number;
    /** For L2/L1: keep only hits with distance <= maxDistance */
    maxDistance?: number;
    /** Optional subset restriction by ids */
    restrictToIds?: string[];
}

export interface SearchHit<M extends Record<string, any> = Record<string, any>> {
    id: string;
    score: number;              // cosine/dot similarity; for L2/L1 we return NEGATIVE distance (higher is better)
    index: number;              // internal index at query time
    meta?: M;
    vec?: Float32Array;         // present when returnVectors=true
}

export interface EmbeddingStoreJSON<M extends Record<string, any> = Record<string, any>> {
    dim: number;
    /** If true, stored 'vec' arrays are L2-normalized (unit length) */
    normalized: boolean;
    /** If true, payload includes 'raw' arrays with the original unnormalized vectors */
    alsoStoredRaw?: boolean;
    /** Optional capacity for ring buffer behavior */
    capacity?: number;
    items: Array<{ id: string; vec: number[]; raw?: number[]; meta?: M }>;
    __version: string;
}

const EPS = 1e-12;

/* ================= math utils ================= */

function l2Norm(v: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
}
function l1Dist(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
}
function dot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}
function normalizeToUnit(v: ArrayLike<number>): Float32Array {
    const out = new Float32Array(v.length);
    const n = l2Norm(v);
    if (n < EPS) return out; // zero vector → stay zero; cosine with zero returns 0
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
}

/** Quickselect (nth_element) on-place for top-k largest by score. Returns cutoff value index. */
function quickselectTopK<T>(
    arr: T[],
    k: number,
    scoreOf: (x: T) => number
): number {
    if (k <= 0 || k >= arr.length) return arr.length - 1;

    let left = 0, right = arr.length - 1;
    const target = k - 1; // 0-based index of kth largest after partition

    function swap(i: number, j: number) {
        const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    function partition(l: number, r: number, pivotIdx: number) {
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
        if (idx === target) return idx;
        if (target < idx) right = idx - 1;
        else left = idx + 1;
    }
}

/* ================= store ================= */

export class EmbeddingStore<M extends Record<string, any> = Record<string, any>> {
    private readonly dim: number;
    /** If true, 'vecs' are unit vectors (good for cosine); otherwise raw are stored in 'vecs' and norms[] is cached. */
    private readonly storeUnit: boolean;
    /** If true, keep original unnormalized vectors alongside normalized ones (enables Euclidean even when storeUnit=true). */
    private readonly alsoStoreRaw: boolean;
    private capacity?: number;

    // Data
    private ids: string[] = [];
    private metas: (M | undefined)[] = [];
    private vecs: Float32Array[] = [];           // if storeUnit=true -> unit vectors; else raw vectors
    private rawVecs?: Float32Array[];            // present iff alsoStoreRaw=true
    private norms?: Float32Array;                // cached L2 norms for cosine on raw storage
    private rawNorms?: Float32Array;             // cached L2 norms for raw when we also store raw

    // Index
    private idToIdx: Map<string, number> = new Map();

    constructor(
        dim: number,
        opts?: { storeUnit?: boolean; capacity?: number; alsoStoreRaw?: boolean }
    ) {
        if (!Number.isFinite(dim) || dim <= 0) throw new Error(`EmbeddingStore: invalid dim=${dim}`);
        this.dim = dim | 0;
        this.storeUnit = opts?.storeUnit ?? true;
        this.alsoStoreRaw = opts?.alsoStoreRaw ?? this.storeUnit; // default: if normalizing, also keep raw so Euclidean is valid
        if (opts?.capacity !== undefined) {
            if (!Number.isFinite(opts.capacity) || opts.capacity <= 0) throw new Error(`capacity must be > 0`);
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

    size(): number { return this.ids.length; }
    dimension(): number { return this.dim; }
    isUnitStored(): boolean { return this.storeUnit; }
    keepsRaw(): boolean { return !!this.rawVecs; }
    getCapacity(): number | undefined { return this.capacity; }

    setCapacity(capacity?: number): void {
        if (capacity === undefined) { this.capacity = undefined; return; }
        if (!Number.isFinite(capacity) || capacity <= 0) throw new Error(`capacity must be > 0`);
        this.capacity = Math.floor(capacity);
        this.enforceCapacity();
    }

    clear(): void {
        this.ids = [];
        this.vecs = [];
        this.metas = [];
        this.idToIdx.clear();
        if (this.rawVecs) this.rawVecs = [];
        if (this.norms) this.norms = new Float32Array(0);
        if (this.rawNorms) this.rawNorms = new Float32Array(0);
    }

    has(id: string): boolean { return this.idToIdx.has(id); }

    get(id: string): { id: string; vec: Float32Array; raw?: Float32Array; meta?: M } | undefined {
        const idx = this.idToIdx.get(id);
        if (idx === undefined) return undefined;
        return {
            id,
            vec: this.vecs[idx],
            raw: this.rawVecs ? this.rawVecs[idx] : undefined,
            meta: this.metas[idx],
        };
    }

    /** Remove by id. Returns true if removed. */
    remove(id: string): boolean {
        const idx = this.idToIdx.get(id);
        if (idx === undefined) return false;
        // capture id, splice arrays
        this.ids.splice(idx, 1);
        this.vecs.splice(idx, 1);
        this.metas.splice(idx, 1);
        if (this.rawVecs) this.rawVecs.splice(idx, 1);
        if (this.norms) this.norms = this.removeFromNorms(this.norms, idx);
        if (this.rawNorms) this.rawNorms = this.removeFromNorms(this.rawNorms, idx);
        this.idToIdx.delete(id);
        this.rebuildIndex(idx);
        return true;
    }

    /** Add or replace an item by id. Returns true if added, false if replaced. */
    upsert(item: EmbeddingItem<M>): boolean {
        const { id, vec, meta } = item;
        if (!id) throw new Error('upsert: id is required');
        if (!vec || vec.length !== this.dim) {
            throw new Error(`upsert: vector dim ${vec?.length ?? 'n/a'} != store dim ${this.dim}`);
        }

        const raw = new Float32Array(vec);
        const unit = this.storeUnit ? normalizeToUnit(raw) : raw;

        const idx = this.idToIdx.get(id);
        if (idx !== undefined) {
            // replace in place
            this.vecs[idx] = unit;
            this.metas[idx] = meta;
            if (this.rawVecs) this.rawVecs[idx] = raw;
            if (this.norms && !this.storeUnit) this.norms[idx] = l2Norm(raw);
            if (this.rawNorms && this.rawVecs) this.rawNorms[idx] = l2Norm(raw);
            return false;
        } else {
            this.ids.push(id);
            this.vecs.push(unit);
            this.metas.push(meta);
            if (this.rawVecs) this.rawVecs.push(raw);

            if (this.norms && !this.storeUnit) {
                // append norm
                const n = l2Norm(raw);
                const newNorms = new Float32Array(this.ids.length);
                newNorms.set(this.norms, 0);
                newNorms[this.ids.length - 1] = n;
                this.norms = newNorms;
            }
            if (this.rawNorms && this.rawVecs) {
                const n = l2Norm(raw);
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

    add(item: EmbeddingItem<M>): void {
        const added = this.upsert(item);
        if (!added) throw new Error(`add: id "${item.id}" already exists (use upsert instead)`);
    }

    addAll(items: EmbeddingItem<M>[], allowUpsert = true): void {
        for (const it of items) {
            if (allowUpsert) this.upsert(it);
            else this.add(it);
        }
    }

    /** Merge another store (same dim & normalization strategy) into this one. */
    merge(other: EmbeddingStore<M>, allowOverwrite = true): void {
        if (other.dimension() !== this.dim) throw new Error('merge: dimension mismatch');
        if (other.isUnitStored() !== this.storeUnit) throw new Error('merge: normalized flag mismatch');
        if (other.keepsRaw() !== this.keepsRaw()) throw new Error('merge: raw retention mismatch');

        for (let i = 0; i < other.ids.length; i++) {
            const id = (other as any).ids[i] as string;
            const vec = (other as any).vecs[i] as Float32Array;
            const raw = (other as any).rawVecs?.[i] as Float32Array | undefined;
            const meta = (other as any).metas[i] as M | undefined;
            if (!allowOverwrite && this.has(id)) continue;
            // Use upsert path, but avoid double-normalizing when both stores have unit vectors:
            this.upsert({ id, vec, meta });
            if (this.rawVecs && raw) this.rawVecs[this.idToIdx.get(id)!] = new Float32Array(raw);
        }
    }

    /* ========== querying ========== */

    /** Top-K KNN query. For L2/L1 we return NEGATIVE distance so higher is better. */
    query(queryVec: ArrayLike<number>, k = 10, opts?: QueryOptions<M>): SearchHit<M>[] {
        if (queryVec.length !== this.dim) {
            throw new Error(`query: vector dim ${queryVec.length} != store dim ${this.dim}`);
        }
        const metric: Metric = opts?.metric ?? 'cosine';
        const filter = opts?.filter;
        const returnVectors = opts?.returnVectors ?? false;
        const minScore = opts?.minScore;
        const maxDistance = opts?.maxDistance;
        const restrictSet = opts?.restrictToIds ? new Set(opts.restrictToIds) : undefined;

        let q: Float32Array;
        let qNorm = 0;

        if (metric === 'cosine') {
            // cosine → normalize query; stored data either unit (fast) or raw (use cached norms)
            q = normalizeToUnit(queryVec);
        } else if (metric === 'dot') {
            q = new Float32Array(queryVec);
            qNorm = l2Norm(q); // only used for potential future scoring transforms
        } else {
            // L2/L1 use RAW query
            q = new Float32Array(queryVec);
            qNorm = l2Norm(q);
        }

        const hits: SearchHit<M>[] = [];
        const N = this.vecs.length;

        // helpers
        const pushHit = (i: number, score: number) => {
            if (restrictSet && !restrictSet.has(this.ids[i])) return;
            if (filter && !filter(this.metas[i], this.ids[i])) return;

            // Apply thresholds
            if (metric === 'euclidean' || metric === 'manhattan') {
                const dist = -score; // score is negative distance
                if (maxDistance !== undefined && dist > maxDistance) return;
            } else {
                if (minScore !== undefined && score < minScore) return;
            }

            hits.push(
                returnVectors
                    ? { id: this.ids[i], score, index: i, meta: this.metas[i], vec: this.vecs[i] }
                    : { id: this.ids[i], score, index: i, meta: this.metas[i] }
            );
        };

        if (metric === 'cosine') {
            if (this.storeUnit) {
                // both unit → score = dot
                for (let i = 0; i < N; i++) {
                    const s = dot(q, this.vecs[i]);
                    pushHit(i, s);
                }
            } else {
                // stored raw in vecs → use cached norms (if available) for cos = dot / (||q||*||v||)
                if (!this.norms || this.norms.length !== N) {
                    // build norms on-demand once
                    this.norms = new Float32Array(N);
                    for (let i = 0; i < N; i++) this.norms[i] = l2Norm(this.vecs[i]);
                }
                const qn = l2Norm(q) || 1; // guard
                for (let i = 0; i < N; i++) {
                    const dn = this.norms[i] || 1;
                    const s = dn < EPS ? 0 : dot(q, this.vecs[i]) / (qn * dn);
                    pushHit(i, s);
                }
            }
        } else if (metric === 'dot') {
            for (let i = 0; i < N; i++) {
                const s = dot(q, this.storeUnit ? this.vecs[i] : this.vecs[i]); // same storage
                pushHit(i, s);
            }
        } else if (metric === 'euclidean') {
            // Need RAW vectors
            const base = this.rawVecs ?? (!this.storeUnit ? this.vecs : null);
            if (!base) throw new Error('euclidean query requires raw vectors; create store with alsoStoreRaw=true or storeUnit=false');
            // Use fast formula: ||q - v|| = sqrt(||q||^2 + ||v||^2 - 2 q·v)
            const vNorms =
                this.rawVecs ? (this.rawNorms ?? this.buildRawNorms()) :
                    this.norms ?? this.buildNorms();

            const q2 = qNorm * qNorm;
            for (let i = 0; i < N; i++) {
                const d2 = Math.max(q2 + vNorms[i] * vNorms[i] - 2 * dot(q, base[i]), 0);
                const dist = Math.sqrt(d2);
                pushHit(i, -dist); // NEGATIVE distance so higher is better
            }
        } else { // 'manhattan'
            const base = this.rawVecs ?? (!this.storeUnit ? this.vecs : null);
            if (!base) throw new Error('manhattan query requires raw vectors; create store with alsoStoreRaw=true or storeUnit=false');
            for (let i = 0; i < N; i++) {
                const dist = l1Dist(q, base[i]);
                pushHit(i, -dist); // NEGATIVE distance
            }
        }

        if (hits.length === 0) return [];
        const kk = Math.max(1, Math.min(k, hits.length));

        // Use quickselect to avoid full O(n log n) sort
        const cutoffIdx = quickselectTopK(hits, kk, (h) => h.score);
        // Now sort just the top-K region for stable ordering
        hits
            .slice(0, kk)
            .sort((a, b) => b.score - a.score)
            .forEach((h, i) => (hits[i] = h));

        return hits.slice(0, kk);
    }

    /** Batch query helper. Returns array of results aligned to input queries. */
    queryBatch(queries: ArrayLike<number>[], k = 10, opts?: QueryOptions<M>): SearchHit<M>[][] {
        return queries.map(q => this.query(q, k, opts));
    }

    /** Convenience: query by id */
    queryById(id: string, k = 10, opts?: QueryOptions<M>): SearchHit<M>[] {
        const rec = this.get(id);
        if (!rec) return [];
        const use = (opts?.metric === 'euclidean' || opts?.metric === 'manhattan')
            ? (rec.raw ?? rec.vec) // prefer raw for distance
            : rec.vec;
        return this.query(use, k, opts);
    }

    /* ========== export / import ========== */

    toJSON(): EmbeddingStoreJSON<M> {
        const includeRaw = !!this.rawVecs;
        return {
            dim: this.dim,
            normalized: this.storeUnit,
            alsoStoredRaw: includeRaw,
            capacity: this.capacity,
            items: this.ids.map((id, i) => ({
                id,
                vec: Array.from(this.vecs[i]),
                raw: includeRaw ? Array.from(this.rawVecs![i]) : undefined,
                meta: this.metas[i],
            })),
            __version: 'embedding-store-2.0.0',
        };
    }

    static fromJSON<M extends Record<string, any> = Record<string, any>>(
        obj: string | EmbeddingStoreJSON<M>
    ): EmbeddingStore<M> {
        const parsed: EmbeddingStoreJSON<M> = typeof obj === 'string' ? JSON.parse(obj) : obj;
        if (!parsed || !parsed.dim || !Array.isArray(parsed.items)) {
            throw new Error('EmbeddingStore.fromJSON: invalid payload');
        }
        const store = new EmbeddingStore<M>(parsed.dim, {
            storeUnit: parsed.normalized,
            capacity: parsed.capacity,
            alsoStoreRaw: parsed.alsoStoredRaw ?? false,
        });

        for (const it of parsed.items) {
            if (!it || typeof it.id !== 'string' || !Array.isArray(it.vec)) continue;
            if (it.vec.length !== parsed.dim) {
                throw new Error(`fromJSON: vector dim ${it.vec.length} != dim ${parsed.dim} for id ${it.id}`);
            }
            // Use public API to keep norms consistent
            store.upsert({ id: it.id, vec: it.raw ?? it.vec, meta: it.meta });
            // If payload includes both vec and raw, ensure both sides are *exactly* respected
            if (store.storeUnit && store.rawVecs && it.raw) {
                const idx = store.idToIdx.get(it.id)!;
                store.rawVecs[idx] = new Float32Array(it.raw);
                if (store.rawNorms) {
                    const newNorms = new Float32Array(store.size());
                    newNorms.set(store.rawNorms, 0);
                    newNorms[idx] = l2Norm(store.rawVecs[idx]);
                    store.rawNorms = newNorms;
                }
            } else if (!store.storeUnit && it.vec) {
                // vecs already raw in this mode; norms handled by upsert path
            }
        }
        return store;
    }

    /* ========== diagnostics / utils ========== */

    /** Estimate memory footprint in bytes (arrays only; metadata excluded). */
    memoryUsageBytes(): number {
        const f32 = 4;
        let bytes = 0;
        for (const v of this.vecs) bytes += v.length * f32;
        if (this.rawVecs) for (const v of this.rawVecs) bytes += v.length * f32;
        if (this.norms) bytes += this.norms.length * f32;
        if (this.rawNorms) bytes += this.rawNorms.length * f32;
        // ids + metas are JS objects; not included
        return bytes;
    }

    /** Re-normalize all vectors in-place (useful if you bulk-updated raw storage). */
    reNormalizeAll(): void {
        if (!this.storeUnit) return; // nothing to do
        for (let i = 0; i < this.vecs.length; i++) {
            const raw = this.rawVecs ? this.rawVecs[i] : this.vecs[i];
            this.vecs[i] = normalizeToUnit(raw);
        }
    }

    /** Iterate over all items */
    *entries(): IterableIterator<{ id: string; vec: Float32Array; raw?: Float32Array; meta?: M }> {
        for (let i = 0; i < this.ids.length; i++) {
            yield { id: this.ids[i], vec: this.vecs[i], raw: this.rawVecs?.[i], meta: this.metas[i] };
        }
    }

    /* ========== internals ========== */

    private removeFromNorms(src: Float32Array, removeIdx: number): Float32Array {
        const out = new Float32Array(src.length - 1);
        if (removeIdx > 0) out.set(src.subarray(0, removeIdx), 0);
        if (removeIdx < src.length - 1) out.set(src.subarray(removeIdx + 1), removeIdx);
        return out;
    }

    /** After a splice at 'start', rebuild id→index for shifted tail */
    private rebuildIndex(start = 0): void {
        if (start <= 0) {
            this.idToIdx.clear();
            for (let i = 0; i < this.ids.length; i++) this.idToIdx.set(this.ids[i], i);
            return;
        }
        for (let i = start; i < this.ids.length; i++) this.idToIdx.set(this.ids[i], i);
    }

    /** Enforce capacity by evicting oldest items (front of arrays) */
    private enforceCapacity(): void {
        if (this.capacity === undefined) return;
        while (this.ids.length > this.capacity) {
            const removedId = this.ids[0];
            // shift( ) is O(n); for very large stores consider a circular buffer
            this.ids.shift();
            this.vecs.shift();
            this.metas.shift();
            if (this.rawVecs) this.rawVecs.shift();
            if (this.norms) this.norms = this.removeFromNorms(this.norms, 0);
            if (this.rawNorms) this.rawNorms = this.removeFromNorms(this.rawNorms, 0);
            this.idToIdx.delete(removedId);
            // rebuild full index (ids shifted)
            this.idToIdx.clear();
            for (let i = 0; i < this.ids.length; i++) this.idToIdx.set(this.ids[i], i);
        }
    }

    private buildNorms(): Float32Array {
        const out = new Float32Array(this.vecs.length);
        for (let i = 0; i < this.vecs.length; i++) out[i] = l2Norm(this.vecs[i]);
        this.norms = out;
        return out;
    }
    private buildRawNorms(): Float32Array {
        if (!this.rawVecs) throw new Error('no raw vectors to build norms for');
        const out = new Float32Array(this.rawVecs.length);
        for (let i = 0; i < this.rawVecs.length; i++) out[i] = l2Norm(this.rawVecs[i]);
        this.rawNorms = out;
        return out;
    }
}
