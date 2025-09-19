// EmbeddingStore.ts — Lightweight vector store with cosine/dot/euclidean KNN

export type Metric = 'cosine' | 'dot' | 'euclidean';

export interface EmbeddingItem {
    id: string;
    vec: number[] | Float32Array;
    meta?: Record<string, any>;
}

export interface QueryOptions {
    /** Similarity/distance metric (default: 'cosine') */
    metric?: Metric;
    /** Optional metadata filter: return only hits where filter(meta, id) is true */
    filter?: (meta: Record<string, any> | undefined, id: string) => boolean;
    /** Include vectors in results (default: false) */
    returnVectors?: boolean;
}

export interface SearchHit {
    id: string;
    score: number;          // similarity (cosine/dot) or negative distance for euclidean (higher is better)
    index: number;          // internal index at time of query
    meta?: Record<string, any>;
    vec?: Float32Array;     // present when returnVectors=true
}

export interface EmbeddingStoreJSON {
    dim: number;
    /** If true, vectors in 'items' are already L2-normalized */
    normalized: boolean;
    /** Optional capacity for ring buffer behavior */
    capacity?: number;
    items: Array<{ id: string; vec: number[]; meta?: Record<string, any> }>;
    __version: string;
}

const EPS = 1e-12;

function l2Norm(v: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
}

function normalizeToUnit(v: ArrayLike<number>): Float32Array {
    const out = new Float32Array(v.length);
    const n = l2Norm(v);
    if (n < EPS) {
        // zero vector → leave as zeros; downstream cosine returns 0
        return out;
    }
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
}

function dot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function euclideanNeg(a: Float32Array, b: Float32Array): number {
    // return negative distance so "higher is better" consistently
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return -Math.sqrt(s);
}

/**
 * In-memory embedding store.
 * - By default stores **unit-normalized** vectors (storeUnit=true),
 *   making cosine similarity a fast dot product.
 * - Capacity (if set) evicts the **oldest** item when exceeded.
 */
export class EmbeddingStore {
    private readonly dim: number;
    private readonly storeUnit: boolean;
    private capacity?: number;

    // Data
    private ids: string[] = [];
    private vecs: Float32Array[] = [];  // stored as unit vectors when storeUnit=true
    private metas: (Record<string, any> | undefined)[] = [];

    // Index
    private idToIdx: Map<string, number> = new Map();

    constructor(dim: number, opts?: { storeUnit?: boolean; capacity?: number }) {
        if (!Number.isFinite(dim) || dim <= 0) throw new Error(`EmbeddingStore: invalid dim=${dim}`);
        this.dim = dim | 0;
        this.storeUnit = opts?.storeUnit ?? true;
        if (opts?.capacity !== undefined) {
            if (!Number.isFinite(opts.capacity) || opts.capacity <= 0) throw new Error(`capacity must be > 0`);
            this.capacity = Math.floor(opts.capacity);
        }
    }

    /* ================= Basic ops ================= */

    size(): number { return this.ids.length; }
    dimension(): number { return this.dim; }
    isUnitStored(): boolean { return this.storeUnit; }
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
    }

    has(id: string): boolean { return this.idToIdx.has(id); }

    get(id: string): { id: string; vec: Float32Array; meta?: Record<string, any> } | undefined {
        const idx = this.idToIdx.get(id);
        if (idx === undefined) return undefined;
        return { id, vec: this.vecs[idx], meta: this.metas[idx] };
    }

    remove(id: string): boolean {
        const idx = this.idToIdx.get(id);
        if (idx === undefined) return false;
        // Remove by splice; update indices after idx
        this.ids.splice(idx, 1);
        this.vecs.splice(idx, 1);
        this.metas.splice(idx, 1);
        this.rebuildIndex(idx);
        return true;
    }

    /** Add or replace an item by id. Returns true if added, false if replaced. */
    upsert(item: EmbeddingItem): boolean {
        const { id, vec, meta } = item;
        if (!id) throw new Error('upsert: id is required');
        if (!vec || vec.length !== this.dim) {
            throw new Error(`upsert: vector dim ${vec?.length ?? 'n/a'} != store dim ${this.dim}`);
        }

        const unit = this.storeUnit ? normalizeToUnit(vec) : new Float32Array(vec);
        const idx = this.idToIdx.get(id);
        if (idx !== undefined) {
            // replace in place
            this.vecs[idx] = unit;
            this.metas[idx] = meta;
            return false;
        } else {
            this.ids.push(id);
            this.vecs.push(unit);
            this.metas.push(meta);
            this.idToIdx.set(id, this.ids.length - 1);
            this.enforceCapacity();
            return true;
        }
    }

    add(item: EmbeddingItem): void {
        const added = this.upsert(item);
        if (!added) throw new Error(`add: id "${item.id}" already exists (use upsert instead)`);
    }

    addAll(items: EmbeddingItem[], allowUpsert = true): void {
        for (const it of items) {
            if (allowUpsert) this.upsert(it);
            else this.add(it);
        }
    }

    /* =============== Querying =============== */

    /**
     * KNN query.
     * For 'cosine' (default), if storeUnit=true we compute dot(query_unit, stored_unit) directly.
     * For 'dot', no normalization is applied.
     * For 'euclidean', returns **negative distance** so higher is better (consistent API).
     */
    query(queryVec: ArrayLike<number>, k = 10, opts?: QueryOptions): SearchHit[] {
        if (queryVec.length !== this.dim) {
            throw new Error(`query: vector dim ${queryVec.length} != store dim ${this.dim}`);
        }
        const metric: Metric = opts?.metric ?? 'cosine';
        const filter = opts?.filter;
        const returnVectors = opts?.returnVectors ?? false;

        let q: Float32Array;
        if (metric === 'cosine') {
            q = normalizeToUnit(queryVec);
        } else {
            q = new Float32Array(queryVec);
        }

        // Score all, keep top-k (for moderate N, full sort is fine)
        const hits: SearchHit[] = [];
        for (let i = 0; i < this.vecs.length; i++) {
            const id = this.ids[i];
            const meta = this.metas[i];
            if (filter && !filter(meta, id)) continue;

            const v = this.vecs[i];
            let score: number;
            if (metric === 'cosine') {
                // If storeUnit=false, cosine = dot(q_unit, v_unit):
                const vUnit = this.storeUnit ? v : normalizeToUnit(v);
                score = dot(q, vUnit);
            } else if (metric === 'dot') {
                score = dot(q, v);
            } else { // euclidean
                const vRaw = this.storeUnit ? v : v; // they are fine; euclidean uses raw coords
                score = euclideanNeg(q, vRaw);
            }

            hits.push(returnVectors ? { id, score, index: i, meta, vec: v } : { id, score, index: i, meta });
        }

        if (hits.length === 0) return [];
        const kk = Math.max(1, Math.min(k, hits.length));
        // Partial selection would be faster for huge N; sort is fine for typical sizes.
        hits.sort((a, b) => b.score - a.score);
        return hits.slice(0, kk);
    }

    /** Convenience: query by id */
    queryById(id: string, k = 10, opts?: QueryOptions): SearchHit[] {
        const rec = this.get(id);
        if (!rec) return [];
        return this.query(rec.vec, k, opts);
    }

    /* =============== Export / Import =============== */

    toJSON(): EmbeddingStoreJSON {
        return {
            dim: this.dim,
            normalized: this.storeUnit,
            capacity: this.capacity,
            items: this.ids.map((id, i) => ({
                id,
                vec: Array.from(this.vecs[i]),
                meta: this.metas[i],
            })),
            __version: 'embedding-store-1.0.0',
        };
    }

    static fromJSON(obj: string | EmbeddingStoreJSON): EmbeddingStore {
        const parsed: EmbeddingStoreJSON = typeof obj === 'string' ? JSON.parse(obj) : obj;
        if (!parsed || !parsed.dim || !Array.isArray(parsed.items)) {
            throw new Error('EmbeddingStore.fromJSON: invalid payload');
        }
        const store = new EmbeddingStore(parsed.dim, { storeUnit: parsed.normalized, capacity: parsed.capacity });
        // Push directly; avoid double-normalization if normalized=true
        for (const it of parsed.items) {
            if (!it || typeof it.id !== 'string' || !Array.isArray(it.vec)) continue;
            if (it.vec.length !== parsed.dim) {
                throw new Error(`fromJSON: vector dim ${it.vec.length} != dim ${parsed.dim} for id ${it.id}`);
            }
            const v = parsed.normalized ? new Float32Array(it.vec) : normalizeToUnit(it.vec);
            store.ids.push(it.id);
            store.vecs.push(v);
            store.metas.push(it.meta);
            store.idToIdx.set(it.id, store.ids.length - 1);
        }
        return store;
    }

    /* =============== Internals =============== */

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
            // Evict oldest (index 0); O(n) shift is OK for moderate sizes
            const removedId = this.ids.shift()!;
            this.vecs.shift();
            this.metas.shift();
            // Rebuild full index (ids have shifted)
            this.idToIdx.clear();
            for (let i = 0; i < this.ids.length; i++) this.idToIdx.set(this.ids[i], i);
            // (optional) If you need a hook: console.debug(`evicted ${removedId}`);
        }
    }
}
