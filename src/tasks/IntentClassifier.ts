// intentClassifier.ts — ELM-based intent classification (text → label)

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';
import type { PredictResult } from '../core/ELM';

type AugmentOpts = {
    suffixes?: string[];
    prefixes?: string[];
    includeNoise?: boolean;
    noiseRate?: number; // default 0.05 per char
    charSet?: string;   // fallback when encoder charset isn't available
};

export class IntentClassifier {
    private model: ELM;
    private config: ELMConfig;
    private categories: string[] = [];

    constructor(config: ELMConfig) {
        // Basic guardrails (common footguns)
        const hidden = (config as any).hiddenUnits;
        const act = (config as any).activation as Activation | undefined;
        if (typeof hidden !== 'number') {
            throw new Error('IntentClassifier requires config.hiddenUnits (number)');
        }
        if (!act) {
            throw new Error('IntentClassifier requires config.activation');
        }

        // Force TEXT mode (tokenizer on). We set categories during train().
        this.config = {
            ...config,
            categories: (config as any).categories ?? [],
            useTokenizer: true,
            log: {
                modelName: 'IntentClassifier',
                verbose: config.log?.verbose ?? false,
                toFile: config.log?.toFile ?? false,
                // @ts-ignore: optional passthrough
                level: (config.log as any)?.level ?? 'info',
            },
        } as any;

        this.model = new ELM(this.config);

        // Optional thresholds/export passthrough
        if ((config as any).metrics) (this.model as any).metrics = (config as any).metrics;
        if ((config as any).exportFileName) (this.model as any).config.exportFileName = (config as any).exportFileName;
    }

    /* ==================== Training ==================== */

    /**
     * Train from (text, label) pairs using closed-form ELM solve.
     * Uses the ELM's UniversalEncoder (token mode).
     */
    public train(
        textLabelPairs: { text: string; label: string }[],
        augmentation?: AugmentOpts
    ): void {
        if (!textLabelPairs?.length) throw new Error('train: empty training data');

        // Build label set
        this.categories = Array.from(new Set(textLabelPairs.map(p => p.label)));
        this.model.setCategories(this.categories);

        // Prepare encoder
        const enc = this.model.getEncoder?.() ?? (this.model as any).encoder;

        if (!enc) throw new Error('IntentClassifier: encoder unavailable on ELM instance.');

        // Inline augmentation (prefix/suffix/noise) — lightweight so we avoid importing Augment here
        const charSet =
            augmentation?.charSet ||
            enc.charSet ||
            'abcdefghijklmnopqrstuvwxyz';

        const makeNoisy = (s: string, rate = augmentation?.noiseRate ?? 0.05) => {
            if (!augmentation?.includeNoise || rate <= 0) return [s];
            const arr = s.split('');
            for (let i = 0; i < arr.length; i++) {
                if (Math.random() < rate) {
                    const r = Math.floor(Math.random() * charSet.length);
                    arr[i] = charSet[r] ?? arr[i];
                }
            }
            return [s, arr.join('')];
        };

        const expanded: { text: string; label: string }[] = [];
        for (const p of textLabelPairs) {
            const base = [p.text];
            const withPrefixes = (augmentation?.prefixes ?? []).map(px => `${px}${p.text}`);
            const withSuffixes = (augmentation?.suffixes ?? []).map(sx => `${p.text}${sx}`);
            const candidates = [...base, ...withPrefixes, ...withSuffixes];
            for (const c of candidates) {
                for (const v of makeNoisy(c)) {
                    expanded.push({ text: v, label: p.label });
                }
            }
        }

        // Encode + one-hot
        const X: number[][] = new Array(expanded.length);
        const Y: number[][] = new Array(expanded.length);
        for (let i = 0; i < expanded.length; i++) {
            const { text, label } = expanded[i];
            const vec = enc.normalize(enc.encode(text));
            X[i] = vec;
            const row = new Array(this.categories.length).fill(0);
            const li = this.categories.indexOf(label);
            if (li >= 0) row[li] = 1;
            Y[i] = row;
        }

        // Closed-form ELM training
        (this.model as any).trainFromData(X, Y);
    }

    /* ==================== Inference ==================== */

    /** Top-K predictions with an optional probability threshold */
    public predict(text: string, topK = 1, threshold = 0): PredictResult[] {
        const res = this.model.predict(text, Math.max(1, topK));
        return threshold > 0 ? res.filter(r => r.prob >= threshold) : res;
    }

    /** Batched predict */
    public predictBatch(texts: string[], topK = 1, threshold = 0): PredictResult[][] {
        return texts.map(t => this.predict(t, topK, threshold));
    }

    /** Convenience: best label + prob (or undefined if below threshold) */
    public predictLabel(text: string, threshold = 0): { label: string; prob: number } | undefined {
        const [top] = this.predict(text, 1, threshold);
        return top;
    }

    /* ==================== Model I/O ==================== */

    public loadModelFromJSON(json: string): void {
        this.model.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.model.saveModelAsJSONFile(filename);
    }
}
