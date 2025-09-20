// CharacterLangEncoderELM.ts — robust char/token text encoder on top of ELM
// Upgrades:
//  • Safe preset extraction (no union-type errors on maxLen/charSet)
//  • Proper (inputs, labels) training via trainFromData()
//  • Hidden-layer embeddings via elm.getEmbedding() (with matrix fallback)
//  • Batch encode(), JSON I/O passthrough, gentle logging
//  • Activation typed, no reliance on private fields

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
// If you have a preset (optional). Otherwise remove this import.
// import { EnglishTokenPreset } from '../config/Presets';

export class CharacterLangEncoderELM {
    private elm: ELM;
    private config: ELMConfig;
    private activation: Activation;

    constructor(config: ELMConfig) {
        // Make sure we have the basics
        if (!config.hiddenUnits) {
            throw new Error('CharacterLangEncoderELM requires hiddenUnits');
        }
        // Activation defaults to 'relu' if not provided
        this.activation = (config as any).activation ?? 'relu';

        // Safely coerce into a *text* config (avoid NumericConfig branch)
        // We do not assume a preset exists; provide conservative defaults.
        const textMaxLen =
            (config as any)?.maxLen ?? /* fallback */ 64;
        const textCharSet =
            (config as any)?.charSet ?? 'abcdefghijklmnopqrstuvwxyz';
        const textTokDelim =
            (config as any)?.tokenizerDelimiter ?? /\s+/;

        // Merge into a TEXT-leaning config object.
        // NOTE: We keep categories if provided, but we will override them in train() from labels.
        this.config = {
            ...config,
            // Force text branch:
            useTokenizer: true,
            maxLen: textMaxLen,
            charSet: textCharSet,
            tokenizerDelimiter: textTokDelim,
            activation: this.activation,
            // Make logging robust:
            log: {
                modelName: 'CharacterLangEncoderELM',
                verbose: config.log?.verbose ?? false,
                toFile: config.log?.toFile ?? false,
                level: (config.log as any)?.level ?? 'info',
            },
        } as any; // cast to any to avoid union friction

        this.elm = new ELM(this.config);

        // Forward thresholds/export if present
        if ((config as any).metrics) {
            (this.elm as any).metrics = (config as any).metrics;
        }
        if ((this.config as any).exportFileName) {
            (this.elm as any).config.exportFileName = (this.config as any).exportFileName;
        }
    }

    /**
     * Train on parallel arrays: inputs (strings) + labels (strings).
     * We:
     *  • dedupe labels → categories
     *  • encode inputs with the ELM’s text encoder
     *  • one-hot the labels
     *  • call trainFromData(X, Y)
     */
    public train(inputStrings: string[], labels: string[]): void {
        if (!inputStrings?.length || !labels?.length || inputStrings.length !== labels.length) {
            throw new Error('train() expects equal-length inputStrings and labels');
        }

        // Build categories from labels
        const categories = Array.from(new Set(labels));
        this.elm.setCategories(categories);

        // Get the encoder (support getEncoder() or .encoder)
        const enc: any =
            (this.elm as any).getEncoder?.() ??
            (this.elm as any).encoder;

        if (!enc?.encode || !enc?.normalize) {
            throw new Error('ELM text encoder is not available. Ensure useTokenizer/maxLen/charSet are set.');
        }

        const X: number[][] = [];
        const Y: number[][] = [];
        for (let i = 0; i < inputStrings.length; i++) {
            const x = enc.normalize(enc.encode(String(inputStrings[i] ?? '')));
            X.push(x);

            const li = categories.indexOf(labels[i]);
            const y = new Array(categories.length).fill(0);
            if (li >= 0) y[li] = 1;
            Y.push(y);
        }

        // Classic ELM closed-form training
        this.elm.trainFromData(X, Y);
    }

    /**
     * Returns a dense embedding for one string.
     * Uses ELM.getEmbedding() if available; otherwise computes H = act(XW^T + b).
     * By design this returns the *hidden* feature (length = hiddenUnits).
     */
    public encode(text: string): number[] {
        // Get encoder
        const enc: any =
            (this.elm as any).getEncoder?.() ??
            (this.elm as any).encoder;

        if (!enc?.encode || !enc?.normalize) {
            throw new Error('ELM text encoder is not available. Train or configure text settings first.');
        }

        const x = enc.normalize(enc.encode(String(text ?? '')));

        // Prefer official embedding API if present
        if (typeof (this.elm as any).getEmbedding === 'function') {
            const E = (this.elm as any).getEmbedding([x]);
            if (Array.isArray(E) && Array.isArray(E[0])) return E[0] as number[];
        }

        // Fallback: compute hidden act via model params (W,b)
        const model = (this.elm as any).model;
        if (!model) throw new Error('Model not trained.');

        const { W, b } = model; // W: hidden x in, b: hidden x 1
        const tempH = Matrix.multiply([x], Matrix.transpose(W)); // (1 x hidden)
        const act = Activations.get(this.activation);
        const H = tempH.map(row => row.map((v, j) => act(v + b[j][0]))); // (1 x hidden)

        // Return hidden vector
        return H[0];
    }

    /** Batch encoding convenience */
    public encodeBatch(texts: string[]): number[][] {
        return texts.map(t => this.encode(t));
    }

    /** Load/save passthroughs */
    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }
    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
