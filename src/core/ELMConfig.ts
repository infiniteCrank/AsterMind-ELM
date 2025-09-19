// ELMConfig.ts - Configuration interfaces, defaults, and helpers for ELM-based models

/* =========================
 * Core Types
 * ========================= */

export type Activation = 'tanh' | 'relu' | 'leakyrelu' | 'sigmoid' | 'linear';

export interface BaseConfig {
    /** Number of hidden units */
    hiddenUnits: number;
    /** Activation function to use */
    activation?: Activation;
    /** Ridge regularization factor */
    ridgeLambda?: number;
    /** Optional random seed for deterministic initialization */
    seed?: number;

    /** Logging options */
    log?: {
        modelName?: string;
        verbose?: boolean;
        toFile?: boolean;
        level?: 'info' | 'debug';
    };
    logFileName?: string;

    /** Regularization & init */
    dropout?: number;
    weightInit?: 'uniform' | 'xavier';

    /** File export */
    exportFileName?: string;

    /** Optional metric thresholds (used to decide if a trained model is "good enough" to save) */
    metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
        f1?: number;
        crossEntropy?: number;
        r2?: number;
    };
}

/** Numeric (vector) input configuration */
export interface NumericConfig extends BaseConfig {
    inputSize: number;
    useTokenizer?: false;         // explicit off
    categories: string[];         // class names / labels
}

/** Text input configuration */
export interface TextConfig extends BaseConfig {
    useTokenizer: true;           // explicit on
    categories: string[];
    maxLen: number;
    charSet?: string;
    tokenizerDelimiter?: RegExp;
    /** Optional pre-built encoder (e.g., UniversalEncoder) */
    encoder?: any;
}

/** Union of numeric + text configs */
export type ELMConfig = NumericConfig | TextConfig;

/* =========================
 * Training contracts
 * ========================= */

export interface TrainOptions {
    task?: 'classification' | 'regression';
    onProgress?: (phase: 'encode' | 'formH' | 'solve' | 'done', pct: number) => void;
}

export interface TrainResult {
    epochs: number;
    loss?: number;
    metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
        [key: string]: number | undefined;
    };
}

/* =========================
 * Serialization
 * (JSON-safe for browser save/load)
 * ========================= */

export interface SerializedELM {
    /**
     * Store tokenizerDelimiter as a string pattern (JSON-safe),
     * and omit encoder instances. Numeric fields are preserved as-is.
     */
    config: (
        Omit<NumericConfig, 'seed' | 'log'> &
        Partial<Omit<TextConfig, 'tokenizerDelimiter' | 'encoder'>>
    ) & {
        tokenizerDelimiter?: string; // serialized pattern if present
    };

    /** Weights */
    W: number[][];  // hidden weights (hiddenUnits x inputSize)
    b: number[][];  // hidden bias    (hiddenUnits x 1)  <-- fixed to matrix
    B: number[][];  // output weights (hiddenUnits x outDim)
}

/** Rehydrate any serialized text bits (e.g., tokenizerDelimiter string → RegExp) */
export function deserializeTextBits<T extends SerializedELM['config']>(cfg: T): T {
    if ((cfg as any).useTokenizer && typeof (cfg as any).tokenizerDelimiter === 'string') {
        (cfg as any).tokenizerDelimiter = new RegExp((cfg as any).tokenizerDelimiter);
    }
    return cfg;
}

/* =========================
 * Type Guards
 * ========================= */

export function isTextConfig(cfg: ELMConfig): cfg is TextConfig {
    return (cfg as TextConfig).useTokenizer === true;
}

export function isNumericConfig(cfg: ELMConfig): cfg is NumericConfig {
    return !isTextConfig(cfg);
}

/* =========================
 * Defaults
 * ========================= */

export const DEFAULT_CHARSET = 'abcdefghijklmnopqrstuvwxyz';

/** Defaults that apply to both modes */
export const defaultBase = {
    hiddenUnits: 50,
    activation: 'relu' as const,
    weightInit: 'uniform' as const,
} satisfies Required<Pick<BaseConfig, 'hiddenUnits' | 'activation' | 'weightInit'>>;

/** Safe partial defaults for numeric mode (no type errors) */
export const defaultNumericConfig = {
    ...defaultBase,
    useTokenizer: false as const,
    // NOTE: inputSize and categories are required by callers; not set here.
} satisfies Partial<NumericConfig>;

/** Safe partial defaults for text mode (no type errors) */
export const defaultTextConfig = {
    ...defaultBase,
    useTokenizer: true as const,
    maxLen: 30,
    charSet: DEFAULT_CHARSET,
    tokenizerDelimiter: /\s+/,
} satisfies Partial<TextConfig>;

/** Merge user config with mode-appropriate defaults, preserving the original type */
export function normalizeConfig(cfg: ELMConfig): NumericConfig | TextConfig {
    if (isTextConfig(cfg)) {
        return { ...defaultTextConfig, ...cfg } as TextConfig;
    }
    return { ...defaultNumericConfig, ...cfg } as NumericConfig;
}
