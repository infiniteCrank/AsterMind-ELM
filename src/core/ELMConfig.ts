// ELMConfig.ts - Configuration interfaces, defaults, helpers for ELM-based models

/* =========== Types =========== */

export type Activation =
    | 'tanh'
    | 'relu'
    | 'leakyrelu'
    | 'sigmoid'
    | 'linear'
    | 'gelu';

export type WeightInit = 'uniform' | 'xavier' | 'he';

export interface BaseConfig {
    /** Random feature width (hidden units) */
    hiddenUnits: number;

    /** Activation function for hidden layer */
    activation?: Activation;

    /** Ridge regularization factor λ (for (HᵀH + λI)) */
    ridgeLambda?: number;

    /** Optional seed for deterministic initialization */
    seed?: number;

    /** Logging options */
    log?: {
        modelName?: string;
        verbose?: boolean;
        toFile?: boolean;
        level?: 'info' | 'debug';
    };
    logFileName?: string;

    /** Regularization */
    dropout?: number;

    /** Weight initialization scheme */
    weightInit?: WeightInit;

    /** Optional export file name for saved model */
    exportFileName?: string;

    /** Optional metrics thresholds (if provided, training saves only when all pass) */
    metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
        f1?: number;
        crossEntropy?: number;
        r2?: number;
        [key: string]: number | undefined;
    };

    /** (Optional) future toggle: 'classification' | 'regression' */
    task?: 'classification' | 'regression';
}

/** Numeric (vector) input configuration */
export interface NumericConfig extends BaseConfig {
    /** Input vector size (required for pure numeric workflows outside text mode) */
    inputSize: number;
    /** Explicitly disable tokenizer for numeric mode */
    useTokenizer?: false;
    /** Output categories (labels) */
    categories: string[];
}

/** Text input configuration */
export interface TextConfig extends BaseConfig {
    /** Enable tokenizer-based text mode */
    useTokenizer: true;
    /** Output categories (labels) */
    categories: string[];
    /** Max sequence/token length for encoder */
    maxLen: number;
    /** Allowed characters for char-mode encoders (if applicable) */
    charSet?: string;
    /** Delimiter regex for tokenization */
    tokenizerDelimiter?: RegExp;
    /** Optional prebuilt encoder instance */
    encoder?: any;
}

/** Union config consumed by ELM */
export type ELMConfig = NumericConfig | TextConfig;

/* =========== Training metadata =========== */

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

/* =========== Serialization =========== */

/**
 * Serialized form for save/load.
 * Note: tokenizerDelimiter is stored as a string (source) to be JSON-safe.
 */
export interface SerializedELM {
    config: (
        Omit<NumericConfig, 'seed' | 'log'> &
        Partial<Omit<TextConfig, 'encoder' | 'tokenizerDelimiter'>> &
        { tokenizerDelimiter?: string }
    );
    W: number[][];  // hidden x input
    b: number[][];  // hidden x 1
    B: number[][];  // hidden x output
}

/* =========== Defaults =========== */

const defaultBase: Required<Pick<BaseConfig,
    'hiddenUnits' | 'activation' | 'ridgeLambda' | 'weightInit'
>> & Partial<BaseConfig> = {
    hiddenUnits: 50,
    activation: 'relu',
    ridgeLambda: 1e-2,
    weightInit: 'xavier',
    seed: 1337,
    dropout: 0,
    log: { verbose: true, toFile: false, modelName: 'Unnamed ELM Model', level: 'info' },
};

export const defaultNumericConfig: Partial<NumericConfig> = {
    ...defaultBase,
    useTokenizer: false,
    // inputSize and categories are required by the caller.
};

export const defaultTextConfig: Partial<TextConfig> = {
    ...defaultBase,
    useTokenizer: true,
    maxLen: 30,
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    tokenizerDelimiter: /\s+/,
    // categories are required by the caller.
};

/* =========== Type guards =========== */

export function isTextConfig(cfg: ELMConfig): cfg is TextConfig {
    return (cfg as TextConfig).useTokenizer === true;
}

export function isNumericConfig(cfg: ELMConfig): cfg is NumericConfig {
    return (cfg as any).useTokenizer !== true;
}

/* =========== Helpers =========== */

/**
 * Normalize a user config with sensible defaults depending on mode.
 * (Keeps the original structural type, only fills in missing optional fields.)
 */
export function normalizeConfig<T extends ELMConfig>(cfg: T): T {
    if (isTextConfig(cfg)) {
        const merged: TextConfig = {
            ...(defaultTextConfig as TextConfig),
            ...cfg,
            log: { ...(defaultBase.log ?? {}), ...(cfg.log ?? {}) },
        };
        return merged as T;
    } else {
        const merged: NumericConfig = {
            ...(defaultNumericConfig as NumericConfig),
            ...cfg,
            log: { ...(defaultBase.log ?? {}), ...(cfg.log ?? {}) },
        };
        return merged as T;
    }
}

/**
 * Rehydrate text-specific fields from a JSON-safe config
 * (e.g., convert tokenizerDelimiter source string → RegExp).
 */
export function deserializeTextBits(config: SerializedELM['config']): ELMConfig {
    // If useTokenizer not true, assume numeric config
    if ((config as any).useTokenizer !== true) {
        const nc: NumericConfig = {
            ...(defaultNumericConfig as NumericConfig),
            ...(config as any),
            log: { ...(defaultBase.log ?? {}), ...((config as any).log ?? {}) },
        };
        return nc;
    }

    // Text config: coerce delimiter
    const tDelim = (config as any).tokenizerDelimiter;
    let delimiter: RegExp | undefined = undefined;
    if (tDelim instanceof RegExp) {
        delimiter = tDelim;
    } else if (typeof tDelim === 'string' && tDelim.length > 0) {
        delimiter = new RegExp(tDelim);
    } else {
        delimiter = defaultTextConfig.tokenizerDelimiter as RegExp;
    }

    const tc: TextConfig = {
        ...(defaultTextConfig as TextConfig),
        ...(config as any),
        tokenizerDelimiter: delimiter,
        log: { ...(defaultBase.log ?? {}), ...((config as any).log ?? {}) },
        useTokenizer: true,
    };

    return tc;
}
