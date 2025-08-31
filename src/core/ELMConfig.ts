// ELMConfig.ts - Configuration interface and defaults for ELM-based models

export interface ELMConfig {
    categories: string[];
    hiddenUnits?: number;
    maxLen?: number;
    activation?: string;
    encoder?: any; // Optional: if you're passing a pre-built UniversalEncoder instance

    // Preprocessing options
    charSet?: string;
    useTokenizer?: boolean;
    tokenizerDelimiter?: RegExp;

    // File export
    exportFileName?: string;

    // Model evaluation
    metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
    };

    // Logging
    log: {
        modelName?: string,
        verbose?: boolean,
        toFile?: boolean,
    }
    logFileName?: string,
    dropout?: number;
    weightInit?: "uniform" | "xavier";
}

export const defaultConfig: Required<Pick<ELMConfig, 'hiddenUnits' | 'maxLen' | 'activation' | 'charSet' | 'useTokenizer' | 'tokenizerDelimiter' | 'weightInit'>> = {
    hiddenUnits: 50,
    maxLen: 30,
    weightInit: "uniform",
    activation: 'relu',
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    useTokenizer: false,
    tokenizerDelimiter: /\s+/,
};
