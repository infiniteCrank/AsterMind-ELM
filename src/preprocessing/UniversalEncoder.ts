// UniversalEncoder.ts - Automatically selects appropriate encoder (char or token based)

import { TextEncoder, TextEncoderConfig } from './TextEncoder';

export interface UniversalEncoderConfig extends TextEncoderConfig {
    mode?: 'char' | 'token';
}

const defaultUniversalConfig: Required<Omit<UniversalEncoderConfig, 'tokenizerDelimiter'>> = {
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    maxLen: 15,
    useTokenizer: false,
    mode: 'char'
};

export class UniversalEncoder {
    private encoder: TextEncoder;

    constructor(config: UniversalEncoderConfig = {}) {
        const merged = { ...defaultUniversalConfig, ...config };
        const useTokenizer = merged.mode === 'token';

        this.encoder = new TextEncoder({
            charSet: merged.charSet,
            maxLen: merged.maxLen,
            useTokenizer,
            tokenizerDelimiter: config.tokenizerDelimiter
        });
    }

    public encode(text: string): number[] {
        return this.encoder.textToVector(text);
    }

    public normalize(v: number[]): number[] {
        return this.encoder.normalizeVector(v);
    }

    public getVectorSize(): number {
        return this.encoder.getVectorSize();
    }
} 
