// TextEncoder.ts - Text preprocessing and one-hot encoding for ELM

import { Tokenizer } from './Tokenizer';

export interface TextEncoderConfig {
    charSet?: string;
    maxLen?: number;
    useTokenizer?: boolean;
    tokenizerDelimiter?: RegExp;
}

const defaultTextEncoderConfig: Required<Omit<TextEncoderConfig, 'tokenizerDelimiter'>> = {
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    maxLen: 15,
    useTokenizer: false
};

export class TextEncoder {
    private charSet: string;
    private charSize: number;
    private maxLen: number;
    private useTokenizer: boolean;
    private tokenizer?: Tokenizer;

    constructor(config: TextEncoderConfig = {}) {
        const cfg = { ...defaultTextEncoderConfig, ...config };
        this.charSet = cfg.charSet;
        this.charSize = cfg.charSet.length;
        this.maxLen = cfg.maxLen;
        this.useTokenizer = cfg.useTokenizer;
        if (this.useTokenizer) {
            this.tokenizer = new Tokenizer(config.tokenizerDelimiter);
        }
    }

    public charToOneHot(c: string): number[] {
        const index = this.charSet.indexOf(c.toLowerCase());
        const vec = Array(this.charSize).fill(0);
        if (index !== -1) vec[index] = 1;
        return vec;
    }

    public textToVector(text: string): number[] {
        let cleaned: string;

        if (this.useTokenizer && this.tokenizer) {
            const tokens = this.tokenizer.tokenize(text).join('');
            cleaned = tokens.slice(0, this.maxLen).padEnd(this.maxLen, ' ');
        } else {
            cleaned = text.toLowerCase().replace(new RegExp(`[^${this.charSet}]`, 'g'), '').padEnd(this.maxLen, ' ').slice(0, this.maxLen);
        }

        const vec: number[] = [];
        for (let i = 0; i < cleaned.length; i++) {
            vec.push(...this.charToOneHot(cleaned[i]));
        }
        return vec;
    }

    public normalizeVector(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
        return norm > 0 ? v.map(x => x / norm) : v;
    }

    public getVectorSize(): number {
        return this.charSize * this.maxLen;
    }

    public getCharSet(): string {
        return this.charSet;
    }

    public getMaxLen(): number {
        return this.maxLen;
    }
} 
