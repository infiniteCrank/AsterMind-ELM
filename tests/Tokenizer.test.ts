import { describe, it, expect } from 'vitest';
import { Tokenizer } from '../src/preprocessing/Tokenizer';

describe('Tokenizer', () => {
    it('tokenizes text using default delimiter', () => {
        const tokenizer = new Tokenizer();
        const input = 'Hello, world! This is a test.';
        const tokens = tokenizer.tokenize(input);
        expect(tokens).toEqual(['hello', 'world', 'this', 'is', 'a', 'test']);
    });

    it('tokenizes using custom delimiter', () => {
        const tokenizer = new Tokenizer(/[\s-]+/);
        const input = 'word1-word2 word3';
        const tokens = tokenizer.tokenize(input);
        expect(tokens).toEqual(['word1', 'word2', 'word3']);
    });

    it('removes empty tokens from result', () => {
        const tokenizer = new Tokenizer(/[\s,]+/);
        const input = 'hello,,world';
        const tokens = tokenizer.tokenize(input);
        expect(tokens).toEqual(['hello', 'world']);
    });

    it('trims and lowercases input before splitting', () => {
        const tokenizer = new Tokenizer();
        const input = '  TeStIng INPUT  ';
        const tokens = tokenizer.tokenize(input);
        expect(tokens).toEqual(['testing', 'input']);
    });

    it('returns empty array when input is empty or whitespace', () => {
        const tokenizer = new Tokenizer();
        expect(tokenizer.tokenize('')).toEqual([]);
        expect(tokenizer.tokenize('    ')).toEqual([]);
    });

    it('generates bigrams (n=2) correctly', () => {
        const tokenizer = new Tokenizer();
        const tokens = ['hello', 'world', 'token'];
        const bigrams = tokenizer.ngrams(tokens, 2);
        expect(bigrams).toEqual(['hello world', 'world token']);
    });

    it('generates trigrams (n=3) correctly', () => {
        const tokenizer = new Tokenizer();
        const tokens = ['a', 'b', 'c', 'd'];
        const trigrams = tokenizer.ngrams(tokens, 3);
        expect(trigrams).toEqual(['a b c', 'b c d']);
    });

    it('returns empty array for ngrams when n is 0 or greater than length', () => {
        const tokenizer = new Tokenizer();
        const tokens = ['a', 'b'];
        expect(tokenizer.ngrams(tokens, 0)).toEqual([]);
        expect(tokenizer.ngrams(tokens, 3)).toEqual([]);
    });
});
