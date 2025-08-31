import { describe, it, expect } from 'vitest';
import { UniversalEncoder } from '../src/preprocessing/UniversalEncoder';

describe('UniversalEncoder', () => {
    it('encodes character-based input correctly', () => {
        const encoder = new UniversalEncoder({
            charSet: 'abc',
            maxLen: 3,
            mode: 'char'
        });

        const vec = encoder.encode('abc');
        expect(vec.length).toBe(9); // 3 chars * 3 one-hot size
        expect(vec.filter(v => v === 1).length).toBe(3); // One-hot positions
    });

    it('encodes with padding if input is too short', () => {
        const encoder = new UniversalEncoder({
            charSet: 'abc',
            maxLen: 5,
            mode: 'char'
        });

        const vec = encoder.encode('a');
        expect(vec.length).toBe(15); // 5 * 3
    });

    it('normalizes vector with non-zero norm', () => {
        const encoder = new UniversalEncoder({
            charSet: 'ab',
            maxLen: 2,
            mode: 'char'
        });

        const vec = encoder.encode('ab');
        const normed = encoder.normalize(vec);
        const magnitude = Math.sqrt(normed.reduce((sum, x) => sum + x * x, 0));
        expect(magnitude).toBeCloseTo(1, 5);
    });

    it('does not divide by zero when normalizing zero vector', () => {
        const encoder = new UniversalEncoder();
        const normed = encoder.normalize([0, 0, 0]);
        expect(normed).toEqual([0, 0, 0]);
    });

    it('returns correct vector size for config', () => {
        const encoder = new UniversalEncoder({
            charSet: 'abc',
            maxLen: 4,
            mode: 'char'
        });

        expect(encoder.getVectorSize()).toBe(12);
    });

    it('supports token mode via config', () => {
        const encoder = new UniversalEncoder({
            charSet: 'abc',
            maxLen: 2,
            mode: 'token',
            tokenizerDelimiter: /[\s]+/
        });

        const vec = encoder.encode('abc def');
        expect(vec.length).toBe(6); // 2 tokens * 3-char set
    });
});

// UniversalEncoder Switching

describe('UniversalEncoder Switching', () => {
    it('encodes input differently in char and token mode with consistent vector sizes', () => {
        const text = 'hello world';

        const charEncoder = new UniversalEncoder({
            charSet: 'abcdefghijklmnopqrstuvwxyz ',
            maxLen: 10,
            mode: 'char'
        });

        const tokenEncoder = new UniversalEncoder({
            charSet: 'abcdefghijklmnopqrstuvwxyz ',
            maxLen: 10,
            mode: 'token',
            tokenizerDelimiter: /\s+/
        });

        const charVec = charEncoder.encode(text);
        const tokenVec = tokenEncoder.encode(text);

        expect(charVec.length).toBe(charEncoder.getVectorSize());
        expect(tokenVec.length).toBe(tokenEncoder.getVectorSize());
        expect(charVec).not.toEqual(tokenVec); // Ensure mode has effect
    });
});
