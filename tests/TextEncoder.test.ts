import { describe, it, expect } from 'vitest';
import { TextEncoder } from '../src/preprocessing/TextEncoder';

describe('TextEncoder', () => {
    const defaultCharSet = 'abc';
    const maxLen = 5;

    it('charToOneHot() correctly encodes known characters', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet });
        expect(encoder.charToOneHot('a')).toEqual([1, 0, 0]);
        expect(encoder.charToOneHot('b')).toEqual([0, 1, 0]);
        expect(encoder.charToOneHot('c')).toEqual([0, 0, 1]);
    });

    it('charToOneHot() returns zero vector for unknown character', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet });
        expect(encoder.charToOneHot('z')).toEqual([0, 0, 0]);
    });

    it('textToVector() encodes and pads lowercase text correctly', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet, maxLen });
        const vec = encoder.textToVector('abc');
        const expected = [
            1, 0, 0,  // a
            0, 1, 0,  // b
            0, 0, 1,  // c
            0, 0, 0,  // padding
            0, 0, 0   // padding
        ];
        expect(vec).toEqual(expected);
    });

    it('textToVector() ignores invalid characters and truncates long input', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet, maxLen: 2 });
        const vec = encoder.textToVector('azb');
        expect(vec).toEqual([
            1, 0, 0, // a
            0, 1, 0  // b
        ]);
    });

    it('normalizeVector() normalizes vectors to unit length', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet });
        const vec = [3, 4]; // magnitude = 5
        const normalized = encoder.normalizeVector(vec);
        expect(normalized[0]).toBeCloseTo(0.6);
        expect(normalized[1]).toBeCloseTo(0.8);
    });

    it('normalizeVector() returns zero vector when norm is zero', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet });
        expect(encoder.normalizeVector([0, 0, 0])).toEqual([0, 0, 0]);
    });

    it('getVectorSize() returns correct total length', () => {
        const encoder = new TextEncoder({ charSet: defaultCharSet, maxLen: 4 });
        expect(encoder.getVectorSize()).toBe(12);
    });

    it('getCharSet() and getMaxLen() return expected values', () => {
        const encoder = new TextEncoder({ charSet: 'xyz', maxLen: 7 });
        expect(encoder.getCharSet()).toBe('xyz');
        expect(encoder.getMaxLen()).toBe(7);
    });

    it('textToVector() uses tokenizer when enabled', () => {
        const encoder = new TextEncoder({
            charSet: 'abc',
            maxLen: 2,
            useTokenizer: true,
            tokenizerDelimiter: /\s+/
        });
        const vec = encoder.textToVector('a b');
        expect(vec.length).toBeGreaterThan(0);
    });
});
