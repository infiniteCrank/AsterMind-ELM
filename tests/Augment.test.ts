// tests/Augment.test.ts

import { describe, it, expect, vi } from 'vitest';
import { Augment } from '../src/utils/Augment';

describe('Augment', () => {
    it('adds suffixes correctly', () => {
        const result = Augment.addSuffix('hello', ['world', 'there']);
        expect(result).toEqual(['hello world', 'hello there']);
    });

    it('adds prefixes correctly', () => {
        const result = Augment.addPrefix('world', ['hello', 'hey']);
        expect(result).toEqual(['hello world', 'hey world']);
    });

    it('adds noise to text with mocked randomness', () => {
        vi.spyOn(Math, 'random').mockImplementation(() => 0.05); // always below noiseRate
        vi.spyOn(Math, 'floor').mockImplementation(() => 1); // choose index 1 from charset
        const result = Augment.addNoise('abc', 'xyz', 0.1);
        expect(result).toEqual('yyy'); // because randomChar = 'y'
        vi.restoreAllMocks();
    });

    it('returns mixed phrases', () => {
        const result = Augment.mix('hello', ['world', 'again']);
        expect(result).toEqual(['hello world', 'hello again']);
    });

    it('generates variants with suffix and prefix', () => {
        const result = Augment.generateVariants('hi', 'abc', {
            suffixes: ['there'],
            prefixes: ['yo'],
        });
        expect(result).toEqual(['hi', 'hi there', 'yo hi']);
    });

    it('generates variants including noise', () => {
        vi.spyOn(Math, 'random').mockImplementation(() => 0.01);
        vi.spyOn(Math, 'floor').mockImplementation(() => 0); // choose first char of charSet
        const result = Augment.generateVariants('test', 'z', { includeNoise: true });
        expect(result.length).toBe(2); // original + noisy
        expect(result[1]).toMatch(/^z+$/);
        vi.restoreAllMocks();
    });

    it('returns only base text if no options provided', () => {
        const result = Augment.generateVariants('plain', 'abc');
        expect(result).toEqual(['plain']);
    });
});

// Augment

describe('Augment utilities', () => {
    it('adds suffixes', () => {
        const variants = Augment.addSuffix('hi', ['there']);
        expect(variants).toContain('hi there');
    });

    it('adds prefixes', () => {
        const variants = Augment.addPrefix('hi', ['yo']);
        expect(variants).toContain('yo hi');
    });

    it('adds noise', () => {
        const noisy = Augment.addNoise('hi', 'xyz', 1);
        expect(noisy).not.toBe('hi');
    });

    it('mixes text with mixins', () => {
        const mixed = Augment.mix('hi', ['there']);
        expect(mixed).toContain('hi there');
    });

    it('generates multiple variants', () => {
        const all = Augment.generateVariants('hi', 'abc', { suffixes: ['you'], includeNoise: true });
        expect(all.length).toBeGreaterThan(1);
    });
});