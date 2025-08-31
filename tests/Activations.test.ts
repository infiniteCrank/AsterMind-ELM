import { describe, it, expect } from 'vitest';
import { Activations } from '../src/core/Activations';

describe('Activations', () => {
    it('relu works correctly', () => {
        expect(Activations.relu(2)).toBe(2);
        expect(Activations.relu(-5)).toBe(0);
    });

    it('leakyRelu works correctly', () => {
        expect(Activations.leakyRelu(3)).toBe(3);
        expect(Activations.leakyRelu(-3)).toBeCloseTo(-0.03);
        expect(Activations.leakyRelu(-3, 0.1)).toBeCloseTo(-0.3);
    });

    it('sigmoid outputs values between 0 and 1', () => {
        expect(Activations.sigmoid(0)).toBeCloseTo(0.5);
        expect(Activations.sigmoid(100)).toBeCloseTo(1);
        expect(Activations.sigmoid(-100)).toBeCloseTo(0);
    });

    it('tanh outputs values between -1 and 1', () => {
        expect(Activations.tanh(0)).toBeCloseTo(0);
        expect(Activations.tanh(100)).toBeCloseTo(1);
        expect(Activations.tanh(-100)).toBeCloseTo(-1);
    });

    it('softmax produces a valid probability distribution', () => {
        const input = [2.0, 1.0, 0.1];
        const output = Activations.softmax(input);
        const sum = output.reduce((acc, val) => acc + val, 0);
        expect(output.length).toBe(input.length);
        expect(sum).toBeCloseTo(1, 5);
        output.forEach(val => expect(val).toBeGreaterThan(0));
    });

    it('apply correctly maps activation across matrix', () => {
        const matrix = [[-1, 2], [3, -4]];
        const result = Activations.apply(matrix, Activations.relu);
        expect(result).toEqual([[0, 2], [3, 0]]);
    });

    it('get returns correct function', () => {
        expect(Activations.get('relu')).toBe(Activations.relu);
        expect(Activations.get('sigmoid')).toBe(Activations.sigmoid);
        expect(Activations.get('tanh')).toBe(Activations.tanh);
        expect(Activations.get('leakyrelu')(-10)).toBeCloseTo(-0.1);
    });

    it('get throws for unknown activation', () => {
        expect(() => Activations.get('unknown')).toThrow();
    });
});
