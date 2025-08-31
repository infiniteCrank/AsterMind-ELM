import { describe, it, expect } from 'vitest';
import { Matrix } from '../src/core/Matrix';

describe('Matrix', () => {
    it('multiply() computes correct matrix product', () => {
        const A = [
            [1, 2],
            [3, 4]
        ];
        const B = [
            [5, 6],
            [7, 8]
        ];
        const result = Matrix.multiply(A, B);
        expect(result).toEqual([
            [19, 22],
            [43, 50]
        ]);
    });

    it('transpose() returns the transposed matrix', () => {
        const A = [
            [1, 2, 3],
            [4, 5, 6]
        ];
        const result = Matrix.transpose(A);
        expect(result).toEqual([
            [1, 4],
            [2, 5],
            [3, 6]
        ]);
    });

    it('identity() returns an identity matrix of correct size', () => {
        const I = Matrix.identity(3);
        expect(I).toEqual([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]);
    });

    it('addRegularization() correctly adds lambda to diagonal', () => {
        const A = [
            [1, 2],
            [3, 4]
        ];
        const result = Matrix.addRegularization(A, 0.1);
        expect(result).toEqual([
            [1.1, 2],
            [3, 4.1]
        ]);
    });

    it('inverse() correctly inverts a simple matrix', () => {
        const A = [
            [4, 7],
            [2, 6]
        ];
        const inv = Matrix.inverse(A);

        const expected = [
            [0.6, -0.7],
            [-0.2, 0.4]
        ];

        for (let i = 0; i < expected.length; i++) {
            for (let j = 0; j < expected[i].length; j++) {
                expect(inv[i][j]).toBeCloseTo(expected[i][j]);
            }
        }
    });

    it('inverse() throws on singular matrix', () => {
        const singular = [
            [1, 2],
            [2, 4]
        ];
        expect(() => Matrix.inverse(singular)).toThrow(/singular/i);
    });

    it('multiply() handles identity matrix correctly', () => {
        const A = [
            [5, 6],
            [7, 8]
        ];
        const I = Matrix.identity(2);
        expect(Matrix.multiply(A, I)).toEqual(A);
        expect(Matrix.multiply(I, A)).toEqual(A);
    });
});
