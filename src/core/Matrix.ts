export class Matrix {
    constructor(public data: number[][]) { }

    static multiply(A: number[][], B: number[][]): number[][] {
        const result: number[][] = [];
        for (let i = 0; i < A.length; i++) {
            result[i] = [];
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < B.length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    static transpose(A: number[][]): number[][] {
        return A[0].map((_, i) => A.map(row => row[i]));
    }

    static identity(size: number): number[][] {
        return Array.from({ length: size }, (_, i) =>
            Array.from({ length: size }, (_, j) => (i === j ? 1 : 0))
        );
    }

    static addRegularization(A: number[][], lambda: number): number[][] {
        return A.map((row, i) =>
            row.map((val, j) => val + (i === j ? lambda : 0))
        );
    }

    static inverse(A: number[][]): number[][] {
        const n = A.length;
        const I = Matrix.identity(n);
        const M = A.map(row => [...row]);

        for (let i = 0; i < n; i++) {
            let maxEl = Math.abs(M[i][i]);
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(M[k][i]) > maxEl) {
                    maxEl = Math.abs(M[k][i]);
                    maxRow = k;
                }
            }

            [M[i], M[maxRow]] = [M[maxRow], M[i]];
            [I[i], I[maxRow]] = [I[maxRow], I[i]];

            const div = M[i][i];
            if (div === 0) throw new Error("Matrix is singular and cannot be inverted");

            for (let j = 0; j < n; j++) {
                M[i][j] /= div;
                I[i][j] /= div;
            }

            for (let k = 0; k < n; k++) {
                if (k === i) continue;
                const factor = M[k][i];
                for (let j = 0; j < n; j++) {
                    M[k][j] -= factor * M[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }

        return I;
    }

    static random(rows: number, cols: number, min: number, max: number): Matrix {
        const data: number[][] = [];
        for (let i = 0; i < rows; i++) {
            const row: number[] = [];
            for (let j = 0; j < cols; j++) {
                row.push(Math.random() * (max - min) + min);
            }
            data.push(row);
        }
        return new Matrix(data);
    }

    static fromArray(array: number[][]): Matrix {
        return new Matrix(array);
    }

    toArray(): number[][] {
        return this.data;
    }
}
