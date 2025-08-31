// Activations.ts - Common activation functions

export class Activations {
    static relu(x: number): number {
        return Math.max(0, x);
    }

    static leakyRelu(x: number, alpha = 0.01): number {
        return x >= 0 ? x : alpha * x;
    }

    static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    static tanh(x: number): number {
        return Math.tanh(x);
    }

    static softmax(arr: number[]): number[] {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    static apply(matrix: number[][], fn: (x: number) => number): number[][] {
        return matrix.map(row => row.map(fn));
    }

    static get(name: string): (x: number) => number {
        switch (name.toLowerCase()) {
            case 'relu': return this.relu;
            case 'leakyrelu': return x => this.leakyRelu(x);
            case 'sigmoid': return this.sigmoid;
            case 'tanh': return this.tanh;
            default: throw new Error(`Unknown activation: ${name}`);
        }
    }
}
