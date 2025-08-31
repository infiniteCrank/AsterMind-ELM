// ✅ AutoComplete.ts patched to support (input, label) training and evaluation

import { ELM } from '../core/ELM';
import { bindAutocompleteUI } from '../ui/components/BindUI';
import { EnglishTokenPreset } from '../config/Presets';
import { Matrix } from '../core/Matrix'
import { Activations } from '../core/Activations'

export interface TrainPair {
    input: string;
    label: string;
}

interface AutoCompleteOptions {
    activation?: string;
    inputElement: HTMLInputElement;
    outputElement: HTMLElement;
    topK?: number;
    metrics?: { rmse?: number; mae?: number; accuracy?: number; top1Accuracy?: number; crossEntropy?: number };
    verbose?: boolean;
    exportFileName?: string;
    augmentationOptions?: {
        suffixes?: string[];
        prefixes?: string[];
        includeNoise?: boolean;
    };
}

export class AutoComplete {
    public elm: ELM;
    private trainPairs: TrainPair[];

    public activation: string;

    constructor(pairs: TrainPair[], options: AutoCompleteOptions) {
        this.trainPairs = pairs;
        this.activation = options.activation ?? 'relu';

        const categories = Array.from(new Set(pairs.map(p => p.label)));

        this.elm = new ELM({
            ...EnglishTokenPreset,
            categories,
            activation: this.activation,
            metrics: options.metrics,
            log: {
                modelName: "AutoComplete",
                verbose: options.verbose
            },
            exportFileName: options.exportFileName,
        });

        bindAutocompleteUI({
            model: this.elm,
            inputElement: options.inputElement,
            outputElement: options.outputElement,
            topK: options.topK
        });
    }

    public train(): void {
        const X: number[][] = [];
        const Y: number[][] = [];

        for (const { input, label } of this.trainPairs) {
            const vec = this.elm.encoder.normalize(this.elm.encoder.encode(input));
            const labelIndex = this.elm.categories.indexOf(label);
            if (labelIndex === -1) continue;
            X.push(vec);
            Y.push(this.elm.oneHot(this.elm.categories.length, labelIndex));
        }

        this.elm.trainFromData(X, Y);
    }

    predict(input: string, topN = 1): { completion: string; prob: number }[] {
        return this.elm.predict(input, topN).map(p => ({
            completion: p.label,
            prob: p.prob
        }));
    }

    public getModel(): ELM {
        return this.elm;
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    public top1Accuracy(pairs: TrainPair[]): number {
        let correct = 0;
        for (const { input, label } of pairs) {
            const [pred] = this.predict(input, 1);
            if (pred?.completion?.toLowerCase().trim() === label.toLowerCase().trim()) {
                correct++;
            }
        }
        return correct / pairs.length;
    }

    public crossEntropy(pairs: TrainPair[]): number {
        let totalLoss = 0;
        for (const { input, label } of pairs) {
            const preds = this.predict(input, 5);
            const match = preds.find(p => p.completion.toLowerCase().trim() === label.toLowerCase().trim());
            const prob = match?.prob ?? 1e-6;
            totalLoss += -Math.log(prob);  // ⬅ switched from log2 to natural log
        }
        return totalLoss / pairs.length;
    }

    public internalCrossEntropy(verbose: boolean = false): number {
        const { model, encoder, categories } = this.elm;
        if (!model) {
            if (verbose) console.warn("⚠️ Cannot compute internal cross-entropy: model not trained.");
            return Infinity;
        }

        const X: number[][] = [];
        const Y: number[][] = [];

        for (const { input, label } of this.trainPairs) {
            const vec = encoder.normalize(encoder.encode(input));
            const labelIdx = categories.indexOf(label);
            if (labelIdx === -1) continue;
            X.push(vec);
            Y.push(this.elm.oneHot(categories.length, labelIdx));
        }

        const { W, b, beta } = model;
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const preds = Matrix.multiply(H, beta);
        const ce = this.elm.calculateCrossEntropy(Y, preds);

        if (verbose) {
            console.log(`📏 Internal Cross-Entropy (full model eval): ${ce.toFixed(4)}`);
        }

        return ce;
    }

}
