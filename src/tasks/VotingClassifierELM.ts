// VotingClassifierELM.ts — meta-classifier that learns to combine multiple ELMs' predictions

import { ELM } from '../core/ELM';
import { ELMConfig, Activation } from '../core/ELMConfig';

export class VotingClassifierELM {
    private elm?: ELM;                   // created at train time (needs inputSize)
    private modelWeights: number[] = [];
    private categories: string[];
    private usesConfidence = false;
    private inputSize?: number;          // computed at train time

    // Keep constructor shape compatible with your existing calls
    constructor(private baseConfig: ELMConfig) {
        this.categories = (baseConfig as any).categories || ['English', 'French', 'Spanish'];
    }

    setModelWeights(weights: number[]): void {
        this.modelWeights = weights.slice();
    }

    calibrateWeights(predictionLists: string[][], trueLabels: string[]): void {
        const numModels = predictionLists.length;
        const numExamples = trueLabels.length;
        const accuracies = new Array(numModels).fill(0);

        for (let m = 0; m < numModels; m++) {
            let correct = 0;
            for (let i = 0; i < numExamples; i++) {
                if (predictionLists[m][i] === trueLabels[i]) correct++;
            }
            accuracies[m] = correct / Math.max(1, numExamples);
        }
        const total = accuracies.reduce((s, a) => s + a, 0) || 1;
        this.modelWeights = accuracies.map(a => a / total);
        if ((this.baseConfig as any)?.log?.verbose) {
            console.log('🔧 Calibrated model weights:', this.modelWeights);
        }
    }

    /** Train meta-classifier on model predictions (+ optional confidences) and true labels. */
    train(
        predictionLists: string[][],       // shape: [numModels][numExamples]
        confidenceLists: number[][] | null,
        trueLabels: string[]
    ): void {
        if (!Array.isArray(predictionLists) || predictionLists.length === 0 || !trueLabels) {
            throw new Error('VotingClassifierELM.train: invalid inputs');
        }
        const numModels = predictionLists.length;
        const numExamples = predictionLists[0].length;

        for (const list of predictionLists) {
            if (list.length !== numExamples) throw new Error('Prediction list lengths must match');
        }

        this.usesConfidence = Array.isArray(confidenceLists);
        if (this.usesConfidence) {
            if (confidenceLists!.length !== numModels) throw new Error('Confidence list count != numModels');
            for (const list of confidenceLists!) {
                if (list.length !== numExamples) throw new Error('Confidence list length mismatch');
            }
        }

        if (!this.modelWeights.length || this.modelWeights.length !== numModels) {
            this.calibrateWeights(predictionLists, trueLabels);
        }

        // Categories (target space) => from true labels
        this.categories = Array.from(new Set(trueLabels));
        const C = this.categories.length;

        // Compute numeric input size for the meta-ELM:
        // per-model features = one-hot over C + (optional) 1 confidence
        const perModel = C + (this.usesConfidence ? 1 : 0);
        this.inputSize = numModels * perModel;

        // Build X, Y
        const X: number[][] = new Array(numExamples);
        for (let i = 0; i < numExamples; i++) {
            let row: number[] = [];
            for (let m = 0; m < numModels; m++) {
                const predLabel = predictionLists[m][i];
                if (predLabel == null) throw new Error(`Invalid label at predictionLists[${m}][${i}]`);
                const w = this.modelWeights[m] ?? 1;

                // one-hot over final categories (C)
                const idx = this.categories.indexOf(predLabel);
                const oh = new Array(C).fill(0);
                if (idx >= 0) oh[idx] = 1;
                row = row.concat(oh.map(x => x * w));

                if (this.usesConfidence) {
                    const conf = confidenceLists![m][i];
                    const norm = Math.max(0, Math.min(1, Number(conf) || 0));
                    row.push(norm * w);
                }
            }
            X[i] = row;
        }

        const Y: number[][] = trueLabels.map(lbl => {
            const idx = this.categories.indexOf(lbl);
            const oh = new Array(C).fill(0);
            if (idx >= 0) oh[idx] = 1;
            return oh;
        });

        // Construct numeric ELM config now that we know inputSize
        const cfg: ELMConfig = {
            useTokenizer: false,           // numeric mode
            inputSize: this.inputSize!,
            categories: this.categories,

            hiddenUnits: (this.baseConfig as any).hiddenUnits ?? 64,
            activation: (this.baseConfig as any).activation ?? 'relu',
            ridgeLambda: (this.baseConfig as any).ridgeLambda,
            dropout: (this.baseConfig as any).dropout,
            weightInit: (this.baseConfig as any).weightInit,

            exportFileName: (this.baseConfig as any).exportFileName,
            log: {
                modelName: (this.baseConfig as any)?.log?.modelName ?? 'VotingClassifierELM',
                verbose: (this.baseConfig as any)?.log?.verbose ?? false,
                toFile: (this.baseConfig as any)?.log?.toFile ?? false,
                level: (this.baseConfig as any)?.log?.level ?? 'info',
            },
        };

        // Create (or recreate) the inner ELM with correct dims
        this.elm = new ELM(cfg);

        // Forward optional metrics gate
        if ((this.baseConfig as any).metrics) {
            (this.elm as any).metrics = (this.baseConfig as any).metrics;
        }

        // Train numerically
        this.elm.trainFromData(X, Y);
    }

    /** Predict final label from a single stacked set of model labels (+ optional confidences). */
    predict(labels: string[], confidences?: number[], topK = 1): Array<{ label: string; prob: number }> {
        if (!this.elm) throw new Error('VotingClassifierELM: call train() before predict().');
        if (!labels?.length) throw new Error('VotingClassifierELM.predict: empty labels');

        const C = this.categories.length;
        const numModels = labels.length;

        // Build numeric input row consistent with training
        let row: number[] = [];
        for (let m = 0; m < numModels; m++) {
            const w = this.modelWeights[m] ?? 1;
            const idx = this.categories.indexOf(labels[m]);
            const oh = new Array(C).fill(0);
            if (idx >= 0) oh[idx] = 1;
            row = row.concat(oh.map(x => x * w));

            if (this.usesConfidence) {
                const norm = Math.max(0, Math.min(1, Number(confidences?.[m]) || 0));
                row.push(norm * w);
            }
        }

        const [res] = this.elm.predictFromVector([row], topK);
        return res;
    }

    loadModelFromJSON(json: string): void {
        if (!this.elm) this.elm = new ELM({
            // minimal placeholder; will be overwritten by fromJSON content
            useTokenizer: false,
            inputSize: 1,
            categories: ['_tmp'],
            hiddenUnits: 1,
            activation: 'relu',
            log: { modelName: 'VotingClassifierELM' },
        } as ELMConfig);
        this.elm.loadModelFromJSON(json);
        // Try to recover categories & inputSize from loaded model
        this.categories = (this.elm as any).categories ?? this.categories;
        this.inputSize = ((this.elm as any).model?.W?.[0]?.length) ?? this.inputSize;
    }

    saveModelAsJSONFile(filename?: string): void {
        if (!this.elm) throw new Error('VotingClassifierELM: no model to save.');
        this.elm.saveModelAsJSONFile(filename);
    }
}
