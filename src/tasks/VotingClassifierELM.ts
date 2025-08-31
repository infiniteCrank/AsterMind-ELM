import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';

/**
 * VotingClassifierELM takes predictions from multiple ELMs
 * and learns to choose the most accurate final label.
 * It can optionally incorporate confidence scores and calibrate model weights.
 */
export class VotingClassifierELM {
    private elm: ELM;
    private modelWeights: number[];
    private categories: string[];

    constructor(config: ELMConfig) {
        this.categories = config.categories || ['English', 'French', 'Spanish'];
        this.modelWeights = [];

        this.elm = new ELM({
            ...config,
            useTokenizer: false,
            categories: this.categories,
            log: {
                modelName: "IntentClassifier",
                verbose: config.log.verbose
            },
        });

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    setModelWeights(weights: number[]): void {
        this.modelWeights = weights;
    }

    calibrateWeights(predictionLists: string[][], trueLabels: string[]): void {
        const numModels = predictionLists.length;
        const numExamples = trueLabels.length;
        const accuracies = new Array(numModels).fill(0);

        for (let m = 0; m < numModels; m++) {
            let correct = 0;
            for (let i = 0; i < numExamples; i++) {
                if (predictionLists[m][i] === trueLabels[i]) {
                    correct++;
                }
            }
            accuracies[m] = correct / numExamples;
        }

        const total = accuracies.reduce((sum, acc) => sum + acc, 0) || 1;
        this.modelWeights = accuracies.map(a => a / total);

        console.log('🔧 Calibrated model weights based on accuracy:', this.modelWeights);
    }

    train(
        predictionLists: string[][],
        confidenceLists: number[][] | null,
        trueLabels: string[]
    ): void {
        if (!Array.isArray(predictionLists) || predictionLists.length === 0 || !trueLabels) {
            throw new Error('Invalid inputs to VotingClassifierELM.train');
        }

        const numModels = predictionLists.length;
        const numExamples = predictionLists[0].length;

        for (let list of predictionLists) {
            if (list.length !== numExamples) {
                throw new Error('Inconsistent prediction lengths across models');
            }
        }

        if (confidenceLists) {
            if (confidenceLists.length !== numModels) {
                throw new Error('Confidence list count must match number of models');
            }
            for (let list of confidenceLists) {
                if (list.length !== numExamples) {
                    throw new Error('Inconsistent confidence lengths across models');
                }
            }
        }

        if (!this.modelWeights || this.modelWeights.length !== numModels) {
            this.calibrateWeights(predictionLists, trueLabels);
        }

        const inputs: number[][] = [];
        for (let i = 0; i < numExamples; i++) {
            let inputRow: number[] = [];
            for (let m = 0; m < numModels; m++) {
                const label = predictionLists[m][i];
                if (typeof label === 'undefined') {
                    console.error(`Undefined label from model ${m} at index ${i}`);
                    throw new Error(`Invalid label in predictionLists[${m}][${i}]`);
                }
                const weight = this.modelWeights[m];
                inputRow = inputRow.concat(this.oneHot(label).map(x => x * weight));

                if (confidenceLists) {
                    const conf = confidenceLists[m][i];
                    const normalizedConf = Math.min(1, Math.max(0, conf));
                    inputRow.push(normalizedConf * weight);
                }
            }
            inputs.push(inputRow);
        }

        const examples = inputs.map((input, i) => ({ input, label: trueLabels[i] }));
        console.log(`📊 VotingClassifierELM training on ${examples.length} examples with ${numModels} models.`);
        this.elm.train(examples as any);
    }

    predict(labels: string[], confidences?: number[]): { label: string; prob: number }[] {
        if (!Array.isArray(labels) || labels.length === 0) {
            throw new Error('No labels provided to VotingClassifierELM.predict');
        }

        let input: number[] = [];
        for (let i = 0; i < labels.length; i++) {
            const weight = this.modelWeights[i] || 1;
            input = input.concat(this.oneHot(labels[i]).map(x => x * weight));
            if (confidences && typeof confidences[i] === 'number') {
                const norm = Math.min(1, Math.max(0, confidences[i]));
                input.push(norm * weight);
            }
        }

        return this.elm.predict(JSON.stringify(input), 1);
    }

    private oneHot(label: string): number[] {
        const index = this.categories.indexOf(label);
        if (index === -1) {
            console.warn(`Unknown label in oneHot: ${label}`);
            return new Array(this.categories.length).fill(0);
        }
        return this.categories.map((_, i) => (i === index ? 1 : 0));
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
