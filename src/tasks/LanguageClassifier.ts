import { ELM } from '../core/ELM';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';
import { ELMConfig } from '../core/ELMConfig';
import { IO, LabeledExample } from '../utils/IO';

export class LanguageClassifier {
    private elm: ELM;
    private config: ELMConfig;
    private trainSamples: Record<string, string[]> = {};

    constructor(config: ELMConfig) {
        this.config = {
            ...config,
            log: {
                modelName: "IntentClassifier",
                verbose: config.log.verbose
            },
        };
        this.elm = new ELM(config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    loadTrainingData(raw: string, format: 'json' | 'csv' | 'tsv' = 'json'): LabeledExample[] {
        switch (format) {
            case 'csv':
                return IO.importCSV(raw);
            case 'tsv':
                return IO.importTSV(raw);
            case 'json':
            default:
                return IO.importJSON(raw);
        }
    }

    train(data: LabeledExample[]): void {
        const categories = [...new Set(data.map(d => d.label))];
        this.elm.setCategories(categories);
        data.forEach(({ text, label }) => {
            if (!this.trainSamples[label]) this.trainSamples[label] = [];
            this.trainSamples[label].push(text);
        });
        this.elm.train();
    }

    predict(text: string, topK = 3) {
        return this.elm.predict(text, topK);
    }

    /**
     * Train the classifier using already-encoded vectors.
     * Each vector must be paired with its label.
     */
    trainVectors(data: { vector: number[]; label: string }[]) {
        const categories = [...new Set(data.map(d => d.label))];
        this.elm.setCategories(categories);

        const X: number[][] = data.map(d => d.vector);
        const Y: number[][] = data.map(d =>
            this.elm.oneHot(categories.length, categories.indexOf(d.label))
        );

        const W = this.elm['randomMatrix'](this.config.hiddenUnits!, X[0].length);
        const b = this.elm['randomMatrix'](this.config.hiddenUnits!, 1);
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);
        const H_pinv = this.elm['pseudoInverse'](H);
        const beta = Matrix.multiply(H_pinv, Y);

        this.elm['model'] = { W, b, beta };
    }

    /**
     * Predict language directly from a dense vector representation.
     */
    predictFromVector(vec: number[], topK = 1) {
        const model = this.elm['model'];
        if (!model) {
            throw new Error('EncoderELM model has not been trained yet.');
        }

        const { W, b, beta } = model;
        const tempH = Matrix.multiply([vec], Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const rawOutput = Matrix.multiply(H, beta)[0];
        const probs = Activations.softmax(rawOutput);

        return probs
            .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}