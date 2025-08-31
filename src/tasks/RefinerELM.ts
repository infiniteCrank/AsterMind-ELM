import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';

export class RefinerELM {
    private elm: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        this.config = {
            ...config,
            useTokenizer: false,
            categories: [],
            log: {
                modelName: "IntentClassifier",
                verbose: config.log.verbose
            },
        };

        this.elm = new ELM(this.config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    train(inputs: number[][], labels: string[]) {
        const categories = [...new Set(labels)];
        this.elm.setCategories(categories);

        const Y = labels.map(label =>
            this.elm.oneHot(categories.length, categories.indexOf(label))
        );

        const W = this.elm['randomMatrix'](this.config.hiddenUnits!, inputs[0].length);
        const b = this.elm['randomMatrix'](this.config.hiddenUnits!, 1);
        const tempH = Matrix.multiply(inputs, Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);
        const H_pinv = this.elm['pseudoInverse'](H);
        const beta = Matrix.multiply(H_pinv, Y);

        this.elm['model'] = { W, b, beta };
    }

    predict(vec: number[]): { label: string; prob: number }[] {
        const input = [vec];
        const model = this.elm['model'];
        if (!model) {
            throw new Error('EncoderELM model has not been trained yet.');
        }

        const { W, b, beta } = model;
        const tempH = Matrix.multiply(input, Matrix.transpose(W));
        const activationFn = Activations.get(this.config.activation!);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);
        const rawOutput = Matrix.multiply(H, beta)[0];
        const probs = Activations.softmax(rawOutput);

        return probs
            .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob);
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
