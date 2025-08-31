import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';

export class FeatureCombinerELM {
    private elm: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        if (typeof config.hiddenUnits !== 'number') {
            throw new Error('FeatureCombinerELM requires hiddenUnits');
        }
        if (!config.activation) {
            throw new Error('FeatureCombinerELM requires activation');
        }

        this.config = {
            ...config,
            categories: [],
            useTokenizer: false, // this ELM takes numeric vectors
            log: {
                modelName: "FeatureCombinerELM",
                verbose: config.log.verbose
            },
        };

        this.elm = new ELM(this.config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    /**
     * Combines encoder vector and metadata into one input vector
     */
    static combineFeatures(encodedVec: number[], meta: number[]): number[] {
        return [...encodedVec, ...meta];
    }

    /**
     * Train the ELM using combined features and labels
     */
    train(encoded: number[][], metas: number[][], labels: string[]): void {
        if (!this.config.hiddenUnits || !this.config.activation) {
            throw new Error("FeatureCombinerELM: config.hiddenUnits or activation is undefined.");
        }

        const X = encoded.map((vec, i) =>
            FeatureCombinerELM.combineFeatures(vec, metas[i])
        );

        const categories = [...new Set(labels)];
        this.elm.setCategories(categories);

        const Y = labels.map(label =>
            this.elm.oneHot(categories.length, categories.indexOf(label))
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
     * Predict from combined input and metadata
     */
    predict(encodedVec: number[], meta: number[], topK = 1) {
        const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
        const [results] = this.elm.predictFromVector(input, topK);
        return results;
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
