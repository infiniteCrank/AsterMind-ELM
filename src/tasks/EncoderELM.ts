import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';

/**
 * EncoderELM: Uses an ELM to convert strings into dense feature vectors.
 */
export class EncoderELM {
    public elm: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        if (typeof config.hiddenUnits !== 'number') {
            throw new Error('EncoderELM requires config.hiddenUnits to be defined as a number');
        }
        if (!config.activation) {
            throw new Error('EncoderELM requires config.activation to be defined');
        }

        this.config = {
            ...config,
            categories: [],
            useTokenizer: true,
            log: {
                modelName: "EncoderELM",
                verbose: config.log.verbose
            },
        };

        this.elm = new ELM(this.config);

        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    /**
     * Custom training method for string → vector encoding.
     */
    train(inputStrings: string[], targetVectors: number[][]): void {
        const X: number[][] = inputStrings.map(s =>
            this.elm.encoder.normalize(this.elm.encoder.encode(s))
        );
        const Y = targetVectors;

        const hiddenUnits = this.config.hiddenUnits!;
        const inputDim = X[0].length;

        const W = this.elm['randomMatrix'](hiddenUnits, inputDim);
        const b = this.elm['randomMatrix'](hiddenUnits, 1);

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
     * Encodes an input string into a dense feature vector using the trained model.
     */
    encode(text: string): number[] {
        const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
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

        return Matrix.multiply(H, beta)[0];
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
