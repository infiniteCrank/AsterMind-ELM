import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { Matrix } from '../core/Matrix';
import { Activations } from '../core/Activations';

export class CharacterLangEncoderELM {
    private elm: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        if (!config.hiddenUnits || !config.activation) {
            throw new Error("CharacterLangEncoderELM requires defined hiddenUnits and activation");
        }

        this.config = {
            ...config,
            log: {
                modelName: "CharacterLangEncoderELM",
                verbose: config.log.verbose
            },
            useTokenizer: true
        };

        this.elm = new ELM(this.config);

        // Forward ELM-specific options
        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    train(inputStrings: string[], labels: string[]) {
        const categories = [...new Set(labels)];
        this.elm.setCategories(categories);
        this.elm.train(); // assumes encoder + categories are set
    }

    /**
     * Returns dense vector (embedding) rather than label prediction
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

        // dense feature vector
        return Matrix.multiply(H, beta)[0];
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
} 
