import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { FeatureCombinerELM } from './FeatureCombinerELM';

/**
 * ConfidenceClassifierELM is a lightweight ELM wrapper
 * designed to classify whether an input prediction is likely to be high or low confidence.
 * It uses the same input format as FeatureCombinerELM (vector + meta).
 */
export class ConfidenceClassifierELM {
    private elm: ELM;

    constructor(private config: ELMConfig) {
        this.elm = new ELM({
            ...config,
            categories: ['low', 'high'],
            useTokenizer: false,
            log: {
                modelName: "ConfidenceClassifierELM",
                verbose: config.log.verbose
            },
        });

        // Forward optional ELM config extensions
        if (config.metrics) this.elm.metrics = config.metrics;
        if (config.exportFileName) this.elm.config.exportFileName = config.exportFileName;
    }

    train(vectors: number[][], metas: number[][], labels: string[]): void {
        const inputs = vectors.map((vec, i) =>
            FeatureCombinerELM.combineFeatures(vec, metas[i])
        );
        const examples = vectors.map((vec, i) => ({
            input: FeatureCombinerELM.combineFeatures(vec, metas[i]),
            label: labels[i]
        }));

        this.elm.train(examples as any);
    }

    predict(vec: number[], meta: number[]): { label: string; prob: number }[] {
        const input = FeatureCombinerELM.combineFeatures(vec, meta);
        const inputStr = JSON.stringify(input);
        return this.elm.predict(inputStr, 1);
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }
}
