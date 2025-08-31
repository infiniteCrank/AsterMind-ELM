// intentClassifier.ts - ELM-based intent classification engine

import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { PredictResult } from '../core/ELM';

export class IntentClassifier {
    private model: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        this.config = {
            ...config,
            log: {
                modelName: "IntentClassifier",
                verbose: config.log.verbose
            },
        };
        this.model = new ELM(config);

        if (config.metrics) this.model.metrics = config.metrics;
        if (config.exportFileName) this.model.config.exportFileName = config.exportFileName;
    }

    public train(textLabelPairs: { text: string, label: string }[], augmentationOptions?: {
        suffixes?: string[];
        prefixes?: string[];
        includeNoise?: boolean;
    }): void {
        const labelSet = Array.from(new Set(textLabelPairs.map(p => p.label)));
        this.model.setCategories(labelSet);
        this.model.train(augmentationOptions);
    }

    public predict(text: string, topK = 1, threshold = 0): PredictResult[] {
        return this.model.predict(text, topK).filter(r => r.prob >= threshold);
    }

    public predictBatch(texts: string[], topK = 1, threshold = 0): PredictResult[][] {
        return texts.map(text => this.predict(text, topK, threshold));
    }

    private oneHot(n: number, index: number): number[] {
        return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
    }

    public loadModelFromJSON(json: string): void {
        this.model.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.model.saveModelAsJSONFile(filename);
    }
}
