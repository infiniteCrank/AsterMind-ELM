import { ELM } from './ELM';

export class ELMChain {
    encoders: ELM[];

    constructor(encoders: ELM[]) {
        this.encoders = encoders;
    }

    getEmbedding(input: number[][]): number[][] {
        let out = input;
        for (const encoder of this.encoders) {
            out = encoder.getEmbedding(out);
        }
        return out;
    }
}
