// Augment.ts - Basic augmentation utilities for category training examples

export class Augment {
    static addSuffix(text: string, suffixes: string[]): string[] {
        return suffixes.map(suffix => `${text} ${suffix}`);
    }

    static addPrefix(text: string, prefixes: string[]): string[] {
        return prefixes.map(prefix => `${prefix} ${text}`);
    }

    static addNoise(text: string, charSet: string, noiseRate = 0.1): string {
        const chars = text.split('');
        for (let i = 0; i < chars.length; i++) {
            if (Math.random() < noiseRate) {
                const randomChar = charSet[Math.floor(Math.random() * charSet.length)];
                chars[i] = randomChar;
            }
        }
        return chars.join('');
    }

    static mix(text: string, mixins: string[]): string[] {
        return mixins.map(m => `${text} ${m}`);
    }

    static generateVariants(
        text: string,
        charSet: string,
        options?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        }
    ): string[] {
        const variants: string[] = [text];

        if (options?.suffixes) {
            variants.push(...this.addSuffix(text, options.suffixes));
        }

        if (options?.prefixes) {
            variants.push(...this.addPrefix(text, options.prefixes));
        }

        if (options?.includeNoise) {
            variants.push(this.addNoise(text, charSet));
        }

        return variants;
    }
}
