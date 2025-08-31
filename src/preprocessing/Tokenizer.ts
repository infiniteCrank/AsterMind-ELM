export class Tokenizer {
    constructor(customDelimiter?: RegExp) {
        this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
    }

    tokenize(text: any): string[] {
        if (typeof text !== 'string') {
            console.warn('[Tokenizer] Expected a string, got:', typeof text, text);
            try {
                text = String(text ?? '');
            } catch {
                return [];
            }
        }

        return text
            .trim()
            .toLowerCase()
            .split(this.delimiter)
            .filter(Boolean);
    }

    ngrams(tokens: string[], n: number): string[] {
        if (n <= 0 || tokens.length < n) return [];
        const result = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            result.push(tokens.slice(i, i + n).join(' '));
        }
        return result;
    }

    private delimiter: RegExp;
}
