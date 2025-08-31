// IO.ts - Import/export utilities for labeled training data

export interface LabeledExample {
    text: string;
    label: string;
}

export interface SchemaField {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'unknown';
}

export interface InferredSchema {
    fields: SchemaField[];
    suggestedMapping?: Record<'text' | 'label', string>;
}

export class IO {
    static importJSON(json: string): LabeledExample[] {
        try {
            const data = JSON.parse(json);
            if (!Array.isArray(data)) throw new Error('Invalid format');
            return data.filter(item => typeof item.text === 'string' && typeof item.label === 'string');
        } catch (err) {
            console.error('Failed to parse training data JSON:', err);
            return [];
        }
    }

    static exportJSON(pairs: LabeledExample[]): string {
        return JSON.stringify(pairs, null, 2);
    }

    static importDelimited(text: string, delimiter: ',' | '\t' = ',', hasHeader = true): LabeledExample[] {
        const lines = text.trim().split('\n');
        const examples: LabeledExample[] = [];

        const headers = hasHeader
            ? lines[0].split(delimiter).map(h => h.trim().toLowerCase())
            : lines[0].split(delimiter).length === 1
                ? ['label']
                : ['text', 'label'];

        const startIndex = hasHeader ? 1 : 0;

        for (let i = startIndex; i < lines.length; i++) {
            const parts = lines[i].split(delimiter);
            if (parts.length === 1) {
                examples.push({ text: parts[0].trim(), label: parts[0].trim() });
            } else {
                const textIdx = headers.indexOf('text');
                const labelIdx = headers.indexOf('label');
                const text = textIdx !== -1 ? parts[textIdx]?.trim() : parts[0]?.trim();
                const label = labelIdx !== -1 ? parts[labelIdx]?.trim() : parts[1]?.trim();
                if (text && label) {
                    examples.push({ text, label });
                }
            }
        }

        return examples;
    }

    static exportDelimited(pairs: LabeledExample[], delimiter: ',' | '\t' = ',', includeHeader = true): string {
        const header = includeHeader ? `text${delimiter}label\n` : '';
        const rows = pairs.map(p => `${p.text.replace(new RegExp(delimiter, 'g'), '')}${delimiter}${p.label.replace(new RegExp(delimiter, 'g'), '')}`);
        return header + rows.join('\n');
    }

    static importCSV(csv: string, hasHeader = true): LabeledExample[] {
        return this.importDelimited(csv, ',', hasHeader);
    }

    static exportCSV(pairs: LabeledExample[], includeHeader = true): string {
        return this.exportDelimited(pairs, ',', includeHeader);
    }

    static importTSV(tsv: string, hasHeader = true): LabeledExample[] {
        return this.importDelimited(tsv, '\t', hasHeader);
    }

    static exportTSV(pairs: LabeledExample[], includeHeader = true): string {
        return this.exportDelimited(pairs, '\t', includeHeader);
    }

    static inferSchemaFromCSV(csv: string): InferredSchema {
        const lines = csv.trim().split('\n');
        if (lines.length === 0) return { fields: [] };

        const header = lines[0].split(',').map(h => h.trim().toLowerCase());
        const row = lines[1]?.split(',') || [];

        const fields: SchemaField[] = header.map((name, i) => {
            const sample = row[i]?.trim();
            let type: 'string' | 'number' | 'boolean' | 'unknown' = 'unknown';
            if (!sample) type = 'unknown';
            else if (!isNaN(Number(sample))) type = 'number';
            else if (sample === 'true' || sample === 'false') type = 'boolean';
            else type = 'string';
            return { name, type };
        });

        const suggestedMapping: Record<'text' | 'label', string> = {
            text: header.find(h => h.includes('text') || h.includes('utterance') || h.includes('input')) || header[0],
            label: header.find(h => h.includes('label') || h.includes('intent') || h.includes('tag')) || header[1] || header[0],
        };

        return { fields, suggestedMapping };
    }

    static inferSchemaFromJSON(json: string): InferredSchema {
        try {
            const data = JSON.parse(json);
            if (!Array.isArray(data) || data.length === 0 || typeof data[0] !== 'object') return { fields: [] };

            const keys = Object.keys(data[0]);
            const fields: SchemaField[] = keys.map(key => {
                const val = data[0][key];
                let type: 'string' | 'number' | 'boolean' | 'unknown' = 'unknown';
                if (typeof val === 'string') type = 'string';
                else if (typeof val === 'number') type = 'number';
                else if (typeof val === 'boolean') type = 'boolean';
                return { name: key.toLowerCase(), type };
            });

            const suggestedMapping: Record<'text' | 'label', string> = {
                text: keys.find(k => k.toLowerCase().includes('text') || k.toLowerCase().includes('utterance') || k.toLowerCase().includes('input')) || keys[0],
                label: keys.find(k => k.toLowerCase().includes('label') || k.toLowerCase().includes('intent') || k.toLowerCase().includes('tag')) || keys[1] || keys[0],
            };

            return { fields, suggestedMapping };
        } catch (err) {
            console.error('Failed to infer schema from JSON:', err);
            return { fields: [] };
        }
    }
}
