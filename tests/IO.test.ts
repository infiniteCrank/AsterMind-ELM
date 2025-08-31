// IO.test.ts - Unit tests for IO utilities

import { describe, it, expect } from 'vitest';
import { IO } from '../src/utils/IO';

const sampleJSON = `[
  {"text": "hello", "label": "greeting"},
  {"text": "bye", "label": "farewell"}
]`;

const sampleCSV = `text,label\nhello,greeting\nbye,farewell`;
const sampleTSV = `text\tlabel\nhello\tgreeting\nbye\tfarewell`;

describe('IO', () => {
    it('imports JSON correctly', () => {
        const data = IO.importJSON(sampleJSON);
        expect(data).toEqual([
            { text: 'hello', label: 'greeting' },
            { text: 'bye', label: 'farewell' },
        ]);
    });

    it('exports JSON correctly', () => {
        const json = IO.exportJSON([
            { text: 'hi', label: 'greet' },
        ]);
        expect(json).toContain('hi');
        expect(json).toContain('greet');
    });

    it('imports CSV correctly', () => {
        const data = IO.importCSV(sampleCSV);
        expect(data.length).toBe(2);
        expect(data[0].text).toBe('hello');
    });

    it('exports CSV correctly', () => {
        const csv = IO.exportCSV([
            { text: 'yo', label: 'greet' },
        ]);
        expect(csv).toContain('yo,greet');
    });

    it('imports TSV correctly', () => {
        const data = IO.importTSV(sampleTSV);
        expect(data.length).toBe(2);
        expect(data[1].label).toBe('farewell');
    });

    it('exports TSV correctly', () => {
        const tsv = IO.exportTSV([
            { text: 'hey', label: 'greeting' },
        ]);
        expect(tsv).toContain('hey\tgreeting');
    });

    it('infers schema from CSV correctly', () => {
        const schema = IO.inferSchemaFromCSV(sampleCSV);
        expect(schema.fields).toEqual([
            { name: 'text', type: 'string' },
            { name: 'label', type: 'string' },
        ]);
        expect(schema.suggestedMapping?.text).toBe('text');
        expect(schema.suggestedMapping?.label).toBe('label');
    });

    it('infers schema from JSON correctly', () => {
        const schema = IO.inferSchemaFromJSON(sampleJSON);
        expect(schema.fields).toEqual([
            { name: 'text', type: 'string' },
            { name: 'label', type: 'string' },
        ]);
        expect(schema.suggestedMapping?.text).toBe('text');
    });

    it('returns empty array on invalid JSON import', () => {
        const data = IO.importJSON('{invalid json');
        expect(data).toEqual([]);
    });
});

// IO Utilities

describe('IO utilities', () => {
    it('parses JSON correctly', () => {
        const json = '[{"text":"hi","label":"greet"}]';
        const parsed = IO.importJSON(json);
        expect(parsed.length).toBe(1);
        expect(parsed[0].text).toBe('hi');
    });

    it('parses CSV correctly', () => {
        const csv = 'text,label\nhello,English';
        const parsed = IO.importCSV(csv);
        expect(parsed.length).toBe(1);
        expect(parsed[0].label).toBe('English');
    });

    it('parses TSV correctly', () => {
        const tsv = 'text\tlabel\nbonjour\tFrench';
        const parsed = IO.importTSV(tsv);
        expect(parsed.length).toBe(1);
        expect(parsed[0].label).toBe('French');
    });

    it('exports to JSON', () => {
        const data = [{ text: 'hi', label: 'greet' }];
        const json = IO.exportJSON(data);
        expect(json).toContain('hi');
    });

    it('infers schema correctly', () => {
        const csv = 'text,label\nhello,English';
        const schema = IO.inferSchemaFromCSV(csv);
        expect(schema.fields.length).toBeGreaterThan(0);
        expect(Object.keys(schema.suggestedMapping || {})).toContain('text');
    });
});

// IO Round-trip

describe('IO round-trip', () => {
    const original = [
        { text: 'hello', label: 'English' },
        { text: 'bonjour', label: 'French' }
    ];

    it('CSV round-trip preserves data', () => {
        const csv = IO.exportCSV(original);
        const parsed = IO.importCSV(csv);
        expect(parsed).toEqual(original);
    });

    it('TSV round-trip preserves data', () => {
        const tsv = IO.exportTSV(original);
        const parsed = IO.importTSV(tsv);
        expect(parsed).toEqual(original);
    });

    it('JSON round-trip preserves data', () => {
        const json = IO.exportJSON(original);
        const parsed = IO.importJSON(json);
        expect(parsed).toEqual(original);
    });
});