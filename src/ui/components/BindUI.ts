// BindUI.ts - Utility to bind ELM model to HTML inputs and outputs

import { ELM } from '../../core/ELM';
import { PredictResult } from '../../core/ELM';

interface BindOptions {
    model: ELM;
    inputElement: HTMLInputElement;
    outputElement: HTMLElement;
    topK?: number;
}

export function bindAutocompleteUI({ model, inputElement, outputElement, topK = 5 }: BindOptions): void {
    inputElement.addEventListener('input', () => {
        const typed = inputElement.value.trim();

        if (typed.length === 0) {
            outputElement.innerHTML = '<em>Start typing...</em>';
            return;
        }

        try {
            const results: PredictResult[] = model.predict(typed, topK);
            outputElement.innerHTML = results.map(r => `
                <div><strong>${r.label}</strong>: ${(r.prob * 100).toFixed(1)}%</div>
            `).join('');
        } catch (e: unknown) {
            const message = e instanceof Error ? e.message : 'Unknown error';
            outputElement.innerHTML = `<span style="color: red;">Error: ${message}</span>`;
        }
    });
}
