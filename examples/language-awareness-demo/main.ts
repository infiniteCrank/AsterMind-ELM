// @ts-ignore
const AM = (window as any).astermind || {};
const { LanguageClassifier } = AM;

// DOM
const input = document.getElementById('langInput') as HTMLInputElement;
const fill = document.getElementById('langFill') as HTMLDivElement;

// Keep diacritics; they help FR/ES separation
const charSet =
    'abcdefghijklmnopqrstuvwxyzçàâêëéèîïôœùûüñáíóúü¿¡ ’-.,!?;:()[]{}"\'\t ';

const CATEGORIES = ['English', 'French', 'Spanish'] as const;

const config = {
    categories: [...CATEGORIES],
    hiddenUnits: 256,
    maxLen: 48,
    activation: 'gelu',
    weightInit: 'xavier',
    ridgeLambda: 1e-2,
    charSet,
    useTokenizer: true,          // <<— enable built-in text encoder
    log: { verbose: false },
};

function safeCSVTwoCols(line: string) {
    const i = line.lastIndexOf(',');
    if (i <= 0 || i >= line.length - 1) return null;
    const text = line.slice(0, i).trim().replace(/^"|"$/g, '');
    const label = line.slice(i + 1).trim().replace(/^"|"$/g, '');
    if (!text || !label) return null;
    return { text, label };
}

function toXY(rows: Array<{ text: string; label: string }>) {
    const map: Record<string, number> = { English: 0, French: 1, Spanish: 2 };
    const X: string[] = [];
    const Y: number[] = [];
    for (const r of rows) {
        const y = map[r.label];
        if (y == null) continue;
        X.push(r.text.replace(/\s+/g, ' ').trim().toLowerCase());
        Y.push(y);
    }
    return { X, Y };
}

function softmax(v: number[]) {
    const m = Math.max(...v);
    const ex = v.map(x => Math.exp(x - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(e => e / s);
}

(async function main() {
    const csv = await fetch('/language_greetings_1500.csv').then(r => r.text());
    const lines = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
    const raw = lines.map(safeCSVTwoCols).filter(Boolean) as { text: string, label: string }[];

    const clf = new LanguageClassifier(config);

    // Prefer the newer API if available
    const { X, Y } = toXY(raw);
    if (typeof (clf as any).trainFromData === 'function') {
        await (clf as any).trainFromData(X, Y, { batchSize: 128, epochs: 1, ridgeLambda: 1e-2 });
    } else {
        await (clf as any).train(raw);
    }

    input.addEventListener('input', () => {
        const typed0 = input.value.trim();
        if (!typed0) {
            fill.style.width = '0%';
            fill.textContent = '';
            fill.style.background = '#ccc';
            return;
        }
        const typed = typed0.toLowerCase();

        let topLabel = 'Unknown', topProb = 0;
        let results: Array<{ label: string; prob: number }> | undefined;

        const r = (clf as any).predict(typed, 3);
        if (Array.isArray(r) && r.length && r[0]?.label != null) {
            results = r;
            topLabel = r[0].label;
            topProb = r[0].prob;
        } else if (Array.isArray(r) && typeof r[0] === 'number') {
            const probs = softmax(r as number[]);
            const idx = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])[0][1];
            topLabel = CATEGORIES[idx];
            topProb = probs[idx];
            results = probs.map((p, i) => ({ label: CATEGORIES[i], prob: p })).sort((a, b) => b.prob - a.prob).slice(0, 3);
        }

        const percent = Math.round((topProb || 0) * 100);
        fill.style.width = `${percent}%`;
        fill.textContent = percent < 40 ? '🤔 Not sure' : `${topLabel} (${percent}%)`;
        fill.style.background = ({
            English: 'linear-gradient(to right, green, lime)',
            French: 'linear-gradient(to right, blue, cyan)',
            Spanish: 'linear-gradient(to right, red, orange)',
        } as any)[topLabel] || '#999';

        console.log('Top predictions:', results);
    });
})().catch(console.error);
