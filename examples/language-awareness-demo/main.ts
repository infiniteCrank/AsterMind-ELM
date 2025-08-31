// @ts-ignore
const { LanguageClassifier } = window.astermind;

const input = document.getElementById('langInput') as HTMLInputElement;
const fill = document.getElementById('langFill') as HTMLDivElement;

const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
const config = {
    categories: ['English', 'French', 'Spanish'],
    hiddenUnits: 100,
    maxLen: 30,
    activation: 'relu',
    charSet,
    useTokenizer: false,
    log: {
        verbose: false
    },
};

fetch('/language_greetings_1500.csv')
    .then(res => res.text())
    .then(csv => {
        const lines = csv
            .split('\n')
            .map(l => l.trim())
            .filter(Boolean)
            .slice(1); // skip header

        const rawData = lines.map(line => {
            const [text = '', label = ''] = line.split(',');
            return {
                text: text.trim(), // preserve case
                label: label.trim()
            };
        }).filter(d => d.text && d.label);

        // OPTIONAL: Print counts for sanity check
        const counts = rawData.reduce((acc, d) => {
            acc[d.label] = (acc[d.label] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);
        console.log('Label distribution:', counts);

        const classifier = new LanguageClassifier(config);
        classifier.train(rawData);

        input.addEventListener('input', () => {
            const typed = input.value.trim();
            if (!typed) {
                fill.style.width = '0%';
                fill.textContent = '';
                fill.style.background = '#ccc';
                return;
            }

            const results = classifier.predict(typed, 3); // top 3 for debug
            const [top] = results;
            const percent = Math.round(top.prob * 100);

            fill.style.width = `${percent}%`;
            fill.textContent = percent < 40
                ? '🤔 Not sure'
                : `${top.label} (${percent}%)`;

            fill.style.background = {
                English: 'linear-gradient(to right, green, lime)',
                French: 'linear-gradient(to right, blue, cyan)',
                Spanish: 'linear-gradient(to right, red, orange)'
            }[top.label] || '#999';

            // Debug
            console.log('Top predictions:', results);
        });
    });
