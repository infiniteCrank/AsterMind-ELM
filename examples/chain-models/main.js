// @ts-ignore
const { LanguageClassifier, AutoComplete, IO } = window.astermind;

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';

    const acConfig = {
        charSet,
        maxLen: 30,
        hiddenUnits: 40,
        activation: 'relu',
        useTokenizer: true,
        tokenizerDelimiter: /\s+/
    };

    const langConfig = {
        categories: ['English', 'French', 'Spanish'],
        charSet,
        maxLen: 30,
        hiddenUnits: 50,
        activation: 'relu',
        useTokenizer: true,
        tokenizerDelimiter: /\s+/
    };

    fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const allData = rows.map(l => {
                const [text = '', label = ''] = l.split(',');
                return { text: text.trim().toLowerCase(), label: label.trim() };
            });

            const uniqueCategories = [...new Set(allData.map(row => row.text))];
            const ac = new AutoComplete(uniqueCategories, {
                ...acConfig,
                inputElement: input,
                outputElement: output
            });
            const lang = new LanguageClassifier(langConfig);
            lang.train(allData);

            input.addEventListener('input', () => {
                const val = input.value.trim().toLowerCase();
                if (!val) {
                    output.textContent = '';
                    fill.style.width = '0%';
                    fill.textContent = '';
                    fill.style.background = '#ccc';
                    return;
                }

                const [acResult] = ac.predict(val, 1);
                output.textContent = `🔮 Autocomplete: ${acResult.completion}`;

                const [langResult] = lang.predict(acResult.completion);
                const percent = Math.round(langResult.prob * 100);
                fill.style.width = `${percent}%`;
                fill.textContent = `${langResult.label} (${percent}%)`;

                fill.style.background = {
                    English: 'linear-gradient(to right, green, lime)',
                    French: 'linear-gradient(to right, blue, cyan)',
                    Spanish: 'linear-gradient(to right, red, orange)'
                }[langResult.label] || '#999';
            });
        });
})