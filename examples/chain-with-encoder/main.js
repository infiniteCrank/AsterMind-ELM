const { AutoComplete, EncoderELM, LanguageClassifier } = window.astermind;

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
    const maxLen = 30;

    const acConfig = {
        charSet,
        maxLen,
        hiddenUnits: 40,
        activation: 'relu',
        useTokenizer: true,
        tokenizerDelimiter: /\s+/
    };

    const encoderConfig = {
        charSet,
        maxLen,
        hiddenUnits: 32,
        activation: 'relu',
        useTokenizer: true,
        tokenizerDelimiter: /\s+/
    };

    const langConfig = {
        categories: ['English', 'French', 'Spanish'],
        charSet,
        maxLen,
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

            const greetings = allData.map(row => row.text);
            const languages = allData.map(row => row.label);

            // 1. Train Autocomplete
            const ac = new AutoComplete([...new Set(greetings)], {
                ...acConfig,
                inputElement: input,
                outputElement: output
            });

            // 2. Train Encoder: greeting → vector
            const encoder = new EncoderELM(encoderConfig);
            const inputVectors = greetings.map(g =>
                encoder.elm.encoder.normalize(encoder.elm.encoder.encode(g))
            );
            encoder.train(greetings, inputVectors);

            const dim = 16; // output size of encoder
            const labelMap = new Map();
            const encodedTargets = greetings.map((_, i) => {
                const label = languages[i];
                if (!labelMap.has(label)) {
                    const vec = new Array(dim).fill(0);
                    vec[labelMap.size] = 1;
                    labelMap.set(label, vec);
                }
                return labelMap.get(label);
            });

            encoder.train(greetings, encodedTargets);

            // 3. Train Language Classifier: vector → label
            const classifier = new LanguageClassifier(langConfig);

            const classifierTrainingData = greetings.map((g, i) => {
                const vec = encoder.encode(g);
                return {
                    vector: vec,
                    label: languages[i]
                };
            });

            classifier.trainVectors(classifierTrainingData);

            // Event Handler
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

                const encoded = encoder.encode(acResult.completion);
                const [langResult] = classifier.predictFromVector(encoded);

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
});
