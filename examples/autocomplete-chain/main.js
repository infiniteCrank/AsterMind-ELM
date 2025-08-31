// main.js — multi-stage ELM pipeline: char → token → sentence prediction

const {
    AutoComplete,
    EncoderELM,
    CharacterLangEncoderELM,
    FeatureCombinerELM,
    RefinerELM,
    ConfidenceClassifierELM,
    LanguageClassifier
} = window.astermind;

function tryLoadOrTrain(model, key, trainFn, evalFn = () => true) {
    const saved = localStorage.getItem(key);
    if (saved) {
        model.loadModelFromJSON(saved);
        return;
    }
    trainFn();
    if (evalFn()) {
        const json = model.elm?.savedModelJSON || model.savedModelJSON;
        if (json) localStorage.setItem(key, json);
    } else {
        console.warn("❌ Model not saved: thresholds not met.");
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
    const maxLen = 30;

    const baseConfig = (hiddenUnits, exportFileName, useTokenizer = false, name = "Unnamed", metrics = { accuracy: 0.85 }) => ({
        charSet,
        maxLen,
        hiddenUnits,
        activation: 'relu',
        useTokenizer,
        tokenizerDelimiter: /\s+/,
        exportFileName,
        log: {
            verbose: true,
            name
        },
        metrics
    });

    fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const allData = rows.map(line => {
                const [text = '', label = ''] = line.split(',');
                return { text: text.trim().toLowerCase(), label: label.trim() };
            }).filter(d => d.text && d.label);

            const greetings = allData.map(d => d.text);

            // Stage 1: character → next character prediction
            const charPairs = greetings.flatMap(g => {
                const pairs = [];
                for (let i = 1; i < g.length; i++) {
                    pairs.push({ input: g.slice(0, i), label: g[i] });
                }
                return pairs;
            });

            const charPredictor = new AutoComplete(charPairs, {
                ...baseConfig(64, 'char_predictor.json', false, "CharPredictor", {
                    accuracy: 0.8
                }),
                inputElement: input,
                outputElement: output,
                activation: 'relu'
            });

            tryLoadOrTrain(charPredictor, 'char_model', () => charPredictor.train());

            // Stage 2: char sequence → full word prediction
            const wordPairs = greetings.map(g => ({
                input: g.slice(0, Math.floor(g.length * 0.6)),
                label: g
            }));

            const wordPredictor = new AutoComplete(wordPairs, {
                ...baseConfig(64, 'word_predictor.json', false, "WordPredictor", {
                    accuracy: 0.8
                }),
                inputElement: input,
                outputElement: output,
                activation: 'relu'
            });

            tryLoadOrTrain(wordPredictor, 'word_model', () => wordPredictor.train());

            // Stage 3: word → full sentence prediction (placeholder)
            const sentencePairs = greetings.map(g => ({
                input: g,
                label: `${g}! how are you today?`  // synthetic full sentence completion
            }));

            const sentencePredictor = new AutoComplete(sentencePairs, {
                ...baseConfig(64, 'sentence_predictor.json', false, "SentencePredictor", {
                    accuracy: 0.8
                }),
                inputElement: input,
                outputElement: output,
                activation: 'relu'
            });

            tryLoadOrTrain(sentencePredictor, 'sentence_model', () => sentencePredictor.train());

            // Wire up live inference through pipeline
            input.addEventListener('input', () => {
                const val = input.value.trim().toLowerCase();
                if (!val) {
                    output.textContent = '';
                    fill.style.width = '0%';
                    fill.textContent = '';
                    fill.style.background = '#ccc';
                    return;
                }

                const [nextChar] = charPredictor.predict(val, 1);
                const nextInput = val + (nextChar?.completion || '');

                const [word] = wordPredictor.predict(nextInput, 1);
                const [sentence] = sentencePredictor.predict(word?.completion || '', 1);

                const final = sentence?.completion || word?.completion || '';
                output.textContent = `🧠 ${final}`;

                fill.style.width = '100%';
                fill.textContent = final;
                fill.style.background = 'linear-gradient(to right, #3f87a6, #ebf8e1)';
            });
        });
})