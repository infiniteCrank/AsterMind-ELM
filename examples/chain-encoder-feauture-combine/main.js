// @ts-ignore
const { AutoComplete, EncoderELM, CharacterLangEncoderELM, FeatureCombinerELM, RefinerELM, ConfidenceClassifierELM } = window.astermind;

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáíóúü¿¡ ';
    const maxLen = 30;

    const acConfig = {
        charSet,
        maxLen,
        hiddenUnits: 40,
        activation: 'relu',
        useTokenizer: true
    };

    const encoderConfig = {
        charSet,
        maxLen,
        hiddenUnits: 64,
        activation: 'relu',
        useTokenizer: true
    };

    const combinerConfig = {
        charSet,
        maxLen,
        hiddenUnits: 128,
        activation: 'relu',
        useTokenizer: false
    };

    const refinerConfig = {
        charSet,
        maxLen,
        hiddenUnits: 64,
        activation: 'relu',
        useTokenizer: false
    };

    fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const allData = rows.map((line, i) => {
                const parts = line.split(',');
                const text = (parts[0] || '').toString().trim().toLowerCase();
                const label = (parts[1] || '').toString().trim();
                return { text, label };
            }).filter(d => d.text && d.label);

            const conflicts = {};
            allData.forEach(({ text, label }) => {
                if (!conflicts[text]) conflicts[text] = new Set();
                conflicts[text].add(label);
            });
            console.log("⚠️ Conflicts:", Object.entries(conflicts).filter(([k, v]) => v.size > 1));

            const greetings = allData.map(d => d.text);
            const labels = allData.map(d => d.label);

            const ac = new AutoComplete([...new Set(greetings)], {
                ...acConfig,
                inputElement: input,
                outputElement: output
            });

            const langEncoder = new CharacterLangEncoderELM(encoderConfig);

            langEncoder.train(greetings, labels);

            function normalize(vec) {
                const mag = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
                return vec.map(x => x / mag);
            }

            function normalizeMeta(meta) {
                const [len, diversity, vowels, punct] = meta;
                return [
                    len / maxLen,
                    diversity,
                    vowels,
                    punct
                ];
            }

            const vectors = greetings.map(g => {
                const langVec = normalize(langEncoder.encode(g));
                return langVec;
            });

            const labelCounts = labels.reduce((acc, label) => {
                acc[label] = (acc[label] || 0) + 1;
                return acc;
            }, {});
            console.log('📊 Label Distribution:', labelCounts);


            const metas = greetings.map(g => normalizeMeta([
                g.length,
                new Set(g).size / g.length,
                (g.match(/[aeiou]/g) || []).length / g.length,
                (g.match(/[.,!?]/g) || []).length / g.length
            ]));

            console.log('🧪 Sample Combined Input:', FeatureCombinerELM.combineFeatures(vectors[0], metas[0]));

            const combiner = new FeatureCombinerELM(combinerConfig);
            combiner.train(vectors, metas, labels);

            const combinedInputs = vectors.map((vec, i) =>
                FeatureCombinerELM.combineFeatures(vec, metas[i])
            );

            const combinerResults = vectors.map((vec, i) =>
                combiner.predict(vec, metas[i])[0]
            );

            const confidenceClassifier = new ConfidenceClassifierELM(refinerConfig);  // reuse config

            const confidenceLabels = combinerResults.map((res, i) => {
                const incorrect = res.label !== labels[i];
                const lowProb = res.prob < 0.8;
                return (lowProb || incorrect) ? 'low' : 'high';
            });

            confidenceClassifier.train(vectors, metas, confidenceLabels);
            console.log('🧠 ConfidenceClassifierELM trained.');

            const LOW_CONFIDENCE_THRESHOLD = 0.8;
            const lowConfidence = combinerResults
                .map((res, i) => ({
                    vector: combinedInputs[i],
                    actual: labels[i],
                    predicted: res.label,
                    label: labels[i],
                    prob: res.prob
                }))
                .filter(d => d.prob < LOW_CONFIDENCE_THRESHOLD || d.predicted !== d.actual);

            const refiner = new RefinerELM(refinerConfig);
            if (lowConfidence.length > 0) {
                refiner.train(
                    lowConfidence.map(d => d.vector),
                    lowConfidence.map(d => d.label)
                );
                console.log(`🔁 Refiner trained on ${lowConfidence.length} low-confidence samples.`);
            } else {
                console.log('⚠️ No low-confidence samples to train the Refiner.');
            }

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
                const prediction = acResult.completion;
                output.textContent = `🔮 Autocomplete: ${prediction}`;

                const langVec = normalize(langEncoder.encode(prediction));
                const vec = langVec;

                const meta = normalizeMeta([
                    val.length,
                    new Set(val).size / val.length,
                    (val.match(/[aeiou]/g) || []).length / val.length,
                    (val.match(/[.,!?]/g) || []).length / val.length
                ]);

                const combinedVec = FeatureCombinerELM.combineFeatures(vec, meta);
                const [combinerResult] = combiner.predict(vec, meta);

                let finalResult = combinerResult;
                let refined;

                const [confidenceResult] = confidenceClassifier.predict(vec, meta);
                const isUncertain = confidenceResult.label === 'low';

                if (combinerResult.prob < 0.6 || isUncertain) {
                    try {
                        [refined] = refiner.predict(combinedVec);
                        if (refined) {
                            finalResult = refined;
                            console.log('🔁 Used Refiner');
                        }
                    } catch (err) {
                        console.warn('⚠️ Refiner not available or failed:', err);
                    }
                }

                const percent = Math.round(finalResult.prob * 100);
                fill.style.width = `${percent}%`;
                fill.textContent = `${finalResult.label} (${percent}%)` + (finalResult === combinerResult ? ' [C]' : ' [R]');
                fill.dataset.model = finalResult === combinerResult ? 'combiner' : 'refiner';
                fill.style.background = {
                    English: 'linear-gradient(to right, green, lime)',
                    French: 'linear-gradient(to right, blue, cyan)',
                    Spanish: 'linear-gradient(to right, red, orange)'
                }[finalResult.label] || '#999';
            });
        });
});
