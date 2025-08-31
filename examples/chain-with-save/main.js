// ✅ main.js patched to load all ELM models from /models instead of localStorage
const {
    AutoComplete,
    EncoderELM,
    CharacterLangEncoderELM,
    FeatureCombinerELM,
    RefinerELM,
    ConfidenceClassifierELM,
    LanguageClassifier
} = window.astermind;

async function tryLoadOrTrain(model, key, trainFn, evalFn = () => ({ passed: true, metrics: {} })) {
    let trained = false;
    try {
        const res = await fetch(`/models/${key}.json`);
        if (!res.ok) throw new Error(`Fetch failed for ${key}.json`);
        const json = await res.text();
        if (json.trim().startsWith('<!DOCTYPE')) throw new Error(`Received HTML instead of JSON`);
        model.loadModelFromJSON(json);
        console.log(`✅ Loaded ${key} from /models/${key}.json`);
        trained = true;
    } catch (e) {
        console.warn(`⚠️ Could not load trained model for ${key}. Will train from scratch.\nReason: ${e.message}`);
    }

    if (!trained) {
        trainFn();

        const result = evalFn();
        const passed = result.passed;
        const metrics = result.metrics || {};

        // Report metrics
        console.log(`📊 Evaluation for ${key}:`);
        for (const [metric, { value, threshold, passed }] of Object.entries(metrics)) {
            console.log(` - ${metric}: ${value} (threshold: ${threshold}) ${passed ? '✅' : '❌'}`);
        }

        if (passed) {
            const json = model.elm?.savedModelJSON || model.savedModelJSON;
            if (json) {
                model.saveModelAsJSONFile(`${key}.json`);
                console.log(`📦 Model saved locally as ${key}.json — please deploy to /models/ manually`);
            }
        } else {
            console.warn(`❌ Model not saved: one or more thresholds not met for ${key}`);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
    const maxLen = 30;

    const baseConfig = (hiddenUnits, exportFileName, useTokenizer = true, name = "Unnamed", metrics = { accuracy: 0.85 }) => ({
        charSet,
        maxLen,
        hiddenUnits,
        activation: 'relu',
        useTokenizer,
        tokenizerDelimiter: /\s+/,
        exportFileName,
        log: {
            verbose: true,
            name: name
        },
        metrics: metrics
    });

    fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(async csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const allData = rows.map(line => {
                const [text = '', label = ''] = line.split(',');
                return { text: text.trim().toLowerCase(), label: label.trim() };
            }).filter(d => d.text && d.label);

            const greetings = allData.map(d => d.text);
            const labels = allData.map(d => d.label);

            const acTrainPairs = greetings.flatMap(g => {
                const cuts = [0.3, 0.6, 0.9];
                return cuts.map(p => ({
                    input: g.slice(0, Math.floor(g.length * p)).trim(),
                    label: g.trim()
                }));
            });

            const acTestPairs = greetings.map(g => ({
                input: g.slice(0, Math.floor(g.length * 0.6)).trim(),
                label: g.trim()
            }));

            const ac = new AutoComplete(acTrainPairs, {
                ...baseConfig(128, 'ac_model.json', false, "AutoCompleteELM", {
                    crossEntropy: 2.04,
                    top1Accuracy: 0.85
                }),
                inputElement: input,
                outputElement: output,
                activation: 'relu'
            });

            await tryLoadOrTrain(ac, 'ac_model', () => ac.train(), () => {
                const acc = ac.top1Accuracy(acTestPairs);
                const xent = ac.crossEntropy(acTestPairs);
                const internalCE = ac.internalCrossEntropy(true);
                console.log(`📊 AutoComplete — Accuracy: ${acc.toFixed(4)}, External CE: ${xent.toFixed(4)}, Internal CE: ${internalCE.toFixed(4)}`);
                return acc >= 0.85 || internalCE <= 2.04;
            });

            const encoder = new EncoderELM(baseConfig(32, 'encoder_model.json', true, "EncoderELM"));
            const inputVectors = greetings.map(g => encoder.elm.encoder.normalize(encoder.elm.encoder.encode(g)));
            await tryLoadOrTrain(encoder, 'encoder_model', () => encoder.train(greetings, inputVectors));

            const dim = 16;
            const labelMap = new Map();
            const encodedTargets = greetings.map((_, i) => {
                const label = labels[i];
                if (!labelMap.has(label)) {
                    const vec = new Array(dim).fill(0);
                    vec[labelMap.size] = 1;
                    labelMap.set(label, vec);
                }
                return labelMap.get(label);
            });
            encoder.train(greetings, encodedTargets);

            const classifier = new LanguageClassifier({
                ...baseConfig(50, 'lang_model.json', true, "LanguageClassifier"),
                categories: ['English', 'French', 'Spanish']
            });
            const classifierData = greetings.map((g, i) => ({ vector: encoder.encode(g), label: labels[i] }));
            await tryLoadOrTrain(classifier, 'lang_model', () => classifier.trainVectors(classifierData));

            const langEncoder = new CharacterLangEncoderELM(baseConfig(64, 'langEncoder_model.json', true, "CharacterLangEncoderELM"));
            await tryLoadOrTrain(langEncoder, 'langEncoder_model', () => langEncoder.train(greetings, labels));

            const normalize = vec => {
                const mag = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
                return vec.map(x => x / mag);
            };
            const normalizeMeta = ([len, diversity, vowels, punct]) => [len / maxLen, diversity, vowels, punct];

            const vectors = greetings.map(g => normalize(langEncoder.encode(g)));
            const metas = greetings.map(g => normalizeMeta([
                g.length,
                new Set(g).size / g.length,
                (g.match(/[aeiou]/g) || []).length / g.length,
                (g.match(/[.,!?]/g) || []).length / g.length
            ]));

            const combiner = new FeatureCombinerELM(baseConfig(128, 'combiner_model.json', false, "FeatureCombinerELM"));
            await tryLoadOrTrain(combiner, 'combiner_model', () => combiner.train(vectors, metas, labels));

            const combinedInputs = vectors.map((vec, i) => FeatureCombinerELM.combineFeatures(vec, metas[i]));
            const combinerResults = vectors.map((vec, i) => combiner.predict(vec, metas[i])[0]);

            const confidenceLabels = combinerResults.map((res, i) => {
                const incorrect = res.label !== labels[i];
                const lowProb = res.prob < 0.8;
                return (lowProb || incorrect) ? 'low' : 'high';
            });

            const confidenceClassifier = new ConfidenceClassifierELM(baseConfig(64, 'conf_model.json', false, "ConfidenceClassifierELM"));
            await tryLoadOrTrain(confidenceClassifier, 'conf_model', () => confidenceClassifier.train(vectors, metas, confidenceLabels));

            const LOW_CONF = 0.8;
            const lowConfidence = combinerResults
                .map((res, i) => ({ vector: combinedInputs[i], actual: labels[i], predicted: res.label, label: labels[i], prob: res.prob }))
                .filter(d => d.prob < LOW_CONF || d.predicted !== d.actual);

            const refiner = new RefinerELM(baseConfig(64, 'refiner_model.json', false, "RefinerELM"));
            await tryLoadOrTrain(refiner, 'refiner_model', () => {
                if (lowConfidence.length > 0) {
                    refiner.train(
                        lowConfidence.map(d => d.vector),
                        lowConfidence.map(d => d.label)
                    );
                }
            });

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

                const encoded = encoder.encode(acResult.completion);
                const [langResult] = classifier.predictFromVector(encoded);

                const langVec = normalize(langEncoder.encode(prediction));
                const meta = normalizeMeta([
                    val.length,
                    new Set(val).size / val.length,
                    (val.match(/[aeiou]/g) || []).length / val.length,
                    (val.match(/[.,!?]/g) || []).length / val.length
                ]);

                const combinedVec = FeatureCombinerELM.combineFeatures(langVec, meta);
                const [combinerResult] = combiner.predict(langVec, meta);

                const [confidenceResult] = confidenceClassifier.predict(langVec, meta);
                const isUncertain = confidenceResult.label === 'low';

                let finalResult = combinerResult;
                let refined;

                if (combinerResult.prob < 0.6 || isUncertain) {
                    try {
                        [refined] = refiner.predict(combinedVec);
                        if (refined) {
                            finalResult = langResult.label === combinerResult.label
                                ? {
                                    label: langResult.label,
                                    prob: (langResult.prob + combinerResult.prob) / 2,
                                }
                                : refined;
                        }
                    } catch (err) {
                        console.warn('⚠️ Refiner failed:', err);
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
