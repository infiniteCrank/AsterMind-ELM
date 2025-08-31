// agnews-demo.js
const { EncoderELM, LanguageClassifier } = window.astermind;

async function tryLoadOrTrain(model, key, trainFn, evalFn = () => ({ passed: true })) {
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
        console.warn(`⚠️ Could not load trained model for ${key}. Will train from scratch. Reason: ${e.message}`);
    }

    if (!trained) {
        trainFn();
        const result = evalFn();
        if (result.passed) {
            const json = model.elm?.savedModelJSON || model.savedModelJSON;
            if (json) {
                model.saveModelAsJSONFile(`${key}.json`);
                console.log(`📦 Model saved locally as ${key}.json — please deploy to /models/ manually.`);
            }
        } else {
            console.warn(`❌ Model not saved: Evaluation thresholds not met.`);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('headlineInput');
    const output = document.getElementById('predictionOutput');
    const fill = document.getElementById('confidenceFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,.;:\'"!?()-';
    const maxLen = 50;
    const categories = ['World', 'Sports', 'Business', 'Sci/Tech'];

    const baseConfig = (hiddenUnits, exportFileName) => ({
        charSet,
        maxLen,
        hiddenUnits,
        activation: 'relu',
        useTokenizer: true,
        tokenizerDelimiter: /\s+/,
        exportFileName,
        categories,
        log: {
            verbose: true,
            modelName: exportFileName
        },
        metrics: { accuracy: 0.80 }
    });

    fetch('/ag-news-classification-dataset/train.csv')
        .then(r => r.text())
        .then(async csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const data = rows.map(line => {
                const [label, text] = line.split(',');
                return { text: text.trim().toLowerCase(), label: label.trim() };
            }).filter(d => d.text && d.label);

            const texts = data.map(d => d.text);
            const labels = data.map(d => d.label);

            const encoder = new EncoderELM(baseConfig(64, 'agnews_encoder.json'));
            await tryLoadOrTrain(encoder, 'agnews_encoder', () => encoder.train(texts, labels));

            const classifier = new LanguageClassifier(baseConfig(128, 'agnews_classifier.json'));
            await tryLoadOrTrain(classifier, 'agnews_classifier', () => classifier.train(texts, labels));

            input.addEventListener('input', () => {
                const val = input.value.trim().toLowerCase();
                if (!val) {
                    output.textContent = '';
                    fill.style.width = '0%';
                    fill.textContent = '';
                    fill.style.background = '#ccc';
                    return;
                }

                const encoded = encoder.encode(val);
                const [result] = classifier.predictFromVector(encoded);

                const percent = Math.round(result.prob * 100);
                output.textContent = `🔍 Predicted: ${result.label}`;
                fill.style.width = `${percent}%`;
                fill.textContent = `${result.label} (${percent}%)`;
                fill.style.background = {
                    World: 'linear-gradient(to right, teal, cyan)',
                    Sports: 'linear-gradient(to right, green, lime)',
                    Business: 'linear-gradient(to right, goldenrod, yellow)',
                    'Sci/Tech': 'linear-gradient(to right, purple, magenta)'
                }[result.label] || '#999';
            });
        });
});
