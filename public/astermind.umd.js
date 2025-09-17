(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.astermind = {}));
})(this, (function (exports) { 'use strict';

    // Activations.ts - Common activation functions
    class Activations {
        static relu(x) {
            return Math.max(0, x);
        }
        static leakyRelu(x, alpha = 0.01) {
            return x >= 0 ? x : alpha * x;
        }
        static sigmoid(x) {
            return 1 / (1 + Math.exp(-x));
        }
        static tanh(x) {
            return Math.tanh(x);
        }
        static softmax(arr) {
            const max = Math.max(...arr);
            const exps = arr.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(e => e / sum);
        }
        static apply(matrix, fn) {
            return matrix.map(row => row.map(fn));
        }
        static get(name) {
            switch (name.toLowerCase()) {
                case 'relu': return this.relu;
                case 'leakyrelu': return x => this.leakyRelu(x);
                case 'sigmoid': return this.sigmoid;
                case 'tanh': return this.tanh;
                default: throw new Error(`Unknown activation: ${name}`);
            }
        }
    }

    class Matrix {
        constructor(data) {
            this.data = data;
        }
        static multiply(A, B) {
            const result = [];
            for (let i = 0; i < A.length; i++) {
                result[i] = [];
                for (let j = 0; j < B[0].length; j++) {
                    let sum = 0;
                    for (let k = 0; k < B.length; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }
        static transpose(A) {
            return A[0].map((_, i) => A.map(row => row[i]));
        }
        static identity(size) {
            return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)));
        }
        static addRegularization(A, lambda) {
            return A.map((row, i) => row.map((val, j) => val + (i === j ? lambda : 0)));
        }
        static inverse(A) {
            const n = A.length;
            const I = Matrix.identity(n);
            const M = A.map(row => [...row]);
            for (let i = 0; i < n; i++) {
                let maxEl = Math.abs(M[i][i]);
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(M[k][i]) > maxEl) {
                        maxEl = Math.abs(M[k][i]);
                        maxRow = k;
                    }
                }
                [M[i], M[maxRow]] = [M[maxRow], M[i]];
                [I[i], I[maxRow]] = [I[maxRow], I[i]];
                const div = M[i][i];
                if (div === 0)
                    throw new Error("Matrix is singular and cannot be inverted");
                for (let j = 0; j < n; j++) {
                    M[i][j] /= div;
                    I[i][j] /= div;
                }
                for (let k = 0; k < n; k++) {
                    if (k === i)
                        continue;
                    const factor = M[k][i];
                    for (let j = 0; j < n; j++) {
                        M[k][j] -= factor * M[i][j];
                        I[k][j] -= factor * I[i][j];
                    }
                }
            }
            return I;
        }
        static random(rows, cols, min, max) {
            const data = [];
            for (let i = 0; i < rows; i++) {
                const row = [];
                for (let j = 0; j < cols; j++) {
                    row.push(Math.random() * (max - min) + min);
                }
                data.push(row);
            }
            return new Matrix(data);
        }
        static fromArray(array) {
            return new Matrix(array);
        }
        toArray() {
            return this.data;
        }
    }

    // ELMConfig.ts - Configuration interface and defaults for ELM-based models
    const defaultConfig = {
        hiddenUnits: 50,
        maxLen: 30,
        weightInit: "uniform",
        activation: 'relu',
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: false,
        tokenizerDelimiter: /\s+/,
    };

    class Tokenizer {
        constructor(customDelimiter) {
            this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
        }
        tokenize(text) {
            if (typeof text !== 'string') {
                console.warn('[Tokenizer] Expected a string, got:', typeof text, text);
                try {
                    text = String(text !== null && text !== void 0 ? text : '');
                }
                catch (_a) {
                    return [];
                }
            }
            return text
                .trim()
                .toLowerCase()
                .split(this.delimiter)
                .filter(Boolean);
        }
        ngrams(tokens, n) {
            if (n <= 0 || tokens.length < n)
                return [];
            const result = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                result.push(tokens.slice(i, i + n).join(' '));
            }
            return result;
        }
    }

    // TextEncoder.ts - Text preprocessing and one-hot encoding for ELM
    const defaultTextEncoderConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false
    };
    class TextEncoder {
        constructor(config = {}) {
            const cfg = Object.assign(Object.assign({}, defaultTextEncoderConfig), config);
            this.charSet = cfg.charSet;
            this.charSize = cfg.charSet.length;
            this.maxLen = cfg.maxLen;
            this.useTokenizer = cfg.useTokenizer;
            if (this.useTokenizer) {
                this.tokenizer = new Tokenizer(config.tokenizerDelimiter);
            }
        }
        charToOneHot(c) {
            const index = this.charSet.indexOf(c.toLowerCase());
            const vec = Array(this.charSize).fill(0);
            if (index !== -1)
                vec[index] = 1;
            return vec;
        }
        textToVector(text) {
            let cleaned;
            if (this.useTokenizer && this.tokenizer) {
                const tokens = this.tokenizer.tokenize(text).join('');
                cleaned = tokens.slice(0, this.maxLen).padEnd(this.maxLen, ' ');
            }
            else {
                cleaned = text.toLowerCase().replace(new RegExp(`[^${this.charSet}]`, 'g'), '').padEnd(this.maxLen, ' ').slice(0, this.maxLen);
            }
            const vec = [];
            for (let i = 0; i < cleaned.length; i++) {
                vec.push(...this.charToOneHot(cleaned[i]));
            }
            return vec;
        }
        normalizeVector(v) {
            const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
            return norm > 0 ? v.map(x => x / norm) : v;
        }
        getVectorSize() {
            return this.charSize * this.maxLen;
        }
        getCharSet() {
            return this.charSet;
        }
        getMaxLen() {
            return this.maxLen;
        }
    }

    // UniversalEncoder.ts - Automatically selects appropriate encoder (char or token based)
    const defaultUniversalConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false,
        mode: 'char'
    };
    class UniversalEncoder {
        constructor(config = {}) {
            const merged = Object.assign(Object.assign({}, defaultUniversalConfig), config);
            const useTokenizer = merged.mode === 'token';
            this.encoder = new TextEncoder({
                charSet: merged.charSet,
                maxLen: merged.maxLen,
                useTokenizer,
                tokenizerDelimiter: config.tokenizerDelimiter
            });
        }
        encode(text) {
            return this.encoder.textToVector(text);
        }
        normalize(v) {
            return this.encoder.normalizeVector(v);
        }
        getVectorSize() {
            return this.encoder.getVectorSize();
        }
    }

    // Augment.ts - Basic augmentation utilities for category training examples
    class Augment {
        static addSuffix(text, suffixes) {
            return suffixes.map(suffix => `${text} ${suffix}`);
        }
        static addPrefix(text, prefixes) {
            return prefixes.map(prefix => `${prefix} ${text}`);
        }
        static addNoise(text, charSet, noiseRate = 0.1) {
            const chars = text.split('');
            for (let i = 0; i < chars.length; i++) {
                if (Math.random() < noiseRate) {
                    const randomChar = charSet[Math.floor(Math.random() * charSet.length)];
                    chars[i] = randomChar;
                }
            }
            return chars.join('');
        }
        static mix(text, mixins) {
            return mixins.map(m => `${text} ${m}`);
        }
        static generateVariants(text, charSet, options) {
            const variants = [text];
            if (options === null || options === void 0 ? void 0 : options.suffixes) {
                variants.push(...this.addSuffix(text, options.suffixes));
            }
            if (options === null || options === void 0 ? void 0 : options.prefixes) {
                variants.push(...this.addPrefix(text, options.prefixes));
            }
            if (options === null || options === void 0 ? void 0 : options.includeNoise) {
                variants.push(this.addNoise(text, charSet));
            }
            return variants;
        }
    }

    // ELM.ts - Core ELM logic with TypeScript types
    class ELM {
        constructor(config) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j;
            const cfg = Object.assign(Object.assign({}, defaultConfig), config);
            this.categories = cfg.categories;
            this.hiddenUnits = cfg.hiddenUnits;
            this.maxLen = cfg.maxLen;
            this.activation = cfg.activation;
            this.charSet = (_a = cfg.charSet) !== null && _a !== void 0 ? _a : 'abcdefghijklmnopqrstuvwxyz';
            this.useTokenizer = (_b = cfg.useTokenizer) !== null && _b !== void 0 ? _b : false;
            this.tokenizerDelimiter = cfg.tokenizerDelimiter;
            this.config = cfg;
            this.metrics = this.config.metrics;
            this.verbose = (_d = (_c = cfg.log) === null || _c === void 0 ? void 0 : _c.verbose) !== null && _d !== void 0 ? _d : true;
            this.modelName = (_f = (_e = cfg.log) === null || _e === void 0 ? void 0 : _e.modelName) !== null && _f !== void 0 ? _f : 'Unnamed ELM Model';
            this.logToFile = (_h = (_g = cfg.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false;
            this.dropout = (_j = cfg.dropout) !== null && _j !== void 0 ? _j : 0;
            this.encoder = new UniversalEncoder({
                charSet: this.charSet,
                maxLen: this.maxLen,
                useTokenizer: this.useTokenizer,
                tokenizerDelimiter: this.tokenizerDelimiter,
                mode: this.useTokenizer ? 'token' : 'char'
            });
            this.inputWeights = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, cfg.maxLen));
            this.biases = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, 1));
            this.model = null;
        }
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
        pseudoInverse(H, lambda = 1e-3) {
            const Ht = Matrix.transpose(H);
            const HtH = Matrix.multiply(Ht, H);
            const HtH_reg = Matrix.addRegularization(HtH, lambda);
            const HtH_inv = Matrix.inverse(HtH_reg);
            return Matrix.multiply(HtH_inv, Ht);
        }
        randomMatrix(rows, cols) {
            if (this.config.weightInit === "xavier") {
                if (this.verbose)
                    console.log(`✨ Xavier init with limit sqrt(6/${rows}+${cols})`);
                const limit = Math.sqrt(6 / (rows + cols));
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 * limit - limit));
            }
            else {
                if (this.verbose)
                    console.log(`✨ Uniform init [-1,1]`);
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 - 1));
            }
        }
        setCategories(categories) {
            this.categories = categories;
        }
        loadModelFromJSON(json) {
            try {
                const parsed = JSON.parse(json);
                this.model = parsed;
                this.savedModelJSON = json;
                if (this.verbose)
                    console.log(`✅ ${this.modelName} Model loaded from JSON`);
            }
            catch (e) {
                console.error(`❌ Failed to load ${this.modelName} model from JSON:`, e);
            }
        }
        trainFromData(X, Y, options) {
            const reuseWeights = (options === null || options === void 0 ? void 0 : options.reuseWeights) === true;
            let W, b;
            if (reuseWeights && this.model) {
                W = this.model.W;
                b = this.model.b;
                if (this.verbose)
                    console.log("🔄 Reusing existing weights/biases for training.");
            }
            else {
                W = this.randomMatrix(this.hiddenUnits, X[0].length);
                b = this.randomMatrix(this.hiddenUnits, 1);
                if (this.verbose)
                    console.log("✨ Initializing fresh weights/biases for training.");
            }
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            let H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            if (this.dropout > 0) {
                const keepProb = 1 - this.dropout;
                for (let i = 0; i < H.length; i++) {
                    for (let j = 0; j < H[0].length; j++) {
                        if (Math.random() < this.dropout) {
                            H[i][j] = 0;
                        }
                        else {
                            H[i][j] /= keepProb;
                        }
                    }
                }
            }
            if (options === null || options === void 0 ? void 0 : options.weights) {
                const W_arr = options.weights;
                if (W_arr.length !== H.length) {
                    throw new Error(`Weight array length ${W_arr.length} does not match sample count ${H.length}`);
                }
                // Scale each row by sqrt(weight)
                H = H.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
                Y = Y.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
            }
            const H_pinv = this.pseudoInverse(H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.model = { W, b, beta };
            const predictions = Matrix.multiply(H, beta);
            if (this.metrics) {
                const rmse = this.calculateRMSE(Y, predictions);
                const mae = this.calculateMAE(Y, predictions);
                const acc = this.calculateAccuracy(Y, predictions);
                const f1 = this.calculateF1Score(Y, predictions);
                const ce = this.calculateCrossEntropy(Y, predictions);
                const r2 = this.calculateR2Score(Y, predictions);
                const results = {};
                let allPassed = true;
                if (this.metrics.rmse !== undefined) {
                    results.rmse = rmse;
                    if (rmse > this.metrics.rmse)
                        allPassed = false;
                }
                if (this.metrics.mae !== undefined) {
                    results.mae = mae;
                    if (mae > this.metrics.mae)
                        allPassed = false;
                }
                if (this.metrics.accuracy !== undefined) {
                    results.accuracy = acc;
                    if (acc < this.metrics.accuracy)
                        allPassed = false;
                }
                if (this.metrics.f1 !== undefined) {
                    results.f1 = f1;
                    if (f1 < this.metrics.f1)
                        allPassed = false;
                }
                if (this.metrics.crossEntropy !== undefined) {
                    results.crossEntropy = ce;
                    if (ce > this.metrics.crossEntropy)
                        allPassed = false;
                }
                if (this.metrics.r2 !== undefined) {
                    results.r2 = r2;
                    if (r2 < this.metrics.r2)
                        allPassed = false;
                }
                if (this.verbose)
                    this.logMetrics(results);
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify(this.model);
                    if (this.verbose)
                        console.log("✅ Model passed thresholds and was saved to JSON.");
                    if (this.config.exportFileName) {
                        this.saveModelAsJSONFile(this.config.exportFileName);
                    }
                }
                else {
                    if (this.verbose)
                        console.log("❌ Model not saved: One or more thresholds not met.");
                }
            }
            else {
                // No metrics—always save the model
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose)
                    console.log("✅ Model trained with no metrics—saved by default.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            }
        }
        train(augmentationOptions, weights) {
            const X = [];
            let Y = [];
            this.categories.forEach((cat, i) => {
                const variants = Augment.generateVariants(cat, this.charSet, augmentationOptions);
                for (const variant of variants) {
                    const vec = this.encoder.normalize(this.encoder.encode(variant));
                    X.push(vec);
                    Y.push(this.oneHot(this.categories.length, i));
                }
            });
            const W = this.randomMatrix(this.hiddenUnits, X[0].length);
            const b = this.randomMatrix(this.hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            let H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            if (this.dropout > 0) {
                const keepProb = 1 - this.dropout;
                for (let i = 0; i < H.length; i++) {
                    for (let j = 0; j < H[0].length; j++) {
                        if (Math.random() < this.dropout) {
                            H[i][j] = 0;
                        }
                        else {
                            H[i][j] /= keepProb;
                        }
                    }
                }
            }
            if (weights) {
                if (weights.length !== H.length) {
                    throw new Error(`Weight array length ${weights.length} does not match sample count ${H.length}`);
                }
                // Scale each row of H and Y by sqrt(weight)
                H = H.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
                Y = Y.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
            }
            const H_pinv = this.pseudoInverse(H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.model = { W, b, beta };
            const predictions = Matrix.multiply(H, beta);
            if (this.metrics) {
                const rmse = this.calculateRMSE(Y, predictions);
                const mae = this.calculateMAE(Y, predictions);
                const acc = this.calculateAccuracy(Y, predictions);
                const f1 = this.calculateF1Score(Y, predictions);
                const ce = this.calculateCrossEntropy(Y, predictions);
                const r2 = this.calculateR2Score(Y, predictions);
                const results = {};
                let allPassed = true;
                if (this.metrics.rmse !== undefined) {
                    results.rmse = rmse;
                    if (rmse > this.metrics.rmse)
                        allPassed = false;
                }
                if (this.metrics.mae !== undefined) {
                    results.mae = mae;
                    if (mae > this.metrics.mae)
                        allPassed = false;
                }
                if (this.metrics.accuracy !== undefined) {
                    results.accuracy = acc;
                    if (acc < this.metrics.accuracy)
                        allPassed = false;
                }
                if (this.metrics.f1 !== undefined) {
                    results.f1 = f1;
                    if (f1 < this.metrics.f1)
                        allPassed = false;
                }
                if (this.metrics.crossEntropy !== undefined) {
                    results.crossEntropy = ce;
                    if (ce > this.metrics.crossEntropy)
                        allPassed = false;
                }
                if (this.metrics.r2 !== undefined) {
                    results.r2 = r2;
                    if (r2 < this.metrics.r2)
                        allPassed = false;
                }
                if (this.verbose) {
                    this.logMetrics(results);
                }
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify(this.model);
                    if (this.verbose)
                        console.log("✅ Model passed thresholds and was saved to JSON.");
                    if (this.config.exportFileName) {
                        this.saveModelAsJSONFile(this.config.exportFileName);
                    }
                }
                else {
                    if (this.verbose)
                        console.log("❌ Model not saved: One or more thresholds not met.");
                }
            }
            else {
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose)
                    console.log("✅ Model trained with no metrics—saved by default.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            }
        }
        logMetrics(results) {
            var _a, _b, _c, _d, _e, _f;
            const logLines = [`📋 ${this.modelName} — Metrics Summary:`];
            const push = (label, value, threshold, cmp) => {
                if (threshold !== undefined)
                    logLines.push(`  ${label}: ${value.toFixed(4)} (threshold: ${cmp} ${threshold})`);
            };
            push('RMSE', results.rmse, (_a = this.metrics) === null || _a === void 0 ? void 0 : _a.rmse, '<=');
            push('MAE', results.mae, (_b = this.metrics) === null || _b === void 0 ? void 0 : _b.mae, '<=');
            push('Accuracy', results.accuracy, (_c = this.metrics) === null || _c === void 0 ? void 0 : _c.accuracy, '>=');
            push('F1 Score', results.f1, (_d = this.metrics) === null || _d === void 0 ? void 0 : _d.f1, '>=');
            push('Cross-Entropy', results.crossEntropy, (_e = this.metrics) === null || _e === void 0 ? void 0 : _e.crossEntropy, '<=');
            push('R² Score', results.r2, (_f = this.metrics) === null || _f === void 0 ? void 0 : _f.r2, '>=');
            if (this.verbose)
                console.log('\n' + logLines.join('\n'));
            if (this.logToFile) {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const logFile = this.config.logFileName || `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_metrics_${timestamp}.txt`;
                const blob = new Blob([logLines.join('\n')], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = logFile;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        }
        saveModelAsJSONFile(filename) {
            if (!this.savedModelJSON) {
                if (this.verbose)
                    console.warn("No model saved — did not meet metric thresholds.");
                return;
            }
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const fallback = `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.json`;
            const finalName = filename || this.config.exportFileName || fallback;
            const blob = new Blob([this.savedModelJSON], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = finalName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            if (this.verbose)
                console.log(`📦 Model exported as ${finalName}`);
        }
        predict(text, topK = 5) {
            if (!this.model)
                throw new Error("Model not trained.");
            const vec = this.encoder.normalize(this.encoder.encode(text));
            const { W, b, beta } = this.model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        }
        predictFromVector(inputVec, topK = 5) {
            if (!this.model)
                throw new Error("Model not trained.");
            const { W, b, beta } = this.model;
            const tempH = Matrix.multiply(inputVec, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            return Matrix.multiply(H, beta).map(rawOutput => {
                const probs = Activations.softmax(rawOutput);
                return probs
                    .map((p, i) => ({ label: this.categories[i], prob: p }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, topK);
            });
        }
        calculateRMSE(Y, P) {
            const N = Y.length;
            let sum = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    const diff = Y[i][j] - P[i][j];
                    sum += diff * diff;
                }
            }
            return Math.sqrt(sum / (N * Y[0].length));
        }
        calculateMAE(Y, P) {
            const N = Y.length;
            let sum = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    sum += Math.abs(Y[i][j] - P[i][j]);
                }
            }
            return sum / (N * Y[0].length);
        }
        calculateAccuracy(Y, P) {
            let correct = 0;
            for (let i = 0; i < Y.length; i++) {
                const yMax = Y[i].indexOf(Math.max(...Y[i]));
                const pMax = P[i].indexOf(Math.max(...P[i]));
                if (yMax === pMax)
                    correct++;
            }
            return correct / Y.length;
        }
        calculateF1Score(Y, P) {
            let tp = 0, fp = 0, fn = 0;
            for (let i = 0; i < Y.length; i++) {
                const yIdx = Y[i].indexOf(1);
                const pIdx = P[i].indexOf(Math.max(...P[i]));
                if (yIdx === pIdx)
                    tp++;
                else {
                    fp++;
                    fn++;
                }
            }
            const precision = tp / (tp + fp || 1);
            const recall = tp / (tp + fn || 1);
            return 2 * (precision * recall) / (precision + recall || 1);
        }
        calculateCrossEntropy(Y, P) {
            let loss = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    const pred = Math.min(Math.max(P[i][j], 1e-15), 1 - 1e-15);
                    loss += -Y[i][j] * Math.log(pred);
                }
            }
            return loss / Y.length;
        }
        calculateR2Score(Y, P) {
            const Y_mean = Y[0].map((_, j) => Y.reduce((sum, y) => sum + y[j], 0) / Y.length);
            let ssRes = 0, ssTot = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    ssRes += Math.pow(Y[i][j] - P[i][j], 2);
                    ssTot += Math.pow(Y[i][j] - Y_mean[j], 2);
                }
            }
            return 1 - ssRes / ssTot;
        }
        computeHiddenLayer(X) {
            if (!this.model)
                throw new Error("Model not trained.");
            const WX = Matrix.multiply(X, Matrix.transpose(this.model.W));
            const WXb = WX.map(row => row.map((val, j) => val + this.model.b[j][0]));
            const activationFn = Activations.get(this.activation);
            return WXb.map(row => row.map(activationFn));
        }
        getEmbedding(X) {
            return this.computeHiddenLayer(X);
        }
    }

    class ELMChain {
        constructor(encoders) {
            this.encoders = encoders;
        }
        getEmbedding(input) {
            let out = input;
            for (const encoder of this.encoders) {
                out = encoder.getEmbedding(out);
            }
            return out;
        }
    }

    const embeddingStore = [];
    function addEmbedding(record) {
        embeddingStore.push(record);
    }
    function searchEmbeddings$1(queryEmbedding, topK = 5) {
        const scored = embeddingStore.map(r => (Object.assign(Object.assign({}, r), { score: cosineSimilarity(queryEmbedding, r.embedding) })));
        return scored.sort((a, b) => b.score - a.score).slice(0, topK);
    }
    function cosineSimilarity(a, b) {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
        return dot / (normA * normB);
    }

    function evaluateEnsembleRetrieval(queries, reference, chains, k) {
        let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
        function cosine(a, b) {
            const dot = a.reduce((s, ai, i) => s + ai * b[i], 0);
            const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
            const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
            return dot / (normA * normB);
        }
        console.log("🔹 Precomputing embeddings...");
        // Precompute embeddings for each chain
        const chainQueryEmbeddings = chains.map(chain => chain.getEmbedding(queries.map(q => q.embedding)));
        const chainReferenceEmbeddings = chains.map(chain => chain.getEmbedding(reference.map(r => r.embedding)));
        console.log("✅ Precomputation complete. Starting retrieval evaluation...");
        queries.forEach((q, i) => {
            if (i % 10 === 0)
                console.log(`🔍 Query ${i + 1}/${queries.length}`);
            const ensembleScores = [];
            for (let j = 0; j < reference.length; j++) {
                let sum = 0;
                for (let c = 0; c < chains.length; c++) {
                    const qEmb = chainQueryEmbeddings[c][i];
                    const rEmb = chainReferenceEmbeddings[c][j];
                    sum += cosine(qEmb, rEmb);
                }
                ensembleScores.push({
                    label: reference[j].metadata.label || "",
                    score: sum / chains.length
                });
            }
            ensembleScores.sort((a, b) => b.score - a.score);
            const ranked = ensembleScores.map(s => s.label);
            const correctLabel = q.metadata.label || "";
            if (ranked[0] === correctLabel)
                hitsAt1++;
            if (ranked.slice(0, k).includes(correctLabel))
                hitsAtK++;
            const rank = ranked.indexOf(correctLabel);
            reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
        });
        return {
            recallAt1: hitsAt1 / queries.length,
            recallAtK: hitsAtK / queries.length,
            mrr: reciprocalRanks / queries.length
        };
    }

    function evaluateRetrieval(queries, store, chain, k = 5) {
        let recallHits = 0;
        let reciprocalRanks = [];
        for (const query of queries) {
            const queryEmbedding = chain.getEmbedding([query.embedding])[0];
            const results = searchEmbeddings(queryEmbedding, k, store);
            const trueLabel = query.metadata.label;
            const labels = results.map(r => r.metadata.label);
            // Recall@K
            if (labels.includes(trueLabel)) {
                recallHits++;
            }
            // MRR
            const rank = labels.indexOf(trueLabel);
            if (rank !== -1) {
                reciprocalRanks.push(1 / (rank + 1));
            }
            else {
                reciprocalRanks.push(0);
            }
        }
        const recallAtK = recallHits / queries.length;
        const mrr = reciprocalRanks.reduce((a, b) => a + b, 0) / queries.length;
        return { recallAtK, mrr };
    }
    function searchEmbeddings(queryEmbedding, topK, store) {
        const scored = store.map(r => (Object.assign(Object.assign({}, r), { score: CosineSimilarity(queryEmbedding, r.embedding) })));
        return scored.sort((a, b) => b.score - a.score).slice(0, topK);
    }
    function CosineSimilarity(a, b) {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
        return dot / (normA * normB);
    }
    function evaluateRetrievalSimple(queries, store, embedFn, k = 5) {
        let recallHits = 0;
        let reciprocalRanks = [];
        for (const query of queries) {
            const queryEmbedding = embedFn([query.embedding])[0];
            const results = searchEmbeddings(queryEmbedding, k, store);
            const trueLabel = query.metadata.label;
            const labels = results.map(r => r.metadata.label);
            if (labels.includes(trueLabel))
                recallHits++;
            const rank = labels.indexOf(trueLabel);
            reciprocalRanks.push(rank !== -1 ? 1 / (rank + 1) : 0);
        }
        const recallAtK = recallHits / queries.length;
        const mrr = reciprocalRanks.reduce((a, b) => a + b, 0) / queries.length;
        return { recallAtK, mrr };
    }

    class OnlineELM {
        // NOTE: We keep P in hidden-space only: (hiddenUnits x hiddenUnits).
        // Then apply it blockwise for multi-output targets during updates.
        constructor(inputDim, hiddenUnits, outputDim, act, lambda = 1e-3) {
            this.inputDim = inputDim;
            this.hiddenUnits = hiddenUnits;
            this.outputDim = outputDim;
            this.lambda = lambda;
            this.act = act;
            this.W = new Float64Array(hiddenUnits * inputDim);
            this.b = new Float64Array(hiddenUnits);
            // Kaiming-ish init for W; small bias
            const scale = Math.sqrt(2 / inputDim);
            for (let i = 0; i < this.W.length; i++)
                this.W[i] = (Math.random() * 2 - 1) * scale;
            for (let i = 0; i < hiddenUnits; i++)
                this.b[i] = (Math.random() * 2 - 1) * 0.01;
            // beta: [hiddenUnits x outputDim]
            this.beta = new Float64Array(hiddenUnits * outputDim);
            // P starts as (1/λ) I in hidden space
            this.P = new Float64Array(hiddenUnits * hiddenUnits);
            for (let i = 0; i < hiddenUnits; i++)
                this.P[i * hiddenUnits + i] = 1 / lambda;
        }
        // Compute hidden activations H for a batch X (rows = batch, cols = inputDim)
        // Returns Float64Array [batchSize x hiddenUnits]
        hiddenBatch(X, batchSize) {
            const H = new Float64Array(batchSize * this.hiddenUnits);
            for (let r = 0; r < batchSize; r++) {
                const baseX = r * this.inputDim;
                for (let h = 0; h < this.hiddenUnits; h++) {
                    let sum = this.b[h];
                    const baseW = h * this.inputDim;
                    for (let c = 0; c < this.inputDim; c++) {
                        sum += this.W[baseW + c] * X[baseX + c];
                    }
                    H[r * this.hiddenUnits + h] = this.act(sum);
                }
            }
            return H;
        }
        // OS-ELM partial fit on a chunk:
        // X: [batchSize x inputDim], T: [batchSize x outputDim]
        partialFit(X, T, batchSize) {
            const H = this.hiddenBatch(X, batchSize); // [B x H]
            // Compute common terms in hidden space once: S = I + H P H^T  => [B x B]
            const HPHt = symm_BxB_from_HPHt(H, this.P, this.hiddenUnits, batchSize); // BxB
            addIdentityInPlace(HPHt, batchSize); // S
            const S_inv = invSymmetric(HPHt, batchSize); // [B x B]
            // K = P H^T S^{-1}    => [H x B]
            const PHt = mul_P_Ht(this.P, H, this.hiddenUnits, batchSize); // [H x B]
            const K = mul(false, PHt, S_inv, this.hiddenUnits, batchSize, batchSize); // [H x B]
            // Compute residual (T - H beta)  => [B x O]
            const HB = mul(false, H, this.beta, batchSize, this.hiddenUnits, this.outputDim); // [B x O]
            for (let i = 0; i < HB.length; i++)
                HB[i] = T[i] - HB[i];
            // beta += K * (T - H beta)   => [H x O]
            const delta = mul(false, K, HB, this.hiddenUnits, batchSize, this.outputDim);
            for (let i = 0; i < this.beta.length; i++)
                this.beta[i] += delta[i];
            // P -= K H P  => KHP: [H x H]
            const KH = mul(false, K, H, this.hiddenUnits, batchSize, this.hiddenUnits);
            const KHP = mul(false, KH, this.P, this.hiddenUnits, this.hiddenUnits, this.hiddenUnits);
            for (let i = 0; i < this.P.length; i++)
                this.P[i] -= KHP[i];
            // help GC
            // (arrays go out of scope)
        }
        // Predict for a single feature vector x (length = inputDim)
        predictOne(x) {
            const h = new Float64Array(this.hiddenUnits);
            for (let j = 0; j < this.hiddenUnits; j++) {
                let s = this.b[j];
                const baseW = j * this.inputDim;
                for (let c = 0; c < this.inputDim; c++)
                    s += this.W[baseW + c] * x[c];
                h[j] = this.act(s);
            }
            // y = h^T beta
            const y = new Float64Array(this.outputDim);
            for (let o = 0; o < this.outputDim; o++) {
                let s = 0;
                for (let j = 0; j < this.hiddenUnits; j++)
                    s += h[j] * this.beta[j * this.outputDim + o];
                y[o] = s;
            }
            return y;
        }
        toJSON() {
            return JSON.stringify({
                inputDim: this.inputDim,
                hiddenUnits: this.hiddenUnits,
                outputDim: this.outputDim,
                lambda: this.lambda,
                W: Array.from(this.W),
                b: Array.from(this.b),
                beta: Array.from(this.beta),
                P: Array.from(this.P)
            });
        }
        static fromJSON(json, act) {
            const o = JSON.parse(json);
            const mdl = new OnlineELM(o.inputDim, o.hiddenUnits, o.outputDim, act, o.lambda);
            mdl.W.set(o.W);
            mdl.b.set(o.b);
            mdl.beta.set(o.beta);
            mdl.P.set(o.P);
            return mdl;
        }
    }
    /* ---------- tiny linear algebra helpers (hidden-space) ---------- */
    // Build S = I + H P H^T  (B x B), where H: [B x H], P: [H x H]
    function symm_BxB_from_HPHt(H, P, Hdim, B) {
        // tmp = H P  => [B x H]
        const tmp = mul(false, H, P, B, Hdim, Hdim);
        // S = tmp H^T => [B x B]
        const S = mul(false, tmp, H, B, Hdim, B, true); // treat second as transposed
        return S;
    }
    // Generic multiply: A [m x k] * B [k x n] => [m x n]
    // If B_is_transposed is true, interpret B as [n x k] transposed
    function mul(A_t, A, B, m, k, n, B_is_transposed = false) {
        const out = new Float64Array(m * n);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let s = 0;
                for (let t = 0; t < k; t++) {
                    const a = A[i * k + t];
                    const b = B_is_transposed ? B[j * k + t] : B[t * n + j];
                    s += a * b;
                }
                out[i * n + j] = s;
            }
        }
        return out;
    }
    function addIdentityInPlace(M, n) {
        for (let i = 0; i < n; i++)
            M[i * n + i] += 1;
    }
    // Small symmetric positive-definite inverse (Cholesky)
    // n = batch size (keep small, e.g. 64-1024)
    function invSymmetric(S, n) {
        // Cholesky decomposition S = L L^T
        const L = S.slice();
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let sum = L[i * n + j];
                for (let k = 0; k < j; k++)
                    sum -= L[i * n + k] * L[j * n + k];
                if (i === j) {
                    L[i * n + j] = Math.sqrt(Math.max(sum, 1e-12));
                }
                else {
                    L[i * n + j] = sum / L[j * n + j];
                }
            }
            for (let j = i + 1; j < n; j++)
                L[i * n + j] = 0;
        }
        // Solve L Y = I, then L^T X = Y => X = S^{-1}
        const inv = new Float64Array(n * n);
        // initialize inv to identity for RHS
        for (let i = 0; i < n; i++)
            inv[i * n + i] = 1;
        // forward solve Y: overwrite inv with Y
        for (let col = 0; col < n; col++) {
            for (let i = 0; i < n; i++) {
                let sum = inv[i * n + col];
                for (let k = 0; k < i; k++)
                    sum -= L[i * n + k] * inv[k * n + col];
                inv[i * n + col] = sum / L[i * n + i];
            }
            // back solve X
            for (let i = n - 1; i >= 0; i--) {
                let sum = inv[i * n + col];
                for (let k = i + 1; k < n; k++)
                    sum -= L[k * n + i] * inv[k * n + col];
                inv[i * n + col] = sum / L[i * n + i];
            }
        }
        return inv;
    }
    // PH^T = P * H^T  where P: [H x H], H: [B x H]  => [H x B]
    function mul_P_Ht(P, H, Hdim, B) {
        const out = new Float64Array(Hdim * B);
        for (let i = 0; i < Hdim; i++) {
            for (let j = 0; j < B; j++) {
                let s = 0;
                for (let k = 0; k < Hdim; k++)
                    s += P[i * Hdim + k] * H[j * Hdim + k];
                out[i * B + j] = s;
            }
        }
        return out;
    }

    class TFIDF {
        constructor(corpusDocs) {
            this.termFrequency = {};
            this.inverseDocFreq = {};
            this.wordsInDoc = [];
            this.processedWords = [];
            this.scores = {};
            this.corpus = "";
            this.corpus = corpusDocs.join(" ");
            const wordsFinal = [];
            const re = /[^a-zA-Z0-9]+/g;
            corpusDocs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                tokens.forEach(word => {
                    const cleaned = word.replace(re, " ");
                    wordsFinal.push(...cleaned.split(/\s+/).filter(Boolean));
                });
            });
            this.wordsInDoc = wordsFinal;
            this.processedWords = TFIDF.processWords(wordsFinal);
            // Compute term frequency
            this.processedWords.forEach(token => {
                this.termFrequency[token] = (this.termFrequency[token] || 0) + 1;
            });
            // Compute inverse document frequency
            for (const term in this.termFrequency) {
                const count = TFIDF.countDocsContainingTerm(corpusDocs, term);
                this.inverseDocFreq[term] = Math.log(corpusDocs.length / (1 + count));
            }
        }
        static countDocsContainingTerm(corpusDocs, term) {
            return corpusDocs.reduce((acc, doc) => (doc.includes(term) ? acc + 1 : acc), 0);
        }
        static processWords(words) {
            const filtered = TFIDF.removeStopWordsAndStem(words).map(w => TFIDF.lemmatize(w));
            const bigrams = TFIDF.generateNGrams(filtered, 2);
            const trigrams = TFIDF.generateNGrams(filtered, 3);
            return [...filtered, ...bigrams, ...trigrams];
        }
        static removeStopWordsAndStem(words) {
            const stopWords = new Set([
                "a", "and", "the", "is", "to", "of", "in", "it", "that", "you",
                "this", "for", "on", "are", "with", "as", "be", "by", "at", "from",
                "or", "an", "but", "not", "we"
            ]);
            return words.filter(w => !stopWords.has(w)).map(w => TFIDF.advancedStem(w));
        }
        static advancedStem(word) {
            const programmingKeywords = new Set([
                "func", "package", "import", "interface", "go",
                "goroutine", "channel", "select", "struct",
                "map", "slice", "var", "const", "type",
                "defer", "fallthrough"
            ]);
            if (programmingKeywords.has(word))
                return word;
            const suffixes = ["es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"];
            for (const suffix of suffixes) {
                if (word.endsWith(suffix)) {
                    if (suffix === "es" && word.length > 2 && word[word.length - 3] === "i") {
                        return word.slice(0, -2);
                    }
                    return word.slice(0, -suffix.length);
                }
            }
            return word;
        }
        static lemmatize(word) {
            const rules = {
                execute: "execute",
                running: "run",
                returns: "return",
                defined: "define",
                compiles: "compile",
                calls: "call",
                creating: "create",
                invoke: "invoke",
                declares: "declare",
                references: "reference",
                implements: "implement",
                utilizes: "utilize",
                tests: "test",
                loops: "loop",
                deletes: "delete",
                functions: "function"
            };
            if (rules[word])
                return rules[word];
            if (word.endsWith("ing"))
                return word.slice(0, -3);
            if (word.endsWith("ed"))
                return word.slice(0, -2);
            return word;
        }
        static generateNGrams(tokens, n) {
            if (tokens.length < n)
                return [];
            const ngrams = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                ngrams.push(tokens.slice(i, i + n).join(" "));
            }
            return ngrams;
        }
        calculateScores() {
            const totalWords = this.processedWords.length;
            const scores = {};
            this.processedWords.forEach(token => {
                const tf = this.termFrequency[token] || 0;
                scores[token] = (tf / totalWords) * (this.inverseDocFreq[token] || 0);
            });
            this.scores = scores;
            return scores;
        }
        extractKeywords(topN) {
            const entries = Object.entries(this.scores).sort((a, b) => b[1] - a[1]);
            return Object.fromEntries(entries.slice(0, topN));
        }
        processedWordsIndex(word) {
            return this.processedWords.indexOf(word);
        }
    }
    class TFIDFVectorizer {
        constructor(docs, maxVocabSize = 2000) {
            this.docTexts = docs;
            this.tfidf = new TFIDF(docs);
            // Collect all unique terms with frequencies
            const termFreq = {};
            docs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
                const processed = TFIDF.processWords(cleaned);
                processed.forEach(t => {
                    termFreq[t] = (termFreq[t] || 0) + 1;
                });
            });
            // Sort terms by frequency descending
            const sortedTerms = Object.entries(termFreq)
                .sort((a, b) => b[1] - a[1])
                .slice(0, maxVocabSize)
                .map(([term]) => term);
            this.vocabulary = sortedTerms;
            console.log(`✅ TFIDFVectorizer vocabulary capped at: ${this.vocabulary.length} terms.`);
        }
        /**
         * Returns the dense TFIDF vector for a given document text.
         */
        vectorize(doc) {
            const tokens = doc.split(/\s+/);
            const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
            const processed = TFIDF.processWords(cleaned);
            // Compute term frequency in this document
            const termFreq = {};
            processed.forEach(token => {
                termFreq[token] = (termFreq[token] || 0) + 1;
            });
            const totalTerms = processed.length;
            return this.vocabulary.map(term => {
                const tf = totalTerms > 0 ? (termFreq[term] || 0) / totalTerms : 0;
                const idf = this.tfidf.inverseDocFreq[term] || 0;
                return tf * idf;
            });
        }
        /**
         * Returns vectors for all original training docs.
         */
        vectorizeAll() {
            return this.docTexts.map(doc => this.vectorize(doc));
        }
        /**
         * Optional L2 normalization utility.
         */
        static l2normalize(vec) {
            const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
            return norm === 0 ? vec : vec.map(x => x / norm);
        }
    }

    class KNN {
        /**
         * Compute cosine similarity between two numeric vectors.
         */
        static cosineSimilarity(vec1, vec2) {
            let dot = 0, norm1 = 0, norm2 = 0;
            for (let i = 0; i < vec1.length; i++) {
                dot += vec1[i] * vec2[i];
                norm1 += vec1[i] * vec1[i];
                norm2 += vec2[i] * vec2[i];
            }
            if (norm1 === 0 || norm2 === 0)
                return 0;
            return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        }
        /**
         * Compute Euclidean distance between two numeric vectors.
         */
        static euclideanDistance(vec1, vec2) {
            let sum = 0;
            for (let i = 0; i < vec1.length; i++) {
                const diff = vec1[i] - vec2[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
        /**
         * Find k nearest neighbors.
         * @param queryVec - Query vector
         * @param dataset - Dataset to search
         * @param k - Number of neighbors
         * @param topX - Number of top results to return
         * @param metric - Similarity metric
         */
        static find(queryVec, dataset, k = 5, topX = 3, metric = "cosine") {
            const similarities = dataset.map((item, idx) => {
                let score;
                if (metric === "cosine") {
                    score = this.cosineSimilarity(queryVec, item.vector);
                }
                else {
                    // For Euclidean, invert distance so higher = closer
                    const dist = this.euclideanDistance(queryVec, item.vector);
                    score = -dist;
                }
                return { index: idx, score };
            });
            similarities.sort((a, b) => b.score - a.score);
            const labelWeights = {};
            for (let i = 0; i < Math.min(k, similarities.length); i++) {
                const label = dataset[similarities[i].index].label;
                const weight = similarities[i].score;
                labelWeights[label] = (labelWeights[label] || 0) + weight;
            }
            const weightedLabels = Object.entries(labelWeights)
                .map(([label, weight]) => ({ label, weight }))
                .sort((a, b) => b.weight - a.weight);
            return weightedLabels.slice(0, topX);
        }
    }

    // BindUI.ts - Utility to bind ELM model to HTML inputs and outputs
    function bindAutocompleteUI({ model, inputElement, outputElement, topK = 5 }) {
        inputElement.addEventListener('input', () => {
            const typed = inputElement.value.trim();
            if (typed.length === 0) {
                outputElement.innerHTML = '<em>Start typing...</em>';
                return;
            }
            try {
                const results = model.predict(typed, topK);
                outputElement.innerHTML = results.map(r => `
                <div><strong>${r.label}</strong>: ${(r.prob * 100).toFixed(1)}%</div>
            `).join('');
            }
            catch (e) {
                const message = e instanceof Error ? e.message : 'Unknown error';
                outputElement.innerHTML = `<span style="color: red;">Error: ${message}</span>`;
            }
        });
    }

    // Presets.ts - Reusable configuration presets for ELM
    const EnglishTokenPreset = {
        categories: [],
        hiddenUnits: 120,
        maxLen: 20,
        activation: 'relu',
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: true,
        tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/,
        log: {}
    };

    // ✅ AutoComplete.ts patched to support (input, label) training and evaluation
    class AutoComplete {
        constructor(pairs, options) {
            var _a;
            this.trainPairs = pairs;
            this.activation = (_a = options.activation) !== null && _a !== void 0 ? _a : 'relu';
            const categories = Array.from(new Set(pairs.map(p => p.label)));
            this.elm = new ELM(Object.assign(Object.assign({}, EnglishTokenPreset), { categories, activation: this.activation, metrics: options.metrics, log: {
                    modelName: "AutoComplete",
                    verbose: options.verbose
                }, exportFileName: options.exportFileName }));
            bindAutocompleteUI({
                model: this.elm,
                inputElement: options.inputElement,
                outputElement: options.outputElement,
                topK: options.topK
            });
        }
        train() {
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = this.elm.encoder.normalize(this.elm.encoder.encode(input));
                const labelIndex = this.elm.categories.indexOf(label);
                if (labelIndex === -1)
                    continue;
                X.push(vec);
                Y.push(this.elm.oneHot(this.elm.categories.length, labelIndex));
            }
            this.elm.trainFromData(X, Y);
        }
        predict(input, topN = 1) {
            return this.elm.predict(input, topN).map(p => ({
                completion: p.label,
                prob: p.prob
            }));
        }
        getModel() {
            return this.elm;
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
        top1Accuracy(pairs) {
            var _a;
            let correct = 0;
            for (const { input, label } of pairs) {
                const [pred] = this.predict(input, 1);
                if (((_a = pred === null || pred === void 0 ? void 0 : pred.completion) === null || _a === void 0 ? void 0 : _a.toLowerCase().trim()) === label.toLowerCase().trim()) {
                    correct++;
                }
            }
            return correct / pairs.length;
        }
        crossEntropy(pairs) {
            var _a;
            let totalLoss = 0;
            for (const { input, label } of pairs) {
                const preds = this.predict(input, 5);
                const match = preds.find(p => p.completion.toLowerCase().trim() === label.toLowerCase().trim());
                const prob = (_a = match === null || match === void 0 ? void 0 : match.prob) !== null && _a !== void 0 ? _a : 1e-6;
                totalLoss += -Math.log(prob); // ⬅ switched from log2 to natural log
            }
            return totalLoss / pairs.length;
        }
        internalCrossEntropy(verbose = false) {
            const { model, encoder, categories } = this.elm;
            if (!model) {
                if (verbose)
                    console.warn("⚠️ Cannot compute internal cross-entropy: model not trained.");
                return Infinity;
            }
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = encoder.normalize(encoder.encode(input));
                const labelIdx = categories.indexOf(label);
                if (labelIdx === -1)
                    continue;
                X.push(vec);
                Y.push(this.elm.oneHot(categories.length, labelIdx));
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const preds = Matrix.multiply(H, beta);
            const ce = this.elm.calculateCrossEntropy(Y, preds);
            if (verbose) {
                console.log(`📏 Internal Cross-Entropy (full model eval): ${ce.toFixed(4)}`);
            }
            return ce;
        }
    }

    class CharacterLangEncoderELM {
        constructor(config) {
            if (!config.hiddenUnits || !config.activation) {
                throw new Error("CharacterLangEncoderELM requires defined hiddenUnits and activation");
            }
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "CharacterLangEncoderELM",
                    verbose: config.log.verbose
                }, useTokenizer: true });
            this.elm = new ELM(this.config);
            // Forward ELM-specific options
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        train(inputStrings, labels) {
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            this.elm.train(); // assumes encoder + categories are set
        }
        /**
         * Returns dense vector (embedding) rather than label prediction
         */
        encode(text) {
            const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            // dense feature vector
            return Matrix.multiply(H, beta)[0];
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    class FeatureCombinerELM {
        constructor(config) {
            if (typeof config.hiddenUnits !== 'number') {
                throw new Error('FeatureCombinerELM requires hiddenUnits');
            }
            if (!config.activation) {
                throw new Error('FeatureCombinerELM requires activation');
            }
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: false, log: {
                    modelName: "FeatureCombinerELM",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        /**
         * Combines encoder vector and metadata into one input vector
         */
        static combineFeatures(encodedVec, meta) {
            return [...encodedVec, ...meta];
        }
        /**
         * Train the ELM using combined features and labels
         */
        train(encoded, metas, labels) {
            if (!this.config.hiddenUnits || !this.config.activation) {
                throw new Error("FeatureCombinerELM: config.hiddenUnits or activation is undefined.");
            }
            const X = encoded.map((vec, i) => FeatureCombinerELM.combineFeatures(vec, metas[i]));
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            const Y = labels.map(label => this.elm.oneHot(categories.length, categories.indexOf(label)));
            const W = this.elm['randomMatrix'](this.config.hiddenUnits, X[0].length);
            const b = this.elm['randomMatrix'](this.config.hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        /**
         * Predict from combined input and metadata
         */
        predict(encodedVec, meta, topK = 1) {
            const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
            const [results] = this.elm.predictFromVector(input, topK);
            return results;
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    /**
     * ConfidenceClassifierELM is a lightweight ELM wrapper
     * designed to classify whether an input prediction is likely to be high or low confidence.
     * It uses the same input format as FeatureCombinerELM (vector + meta).
     */
    class ConfidenceClassifierELM {
        constructor(config) {
            this.config = config;
            this.elm = new ELM(Object.assign(Object.assign({}, config), { categories: ['low', 'high'], useTokenizer: false, log: {
                    modelName: "ConfidenceClassifierELM",
                    verbose: config.log.verbose
                } }));
            // Forward optional ELM config extensions
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        train(vectors, metas, labels) {
            vectors.map((vec, i) => FeatureCombinerELM.combineFeatures(vec, metas[i]));
            const examples = vectors.map((vec, i) => ({
                input: FeatureCombinerELM.combineFeatures(vec, metas[i]),
                label: labels[i]
            }));
            this.elm.train(examples);
        }
        predict(vec, meta) {
            const input = FeatureCombinerELM.combineFeatures(vec, meta);
            const inputStr = JSON.stringify(input);
            return this.elm.predict(inputStr, 1);
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    /**
     * EncoderELM: Uses an ELM to convert strings into dense feature vectors.
     */
    class EncoderELM {
        constructor(config) {
            if (typeof config.hiddenUnits !== 'number') {
                throw new Error('EncoderELM requires config.hiddenUnits to be defined as a number');
            }
            if (!config.activation) {
                throw new Error('EncoderELM requires config.activation to be defined');
            }
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: true, log: {
                    modelName: "EncoderELM",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        /**
         * Custom training method for string → vector encoding.
         */
        train(inputStrings, targetVectors) {
            const X = inputStrings.map(s => this.elm.encoder.normalize(this.elm.encoder.encode(s)));
            const Y = targetVectors;
            const hiddenUnits = this.config.hiddenUnits;
            const inputDim = X[0].length;
            const W = this.elm['randomMatrix'](hiddenUnits, inputDim);
            const b = this.elm['randomMatrix'](hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        /**
         * Encodes an input string into a dense feature vector using the trained model.
         */
        encode(text) {
            const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            return Matrix.multiply(H, beta)[0];
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
        // ===================== NEW: Online / Incremental encoding API =====================
        /**
         * Initialize an online (OS-ELM/RLS) run for string→vector encoding.
         *
         * Provide outputDim, and EITHER inputDim OR a sampleText we can encode to infer inputDim.
         * hiddenUnits defaults to config.hiddenUnits; activation defaults to config.activation; lambda defaults to 1e-2.
         */
        beginOnline(opts) {
            var _a, _b, _c;
            const outputDim = opts.outputDim;
            if (!Number.isFinite(outputDim) || outputDim <= 0) {
                throw new Error('beginOnline: outputDim must be a positive integer.');
            }
            let inputDim = opts.inputDim;
            if (inputDim == null) {
                if (!opts.sampleText) {
                    throw new Error('beginOnline: provide either inputDim or sampleText to infer dimension.');
                }
                const probe = this.elm.encoder.normalize(this.elm.encoder.encode(opts.sampleText));
                inputDim = probe.length;
            }
            const hiddenUnits = (_a = opts.hiddenUnits) !== null && _a !== void 0 ? _a : this.config.hiddenUnits;
            const lambda = (_b = opts.lambda) !== null && _b !== void 0 ? _b : 1e-2;
            const actName = (_c = opts.activation) !== null && _c !== void 0 ? _c : this.config.activation;
            const act = actName === 'tanh' ? Math.tanh
                : actName === 'sigmoid' ? (x) => 1 / (1 + Math.exp(-x))
                    : (x) => (x > 0 ? x : 0); // relu default
            // Spin up the typed-array OnlineELM
            this.onlineMdl = new OnlineELM(inputDim, hiddenUnits, outputDim, act, lambda);
            this.onlineInputDim = inputDim;
            this.onlineOutputDim = outputDim;
        }
        /**
         * Online partial fit with pre-encoded vectors.
         * Each batch element supplies x (length = inputDim) and y (length = outputDim).
         * Memory-friendly: pack into typed arrays, update, discard.
         */
        partialTrainOnlineVectors(batch) {
            if (!this.onlineMdl || this.onlineInputDim == null || this.onlineOutputDim == null) {
                throw new Error('partialTrainOnlineVectors: call beginOnline() first.');
            }
            if (!batch.length)
                return;
            const B = batch.length;
            const D = this.onlineInputDim;
            const O = this.onlineOutputDim;
            const X = new Float64Array(B * D);
            const T = new Float64Array(B * O);
            for (let i = 0; i < B; i++) {
                const { x, y } = batch[i];
                if (x.length !== D)
                    throw new Error(`x length ${x.length} != inputDim ${D}`);
                if (y.length !== O)
                    throw new Error(`y length ${y.length} != outputDim ${O}`);
                X.set(x, i * D);
                T.set(y, i * O);
            }
            this.onlineMdl.partialFit(X, T, B);
        }
        /**
         * Online partial fit with raw texts.
         * We encode+normalize each text internally to build X, and use the given dense target vector y.
         */
        partialTrainOnlineTexts(batch) {
            if (!this.onlineMdl || this.onlineInputDim == null || this.onlineOutputDim == null) {
                throw new Error('partialTrainOnlineTexts: call beginOnline() first.');
            }
            if (!batch.length)
                return;
            const B = batch.length;
            const D = this.onlineInputDim;
            const O = this.onlineOutputDim;
            const X = new Float64Array(B * D);
            const T = new Float64Array(B * O);
            for (let i = 0; i < B; i++) {
                const { text, target } = batch[i];
                const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
                if (vec.length !== D)
                    throw new Error(`encoded text dim ${vec.length} != inputDim ${D}`);
                if (target.length !== O)
                    throw new Error(`target length ${target.length} != outputDim ${O}`);
                X.set(vec, i * D);
                T.set(target, i * O);
            }
            this.onlineMdl.partialFit(X, T, B);
        }
        /**
         * Finalize the online run by publishing learned weights into the standard model shape.
         * After this, the normal encode() path works unchanged.
         */
        endOnline() {
            if (!this.onlineMdl)
                return;
            const H = this.onlineMdl.hiddenUnits;
            const D = this.onlineMdl.inputDim;
            const O = this.onlineMdl.outputDim;
            const W = this.reshapeTo2D(this.onlineMdl.W, H, D); // [H x D]
            const b = this.reshapeCol(this.onlineMdl.b, H); // [H x 1]
            const beta = this.reshapeTo2D(this.onlineMdl.beta, H, O); // [H x O]
            this.elm['model'] = { W, b, beta };
            // clear online state
            this.onlineMdl = undefined;
            this.onlineInputDim = undefined;
            this.onlineOutputDim = undefined;
        }
        // ===================== small helpers =====================
        reshapeTo2D(buf, rows, cols) {
            const out = new Array(rows);
            for (let r = 0; r < rows; r++) {
                const row = new Array(cols);
                for (let c = 0; c < cols; c++)
                    row[c] = buf[r * cols + c];
                out[r] = row;
            }
            return out;
        }
        reshapeCol(buf, rows) {
            const out = new Array(rows);
            for (let r = 0; r < rows; r++)
                out[r] = [buf[r]];
            return out;
        }
    }

    // intentClassifier.ts - ELM-based intent classification engine
    class IntentClassifier {
        constructor(config) {
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.model = new ELM(config);
            if (config.metrics)
                this.model.metrics = config.metrics;
            if (config.exportFileName)
                this.model.config.exportFileName = config.exportFileName;
        }
        train(textLabelPairs, augmentationOptions) {
            const labelSet = Array.from(new Set(textLabelPairs.map(p => p.label)));
            this.model.setCategories(labelSet);
            this.model.train(augmentationOptions);
        }
        predict(text, topK = 1, threshold = 0) {
            return this.model.predict(text, topK).filter(r => r.prob >= threshold);
        }
        predictBatch(texts, topK = 1, threshold = 0) {
            return texts.map(text => this.predict(text, topK, threshold));
        }
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
        loadModelFromJSON(json) {
            this.model.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.model.saveModelAsJSONFile(filename);
        }
    }

    // IO.ts - Import/export utilities for labeled training data
    class IO {
        static importJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data))
                    throw new Error('Invalid format');
                return data.filter(item => typeof item.text === 'string' && typeof item.label === 'string');
            }
            catch (err) {
                console.error('Failed to parse training data JSON:', err);
                return [];
            }
        }
        static exportJSON(pairs) {
            return JSON.stringify(pairs, null, 2);
        }
        static importDelimited(text, delimiter = ',', hasHeader = true) {
            var _a, _b, _c, _d;
            const lines = text.trim().split('\n');
            const examples = [];
            const headers = hasHeader
                ? lines[0].split(delimiter).map(h => h.trim().toLowerCase())
                : lines[0].split(delimiter).length === 1
                    ? ['label']
                    : ['text', 'label'];
            const startIndex = hasHeader ? 1 : 0;
            for (let i = startIndex; i < lines.length; i++) {
                const parts = lines[i].split(delimiter);
                if (parts.length === 1) {
                    examples.push({ text: parts[0].trim(), label: parts[0].trim() });
                }
                else {
                    const textIdx = headers.indexOf('text');
                    const labelIdx = headers.indexOf('label');
                    const text = textIdx !== -1 ? (_a = parts[textIdx]) === null || _a === void 0 ? void 0 : _a.trim() : (_b = parts[0]) === null || _b === void 0 ? void 0 : _b.trim();
                    const label = labelIdx !== -1 ? (_c = parts[labelIdx]) === null || _c === void 0 ? void 0 : _c.trim() : (_d = parts[1]) === null || _d === void 0 ? void 0 : _d.trim();
                    if (text && label) {
                        examples.push({ text, label });
                    }
                }
            }
            return examples;
        }
        static exportDelimited(pairs, delimiter = ',', includeHeader = true) {
            const header = includeHeader ? `text${delimiter}label\n` : '';
            const rows = pairs.map(p => `${p.text.replace(new RegExp(delimiter, 'g'), '')}${delimiter}${p.label.replace(new RegExp(delimiter, 'g'), '')}`);
            return header + rows.join('\n');
        }
        static importCSV(csv, hasHeader = true) {
            return this.importDelimited(csv, ',', hasHeader);
        }
        static exportCSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, ',', includeHeader);
        }
        static importTSV(tsv, hasHeader = true) {
            return this.importDelimited(tsv, '\t', hasHeader);
        }
        static exportTSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, '\t', includeHeader);
        }
        static inferSchemaFromCSV(csv) {
            var _a;
            const lines = csv.trim().split('\n');
            if (lines.length === 0)
                return { fields: [] };
            const header = lines[0].split(',').map(h => h.trim().toLowerCase());
            const row = ((_a = lines[1]) === null || _a === void 0 ? void 0 : _a.split(',')) || [];
            const fields = header.map((name, i) => {
                var _a;
                const sample = (_a = row[i]) === null || _a === void 0 ? void 0 : _a.trim();
                let type = 'unknown';
                if (!sample)
                    type = 'unknown';
                else if (!isNaN(Number(sample)))
                    type = 'number';
                else if (sample === 'true' || sample === 'false')
                    type = 'boolean';
                else
                    type = 'string';
                return { name, type };
            });
            const suggestedMapping = {
                text: header.find(h => h.includes('text') || h.includes('utterance') || h.includes('input')) || header[0],
                label: header.find(h => h.includes('label') || h.includes('intent') || h.includes('tag')) || header[1] || header[0],
            };
            return { fields, suggestedMapping };
        }
        static inferSchemaFromJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data) || data.length === 0 || typeof data[0] !== 'object')
                    return { fields: [] };
                const keys = Object.keys(data[0]);
                const fields = keys.map(key => {
                    const val = data[0][key];
                    let type = 'unknown';
                    if (typeof val === 'string')
                        type = 'string';
                    else if (typeof val === 'number')
                        type = 'number';
                    else if (typeof val === 'boolean')
                        type = 'boolean';
                    return { name: key.toLowerCase(), type };
                });
                const suggestedMapping = {
                    text: keys.find(k => k.toLowerCase().includes('text') || k.toLowerCase().includes('utterance') || k.toLowerCase().includes('input')) || keys[0],
                    label: keys.find(k => k.toLowerCase().includes('label') || k.toLowerCase().includes('intent') || k.toLowerCase().includes('tag')) || keys[1] || keys[0],
                };
                return { fields, suggestedMapping };
            }
            catch (err) {
                console.error('Failed to infer schema from JSON:', err);
                return { fields: [] };
            }
        }
    }

    class LanguageClassifier {
        constructor(config) {
            this.trainSamples = {};
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        loadTrainingData(raw, format = 'json') {
            switch (format) {
                case 'csv':
                    return IO.importCSV(raw);
                case 'tsv':
                    return IO.importTSV(raw);
                case 'json':
                default:
                    return IO.importJSON(raw);
            }
        }
        train(data) {
            const categories = [...new Set(data.map(d => d.label))];
            this.elm.setCategories(categories);
            data.forEach(({ text, label }) => {
                if (!this.trainSamples[label])
                    this.trainSamples[label] = [];
                this.trainSamples[label].push(text);
            });
            this.elm.train();
        }
        predict(text, topK = 3) {
            return this.elm.predict(text, topK);
        }
        /**
         * Train the classifier using already-encoded vectors.
         * Each vector must be paired with its label.
         */
        trainVectors(data) {
            const categories = [...new Set(data.map(d => d.label))];
            this.elm.setCategories(categories);
            const X = data.map(d => d.vector);
            const Y = data.map(d => this.elm.oneHot(categories.length, categories.indexOf(d.label)));
            const W = this.elm['randomMatrix'](this.config.hiddenUnits, X[0].length);
            const b = this.elm['randomMatrix'](this.config.hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        /**
         * Predict language directly from a dense vector representation.
         */
        predictFromVector(vec, topK = 1) {
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
        beginOnline(opts) {
            var _a, _b, _c;
            const categories = opts.categories.slice();
            const inputDim = opts.inputDim;
            const hiddenUnits = (_a = opts.hiddenUnits) !== null && _a !== void 0 ? _a : this.config.hiddenUnits;
            const outputDim = categories.length;
            const lambda = (_b = opts.lambda) !== null && _b !== void 0 ? _b : 1e-2;
            const actName = (_c = opts.activation) !== null && _c !== void 0 ? _c : this.config.activation;
            // map your activation names to a function for OnlineELM
            const act = actName === 'tanh' ? Math.tanh
                : actName === 'sigmoid' ? (x) => 1 / (1 + Math.exp(-x))
                    : (x) => (x > 0 ? x : 0); // relu default
            this.onlineMdl = new OnlineELM(inputDim, hiddenUnits, outputDim, act, lambda);
            this.onlineCats = categories;
            this.onlineInputDim = inputDim;
        }
        partialTrainVectorsOnline(batch) {
            if (!this.onlineMdl || !this.onlineCats || !this.onlineInputDim) {
                throw new Error('Call beginOnline() before partialTrainVectorsOnline().');
            }
            if (!batch.length)
                return;
            const B = batch.length;
            const D = this.onlineInputDim;
            const O = this.onlineCats.length;
            this.onlineMdl.hiddenUnits;
            // Pack X [B x D] and T [B x O] into typed arrays
            const X = new Float64Array(B * D);
            const T = new Float64Array(B * O);
            for (let i = 0; i < B; i++) {
                const { vector, label } = batch[i];
                if (vector.length !== D) {
                    throw new Error(`vector dim ${vector.length} != inputDim ${D}`);
                }
                // X row
                X.set(vector, i * D);
                // one-hot T row
                const li = this.onlineCats.indexOf(label);
                if (li < 0)
                    throw new Error(`Unknown label "${label}" for this online run.`);
                T[i * O + li] = 1;
            }
            this.onlineMdl.partialFit(X, T, B);
        }
        endOnline() {
            if (!this.onlineMdl || !this.onlineCats)
                return;
            // Convert typed arrays -> number[][] expected by existing ELM paths
            const H = this.onlineMdl.hiddenUnits;
            const D = this.onlineMdl.inputDim;
            const O = this.onlineMdl.outputDim;
            const W = this.reshapeTo2D(this.onlineMdl.W, H, D); // [H x D]
            const b = this.reshapeCol(this.onlineMdl.b, H); // [H x 1]
            const beta = this.reshapeTo2D(this.onlineMdl.beta, H, O); // [H x O]
            // Activate categories for this classifier and publish the model
            this.elm.setCategories(this.onlineCats);
            this.elm['model'] = { W, b, beta };
            // clear online state
            this.onlineMdl = undefined;
            this.onlineCats = undefined;
            this.onlineInputDim = undefined;
        }
        reshapeTo2D(buf, rows, cols) {
            const out = new Array(rows);
            for (let r = 0; r < rows; r++) {
                const row = new Array(cols);
                for (let c = 0; c < cols; c++)
                    row[c] = buf[r * cols + c];
                out[r] = row;
            }
            return out;
        }
        reshapeCol(buf, rows) {
            const out = new Array(rows);
            for (let r = 0; r < rows; r++)
                out[r] = [buf[r]];
            return out;
        }
    }

    class RefinerELM {
        constructor(config) {
            this.config = Object.assign(Object.assign({}, config), { useTokenizer: false, categories: [], log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        train(inputs, labels) {
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            const Y = labels.map(label => this.elm.oneHot(categories.length, categories.indexOf(label)));
            const W = this.elm['randomMatrix'](this.config.hiddenUnits, inputs[0].length);
            const b = this.elm['randomMatrix'](this.config.hiddenUnits, 1);
            const tempH = Matrix.multiply(inputs, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        predict(vec) {
            const input = [vec];
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply(input, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob);
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    /**
     * VotingClassifierELM takes predictions from multiple ELMs
     * and learns to choose the most accurate final label.
     * It can optionally incorporate confidence scores and calibrate model weights.
     */
    class VotingClassifierELM {
        constructor(config) {
            this.categories = config.categories || ['English', 'French', 'Spanish'];
            this.modelWeights = [];
            this.elm = new ELM(Object.assign(Object.assign({}, config), { useTokenizer: false, categories: this.categories, log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } }));
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        setModelWeights(weights) {
            this.modelWeights = weights;
        }
        calibrateWeights(predictionLists, trueLabels) {
            const numModels = predictionLists.length;
            const numExamples = trueLabels.length;
            const accuracies = new Array(numModels).fill(0);
            for (let m = 0; m < numModels; m++) {
                let correct = 0;
                for (let i = 0; i < numExamples; i++) {
                    if (predictionLists[m][i] === trueLabels[i]) {
                        correct++;
                    }
                }
                accuracies[m] = correct / numExamples;
            }
            const total = accuracies.reduce((sum, acc) => sum + acc, 0) || 1;
            this.modelWeights = accuracies.map(a => a / total);
            console.log('🔧 Calibrated model weights based on accuracy:', this.modelWeights);
        }
        train(predictionLists, confidenceLists, trueLabels) {
            if (!Array.isArray(predictionLists) || predictionLists.length === 0 || !trueLabels) {
                throw new Error('Invalid inputs to VotingClassifierELM.train');
            }
            const numModels = predictionLists.length;
            const numExamples = predictionLists[0].length;
            for (let list of predictionLists) {
                if (list.length !== numExamples) {
                    throw new Error('Inconsistent prediction lengths across models');
                }
            }
            if (confidenceLists) {
                if (confidenceLists.length !== numModels) {
                    throw new Error('Confidence list count must match number of models');
                }
                for (let list of confidenceLists) {
                    if (list.length !== numExamples) {
                        throw new Error('Inconsistent confidence lengths across models');
                    }
                }
            }
            if (!this.modelWeights || this.modelWeights.length !== numModels) {
                this.calibrateWeights(predictionLists, trueLabels);
            }
            const inputs = [];
            for (let i = 0; i < numExamples; i++) {
                let inputRow = [];
                for (let m = 0; m < numModels; m++) {
                    const label = predictionLists[m][i];
                    if (typeof label === 'undefined') {
                        console.error(`Undefined label from model ${m} at index ${i}`);
                        throw new Error(`Invalid label in predictionLists[${m}][${i}]`);
                    }
                    const weight = this.modelWeights[m];
                    inputRow = inputRow.concat(this.oneHot(label).map(x => x * weight));
                    if (confidenceLists) {
                        const conf = confidenceLists[m][i];
                        const normalizedConf = Math.min(1, Math.max(0, conf));
                        inputRow.push(normalizedConf * weight);
                    }
                }
                inputs.push(inputRow);
            }
            const examples = inputs.map((input, i) => ({ input, label: trueLabels[i] }));
            console.log(`📊 VotingClassifierELM training on ${examples.length} examples with ${numModels} models.`);
            this.elm.train(examples);
        }
        predict(labels, confidences) {
            if (!Array.isArray(labels) || labels.length === 0) {
                throw new Error('No labels provided to VotingClassifierELM.predict');
            }
            let input = [];
            for (let i = 0; i < labels.length; i++) {
                const weight = this.modelWeights[i] || 1;
                input = input.concat(this.oneHot(labels[i]).map(x => x * weight));
                if (confidences && typeof confidences[i] === 'number') {
                    const norm = Math.min(1, Math.max(0, confidences[i]));
                    input.push(norm * weight);
                }
            }
            return this.elm.predict(JSON.stringify(input), 1);
        }
        oneHot(label) {
            const index = this.categories.indexOf(label);
            if (index === -1) {
                console.warn(`Unknown label in oneHot: ${label}`);
                return new Array(this.categories.length).fill(0);
            }
            return this.categories.map((_, i) => (i === index ? 1 : 0));
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    exports.Activations = Activations;
    exports.Augment = Augment;
    exports.AutoComplete = AutoComplete;
    exports.CharacterLangEncoderELM = CharacterLangEncoderELM;
    exports.ConfidenceClassifierELM = ConfidenceClassifierELM;
    exports.CosineSimilarity = CosineSimilarity;
    exports.ELM = ELM;
    exports.ELMChain = ELMChain;
    exports.EncoderELM = EncoderELM;
    exports.FeatureCombinerELM = FeatureCombinerELM;
    exports.IO = IO;
    exports.IntentClassifier = IntentClassifier;
    exports.KNN = KNN;
    exports.LanguageClassifier = LanguageClassifier;
    exports.Matrix = Matrix;
    exports.OnlineELM = OnlineELM;
    exports.RefinerELM = RefinerELM;
    exports.TFIDF = TFIDF;
    exports.TFIDFVectorizer = TFIDFVectorizer;
    exports.TextEncoder = TextEncoder;
    exports.Tokenizer = Tokenizer;
    exports.UniversalEncoder = UniversalEncoder;
    exports.VotingClassifierELM = VotingClassifierELM;
    exports.addEmbedding = addEmbedding;
    exports.bindAutocompleteUI = bindAutocompleteUI;
    exports.defaultConfig = defaultConfig;
    exports.embeddingStore = embeddingStore;
    exports.evaluateEnsembleRetrieval = evaluateEnsembleRetrieval;
    exports.evaluateRetrieval = evaluateRetrieval;
    exports.evaluateRetrievalSimple = evaluateRetrievalSimple;
    exports.searchEmbeddings = searchEmbeddings$1;

}));
//# sourceMappingURL=astermind.umd.js.map
