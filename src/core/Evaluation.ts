// Evaluation.ts — Classification & Regression metrics (no deps)

const EPS = 1e-12;

/* =========================
 * Types
 * ========================= */

export interface PerClassMetrics {
    label: string | number;
    support: number;
    tp: number; fp: number; fn: number; tn: number;
    precision: number;
    recall: number;
    f1: number;
}

export interface ClassificationAverages {
    accuracy: number;
    macroPrecision: number;
    macroRecall: number;
    macroF1: number;
    microPrecision: number;
    microRecall: number;
    microF1: number;
    weightedPrecision: number;
    weightedRecall: number;
    weightedF1: number;
    logLoss?: number;
    topKAccuracy?: number;
}

export interface ClassificationReport {
    confusionMatrix: number[][];
    perClass: PerClassMetrics[];
    averages: ClassificationAverages;
}

export interface RegressionPerOutput {
    index: number;
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
}

export interface RegressionReport {
    perOutput: RegressionPerOutput[];
    // macro over outputs
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
}

/* =========================
 * Helpers
 * ========================= */

function isOneHot(Y: number[] | number[][]): Y is number[][] {
    return Array.isArray(Y[0]);
}

function argmax(a: number[]): number {
    let i = 0;
    for (let k = 1; k < a.length; k++) if (a[k] > a[i]) i = k;
    return i;
}

function toIndexLabels(
    yTrue: number[] | number[][],
    yPred: number[] | number[][],
    numClasses?: number
): { yTrueIdx: number[]; yPredIdx: number[]; C: number } {
    let yTrueIdx: number[];
    let yPredIdx: number[];
    if (isOneHot(yTrue)) yTrueIdx = yTrue.map(argmax);
    else yTrueIdx = yTrue as number[];

    if (isOneHot(yPred)) yPredIdx = (yPred as number[][]).map(argmax);
    else yPredIdx = yPred as number[];

    const C = numClasses ?? 1 + Math.max(
        Math.max(...yTrueIdx),
        Math.max(...yPredIdx)
    );
    return { yTrueIdx, yPredIdx, C };
}

/* =========================
 * Confusion matrix
 * ========================= */

export function confusionMatrixFromIndices(
    yTrueIdx: number[],
    yPredIdx: number[],
    C?: number
): number[][] {
    if (yTrueIdx.length !== yPredIdx.length) {
        throw new Error(`confusionMatrix: length mismatch (${yTrueIdx.length} vs ${yPredIdx.length})`);
    }
    const classes = C ?? 1 + Math.max(
        Math.max(...yTrueIdx),
        Math.max(...yPredIdx)
    );
    const M = Array.from({ length: classes }, () => new Array(classes).fill(0));
    for (let i = 0; i < yTrueIdx.length; i++) {
        const r = yTrueIdx[i] | 0;
        const c = yPredIdx[i] | 0;
        if (r >= 0 && r < classes && c >= 0 && c < classes) M[r][c]++;
    }
    return M;
}

/* =========================
 * Per-class metrics
 * ========================= */

function perClassFromCM(M: number[][], labels?: Array<string | number>): PerClassMetrics[] {
    const C = M.length;
    const totals = new Array(C).fill(0);
    const colTotals = new Array(C).fill(0);
    let N = 0;
    for (let i = 0; i < C; i++) {
        let rsum = 0;
        for (let j = 0; j < C; j++) {
            rsum += M[i][j];
            colTotals[j] += M[i][j];
            N += M[i][j];
        }
        totals[i] = rsum;
    }
    const perClass: PerClassMetrics[] = [];
    for (let k = 0; k < C; k++) {
        const tp = M[k][k];
        const fp = colTotals[k] - tp;
        const fn = totals[k] - tp;
        const tn = N - tp - fp - fn;
        const precision = tp / (tp + fp + EPS);
        const recall = tp / (tp + fn + EPS);
        const f1 = (2 * precision * recall) / (precision + recall + EPS);
        perClass.push({
            label: labels?.[k] ?? k,
            support: totals[k],
            tp, fp, fn, tn,
            precision, recall, f1
        });
    }
    return perClass;
}

/* =========================
 * Averages
 * ========================= */

function averagesFromPerClass(per: PerClassMetrics[], accuracy: number): ClassificationAverages {
    const C = per.length;
    let sumP = 0, sumR = 0, sumF = 0;
    let sumWP = 0, sumWR = 0, sumWF = 0, total = 0;
    let tp = 0, fp = 0, fn = 0; // for micro
    for (const c of per) {
        sumP += c.precision; sumR += c.recall; sumF += c.f1;
        sumWP += c.precision * c.support;
        sumWR += c.recall * c.support;
        sumWF += c.f1 * c.support;
        total += c.support;
        tp += c.tp; fp += c.fp; fn += c.fn;
    }
    const microP = tp / (tp + fp + EPS);
    const microR = tp / (tp + fn + EPS);
    const microF = (2 * microP * microR) / (microP + microR + EPS);
    return {
        accuracy,
        macroPrecision: sumP / C,
        macroRecall: sumR / C,
        macroF1: sumF / C,
        microPrecision: microP,
        microRecall: microR,
        microF1: microF,
        weightedPrecision: sumWP / (total + EPS),
        weightedRecall: sumWR / (total + EPS),
        weightedF1: sumWF / (total + EPS)
    };
}

/* =========================
 * Log loss / cross-entropy
 * ========================= */

export function logLoss(
    yTrue: number[] | number[][],
    yPredProba: number[] | number[][]
): number {
    if (!isOneHot(yTrue) || !isOneHot(yPredProba)) {
        throw new Error('logLoss expects one-hot ground truth and probability matrix (N x C).');
    }
    const Y = yTrue as number[][];
    const P = yPredProba as number[][];
    if (Y.length !== P.length) throw new Error('logLoss: length mismatch');
    const N = Y.length;
    let sum = 0;
    for (let i = 0; i < N; i++) {
        const yi = Y[i];
        const pi = P[i];
        if (yi.length !== pi.length) throw new Error('logLoss: class count mismatch');
        for (let j = 0; j < yi.length; j++) {
            if (yi[j] > 0) {
                const p = Math.min(Math.max(pi[j], EPS), 1 - EPS);
                sum += -Math.log(p);
            }
        }
    }
    return sum / N;
}

/* =========================
 * Top-K accuracy
 * ========================= */

export function topKAccuracy(
    yTrueIdx: number[],
    yPredProba: number[][],
    k = 5
): number {
    const N = yTrueIdx.length;
    let correct = 0;
    for (let i = 0; i < N; i++) {
        const probs = yPredProba[i];
        const idx = probs.map((p, j) => j).sort((a, b) => probs[b] - probs[a]).slice(0, Math.max(1, Math.min(k, probs.length)));
        if (idx.includes(yTrueIdx[i])) correct++;
    }
    return correct / N;
}

/* =========================
 * Binary curves (ROC / PR)
 * ========================= */

export interface BinaryCurve {
    thresholds: number[];
    tpr?: number[]; fpr?: number[];     // ROC
    precision?: number[]; recall?: number[]; // PR
    auc: number; // area under curve
}

function pairSortByScore(yTrue01: number[], yScore: number[]): Array<[number, number]> {
    const pairs = yScore.map((s, i) => [s, yTrue01[i]] as [number, number]);
    pairs.sort((a, b) => b[0] - a[0]);
    return pairs;
}

export function binaryROC(yTrue01: number[], yScore: number[]): BinaryCurve {
    if (yTrue01.length !== yScore.length) throw new Error('binaryROC: length mismatch');
    const pairs = pairSortByScore(yTrue01, yScore);
    const P = yTrue01.reduce((s, v) => s + (v ? 1 : 0), 0);
    const N = yTrue01.length - P;
    let tp = 0, fp = 0;
    const tpr: number[] = [0], fpr: number[] = [0], thr: number[] = [Infinity];
    for (let i = 0; i < pairs.length; i++) {
        const [score, y] = pairs[i];
        if (y === 1) tp++; else fp++;
        tpr.push(tp / (P + EPS));
        fpr.push(fp / (N + EPS));
        thr.push(score);
    }
    tpr.push(1); fpr.push(1); thr.push(-Infinity);

    // Trapezoidal AUC
    let auc = 0;
    for (let i = 1; i < tpr.length; i++) {
        const dx = fpr[i] - fpr[i - 1];
        const yAvg = (tpr[i] + tpr[i - 1]) / 2;
        auc += dx * yAvg;
    }
    return { thresholds: thr, tpr, fpr, auc };
}

export function binaryPR(yTrue01: number[], yScore: number[]): BinaryCurve {
    if (yTrue01.length !== yScore.length) throw new Error('binaryPR: length mismatch');
    const pairs = pairSortByScore(yTrue01, yScore);
    const P = yTrue01.reduce((s, v) => s + (v ? 1 : 0), 0);
    let tp = 0, fp = 0;
    const precision: number[] = [], recall: number[] = [], thr: number[] = [];
    // Add starting point
    precision.push(P > 0 ? P / (P + 0) : 1); recall.push(0); thr.push(Infinity);

    for (let i = 0; i < pairs.length; i++) {
        const [score, y] = pairs[i];
        if (y === 1) tp++; else fp++;
        const prec = tp / (tp + fp + EPS);
        const rec = tp / (P + EPS);
        precision.push(prec); recall.push(rec); thr.push(score);
    }

    // AUPRC via trapezoid over recall axis
    let auc = 0;
    for (let i = 1; i < precision.length; i++) {
        const dx = recall[i] - recall[i - 1];
        const yAvg = (precision[i] + precision[i - 1]) / 2;
        auc += dx * yAvg;
    }
    return { thresholds: thr, precision, recall, auc };
}

/* =========================
 * Main: evaluate classification
 * ========================= */

/**
 * Evaluate multi-class classification.
 * - yTrue can be indices (N) or one-hot (N x C)
 * - yPred can be indices (N) or probabilities (N x C)
 * - If yPred are probabilities, we also compute logLoss and optional topK.
 */
export function evaluateClassification(
    yTrue: number[] | number[][],
    yPred: number[] | number[][],
    opts?: { labels?: Array<string | number>; topK?: number }
): ClassificationReport {
    const labels = opts?.labels;
    const { yTrueIdx, yPredIdx, C } = toIndexLabels(yTrue, yPred);
    const M = confusionMatrixFromIndices(yTrueIdx, yPredIdx, C);
    const per = perClassFromCM(M, labels);

    const correct = yTrueIdx.reduce((s, yt, i) => s + (yt === yPredIdx[i] ? 1 : 0), 0);
    const accuracy = correct / yTrueIdx.length;

    const averages = averagesFromPerClass(per, accuracy);

    // Optional extras if we have probabilities
    if (isOneHot(yTrue) && isOneHot(yPred)) {
        try {
            averages.logLoss = logLoss(yTrue as number[][], yPred as number[][]);
            if (opts?.topK && opts.topK > 1) {
                averages.topKAccuracy = topKAccuracy(yTrueIdx, yPred as number[][], opts.topK);
            }
        } catch { /* ignore extras if shapes disagree */ }
    }

    return { confusionMatrix: M, perClass: per, averages };
}

/* =========================
 * Regression
 * ========================= */

export function evaluateRegression(
    yTrue: number[] | number[][],
    yPred: number[] | number[][]
): RegressionReport {
    const Y = Array.isArray(yTrue[0]) ? (yTrue as number[][]) : (yTrue as number[]).map(v => [v]);
    const P = Array.isArray(yPred[0]) ? (yPred as number[][]) : (yPred as number[]).map(v => [v]);
    if (Y.length !== P.length) throw new Error('evaluateRegression: length mismatch');

    const N = Y.length;
    const D = Y[0].length;

    const perOutput: RegressionPerOutput[] = [];
    let sumMSE = 0, sumMAE = 0, sumR2 = 0;

    for (let d = 0; d < D; d++) {
        let mse = 0, mae = 0;
        // mean of Y[:,d]
        let mean = 0;
        for (let i = 0; i < N; i++) mean += Y[i][d];
        mean /= N;

        let ssTot = 0, ssRes = 0;

        for (let i = 0; i < N; i++) {
            const y = Y[i][d], p = P[i][d];
            const e = y - p;
            mse += e * e;
            mae += Math.abs(e);
            ssRes += e * e;
            const dy = y - mean;
            ssTot += dy * dy;
        }
        mse /= N;
        const rmse = Math.sqrt(mse);
        mae /= N;
        const r2 = 1 - (ssRes / (ssTot + EPS));

        perOutput.push({ index: d, mse, rmse, mae, r2 });
        sumMSE += mse; sumMAE += mae; sumR2 += r2;
    }

    const mse = sumMSE / D;
    const rmse = Math.sqrt(mse);
    const mae = sumMAE / D;
    const r2 = sumR2 / D;

    return { perOutput, mse, rmse, mae, r2 };
}

/* =========================
 * Pretty report (optional)
 * ========================= */

export function formatClassificationReport(rep: ClassificationReport): string {
    const lines: string[] = [];
    lines.push('Class\tSupport\tPrecision\tRecall\tF1');
    for (const c of rep.perClass) {
        lines.push(`${c.label}\t${c.support}\t${c.precision.toFixed(4)}\t${c.recall.toFixed(4)}\t${c.f1.toFixed(4)}`);
    }
    const a = rep.averages;
    lines.push('');
    lines.push(`Accuracy:\t${a.accuracy.toFixed(4)}`);
    lines.push(`Macro P/R/F1:\t${a.macroPrecision.toFixed(4)}\t${a.macroRecall.toFixed(4)}\t${a.macroF1.toFixed(4)}`);
    lines.push(`Micro P/R/F1:\t${a.microPrecision.toFixed(4)}\t${a.microRecall.toFixed(4)}\t${a.microF1.toFixed(4)}`);
    lines.push(`Weighted P/R/F1:\t${a.weightedPrecision.toFixed(4)}\t${a.weightedRecall.toFixed(4)}\t${a.weightedF1.toFixed(4)}`);
    if (a.logLoss !== undefined) lines.push(`LogLoss:\t${a.logLoss.toFixed(6)}`);
    if (a.topKAccuracy !== undefined) lines.push(`TopK Acc:\t${a.topKAccuracy.toFixed(4)}`);
    return lines.join('\n');
}
