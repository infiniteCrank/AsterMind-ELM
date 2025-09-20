import { ID2STR } from '../drums/tokens.js';

export function createConfig(engine, inputSize, hiddenUnits, ridgeLambda, ui) {
  const activation = ui.activationSel.value;
  if (engine === 'kernel') {
    const kernelType = ui.kernelTypeSel.value;
    const cfg = {
      outputDim: ID2STR.length,
      mode: 'nystrom',
      kernel: kernelType === 'poly'
        ? { type: 'poly', degree: parseInt(ui.degreeInp.value, 10) || 2, coef0: parseFloat(ui.coef0Inp.value) || 1 }
        : kernelType === 'linear'
          ? { type: 'linear' }
          : kernelType === 'laplacian'
            ? { type: 'laplacian', gamma: parseFloat(ui.gammaInp.value) || (1 / inputSize) }
            : { type: 'rbf', gamma: parseFloat(ui.gammaInp.value) || (1 / inputSize) },
      nystrom: {
        m: parseInt(ui.mLandmarksInp.value, 10) || 128,
        strategy: 'kmeans++',
        whiten: !!ui.whitenChk.checked,
        jitter: 1e-9,
      },
      ridgeLambda: parseFloat(ui.lambdaInp.value) || ridgeLambda,
      task: 'classification',
      log: { verbose: false, modelName: 'DrumKernelELM' }
    };
    return cfg;
  }
  if (engine === 'online') {
    return {
      inputDim: inputSize,
      outputDim: ID2STR.length,
      hiddenUnits,
      activation,
      ridgeLambda: parseFloat(ui.lambdaInp.value) || ridgeLambda,
      weightInit: 'he',
      forgettingFactor: 0.997,
      log: { verbose: false, modelName: 'DrumOnlineELM' }
    };
  }
  // plain ELM / numeric mode
  return {
    inputSize: inputSize,
    hiddenUnits: hiddenUnits,
    activation: activation,
    ridgeLambda: parseFloat(ui.lambdaInp.value) || ridgeLambda,
    task: 'classification',
    outputDim: ID2STR.length,   // <- 4
    categories: ID2STR.slice(),
    useTokenizer: false,
    weightInit: 'xavier',
    log: { verbose: false, modelName: 'DrumELM' }
  };

}
