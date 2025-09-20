export function predictLogitsOnMain(model, x) {
  if (typeof model.predictLogitsFromVector === 'function') return model.predictLogitsFromVector(x);
  if (typeof model.predictLogits === 'function') return model.predictLogits(x);
  if (typeof model.predictProbaFromVector === 'function') {
    const p = model.predictProbaFromVector(x); return p.map(v => Math.log((v || 1e-12)));
  }
  if (typeof model.predictFromVector === 'function') {
    const r = model.predictFromVector([x])[0];
    if (Array.isArray(r)) return r;
    if (r && Array.isArray(r.probs)) return r.probs.map(v => Math.log((v || 1e-12)));
  }
  throw new Error('No vector-safe predict on model');
}

export function generateTokensSync(model, seed, steps, buildInputWindow, N, V, softmax, sampleTopK, temp, topK) {
  const out = seed.slice();
  for (let i = 0; i < steps; i++) {
    const x = Array.from(buildInputWindow(out, out.length, N, V));
    const logits = predictLogitsOnMain(model, x);
    const probs = softmax(Array.from(logits), temp);
    const id = sampleTopK(probs, topK | 0);
    out.push(id);
  }
  return out.slice(seed.length);
}

export async function generateTokensAsync(workerClient, seed, steps, buildInputWindow, N, V, softmax, sampleTopK, temp, topK) {
  const out = seed.slice();
  for (let i = 0; i < steps; i++) {
    const x = Array.from(buildInputWindow(out, out.length, N, V));
    const logits = await workerClient.call('predictLogits', { x });
    const probs = softmax(Array.from(logits), temp);
    const id = sampleTopK(probs, topK | 0);
    out.push(id);
  }
  return out.slice(seed.length);
}
