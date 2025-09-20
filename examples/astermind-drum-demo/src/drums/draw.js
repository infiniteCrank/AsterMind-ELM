export function resizeCanvas(canvas, g) {
  const DPR = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = Math.floor(W * DPR);
  canvas.height = Math.floor(H * DPR);
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
}

export function drawTimeline(canvas, g, tokens, cursor, stepsPerBar = 16) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  g.clearRect(0, 0, W, H);
  const pad = 20, usableW = W - 2 * pad, laneH = (H - 2 * pad) / 3;
  const totalSteps = stepsPerBar * 4; const barWidth = usableW / 4;
  for (let b = 0; b < 4; b++) { const xs = pad + b * barWidth; g.fillStyle = 'rgba(90,209,255,0.04)'; g.fillRect(xs, pad, barWidth, H - 2 * pad); }
  g.strokeStyle = 'rgba(90,209,255,0.25)';
  for (let i = 0; i <= totalSteps; i++) { const x = pad + (i / totalSteps) * usableW; g.beginPath(); g.moveTo(x, pad); g.lineTo(x, H - pad); g.stroke(); }
  g.fillStyle = 'var(--muted)'; g.font = '12px system-ui';
  ['KICK', 'SNARE', 'HAT'].forEach((L, i) => { g.fillText(L, 6, pad + i * laneH + 14); });

  const colors = { kick: 'var(--accent)', snare: '#6ee7a2', hat: '#facc15' };
  const TOK = { WAIT_1: 0, ON_K: 1, ON_S: 2, ON_H: 3 };
  const windowTokens = tokens.slice(-totalSteps);
  for (let i = 0; i < windowTokens.length; i++) {
    const t = windowTokens[i]; const x = pad + (i / totalSteps) * usableW;
    const yKick = pad + laneH * 0.5, ySnare = pad + laneH * 1.5, yHat = pad + laneH * 2.5;
    if (t === TOK.ON_K) { g.fillStyle = colors.kick; g.beginPath(); g.arc(x, yKick, 6, 0, Math.PI * 2); g.fill(); }
    if (t === TOK.ON_S) { g.fillStyle = colors.snare; g.beginPath(); g.arc(x, ySnare, 6, 0, Math.PI * 2); g.fill(); }
    if (t === TOK.ON_H) { g.fillStyle = colors.hat; g.beginPath(); g.arc(x, yHat, 4, 0, Math.PI * 2); g.fill(); }
  }
  const idx = (cursor % totalSteps);
  const playX = pad + (idx / totalSteps) * usableW;
  g.strokeStyle = 'rgba(250,250,255,0.85)'; g.beginPath(); g.moveTo(playX, pad); g.lineTo(playX, H - pad); g.stroke();
}
