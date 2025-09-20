import { TOK } from './tokens.js';

export class DrumSynth {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.25;
    this.master.connect(this.ctx.destination);
  }
  kick(t) {
    const c = this.ctx; const o = c.createOscillator(), g = c.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(40, t + 0.15);
    g.gain.setValueAtTime(1, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
    o.connect(g).connect(this.master); o.start(t); o.stop(t + 0.16);
  }
  snare(t) {
    const c = this.ctx;
    const buffer = c.createBuffer(1, c.sampleRate * 0.2, c.sampleRate);
    const data = buffer.getChannelData(0); for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const s = c.createBufferSource(); s.buffer = buffer;
    const bp = c.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1800;
    const g = c.createGain(); g.gain.setValueAtTime(0.5, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
    s.connect(bp).connect(g).connect(this.master); s.start(t); s.stop(t + 0.2);
  }
  hat(t) {
    const c = this.ctx;
    const buffer = c.createBuffer(1, c.sampleRate * 0.05, c.sampleRate);
    const data = buffer.getChannelData(0); for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const s = c.createBufferSource(); s.buffer = buffer;
    const hp = c.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 7000;
    const g = c.createGain(); g.gain.setValueAtTime(0.25, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    s.connect(hp).connect(g).connect(this.master); s.start(t); s.stop(t + 0.06);
  }
  playToken(tok, when) {
    if (tok === TOK.ON_K) this.kick(when);
    else if (tok === TOK.ON_S) this.snare(when);
    else if (tok === TOK.ON_H) this.hat(when);
  }
}
