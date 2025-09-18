const input = document.getElementById('langInput');
const fill  = document.getElementById('langFill');
const meta  = document.getElementById('meta');

let w;
if (window.Worker) {
  w = new Worker('./trainer.worker.js');
  w.onmessage = (e) => {
    const { type, payload } = e.data || {};
    if (type === 'ready') {
      meta.textContent = 'Loading dataset…';
      fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(text => w.postMessage({ type: 'init', csvText: text }))
        .catch(err => w.postMessage({ type: 'error', message: String(err) }));
    }
    else if (type === 'progress') {
      meta.innerHTML = payload;
    }
    else if (type === 'trained') {
      meta.innerHTML = payload.html;
    }
    else if (type === 'predictions') {
      const { top, second } = payload;
      const p1 = top?.prob ?? 0, p2 = second?.prob ?? 0;
      const percent = Math.round(p1 * 100);
      const margin  = p1 - p2;
      //const unsure  = percent < 40 || margin < 0.15;
      const unsure  = false 

      fill.style.width = `${Math.max(percent, 8)}%`;
      fill.textContent = unsure ? '🤔 Not sure' : `${top.label} (${percent}%)`;
      fill.style.background = ({
        English: 'linear-gradient(to right, green, lime)',
        French:  'linear-gradient(to right, blue, cyan)',
        Spanish: 'linear-gradient(to right, red, orange)'
      })[top?.label ?? ''] || '#6b7280';
    }
    else if (type === 'error') {
      console.error(payload);
      meta.textContent = 'Error: ' + payload;
    }
  };
} else {
  meta.textContent = 'Your browser does not support Web Workers.';
}

// Send keystrokes to the worker (worker handles normalization + predict)
input.addEventListener('input', () => {
  if (!w) return;
  const text = input.value || '';
  if (!text) {
    fill.style.width = '0%'; fill.textContent = ''; fill.style.background = '#9aa6c7';
    return;
  }
  w.postMessage({ type: 'predict', text });
});
