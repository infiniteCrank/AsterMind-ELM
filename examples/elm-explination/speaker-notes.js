/**
 * Injects detailed speaker notes for each slide.
 * - Left column: verbatim script (what to say)
 * - Right column: bullet points (talking cues)
 */
document.addEventListener('DOMContentLoaded', () => {
  const NOTES = {
    /* =========================
       SLIDE 0 — OVERVIEW
       ========================= */
    slide0: {
      left: `
      <p>Think about the last time you asked Copilot to finish a function, or used ChatGPT to help debug some code. It feels like magic, right? That magic comes from neural networks—the engines behind today’s AI.</p>

      <p>Now, most of those networks run in massive data centers filled with GPUs. That’s why tools like ChatGPT or Copilot don’t actually run on your laptop—they run in the cloud. But here’s the twist: what if you could bring some of that predictive power <em>into the browser</em> with nothing more than JavaScript?</p>

      <p>And here’s the kicker: while traditional neural networks can take hours—or even days—to train, there’s a special type of neural network that can be trained in seconds. Literally seconds. No GPUs, no server farms—just you, your code, and instant results.</p>

      <p>Hello everyone, my name is Julian Duran. I’ve been a software engineer for many years, and over the past few years I’ve become fascinated with machine learning and AI. Along that journey, I stumbled onto this almost-forgotten technology called <strong>Extreme Learning Machines</strong>, or ELMs.</p>

      <p>In this session, I’m going to show you what they are, how they differ from traditional neural networks, and why they’re such a powerful fit for us as JavaScript engineers. We’ll walk through the pieces of an ELM step by step with interactive slides, and by the end, you’ll see how you can start using them yourself—right inside your own projects.</p>
 `,
      right: `
 <h3>SMART Goals for This Presentation</h3>
  <ul>
    <li><strong>Specific:</strong> Introduce Extreme Learning Machines (ELMs) in a way that JavaScript engineers with little ML background can understand.</li>
    <li><strong>Measurable:</strong> By the end, attendees should be able to explain what an ELM is, how it differs from a traditional neural network, and imagine one use case in their own projects.</li>
    <li><strong>Achievable:</strong> Use plain-language explanations, interactive slides, and live demos to make the concepts approachable without heavy math.</li>
    <li><strong>Relevant:</strong> Connect ELMs to tools the audience already knows—like Copilot or ChatGPT—so they see the value for their everyday coding work.</li>
    <li><strong>Time-bound:</strong> Deliver all of this within the session time, keeping the introduction under 2 minutes and dedicating the majority of time to hands-on exploration and practical examples.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE 2a — Brief NN Overview
       ========================= */
    slide2a: {
      left: `
        <p>Quick mental model of a simple neural network. First, we turn text into numbers — think an array of features. That array flows into an <strong>input layer</strong>. Next is a <strong>hidden layer</strong> that mixes numbers and applies a squish function called an <em>activation</em>. Finally, an <strong>output layer</strong> turns the signal into scores or classes.</p>
        <p>Let’s click <strong>Run</strong>. You’ll see particles flow left→right. There’s no feedback here — it’s a straight pipeline, like chaining array operations in JS.</p>
        <p>Keep this picture in mind because ELM will keep the same outer shape — inputs, a hidden layer, an output layer — but we’ll skip the slow, iterative training of the hidden layer.</p>
      `,
      right: `
        <ul>
          <li>Pipeline: text → numbers → hidden “mix & squish” → output scores.</li>
          <li>Analogy: like <code>array.map()</code> steps — one direction only.</li>
          <li>Watch the canvas: particles = data flowing through layers.</li>
          <li>Foreshadow: ELM keeps the pipeline but avoids hidden-layer training.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE 1 — Neuron Demo
       ========================= */
    slide1: {
      left: `
        <p>This is a single neuron. It takes an input <em>x</em>, multiplies by a <strong>weight</strong> <em>w</em>, adds a <strong>bias</strong> <em>b</em>, then applies an <strong>activation</strong> function. In JS terms: compute <code>z = w*x + b</code>, then run <code>y = activation(z)</code>.</p>
        <p>Let’s try a setting. I’ll set <strong>w = -2</strong>, <strong>b = 0.5</strong>. Now I’ll toggle the activation between <strong>ReLU</strong> and <strong>tanh</strong>.</p>
        <p>Notice the shape: ReLU clips everything below zero — it’s piecewise straight. <em>tanh</em> is smooth, squashing outputs into −1..1. <em>sigmoid</em> is similar but squashes into 0..1.</p>
        <p>The gauges below show amplitude before and after activation. The post-activation gauge tells you how much the squish function is limiting the output range.</p>
        <p>Key takeaway: a neuron is just “multiply, add, squish.” Networks are stacks of these.</p>
      `,
      right: `
        <ul>
          <li>Formula: <code>z = w*x + b</code> → <code>y = g(z)</code>.</li>
          <li>Try: <strong>w = -2</strong>, <strong>b = 0.5</strong>; compare ReLU vs tanh vs sigmoid.</li>
          <li>ReLU: zeroes negatives; tanh/sigmoid: smooth squash.</li>
          <li>Watch the two amplitude bars (pre vs post activation).</li>
          <li>Takeaway: building block for all slides ahead.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE 2 — Vectorization
       ========================= */
    slide2: {
      left: `
        <p>Neural nets need numbers, so we turn text into a numeric vector — this is <strong>vectorization</strong>.</p>
        <p><strong>Bag-of-Words</strong>: each unique word becomes a feature; the value is the count. Simple and fast; it ignores order and context.</p>
        <p><strong>TF-IDF</strong>: like counts, but common words (like “the”) are down-weighted and rare, informative words are up-weighted. Often a stronger default for small text datasets.</p>
        <p><strong>Isolated</strong>: here that just means “local counts” without IDF — useful to show the mechanics.</p>
        <p>Workflow: pick a row, choose an encoding, click <strong>Encode text</strong>. The heatmap shows the first N features (use the slider). Hover to see the feature index and value. The token list below prints top tokens.</p>
        <p>ELM tip: it doesn’t care which reasonable encoder you use — it just needs a numeric vector. We’ll reuse this vector on the hidden-layer slide.</p>
      `,
      right: `
        <ul>
          <li>BoW = counts; TF-IDF = counts × informativeness.</li>
          <li>Click <strong>Encode text</strong> → heatmap + tokens below.</li>
          <li>Adjust “Max features” to see truncation effects.</li>
          <li>Any numeric vector works for ELM.</li>
          <li>We’ll project this vector through a random hidden layer next.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE BP — Backprop
       ========================= */
    slideBP: {
      left: `
        <p>Training a regular neural net means running a loop called <strong>backpropagation</strong>. The network predicts, we measure the <strong>loss</strong> (error), and we nudge weights to reduce that loss. Repeat for many <strong>epochs</strong>.</p>
        <p>The tricky knob is the <strong>learning rate</strong>. Too low: training crawls; too high: the loss bounces or even diverges. Two classic failure modes: <em>vanishing gradients</em> (updates get tiny) and <em>exploding gradients</em> (updates blow up).</p>
        <p>Let’s scrub the slider and switch scenarios. Watch the left heatmap (weights changing) and the right chart (loss over time). This is the effort ELM avoids.</p>
      `,
      right: `
        <ul>
          <li>Loop: predict → compute loss → nudge weights → repeat.</li>
          <li>Learning rate: low = slow, high = noisy/unstable.</li>
          <li>Failure modes: vanishing vs exploding gradients.</li>
          <li>Watch: left = weight heatmap; right = loss curve.</li>
          <li>Contrast setup: ELM skips these loops entirely.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE HUANG — The Question
       ========================= */
    slideHuang: {
      left: `
        <p>Here’s the bold idea by <strong>Guang-Bin Huang</strong>: <em>what if we didn’t train the hidden layer at all?</em></p>
        <p>At first that sounds wrong — isn’t the hidden layer where the magic happens? The insight is that a <em>random</em> hidden layer can be <strong>good enough</strong> to create a useful coordinate system for the data. Then we only solve the output layer <em>once</em>.</p>
        <p>So ELM = randomize hidden weights and biases a single time, freeze them, and do a one-shot solve for the output weights.</p>
      `,
      right: `
        <ul>
          <li>Provocation: skip training the hidden layer.</li>
          <li>Use randomness to create a rich coordinate system.</li>
          <li>Train only the output layer (one solve).</li>
          <li>Outcome: seconds instead of minutes/hours.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE GRID — Random Grid Analogy
       ========================= */
    slideGrid: {
      left: `
        <p>Imagine looking down on a messy city. It’s hard to describe locations. If we drop a <strong>random grid</strong> on top, it won’t line up with streets, but it gives us coordinates — “row 3, col 5.” It’s not perfect; it’s <em>useful</em>.</p>
        <p>ELM uses a random hidden layer the same way: it projects each example into random coordinates. With coordinates, it’s easier to separate classes using a simple output mapping.</p>
      `,
      right: `
        <ul>
          <li>Analogy: random grid → usable coordinates.</li>
          <li>ELM mapping: <code>H = g(X·W + b)</code>.</li>
          <li>Once in H-space, separation becomes simpler.</li>
          <li>We’ll solve the output mapping next.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE GPS — Pseudoinverse
       ========================= */
    slideGPS: {
      left: `
        <p>Now the one-shot solve. Let <code>H</code> be the matrix of hidden-layer outputs for all training rows, and <code>Y</code> be the matrix of labels (e.g., one-hot classes). We want weights <code>β</code> that make <code>Hβ ≈ Y</code>.</p>
        <p>The fast way is the <strong>Moore–Penrose pseudoinverse</strong>. In practice you can think: “find the best linear mapping from hidden features to labels in one calculation.”</p>
        <p>In plain terms: the hidden layer gave us coordinates; now we draw straight lines in that space to hit the targets. If features are noisy or correlated, we add a small <strong>ridge</strong> term for stability.</p>
        <p>Dimensions check: if H is <em>n × h</em> and Y is <em>n × k</em>, then β is <em>h × k</em>. That’s the piece we’ll visualize on the next slide.</p>
      `,
      right: `
        <ul>
          <li>Goal: solve <code>Hβ ≈ Y</code> in one step.</li>
          <li>Tool: pseudoinverse (or ridge): <code>β = (HᵀH + λI)⁻¹HᵀY</code>.</li>
          <li>Intuition: “find best straight mapping in H-space.”</li>
          <li>Dims: H(n×h), Y(n×k) ⇒ β(h×k).</li>
          <li>Why fast: no epochs, no LR tuning.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE WHY — Why Randomness Works
       ========================= */
    slideWhy: {
      left: `
        <p>Why does this work at all? Picture a crumpled sheet of paper with two colors mixed together. In the original space they overlap. If we project into a richer, randomly mixed space, the colors often spread so a simple straight line can separate them.</p>
        <p>On the right animation you’ll see points move from “crumpled” to “spread out.” The line that separates them is just a straight cut in the projected space.</p>
        <p>Two practical notes: you usually need a <strong>large enough hidden size</strong> to get a rich projection, and a small <strong>ridge</strong> helps keep the solve stable.</p>
      `,
      right: `
        <ul>
          <li>Random projection can “untangle” classes.</li>
          <li>Then a linear separator is enough.</li>
          <li>Hidden size ↑ → richer features (to a point).</li>
          <li>Add ridge (λ) to stabilize the solve.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE HIDDEN — Random & Project
       ========================= */
    slideHidden: {
      left: `
        <p>Let’s build the random hidden layer. Click <strong>Reseed hidden</strong> to sample new random weights <code>W</code> and biases <code>b</code>. The heatmap shows rows = hidden neurons and columns = input features; color encodes sign and magnitude.</p>
        <p>Now click <strong>Project H</strong>. We take the encoded text from the previous slide, compute <code>z = W·x + b</code>, apply the activation <code>g</code>, and display the resulting hidden vector <code>H</code> as bars.</p>
        <p>Key idea: once we like the hidden size, we can freeze this basis and use it for the one-shot training on the next slide.</p>
      `,
      right: `
        <ul>
          <li>Controls: set <em>Hidden size</em>, then <strong>Reseed hidden</strong>.</li>
          <li>Heatmap = W (hidden × input) values.</li>
          <li>Click <strong>Project H</strong> to see <code>H = g(Wx + b)</code>.</li>
          <li>These H coordinates feed the one-shot solve.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE 4 — One-shot Training
       ========================= */
    slide4: {
      left: `
        <p>Time to train. When I click <strong>Train (one shot)</strong>, we freeze the current hidden basis, build H for all rows, and compute β in one step using the pseudoinverse (with optional ridge).</p>
        <p>The box will print dimensions for H, Y, and β, plus a small sample of β values. The heatmap below visualizes β: rows = hidden features, columns = classes.</p>
        <p>Try changing <em>Hidden size</em> and retraining to see how capacity affects the solution. If things look noisy, a small ridge usually helps.</p>
      `,
      right: `
        <ul>
          <li>Click <strong>Train</strong> → freeze basis → compute β once.</li>
          <li>Outputs: dims, β sample, β heatmap.</li>
          <li>Hidden size is a key capacity knob.</li>
          <li>Use ridge if unstable (regularization).</li>
          <li>Export/reset buttons for iteration.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE COMPARE — Backprop vs ELM
       ========================= */
    slideCompare: {
      left: `
        <p>Think of backprop as sculpting a custom key: you chip, test, chip, test — many iterations until it fits perfectly. That’s great when you need maximum accuracy and you have time and data.</p>
        <p>ELM is like trying a keyring of random keys: we generate many random hidden features, and one linear solve picks the combination that unlocks the task. It’s fast and strong as a baseline or for small/medium problems.</p>
        <p>Rule of thumb: start with ELM to get a baseline in seconds. If you need more accuracy, move to a trained hidden layer or embeddings and compare.</p>
      `,
      right: `
        <ul>
          <li>Backprop: control/fidelity, higher cost/tuning.</li>
          <li>ELM: speed/simplicity, great baseline/prototyping.</li>
          <li>Workflow: start with ELM → escalate if needed.</li>
        </ul>
      `
    },

    /* =========================
       SLIDE PRED — Prediction
       ========================= */
    slidePred: {
      left: `
        <p>Let’s run the trained model. I’ll choose a row and click <strong>Predict</strong>. We encode the text, project through the frozen hidden layer, apply β, and compute class probabilities with a softmax.</p>
        <p>The output shows the predicted class, the ground truth from the sample, and probabilities for each class. If the label is known, we mark ✓/✗.</p>
        <p>In practice you’d use this exactly like any other JS function: encode → project → multiply by β → read off the scores.</p>
      `,
      right: `
        <ul>
          <li>Select a row → <strong>Predict</strong>.</li>
          <li>Pipeline: encode → H = g(Wx + b) → scores = H·β → softmax.</li>
          <li>Read: predicted vs truth + per-class probabilities.</li>
          <li>✓/✗ shows correctness on this tiny dataset.</li>
        </ul>
      `
    }
  };

  function setNotes(panelEl, html) {
    if (!panelEl) return;
    let notes = panelEl.querySelector('.notes');
    if (!notes) {
      notes = document.createElement('div');
      notes.className = 'notes';
      panelEl.appendChild(notes);
    }
    notes.innerHTML = '<span class="label">Speaker notes</span>' + html;
  }

  function apply(slideId, spec) {
    const slide = document.getElementById(slideId);
    if (!slide) return;
    const leftPanel = slide.querySelector('.left .panel');
    const rightPanel = slide.querySelector('.right .panel');
    if (spec.left) setNotes(leftPanel, spec.left);
    if (spec.right) setNotes(rightPanel, spec.right);
  }

  Object.entries(NOTES).forEach(([id, spec]) => apply(id, spec));
});

