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
        <p>So, without further ado, let’s take a 1,000-foot view of a neural network. On the screen, you can see a big web of circles connected by lines—that’s the classic picture of a neural network.</p>

  <p>Those circles are called nodes, or neurons. The first set of neurons is the <strong>input layer</strong>. In the middle, we have the <strong>hidden layer</strong>, and at the end, the <strong>output layer</strong>. The lines connecting them are called <strong>synapses</strong>, and each connection has a number attached to it, called a <strong>weight</strong>, which tells the network how strong that connection is. We’ll dig into weights more later.</p>

  <p>For now, what’s important is how data moves through this system. Raw data—like text in a textbox—gets encoded and passed into the input layer. From there, it flows through the hidden layer, which transforms and maps the data. Finally, the output layer produces a result—usually probabilities—that represent the network’s prediction.</p>

  <p>Don’t worry if this feels abstract—we’re going to break it all down step by step. But for now, go ahead and click ‘Run’ to watch a quick simulation of how data flows through the network. Now that we’ve seen the big picture, let’s zoom in on a single neuron.</p>
`,
      right: `
<h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Introduce the structure of a neural network by naming its key components—input layer, hidden layer, output layer, synapses, and weights.</li>
    <li><strong>Measurable:</strong> Attendees should be able to recall and describe these components in their own words after the explanation.</li>
    <li><strong>Achievable:</strong> Use the visual diagram and simulation to reinforce the explanation without requiring prior ML knowledge.</li>
    <li><strong>Relevant:</strong> Builds the foundation needed to understand how Extreme Learning Machines differ from traditional neural networks.</li>
    <li><strong>Time-bound:</strong> Deliver this overview in under 3 minutes before zooming in on a single neuron.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE 1 — Neuron Demo
       ========================= */
    slide1: {
      left: `
      <p>This is a single neuron. Here’s how it works: it takes an input value, multiplies it by a weight, adds a bias, and then runs the result through an activation function.</p>

  <p>In JavaScript terms: <code>z = w * x + b</code>, and then <code>y = activation(z)</code>.</p>

  <p>Let’s try an example. Set the weight <code>w</code> to -2 and the bias <code>b</code> to 0.5. Now toggle the activation function between ReLU and tanh.</p>

  <p>Notice the difference:
    <ul>
      <li><strong>ReLU:</strong> Clips everything below zero — piecewise straight.</li>
      <li><strong>tanh:</strong> Smooth curve, squashes outputs into the range -1 to 1.</li>
      <li><strong>sigmoid:</strong> Similar to tanh, but squashes into the range 0 to 1.</li>
    </ul>
  </p>

  <p>The gauges on screen show amplitude before and after activation. The post-activation gauge tells you how much the “squish function” is limiting the output range.</p>

  <p><strong>Key takeaway:</strong> A neuron is basically “multiply, add, squish.” Networks are just stacks of these neurons.</p>

       `,
      right: `
 <h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Teach how a single neuron processes input using weights, bias, and an activation function.</li>
    <li><strong>Measurable:</strong> Attendees can describe a neuron in plain terms as “multiply, add, squish” and identify the effect of ReLU, tanh, and sigmoid.</li>
    <li><strong>Achievable:</strong> Use the interactive demo and gauges to make the abstract math visual and intuitive.</li>
    <li><strong>Relevant:</strong> Builds a clear mental model for understanding how networks are stacks of these simple units.</li>
    <li><strong>Time-bound:</strong> Deliver the concept and run the demo in under 3 minutes before moving on to networks of neurons.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE 2 — Vectorization
       ========================= */
    slide2: {
      left: `
       <p>Now that we understand how a single neuron works, let’s talk about the data we feed into it. Data usually starts out raw—like a piece of text, an image, or really anything. But here’s the catch: we can’t just feed raw data into a neuron. It needs to be converted into numbers.</p>

  <p>For text, this process is called <strong>vectorization</strong>. First, we break the text into tokens—these can be words, punctuation, or even characters. Each token is then mapped into a numerical value inside a two-dimensional matrix.</p>

  <p>On this slide, you can experiment with different ways to encode text. When you click the <em>Encode Text</em> button, you’ll see a visualization of that 2D matrix, showing how the tokens were encoded and their weights. If you hover over a box, you can see which word that box represents.</p>

  <p>This is just one way of doing text encoding—there are many others. The good news is, Extreme Learning Machines aren’t picky about which encoding you use. And the library I’ve built comes with an ELM encoder, as well as <strong>TF-IDF</strong>, a popular text encoder, to handle that step for you.</p>
`,
      right: `
      <h3>SMART Goal for This Slide</h3>
<ul>
    <li><strong>Specific:</strong> Explain why raw data cannot go directly into a neural network and introduce vectorization as the solution.</li>
    <li><strong>Measurable:</strong> Attendees can describe vectorization as “breaking text into tokens and mapping them into numbers.”</li>
    <li><strong>Achievable:</strong> Use the interactive visualization to make the concept tangible without requiring math-heavy explanations.</li>
    <li><strong>Relevant:</strong> Builds the foundation for how data is prepared before being processed by Extreme Learning Machines.</li>
    <li><strong>Time-bound:</strong> Deliver this explanation and run the demo in under 3 minutes before moving on to ELM’s flexibility with encoders.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE BP — Backprop
       ========================= */
    slideBP: {
      left: `
      <p>Alright, so we’ve seen how raw data like text gets vectorized—turned into numbers—so it can flow into a neural network. But encoding the data is just the start. The real challenge comes next: training the network.</p>

  <p>Training a traditional neural network means running a process called <strong>backpropagation</strong>. Here’s how it works: the network makes a prediction, we measure the error—called the loss—and then we adjust the weights to reduce that loss. And we don’t just do this once. We repeat it over and over, often thousands of times, across what are called epochs.</p>

  <p>This is where things get expensive. Backpropagation is computationally heavy and time-consuming, especially for large networks. And it comes with its own set of headaches. The most important setting to tune is the <strong>learning rate</strong>. If it’s too low, training crawls. If it’s too high, the loss bounces around or even diverges completely.</p>

  <p>Two classic problems can also crop up:</p>
  <ul>
    <li><strong>Vanishing gradients</strong> — updates become so tiny the network stops learning.</li>
    <li><strong>Exploding gradients</strong> — updates get too large and the network blows up.</li>
  </ul>

  <p>On this slide, you can scrub the slider to switch scenarios. Watch the heatmap on the left—those are the weights changing—and the chart on the right, which shows the loss over time.</p>

  <p>This long, costly loop is the effort that <strong>Extreme Learning Machines</strong> avoid. And that’s why ELMs can train in seconds instead of hours or days.</p>
 `,
      right: `
    <h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Explain what backpropagation is and why it makes training traditional neural networks slow and resource-intensive.</li>
    <li><strong>Measurable:</strong> Attendees can describe the backprop loop as “predict → measure loss → adjust weights → repeat.”</li>
    <li><strong>Achievable:</strong> Use the slider demo with heatmaps and charts to visually reinforce the concepts of weight updates and loss curves.</li>
    <li><strong>Relevant:</strong> Sets up the contrast for why Extreme Learning Machines are different and valuable.</li>
    <li><strong>Time-bound:</strong> Deliver this explanation and demo in under 4 minutes before introducing how ELMs skip this process.</li>
  </ul>

      `
    },

    /* =========================
       SLIDE HUANG — The Question
       ========================= */
    slideHuang: {
      left: `
       <p>Here’s the bold idea from Guang-Bin Huang, the creator of Extreme Learning Machines: what if we didn’t train the hidden layer at all?</p>

  <p>At first, that sounds backwards. Isn’t the hidden layer where all the magic happens? But here’s the key insight: if you randomize the hidden layer just once, it can still provide a rich enough coordinate system for the data.</p>

  <p>That means instead of spending hours adjusting hidden weights with backpropagation, ELMs simply freeze the hidden layer and solve the output layer in a single step.</p>

  <p>One pass, one calculation — and you’re done.</p>`,
      right: `
  <h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Introduce Guang-Bin Huang’s insight that the hidden layer does not need to be trained.</li>
    <li><strong>Measurable:</strong> Attendees can restate ELM’s approach as “randomize the hidden layer once, freeze it, and solve only the output layer.”</li>
    <li><strong>Achievable:</strong> Use a simple explanation and visual to demystify how this shortcut still works.</li>
    <li><strong>Relevant:</strong> Highlights the core difference between ELMs and traditional neural networks, connecting back to the pain of backpropagation.</li>
    <li><strong>Time-bound:</strong> Deliver this key insight in under 3 minutes to keep momentum moving into practical demos.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE GRID — Random Grid Analogy
       ========================= */
    slideGrid: {
      left: `
      <p>On the last slide, we saw Huang’s bold idea: don’t train the hidden layer—just randomize it once and freeze it. That might still feel a little abstract, so let’s make it more concrete with a visual analogy.</p>

  <p>Imagine looking down on a messy city. Without a system, it’s hard to describe where anything is. Now, if we drop a random grid on top, the grid won’t line up with the streets, but it gives us coordinates: “row 3, column 5.” It’s not perfect—but it’s useful.</p>

  <p>Extreme Learning Machines use a random hidden layer in the same way. That random layer projects each example into a set of coordinates. And once you’ve got coordinates, it becomes much easier to separate classes using a simple output mapping.</p>
`,
      right: `
  <h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Use a city-grid analogy to explain how a randomized hidden layer can still provide useful structure for data.</li>
    <li><strong>Measurable:</strong> Attendees can describe the hidden layer as a way of giving “coordinates” that make data easier to separate.</li>
    <li><strong>Achievable:</strong> Leverage a simple visual analogy (city + grid) to make the abstract math idea intuitive.</li>
    <li><strong>Relevant:</strong> Builds on Huang’s insight from the previous slide and reinforces why ELMs don’t need to train hidden layers.</li>
    <li><strong>Time-bound:</strong> Deliver the analogy in under 3 minutes, keeping the audience engaged before diving into demos.</li>
  </ul>
      `
    },

    /* =========================
       SLIDE GPS — Pseudoinverse
       ========================= */
    slideGPS: {
      left: `
       <p>Here’s the one-shot solve. Remember, the hidden layer gave us a set of coordinates for each training example. Now our job is to map those coordinates to the right answers.</p>

  <p>Instead of running loops and nudging weights over and over, we can just draw the best straight lines through those coordinates in a single calculation. That’s it—one step.</p>

  <p>If the data is messy or the features overlap too much, we add a small <strong>stability boost</strong> (called regularization) to keep things well-behaved.</p>

  <p><strong>Key takeaway:</strong> Hidden features in → one calculation → output weights out.</p>

  <p style="font-size: 0.9em; color: #aaa;">
    (Math shorthand for reference: <code>Hβ ≈ Y</code>.
    If <code>H</code> is <code>n × h</code> and <code>Y</code> is <code>n × k</code>, then <code>β</code> is <code>h × k</code>.)
  </p> `,
      right: `
  <h3>SMART Goal for This Slide</h3>
  <ul>
    <li><strong>Specific:</strong> Show how Extreme Learning Machines replace slow, iterative backpropagation with a single one-shot solve for output weights.</li>
    <li><strong>Measurable:</strong> Attendees can restate the ELM process as “random hidden layer → one-step solve for outputs.”</li>
    <li><strong>Achievable:</strong> Use the optimization engine diagram and live demo to make the concept clear without requiring deep math.</li>
    <li><strong>Relevant:</strong> Highlights the core advantage of ELMs: fast training without gradient descent loops.</li>
    <li><strong>Time-bound:</strong> Deliver this insight in under 4 minutes before moving into applied examples and demos.</li>
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

  function setNotes(panelEl, html, position) {
    if (!panelEl) return;
    let notes = panelEl.querySelector('.notes');
    if (!notes) {
      notes = document.createElement('div');
      notes.className = 'notes';
      panelEl.appendChild(notes);
    }
    console.log(notes)
    if (position === "right") {
      notes.innerHTML = '<span class="label">Bullet Points:</span>' + html;
    } else {
      notes.innerHTML = '<span class="label">Speaker Notes:</span>' + html;
    }

  }

  function apply(slideId, spec) {
    const slide = document.getElementById(slideId);
    if (!slide) return;
    const leftPanel = slide.querySelector('.left .panel');
    const rightPanel = slide.querySelector('.right .panel');
    if (spec.left) setNotes(leftPanel, spec.left, "left");
    if (spec.right) setNotes(rightPanel, spec.right, "right");
  }

  Object.entries(NOTES).forEach(([id, spec]) => apply(id, spec));
});

