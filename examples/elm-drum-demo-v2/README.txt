ELM Drum Generator • Demo (v2)
=================================

How to run
----------
1) Build or download your UMD bundle and serve it at **/astermind.umd.js** (site root).
   - For local testing, you can run: `npx http-server . -p 8080` then place the UMD at `./astermind.umd.js` and open http://localhost:8080/elm-drum-demo-v2/

2) Open **index.html** in a local web server (not file://).

What changed in v2
------------------
- WebWorker split into **elm-worker.js** (supports actions: init, train/fit/update/trainFromData, predictLogits, toJSON, dispose)
- Training progress overlay with visible stages and progress bar
- Kernel controls (γ, degree, coef0, m, whiten) + smart defaults
- Safer AudioContext resume on user gesture
- Export model to JSON (works for main-thread and worker paths)
