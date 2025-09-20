# ELM Drum Generator — Main-thread build (uses your library directly)

This version avoids Web Workers and loads your UMD exactly like your working demo does.

## How to run
1. Serve your app with Vite at http://localhost:5173/ (or any static server).
2. Place these files so they’re reachable (e.g., Vite `public/`):
   - /index.html
   - /app.js
   - /astermind.umd.js (your existing file)
3. Visit http://localhost:5173/index.html
4. Click **Train**, then **Generate 8 Bars**.

## Why this build
Your previous error came from the worker trying to `importScripts` the UMD (origin/CORS/export issues). Here we call `window.astermind` directly, like your working reference demo, so those issues disappear.
