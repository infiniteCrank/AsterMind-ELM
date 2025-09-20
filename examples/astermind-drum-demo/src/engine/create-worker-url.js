// Optional helper if you want to inline the worker via Blob later.
export async function createWorkerURLFromFile(pathRelativeToRoot = '/src/worker/elm-worker.js') {
  const res = await fetch(pathRelativeToRoot);
  const src = await res.text();
  const blob = new Blob([src], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}
