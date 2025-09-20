import { TOK } from './tokens.js';

export function grooveFourOnTheFloor() {
  const seq = [];
  for (let bar = 0; bar < 4; bar++) {
    for (let step = 0; step < 16; step++) {
      const pos = step % 16;
      if (pos === 0 || pos === 4 || pos === 8 || pos === 12) seq.push(TOK.ON_K);
      else if (pos === 4 || pos === 12) seq.push(TOK.ON_S);
      else if (pos % 2 === 0) seq.push(TOK.ON_H);
      else seq.push(TOK.WAIT_1);
    }
  }
  return seq;
}
export function grooveBackbeat() {
  const seq = [];
  for (let bar = 0; bar < 4; bar++) {
    for (let step = 0; step < 16; step++) {
      const pos = step % 16;
      if (pos === 0 || pos === 7) seq.push(TOK.ON_K);
      else if (pos === 4 || pos === 12) seq.push(TOK.ON_S);
      else seq.push(TOK.ON_H);
    }
  }
  return seq;
}
export const TRAIN_SEQS = [grooveFourOnTheFloor(), grooveBackbeat()];
