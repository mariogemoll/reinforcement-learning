// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { BreakoutAction } from './breakout';
import type { BreakoutQValues } from './breakout-policy-types';

export function linear(
  input: ArrayLike<number>,
  w: Float32Array,
  b: Float32Array
): number[] {
  const inDim = input.length;
  const outDim = b.length;
  const out: number[] = new Array(outDim) as number[];
  for (let i = 0; i < outDim; i++) {
    let sum = b[i];
    for (let j = 0; j < inDim; j++) {
      sum += w[(j * outDim) + i] * input[j];
    }
    out[i] = sum;
  }
  return out;
}

export function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

export function pickGreedyAction(q: number[]): BreakoutAction {
  let best = 0;
  if (q[1] > q[best]) {best = 1;}
  if (q[2] > q[best]) {best = 2;}
  return best as BreakoutAction;
}

export function toQValues(q: number[]): BreakoutQValues {
  return {
    noop: q[0],
    left: q[1],
    right: q[2]
  };
}
