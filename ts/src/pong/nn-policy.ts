// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from '../shared/safetensors';
import type { PongAction, PongState } from './types';

function linear(input: number[], w: Float32Array, b: Float32Array): number[] {
  const outDim = b.length;
  const inDim = input.length;
  const out: number[] = new Array(outDim) as number[];
  for (let i = 0; i < outDim; i++) {
    let sum = b[i];
    for (let j = 0; j < inDim; j++) {
      sum += w[i * inDim + j] * input[j];
    }
    out[i] = sum;
  }
  return out;
}

function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

function extractFeatures(state: PongState): number[] {
  // Mirrors extract_features() in py/pong.py — normalised to ~[-1, 1].
  return [
    state.p1Center / 15 - 1,
    state.p2Center / 15 - 1,
    state.ballRow  / 15 - 1,
    state.ballCol  / 20 - 1,
    state.ballVRow / 3,
    state.ballVCol
  ];
}

export interface PongQValues { noop: number; up: number; down: number; }

export type PongNNPolicy = (state: PongState) => { qValues: PongQValues; action: PongAction };

export async function loadPongNNPolicy(url: string): Promise<PongNNPolicy> {
  const tensors = await loadSafetensors(url);

  const layers: { w: Float32Array; b: Float32Array }[] = [];
  for (let i = 0; ; i++) {
    const wKey = `w${String(i)}`;
    const bKey = `b${String(i)}`;
    if (!Object.hasOwn(tensors, wKey) || !Object.hasOwn(tensors, bKey)) {break;}
    layers.push({ w: tensors[wKey], b: tensors[bKey] });
  }

  return (state: PongState): { qValues: PongQValues; action: PongAction } => {
    let x = extractFeatures(state);
    for (let i = 0; i < layers.length - 1; i++) {
      x = relu(linear(x, layers[i].w, layers[i].b));
    }
    const q = linear(x, layers[layers.length - 1].w, layers[layers.length - 1].b);
    const qValues: PongQValues = { noop: q[0], up: q[1], down: q[2] };
    let best = 0;
    if (q[1] > q[best]) {best = 1;}
    if (q[2] > q[best]) {best = 2;}
    const action = best as PongAction;
    return { qValues, action };
  };
}
