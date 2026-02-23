// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from './safetensors';
import type { CartPoleAction, CartPoleState } from './types';

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

export interface DqnOutput {
  qLeft: number;
  qRight: number;
  action: CartPoleAction;
}

export type DqnPolicy = (state: Readonly<CartPoleState>) => DqnOutput;

export async function loadDqnPolicy(url: string): Promise<DqnPolicy> {
  const tensors = await loadSafetensors(url);

  // Load all layers: w0/b0, w1/b1, ... until a key is missing.
  const layers: { w: Float32Array; b: Float32Array }[] = [];
  for (let i = 0; ; i++) {
    const wKey = `w${String(i)}`;
    const bKey = `b${String(i)}`;
    if (!Object.hasOwn(tensors, wKey) || !Object.hasOwn(tensors, bKey)) {break;}
    layers.push({ w: tensors[wKey], b: tensors[bKey] });
  }

  return (state: Readonly<CartPoleState>): DqnOutput => {
    let x: number[] = [state[0], state[1], state[2], state[3]];
    for (let i = 0; i < layers.length - 1; i++) {
      x = relu(linear(x, layers[i].w, layers[i].b));
    }
    const q = linear(x, layers[layers.length - 1].w, layers[layers.length - 1].b);
    const action: CartPoleAction = q[1] > q[0] ? 1 : 0;
    return { qLeft: q[0], qRight: q[1], action };
  };
}
