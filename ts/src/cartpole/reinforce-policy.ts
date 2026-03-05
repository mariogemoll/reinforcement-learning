// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from '../shared/safetensors';
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

function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export interface ReinforceOutput {
  probLeft: number;
  probRight: number;
  action: CartPoleAction;
}

export type ReinforcePolicy = (state: Readonly<CartPoleState>) => ReinforceOutput;

export async function loadReinforcePolicy(url: string): Promise<ReinforcePolicy> {
  const tensors = await loadSafetensors(url);

  // Load all layers: w0/b0, w1/b1, ... until a key is missing.
  const layers: { w: Float32Array; b: Float32Array }[] = [];
  for (let i = 0; ; i++) {
    const wKey = `w${String(i)}`;
    const bKey = `b${String(i)}`;
    if (!Object.hasOwn(tensors, wKey) || !Object.hasOwn(tensors, bKey)) {
      break;
    }
    layers.push({ w: tensors[wKey], b: tensors[bKey] });
  }

  return (state: Readonly<CartPoleState>): ReinforceOutput => {
    let x: number[] = [state[0], state[1], state[2], state[3]];
    // Hidden layers with ReLU
    for (let i = 0; i < layers.length - 1; i++) {
      x = relu(linear(x, layers[i].w, layers[i].b));
    }
    // Final layer gives logits
    const logits = linear(x, layers[layers.length - 1].w, layers[layers.length - 1].b);
    // Apply softmax to get action probabilities
    const probs = softmax(logits);
    // Sample action (for deterministic behavior, use argmax)
    const action: CartPoleAction = probs[1] > probs[0] ? 1 : 0;
    return { probLeft: probs[0], probRight: probs[1], action };
  };
}
