// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from './safetensors';
import type { CartPoleAction, CartPoleState } from './types';

// Network dimensions
const IN = 4;
const H1 = 128;
const H2 = 128;
const OUT = 2;

function linear(
  input: number[],
  w: Float32Array,
  b: Float32Array,
  inDim: number,
  outDim: number
): number[] {
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
  const w0 = tensors.w0;
  const b0 = tensors.b0;
  const w2 = tensors.w2;
  const b2 = tensors.b2;
  const w4 = tensors.w4;
  const b4 = tensors.b4;

  return (state: Readonly<CartPoleState>): DqnOutput => {
    const input = [state[0], state[1], state[2], state[3]];
    const h1 = relu(linear(input, w0, b0, IN, H1));
    const h2 = relu(linear(h1, w2, b2, H1, H2));
    const q = linear(h2, w4, b4, H2, OUT);
    const action: CartPoleAction = q[1] > q[0] ? 1 : 0;
    return { qLeft: q[0], qRight: q[1], action };
  };
}
