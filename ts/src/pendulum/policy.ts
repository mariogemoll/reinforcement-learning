// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from '../shared/safetensors';
import type { PendulumObs } from './types';

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

function tanh(x: number[]): number[] {
  return x.map(v => Math.tanh(v));
}

export interface PolicyOutput {
  torque: number;  // greedy action, clipped to [-2, 2]
}

export type PendulumPolicy = (obs: Readonly<PendulumObs>) => PolicyOutput;

export async function loadPendulumPolicy(url: string): Promise<PendulumPolicy> {
  const tensors = await loadSafetensors(url);

  const layers: { w: Float32Array; b: Float32Array }[] = [];
  for (let i = 0; ; i++) {
    const wKey = `w${String(i)}`;
    const bKey = `b${String(i)}`;
    if (!Object.hasOwn(tensors, wKey) || !Object.hasOwn(tensors, bKey)) { break; }
    layers.push({ w: tensors[wKey], b: tensors[bKey] });
  }

  const wMean = tensors.w_mean;
  const bMean = tensors.b_mean;

  return (obs: Readonly<PendulumObs>): PolicyOutput => {
    let x: number[] = [obs[0], obs[1], obs[2]];
    for (const layer of layers) {
      x = tanh(linear(x, layer.w, layer.b));
    }
    const mean = linear(x, wMean, bMean);
    const torque = Math.max(-2.0, Math.min(2.0, mean[0]));
    return { torque };
  };
}
