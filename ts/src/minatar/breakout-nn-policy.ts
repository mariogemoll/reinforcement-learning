// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from '../shared/safetensors';
import {
  type BreakoutAction,
  type BreakoutState,
  getBreakoutObservation
} from './breakout';

interface DenseLayer {
  w: Float32Array;
  b: Float32Array;
}

export interface BreakoutQValues {
  noop: number;
  left: number;
  right: number;
}

export type BreakoutNNPolicy = (
  state: BreakoutState
) => { qValues: BreakoutQValues; action: BreakoutAction };

function linear(input: ArrayLike<number>, w: Float32Array, b: Float32Array): number[] {
  // Breakout weights are exported as (in_dim, out_dim) kernels.
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

function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

function extractFeatures(state: BreakoutState): number[] {
  return Array.from(getBreakoutObservation(state));
}

export async function loadBreakoutNNPolicy(url: string): Promise<BreakoutNNPolicy> {
  const tensors = await loadSafetensors(url);

  const hiddenLayers: DenseLayer[] = [];
  for (let i = 1; ; i++) {
    const wKey = `dense${String(i)}.weight`;
    const bKey = `dense${String(i)}.bias`;
    if (!Object.hasOwn(tensors, wKey) || !Object.hasOwn(tensors, bKey)) {
      break;
    }
    hiddenLayers.push({ w: tensors[wKey], b: tensors[bKey] });
  }

  if (!Object.hasOwn(tensors, 'dense_out.weight') || !Object.hasOwn(tensors, 'dense_out.bias')) {
    throw new Error('Breakout policy weights are missing dense_out tensors');
  }

  const outLayer: DenseLayer = {
    w: tensors['dense_out.weight'],
    b: tensors['dense_out.bias']
  };

  return (state: BreakoutState): { qValues: BreakoutQValues; action: BreakoutAction } => {
    let x = extractFeatures(state);
    for (const layer of hiddenLayers) {
      x = relu(linear(x, layer.w, layer.b));
    }

    const q = linear(x, outLayer.w, outLayer.b);
    const qValues: BreakoutQValues = {
      noop: q[0],
      left: q[1],
      right: q[2]
    };

    let best = 0;
    if (q[1] > q[best]) {best = 1;}
    if (q[2] > q[best]) {best = 2;}

    return { qValues, action: best as BreakoutAction };
  };
}
