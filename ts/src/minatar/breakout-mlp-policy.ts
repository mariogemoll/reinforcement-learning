// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { type BreakoutState,getBreakoutObservation } from './breakout';
import { linear, pickGreedyAction, relu, toQValues } from './breakout-policy-math';
import type { BreakoutPolicy } from './breakout-policy-types';

interface DenseLayer {
  w: Float32Array;
  b: Float32Array;
}

function extractFeatures(state: BreakoutState): number[] {
  return Array.from(getBreakoutObservation(state));
}

export function createBreakoutMLPPolicy(tensors: Record<string, Float32Array>): BreakoutPolicy {
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
    throw new Error('Breakout MLP policy weights are missing dense_out tensors');
  }

  const outLayer: DenseLayer = {
    w: tensors['dense_out.weight'],
    b: tensors['dense_out.bias']
  };

  return (state): ReturnType<BreakoutPolicy> => {
    let x = extractFeatures(state);
    for (const layer of hiddenLayers) {
      x = relu(linear(x, layer.w, layer.b));
    }
    const q = linear(x, outLayer.w, outLayer.b);
    return {
      qValues: toQValues(q),
      action: pickGreedyAction(q)
    };
  };
}
