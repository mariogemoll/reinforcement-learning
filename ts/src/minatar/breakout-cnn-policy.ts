// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { getBreakoutObservation } from './breakout';
import { linear, pickGreedyAction, relu, toQValues } from './breakout-policy-math';
import type { BreakoutPolicy } from './breakout-policy-types';

interface ConvLayer {
  kernel: Float32Array;
  bias: Float32Array;
}

interface ConvOutput {
  data: Float32Array;
  h: number;
  w: number;
  c: number;
}

function conv2dValidRelu(
  input: Float32Array,
  inH: number,
  inW: number,
  inC: number,
  kernel: Float32Array,
  bias: Float32Array
): ConvOutput {
  const kH = 3;
  const kW = 3;
  const outC = bias.length;
  const outH = inH - (kH - 1);
  const outW = inW - (kW - 1);

  if (outH <= 0 || outW <= 0) {
    throw new Error('Invalid CNN shape: conv output dimensions are non-positive');
  }

  const expectedKernelSize = kH * kW * inC * outC;
  if (kernel.length !== expectedKernelSize) {
    throw new Error(
      `Invalid CNN kernel size: got ${String(kernel.length)}, ` +
        `expected ${String(expectedKernelSize)}`
    );
  }

  const out = new Float32Array(outH * outW * outC);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      for (let oc = 0; oc < outC; oc++) {
        let sum = bias[oc];
        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            for (let ic = 0; ic < inC; ic++) {
              const iy = oy + ky;
              const ix = ox + kx;
              const inputIndex = ((iy * inW) + ix) * inC + ic;
              const kernelIndex = (((ky * kW) + kx) * inC + ic) * outC + oc;
              sum += input[inputIndex] * kernel[kernelIndex];
            }
          }
        }
        const outIndex = ((oy * outW) + ox) * outC + oc;
        out[outIndex] = Math.max(0, sum);
      }
    }
  }

  return { data: out, h: outH, w: outW, c: outC };
}

export function createBreakoutCNNPolicy(tensors: Record<string, Float32Array>): BreakoutPolicy {
  const convLayers: ConvLayer[] = [];
  for (let i = 0; ; i++) {
    const kernelKey = `conv_${String(i)}.kernel`;
    const biasKey = `conv_${String(i)}.bias`;
    if (!Object.hasOwn(tensors, kernelKey) || !Object.hasOwn(tensors, biasKey)) {
      break;
    }
    convLayers.push({
      kernel: tensors[kernelKey],
      bias: tensors[biasKey]
    });
  }

  if (convLayers.length === 0) {
    throw new Error('Breakout CNN policy weights are missing conv_* tensors');
  }
  if (!Object.hasOwn(tensors, 'fc.kernel') || !Object.hasOwn(tensors, 'fc.bias')) {
    throw new Error('Breakout CNN policy weights are missing fc tensors');
  }
  if (!Object.hasOwn(tensors, 'out_layer.kernel') || !Object.hasOwn(tensors, 'out_layer.bias')) {
    throw new Error('Breakout CNN policy weights are missing out_layer tensors');
  }

  const fcKernel = tensors['fc.kernel'];
  const fcBias = tensors['fc.bias'];
  const outKernel = tensors['out_layer.kernel'];
  const outBias = tensors['out_layer.bias'];

  return (state): ReturnType<BreakoutPolicy> => {
    let x: Float32Array = new Float32Array(getBreakoutObservation(state));
    let h = 10;
    let w = 10;
    let c = 4;

    for (const layer of convLayers) {
      const out = conv2dValidRelu(x, h, w, c, layer.kernel, layer.bias);
      x = out.data;
      h = out.h;
      w = out.w;
      c = out.c;
    }

    const fc = relu(linear(x, fcKernel, fcBias));
    const q = linear(fc, outKernel, outBias);
    return {
      qValues: toQValues(q),
      action: pickGreedyAction(q)
    };
  };
}
