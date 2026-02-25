// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadSafetensors } from '../shared/safetensors';
import { ENV_HEIGHT, ENV_WIDTH } from './environment';
import type { PongQValues } from './nn-policy';
import type { PixelPongState } from './pixel-env';
import type { PongAction } from './types';

export type { PongQValues } from './nn-policy';

const N_FRAMES = 4;
const SINGLE_FRAME_DIM = ENV_HEIGHT * ENV_WIDTH;

// CNN architecture constants — must match Python QNetwork.
const CONV0 = { inC: N_FRAMES, outC: 16, kH: 5, kW: 5, strideH: 2, strideW: 2 };
const CONV1 = { inC: 16,       outC: 32, kH: 3, kW: 3, strideH: 2, strideW: 2 };

/**
 * 2-D convolution + ReLU (VALID padding, channels-last layout).
 *
 * input: Float32Array of shape (inH, inW, inC) in row-major order.
 * w:     Float32Array of shape (outC, inC, kH, kW) — exported from Python as (out, in, kH, kW).
 * b:     Float32Array of shape (outC,).
 * Returns a Float32Array of shape (outH, outW, outC).
 */
function conv2dRelu(
  input: Float32Array,
  inH: number, inW: number, inC: number,
  w: Float32Array, b: Float32Array,
  kH: number, kW: number, outC: number,
  strideH: number, strideW: number
): Float32Array {
  const outH = Math.floor((inH - kH) / strideH) + 1;
  const outW = Math.floor((inW - kW) / strideW) + 1;
  const out = new Float32Array(outH * outW * outC);
  for (let oc = 0; oc < outC; oc++) {
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        let sum = b[oc];
        for (let ic = 0; ic < inC; ic++) {
          for (let kh = 0; kh < kH; kh++) {
            for (let kw = 0; kw < kW; kw++) {
              const ih = oh * strideH + kh;
              const iw = ow * strideW + kw;
              sum += w[((oc * inC + ic) * kH + kh) * kW + kw] * input[(ih * inW + iw) * inC + ic];
            }
          }
        }
        out[(oh * outW + ow) * outC + oc] = Math.max(0, sum);
      }
    }
  }
  return out;
}

function linear(input: ArrayLike<number>, w: Float32Array, b: Float32Array): number[] {
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

export type PixelPongNNPolicy = (
  state: PixelPongState
) => { qValues: PongQValues; action: PongAction };

export async function loadPixelPongNNPolicy(
  url: string
): Promise<PixelPongNNPolicy> {
  const tensors = await loadSafetensors(url);

  const conv0 = { w: tensors.conv0_w, b: tensors.conv0_b };
  const conv1 = { w: tensors.conv1_w, b: tensors.conv1_b };

  const denseLayers: { w: Float32Array; b: Float32Array }[] = [];
  for (let i = 0; ; i++) {
    if (!Object.hasOwn(tensors, `w${String(i)}`)) {break;}
    denseLayers.push({ w: tensors[`w${String(i)}`], b: tensors[`b${String(i)}`] });
  }

  // Frame stack: (N_FRAMES, SINGLE_FRAME_DIM), oldest frame first.
  // Initialised to zeros; fills up naturally after N_FRAMES steps.
  const frameStack = new Float32Array(N_FRAMES * SINGLE_FRAME_DIM);

  return (state: PixelPongState): { qValues: PongQValues; action: PongAction } => {
    // Push new frame: shift left by one frame, insert at end.
    frameStack.copyWithin(0, SINGLE_FRAME_DIM);
    frameStack.set(state.pixels, (N_FRAMES - 1) * SINGLE_FRAME_DIM);

    // Rearrange (N_FRAMES, H, W) → (H, W, N_FRAMES) for channels-last conv.
    const hwc = new Float32Array(ENV_HEIGHT * ENV_WIDTH * N_FRAMES);
    for (let f = 0; f < N_FRAMES; f++) {
      for (let h = 0; h < ENV_HEIGHT; h++) {
        for (let w = 0; w < ENV_WIDTH; w++) {
          hwc[(h * ENV_WIDTH + w) * N_FRAMES + f] =
            frameStack[f * SINGLE_FRAME_DIM + h * ENV_WIDTH + w];
        }
      }
    }

    // Conv0 (5×5, stride 2, VALID) + ReLU
    const after0 = conv2dRelu(
      hwc,
      ENV_HEIGHT,
      ENV_WIDTH,
      CONV0.inC,
      conv0.w,
      conv0.b,
      CONV0.kH,
      CONV0.kW,
      CONV0.outC,
      CONV0.strideH,
      CONV0.strideW
    );
    const h0 = Math.floor((ENV_HEIGHT - CONV0.kH) / CONV0.strideH) + 1;
    const w0 = Math.floor((ENV_WIDTH  - CONV0.kW) / CONV0.strideW) + 1;

    // Conv1 (3×3, stride 2, VALID) + ReLU
    const after1 = conv2dRelu(
      after0,
      h0,
      w0,
      CONV1.inC,
      conv1.w,
      conv1.b,
      CONV1.kH,
      CONV1.kW,
      CONV1.outC,
      CONV1.strideH,
      CONV1.strideW
    );

    // Dense head
    let x: ArrayLike<number> = after1;
    for (let i = 0; i < denseLayers.length - 1; i++) {
      x = relu(linear(x, denseLayers[i].w, denseLayers[i].b));
    }
    const q = linear(
      x,
      denseLayers[denseLayers.length - 1].w,
      denseLayers[denseLayers.length - 1].b
    );

    const qValues: PongQValues = { noop: q[0], up: q[1], down: q[2] };
    let best = 0;
    if (q[1] > q[best]) {best = 1;}
    if (q[2] > q[best]) {best = 2;}
    const action = best as PongAction;
    return { qValues, action };
  };
}
