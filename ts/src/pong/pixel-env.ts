// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/**
 * Pixel-observation Pong environment.  Mirrors py/pong_pixel_env.py.
 *
 * The agent sees only a flat (HEIGHT × WIDTH) Float32Array of pixels
 * (0 = background, 1 = object).  The underlying engine state is kept
 * internal so it cannot be used as a cheap shortcut for a policy.
 */

import {
  computerPongPlayer,
  ENV_HEIGHT,
  ENV_WIDTH,
  PADDLE_HALF_HEIGHT,
  resetPongState,
  stepPongState
} from './environment';
import type { PongAction, PongState } from './types';

export interface PixelPongState {
  pixels: Float32Array;

  /** Internal engine state — not intended for agent consumption. */
  _engine: PongState;
}

export interface PixelPongStepResult {
  state: PixelPongState;
  done: boolean;
}

function render(engine: PongState): Float32Array {
  const frame = new Float32Array(ENV_HEIGHT * ENV_WIDTH);

  // Ball
  const br = Math.max(0, Math.min(ENV_HEIGHT - 1, Math.round(engine.ballRow)));
  const bc = Math.max(0, Math.min(ENV_WIDTH - 1, Math.round(engine.ballCol)));
  frame[br * ENV_WIDTH + bc] = 1;

  // Player paddle (left, col 0)
  const p1Center = Math.round(engine.p1Center);
  for (let offset = -PADDLE_HALF_HEIGHT; offset <= PADDLE_HALF_HEIGHT; offset++) {
    const r = p1Center + offset;
    if (r >= 0 && r < ENV_HEIGHT) {
      frame[r * ENV_WIDTH] = 1;
    }
  }

  // CPU paddle (right, col WIDTH-1)
  const p2Center = Math.round(engine.p2Center);
  for (let offset = -PADDLE_HALF_HEIGHT; offset <= PADDLE_HALF_HEIGHT; offset++) {
    const r = p2Center + offset;
    if (r >= 0 && r < ENV_HEIGHT) {
      frame[r * ENV_WIDTH + (ENV_WIDTH - 1)] = 1;
    }
  }

  return frame;
}

export function resetPixelPong(initVY: 1 | -1): PixelPongState {
  const engine = resetPongState(initVY);
  return { pixels: render(engine), _engine: engine };
}

export function stepPixelPong(
  state: PixelPongState,
  p1Action: PongAction
): PixelPongStepResult {
  const cpuAction = computerPongPlayer(state._engine);
  const { state: nextEngine, done } = stepPongState(state._engine, p1Action, cpuAction);
  return {
    state: { pixels: render(nextEngine), _engine: nextEngine },
    done
  };
}
