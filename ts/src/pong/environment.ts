// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Mirrors gymnax Pong-misc defaults exactly.
// Physics order: move paddles → move ball → reflect walls → reflect paddles.
// The top-wall reflection is missing in gymnax and is added here.

import type { PongAction, PongPlayer, PongState, PongStepResult } from './types';

export const ENV_WIDTH = 40;
export const ENV_HEIGHT = 30;
export const PADDLE_HALF_HEIGHT = 2;

const BALL_MAX_Y_SPEED = 3;
const PADDLE_Y_SPEED = 1;
const BALL_X_SPEED = 1;
const MAX_STEPS = 1000;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function resetPongState(initVY: 1 | -1): PongState {
  return {
    ballRow: ENV_HEIGHT / 2,
    ballCol: ENV_WIDTH / 2,
    ballVRow: initVY,
    ballVCol: BALL_X_SPEED,
    p1Center: ENV_HEIGHT / 2,
    p2Center: ENV_HEIGHT / 2,
    time: 0
  };
}

// Greedy computer AI: always moves toward the ball.
export const computerPongPlayer: PongPlayer = (state: PongState): PongAction => {
  const { ballRow, p2Center } = state;
  const distIfDown = Math.abs(
    ballRow -
      clamp(
        p2Center + PADDLE_Y_SPEED,
        PADDLE_HALF_HEIGHT,
        ENV_HEIGHT - PADDLE_HALF_HEIGHT - 1
      )
  );
  const distIfUp = Math.abs(
    ballRow -
      clamp(
        p2Center - PADDLE_Y_SPEED,
        PADDLE_HALF_HEIGHT,
        ENV_HEIGHT - PADDLE_HALF_HEIGHT - 1
      )
  );
  return distIfUp < distIfDown ? 1 : 2;
};

export function stepPongState(
  state: PongState,
  p1Action: PongAction,
  p2Action: PongAction
): PongStepResult {
  const { ballRow, ballCol, ballVRow, ballVCol, p1Center, p2Center, time } = state;

  // --- Move paddles ---
  const p1Step = (p1Action === 2 ? 1 : 0) - (p1Action === 1 ? 1 : 0);
  const newP1 = clamp(
    p1Center + p1Step * PADDLE_Y_SPEED,
    PADDLE_HALF_HEIGHT,
    ENV_HEIGHT - PADDLE_HALF_HEIGHT - 1
  );

  const p2Step = (p2Action === 2 ? 1 : 0) - (p2Action === 1 ? 1 : 0);
  const newP2 = clamp(
    p2Center + p2Step * PADDLE_Y_SPEED,
    PADDLE_HALF_HEIGHT,
    ENV_HEIGHT - PADDLE_HALF_HEIGHT - 1
  );

  // --- Move ball ---
  let newRow = ballRow + ballVRow;
  let newCol = ballCol + ballVCol;
  let newVRow = ballVRow;
  let newVCol = ballVCol;

  // --- Reflect on walls ---
  // Bottom wall (gymnax implements this one)
  if (newRow >= ENV_HEIGHT) {
    newRow = 2 * (ENV_HEIGHT - 1) - newRow;
    newVRow = -newVRow;
  }
  // Top wall (missing in gymnax — fixed here)
  if (newRow < 0) {
    newRow = -newRow;
    newVRow = -newVRow;
  }

  // --- Reflect on paddles ---
  const leftReflCol  = 2 * 1 - newCol;
  const rightReflCol = 2 * (ENV_WIDTH - 2) - newCol;
  const distP1 = newRow - newP1;
  const distP2 = newRow - newP2;

  if (leftReflCol >= 1 && Math.abs(distP1) <= PADDLE_HALF_HEIGHT) {
    newCol  = leftReflCol;
    newVCol = -newVCol;
    // Match gymnax int32 truncation for the y-speed nudge
    newVRow = clamp(
      newVRow + Math.trunc(distP1 / PADDLE_HALF_HEIGHT),
      -BALL_MAX_Y_SPEED,
      BALL_MAX_Y_SPEED
    );
  }
  if (rightReflCol < ENV_WIDTH - 2 && Math.abs(distP2) < PADDLE_HALF_HEIGHT + 1) {
    newCol  = rightReflCol;
    newVCol = -newVCol;
    newVRow = clamp(
      newVRow + Math.trunc(distP2 / PADDLE_HALF_HEIGHT),
      -BALL_MAX_Y_SPEED,
      BALL_MAX_Y_SPEED
    );
  }

  const newTime = time + 1;
  const done = newCol < 0 || newCol >= ENV_WIDTH || newTime >= MAX_STEPS;

  return {
    state: {
      ballRow: newRow,
      ballCol: newCol,
      ballVRow: newVRow,
      ballVCol: newVCol,
      p1Center: newP1,
      p2Center: newP2,
      time: newTime
    },
    done
  };
}
