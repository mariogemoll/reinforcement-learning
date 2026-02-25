// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import {
  BREAKOUT_CHANNELS,
  BREAKOUT_GRID_SIZE,
  BREAKOUT_MAX_STEPS,
  type BreakoutState,
  getBreakoutObservation,
  resetBreakoutState,
  stepBreakoutState
} from './breakout';

describe('resetBreakoutState', () => {
  it('initializes to canonical start configuration', () => {
    const state = resetBreakoutState(0);
    expect(state.ballX).toBe(0);
    expect(state.ballY).toBe(3);
    expect(state.ballDir).toBe(2);
    expect(state.pos).toBe(4);
    expect(state.time).toBe(0);
    expect(state.terminal).toBe(false);
  });
});

describe('getBreakoutObservation', () => {
  it('returns a 10x10x4 tensor and encodes paddle/ball/trail', () => {
    const state = resetBreakoutState(1);
    const obs = getBreakoutObservation(state);
    expect(obs.length).toBe(
      BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE * BREAKOUT_CHANNELS
    );

    const paddleIdx = ((9 * BREAKOUT_GRID_SIZE) + state.pos) * BREAKOUT_CHANNELS;
    const ballIdx = (((state.ballY * BREAKOUT_GRID_SIZE) + state.ballX) * BREAKOUT_CHANNELS) + 1;
    const trailIdx = (((state.lastY * BREAKOUT_GRID_SIZE) + state.lastX) * BREAKOUT_CHANNELS) + 2;
    expect(obs[paddleIdx]).toBe(1);
    expect(obs[ballIdx]).toBe(1);
    expect(obs[trailIdx]).toBe(1);
  });
});

describe('stepBreakoutState', () => {
  it('advances time and preserves observation shape', () => {
    const state = resetBreakoutState(0);
    const result = stepBreakoutState(state, 0);
    expect(result.state.time).toBe(1);
    expect(result.observation.length).toBe(
      BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE * BREAKOUT_CHANNELS
    );
  });

  it('moves paddle left/right under actions', () => {
    const state = resetBreakoutState(0);
    const left = stepBreakoutState(state, 1);
    expect(left.state.pos).toBe(3);

    const right = stepBreakoutState(state, 2);
    expect(right.state.pos).toBe(5);
  });

  it('rewards and clears brick on first collision frame', () => {
    const brickMap = new Uint8Array(BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE);
    // Ball moves from (4,2) to (3,1), so place a brick at (3,1).
    brickMap[(3 * BREAKOUT_GRID_SIZE) + 1] = 1;
    const state: BreakoutState = {
      ballY: 4,
      ballX: 2,
      ballDir: 0, // up-left
      pos: 4,
      brickMap,
      strike: false,
      lastY: 4,
      lastX: 2,
      time: 0,
      terminal: false
    };

    const result = stepBreakoutState(state, 0);
    expect(result.reward).toBe(1);
    expect(result.done).toBe(false);
    expect(result.state.brickMap[(3 * BREAKOUT_GRID_SIZE) + 1]).toBe(0);
    expect(result.state.ballY).toBe(4); // bounced back to previous row
    expect(result.state.ballDir).toBe(3); // y-reflected from 0 -> 3
    expect(result.state.strike).toBe(true);
  });

  it('terminates when ball misses paddle at bottom row', () => {
    const state: BreakoutState = {
      ballY: 8,
      ballX: 5,
      ballDir: 2, // down-right to (9,6)
      pos: 0, // paddle far away from old/new x
      brickMap: new Uint8Array(BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE),
      strike: false,
      lastY: 8,
      lastX: 5,
      time: 42,
      terminal: false
    };

    const result = stepBreakoutState(state, 0);
    expect(result.reward).toBe(0);
    expect(result.done).toBe(true);
    expect(result.state.terminal).toBe(true);
  });

  it('terminates at max step limit', () => {
    const state: BreakoutState = {
      ballY: 5,
      ballX: 5,
      ballDir: 0,
      pos: 5,
      brickMap: new Uint8Array(BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE),
      strike: false,
      lastY: 5,
      lastX: 5,
      time: BREAKOUT_MAX_STEPS - 1,
      terminal: false
    };

    const result = stepBreakoutState(state, 0);
    expect(result.done).toBe(true);
    expect(result.state.terminal).toBe(true);
    expect(result.state.time).toBe(BREAKOUT_MAX_STEPS);
  });

  it('reflects on horizontal boundary', () => {
    const state: BreakoutState = {
      ballY: 5,
      ballX: 0,
      ballDir: 0, // up-left, attempts x=-1
      pos: 4,
      brickMap: new Uint8Array(BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE),
      strike: false,
      lastY: 5,
      lastX: 0,
      time: 0,
      terminal: false
    };

    const result = stepBreakoutState(state, 0);
    expect(result.done).toBe(false);
    expect(result.state.ballX).toBe(0);
    expect(result.state.ballDir).toBe(1); // x-reflected from 0 -> 1
  });
});
