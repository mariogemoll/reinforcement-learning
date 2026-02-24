// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { ENV_HEIGHT, ENV_WIDTH, PADDLE_HALF_HEIGHT } from './environment';
import { resetPixelPong, stepPixelPong } from './pixel-env';

/** Collect all (row, col) positions of lit pixels. */
function litPixels(pixels: Float32Array): [number, number][] {
  const result: [number, number][] = [];
  for (let r = 0; r < ENV_HEIGHT; r++) {
    for (let c = 0; c < ENV_WIDTH; c++) {
      if (pixels[r * ENV_WIDTH + c] > 0) {
        result.push([r, c]);
      }
    }
  }
  return result;
}

describe('resetPixelPong', () => {
  it('returns pixels with correct dimensions', () => {
    const state = resetPixelPong(1);
    expect(state.pixels).toBeInstanceOf(Float32Array);
    expect(state.pixels.length).toBe(ENV_HEIGHT * ENV_WIDTH);
  });

  it('has ball at centre', () => {
    const state = resetPixelPong(1);
    const ballRow = ENV_HEIGHT / 2;
    const ballCol = ENV_WIDTH / 2;
    expect(state.pixels[ballRow * ENV_WIDTH + ballCol]).toBe(1);
  });

  it('has left paddle in column 0', () => {
    const state = resetPixelPong(1);
    const center = ENV_HEIGHT / 2;
    for (let offset = -PADDLE_HALF_HEIGHT; offset <= PADDLE_HALF_HEIGHT; offset++) {
      expect(state.pixels[(center + offset) * ENV_WIDTH]).toBe(1);
    }
  });

  it('has right paddle in last column', () => {
    const state = resetPixelPong(1);
    const center = ENV_HEIGHT / 2;
    const col = ENV_WIDTH - 1;
    for (let offset = -PADDLE_HALF_HEIGHT; offset <= PADDLE_HALF_HEIGHT; offset++) {
      expect(state.pixels[(center + offset) * ENV_WIDTH + col]).toBe(1);
    }
  });

  it('has exactly ball + two paddles lit', () => {
    const state = resetPixelPong(1);
    const paddleSize = 2 * PADDLE_HALF_HEIGHT + 1;
    // Ball overlaps left paddle at reset (both at centre, col 20 vs col 0),
    // so ball is a separate pixel.
    const expectedLit = 1 + 2 * paddleSize;
    expect(litPixels(state.pixels).length).toBe(expectedLit);
  });

  it('all pixel values are 0 or 1', () => {
    const state = resetPixelPong(-1);
    for (const v of state.pixels) {
      expect(v === 0 || v === 1).toBe(true);
    }
  });
});

describe('stepPixelPong', () => {
  it('ball moves after a step', () => {
    const s0 = resetPixelPong(1);
    const { state: s1, done } = stepPixelPong(s0, 0);
    expect(done).toBe(false);
    // Ball started at (15, 20) with vRow=1, vCol=1 → now at (16, 21)
    expect(s1.pixels[16 * ENV_WIDTH + 21]).toBe(1);
    // Old ball position should be clear (no paddle there)
    expect(s1.pixels[15 * ENV_WIDTH + 20]).toBe(0);
  });

  it('player paddle moves up on action 1', () => {
    const s0 = resetPixelPong(1);
    const { state: s1 } = stepPixelPong(s0, 1);
    // Paddle center should have moved from 15 to 14
    expect(s1._engine.p1Center).toBe(14);
    // Top of paddle at row 12, col 0 should be lit
    expect(s1.pixels[12 * ENV_WIDTH]).toBe(1);
  });

  it('player paddle moves down on action 2', () => {
    const s0 = resetPixelPong(1);
    const { state: s1 } = stepPixelPong(s0, 2);
    expect(s1._engine.p1Center).toBe(16);
    // Bottom of paddle at row 18, col 0
    expect(s1.pixels[18 * ENV_WIDTH]).toBe(1);
  });

  it('cpu paddle responds to ball', () => {
    const s0 = resetPixelPong(1);
    const { state: s1 } = stepPixelPong(s0, 0);
    // Ball started at row 15 with vRow=1 → moving down.
    // CPU paddle at row 15 should move down toward ball.
    expect(s1._engine.p2Center).toBe(16);
  });

  it('episode ends when ball exits left or right', () => {
    // Run many noop steps until done
    let state = resetPixelPong(1);
    let done = false;
    for (let i = 0; i < 1000; i++) {
      ({ state, done } = stepPixelPong(state, 0));
      if (done) {break;}
    }
    expect(done).toBe(true);
  });

  it('pixels stay in sync with engine after multiple steps', () => {
    let state = resetPixelPong(-1);
    for (let i = 0; i < 10; i++) {
      const { state: next, done } = stepPixelPong(state, 0);
      if (done) {break;}
      state = next;
    }
    // Ball pixel should match engine position
    const br = Math.round(state._engine.ballRow);
    const bc = Math.round(state._engine.ballCol);
    expect(state.pixels[br * ENV_WIDTH + bc]).toBe(1);
  });
});
