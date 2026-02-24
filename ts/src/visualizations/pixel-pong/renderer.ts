// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ENV_HEIGHT, ENV_WIDTH } from '../../pong/environment';

const CELL = 16;

export const CANVAS_WIDTH  = ENV_WIDTH  * CELL;
export const CANVAS_HEIGHT = ENV_HEIGHT * CELL;

/**
 * Draw a pixel-observation Float32Array (ENV_HEIGHT × ENV_WIDTH) onto a canvas.
 * White on black, scaled to fill the canvas dimensions.  Includes score overlay.
 */
export function renderPixelPongCanvas(
  canvas: HTMLCanvasElement,
  pixels: Float32Array,
  playerScore: number,
  aiScore: number
): void {
  const ctx = canvas.getContext('2d');
  if (ctx === null) {return;}

  const cellW = canvas.width / ENV_WIDTH;
  const cellH = canvas.height / ENV_HEIGHT;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = '#fff';
  for (let r = 0; r < ENV_HEIGHT; r++) {
    for (let c = 0; c < ENV_WIDTH; c++) {
      if (pixels[r * ENV_WIDTH + c] > 0) {
        ctx.fillRect(
          Math.floor(c * cellW),
          Math.floor(r * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH)
        );
      }
    }
  }

  // Score
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 20px monospace';
  ctx.textBaseline = 'top';

  ctx.textAlign = 'right';
  ctx.fillText(String(playerScore), canvas.width / 2 - 20, 10);

  ctx.textAlign = 'left';
  ctx.fillText(String(aiScore), canvas.width / 2 + 20, 10);
}
