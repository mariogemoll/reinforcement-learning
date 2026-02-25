// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  BREAKOUT_CHANNELS,
  BREAKOUT_GRID_SIZE
} from '../../minatar/breakout';

const CELL_SIZE = 32;
export const CANVAS_WIDTH = BREAKOUT_GRID_SIZE * CELL_SIZE;
export const CANVAS_HEIGHT = BREAKOUT_GRID_SIZE * CELL_SIZE;

// ch0 paddle, ch1 ball, ch3 brick. (Trail channel intentionally hidden.)
const CHANNEL_COLORS = [
  '#e74c3c',
  '#ffffff',
  '',
  '#2ecc71'
];

export function renderMinAtarBreakout(
  canvas: HTMLCanvasElement,
  observation: Float32Array
): void {
  const ctx = canvas.getContext('2d');
  if (ctx === null) {
    return;
  }

  const cellW = canvas.width / BREAKOUT_GRID_SIZE;
  const cellH = canvas.height / BREAKOUT_GRID_SIZE;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let row = 0; row < BREAKOUT_GRID_SIZE; row++) {
    for (let col = 0; col < BREAKOUT_GRID_SIZE; col++) {
      let activeChannel = -1;
      const base = (row * BREAKOUT_GRID_SIZE + col) * BREAKOUT_CHANNELS;
      for (let channel = 0; channel < BREAKOUT_CHANNELS; channel++) {
        if (channel === 2) {
          continue;
        }
        if (observation[base + channel] > 0) {
          activeChannel = channel;
        }
      }

      if (activeChannel >= 0) {
        ctx.fillStyle = CHANNEL_COLORS[activeChannel];
        ctx.fillRect(
          Math.floor(col * cellW),
          Math.floor(row * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH)
        );
      }
    }
  }

  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 1;
  for (let i = 0; i <= BREAKOUT_GRID_SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cellW, 0);
    ctx.lineTo(i * cellW, canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * cellH);
    ctx.lineTo(canvas.width, i * cellH);
    ctx.stroke();
  }
}
