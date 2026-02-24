// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ENV_HEIGHT, ENV_WIDTH, PADDLE_HALF_HEIGHT } from '../../pong/environment';
import type { PongState } from '../../pong/types';

const CELL = 16; // px per grid cell

export const CANVAS_WIDTH  = ENV_WIDTH  * CELL;
export const CANVAS_HEIGHT = ENV_HEIGHT * CELL;

const BG_COLOR      = '#0d0d1a';
const BALL_COLOR    = '#ffffff';
const DIVIDER_COLOR = 'rgba(255,255,255,0.10)';
const SCORE_COLOR   = '#ffffff';

export function renderPong(
  canvas: HTMLCanvasElement,
  state: PongState,
  playerScore: number,
  aiScore: number,
  p1Color: string,
  p2Color: string
): void {
  const ctx = canvas.getContext('2d');
  if (ctx === null) {return;}

  // Background
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Centre divider
  ctx.strokeStyle = DIVIDER_COLOR;
  ctx.lineWidth = 2;
  ctx.setLineDash([CELL, CELL]);
  ctx.beginPath();
  ctx.moveTo(CANVAS_WIDTH / 2, 0);
  ctx.lineTo(CANVAS_WIDTH / 2, CANVAS_HEIGHT);
  ctx.stroke();
  ctx.setLineDash([]);

  // Paddles
  drawPaddle(ctx, 0, Math.round(state.p1Center), p1Color);
  drawPaddle(ctx, ENV_WIDTH - 1, Math.round(state.p2Center), p2Color);

  // Ball
  const ballC = Math.max(0, Math.min(ENV_WIDTH  - 1, Math.floor(state.ballCol)));
  const ballR = Math.max(0, Math.min(ENV_HEIGHT - 1, Math.floor(state.ballRow)));
  ctx.fillStyle = BALL_COLOR;
  ctx.beginPath();
  ctx.arc(
    ballC * CELL + CELL / 2,
    ballR * CELL + CELL / 2,
    CELL * 0.42,
    0,
    Math.PI * 2
  );
  ctx.fill();

  // Score
  ctx.fillStyle = SCORE_COLOR;
  ctx.font = 'bold 20px monospace';
  ctx.textBaseline = 'top';

  ctx.textAlign = 'right';
  ctx.fillText(String(playerScore), CANVAS_WIDTH / 2 - 20, 10);

  ctx.textAlign = 'left';
  ctx.fillText(String(aiScore), CANVAS_WIDTH / 2 + 20, 10);
}

function drawPaddle(
  ctx: CanvasRenderingContext2D,
  col: number,
  centerRow: number,
  color: string
): void {
  ctx.fillStyle = color;
  for (let r = centerRow - PADDLE_HALF_HEIGHT; r <= centerRow + PADDLE_HALF_HEIGHT; r++) {
    if (r >= 0 && r < ENV_HEIGHT) {
      ctx.fillRect(col * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
    }
  }
}
