// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { CANVAS_COLORS } from '../config/constants';
import type { Action } from '../core/types';

export const directionOffsets: Record<
  Action,
  { dx: number; dy: number }
> = {
  up: { dx: 0, dy: -1 },
  down: { dx: 0, dy: 1 },
  left: { dx: -1, dy: 0 },
  right: { dx: 1, dy: 0 }
};

export function drawGoal(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number
): void {
  const centerX = x + cellSize / 2;
  const centerY = y + cellSize / 2;
  const size = Math.min(cellSize * 0.4, 12);

  ctx.fillStyle = CANVAS_COLORS.icons.goal;
  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const angle = (i * 4 * Math.PI) / 5 - Math.PI / 2;
    const px = centerX + Math.cos(angle) * size;
    const py = centerY + Math.sin(angle) * size;
    if (i === 0) {ctx.moveTo(px, py);}
    else {ctx.lineTo(px, py);}
  }
  ctx.closePath();
  ctx.fill();
}

export function drawTrapCross(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number
): void {
  const margin = cellSize * 0.2;
  ctx.strokeStyle = CANVAS_COLORS.icons.trap;
  ctx.lineWidth = Math.max(2, cellSize * 0.1);
  ctx.beginPath();
  ctx.moveTo(x + margin, y + margin);
  ctx.lineTo(x + cellSize - margin, y + cellSize - margin);
  ctx.moveTo(x + cellSize - margin, y + margin);
  ctx.lineTo(x + margin, y + cellSize - margin);
  ctx.stroke();
}

export function drawPolicyArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  action: Action
): void {
  const centerX = x + cellSize / 2;
  const centerY = y + cellSize / 2;
  const s = cellSize * 0.18;
  const { dx, dy } = directionOffsets[action];

  ctx.fillStyle = 'rgba(31, 41, 55, 0.5)';
  ctx.beginPath();
  ctx.moveTo(centerX + dx * s, centerY + dy * s);
  ctx.lineTo(
    centerX - dx * s + dy * s,
    centerY - dy * s - dx * s
  );
  ctx.lineTo(
    centerX - dx * s - dy * s,
    centerY - dy * s + dx * s
  );
  ctx.closePath();
  ctx.fill();
}
