import { CANVAS_COLORS } from '../config/constants';
import type { Action } from '../core/types';

export function drawAgent(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number
): void {
  const centerX = x + cellSize / 2;
  const centerY = y + cellSize / 2;
  const radius = Math.min(cellSize * 0.3, 10);

  ctx.fillStyle = CANVAS_COLORS.agent.body;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = CANVAS_COLORS.agent.highlight;
  ctx.beginPath();
  ctx.arc(centerX - radius * 0.3, centerY - radius * 0.3, radius * 0.4, 0, Math.PI * 2);
  ctx.fill();
}

export function drawIntendedArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cellSize: number,
  direction: Action,
  slipped: boolean
): void {
  const len = cellSize * 0.35;
  const color = slipped ? CANVAS_COLORS.arrow.slipped : CANVAS_COLORS.arrow.success;
  let dx = 0;
  let dy = 0;
  switch (direction) {
  case 'up': dy = -len; break;
  case 'down': dy = len; break;
  case 'left': dx = -len; break;
  case 'right': dx = len; break;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + dx, cy + dy);
  ctx.stroke();

  // Arrowhead
  const headLen = cellSize * 0.12;
  const angle = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(cx + dx, cy + dy);
  ctx.lineTo(cx + dx - headLen * Math.cos(angle - 0.5), cy + dy - headLen * Math.sin(angle - 0.5));
  ctx.moveTo(cx + dx, cy + dy);
  ctx.lineTo(cx + dx - headLen * Math.cos(angle + 0.5), cy + dy - headLen * Math.sin(angle + 0.5));
  ctx.stroke();
}
