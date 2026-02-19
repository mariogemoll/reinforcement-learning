// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { Effect } from './types';

export function drawAgentCircle(
  ctx: CanvasRenderingContext2D,
  visualRow: number,
  visualCol: number,
  cellSize: number
): void {
  const centerX = visualCol * cellSize + cellSize / 2;
  const centerY = visualRow * cellSize + cellSize / 2;
  const radius = Math.min(cellSize * 0.25, 8);

  ctx.fillStyle = '#8B5CF6';
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#C4B5FD';
  ctx.beginPath();
  ctx.arc(
    centerX - radius * 0.3,
    centerY - radius * 0.3,
    radius * 0.35,
    0,
    Math.PI * 2
  );
  ctx.fill();

  ctx.strokeStyle = '#6D28D9';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.stroke();
}

export function drawEffects(
  ctx: CanvasRenderingContext2D,
  effects: Effect[],
  cellSize: number
): void {
  const now = Date.now();

  for (const effect of effects) {
    const elapsed = now - effect.startTime;
    const progress = Math.min(1, elapsed / effect.duration);
    if (progress >= 1) {
      continue;
    }

    const cx = effect.col * cellSize + cellSize / 2;
    const cy = effect.row * cellSize + cellSize / 2;

    if (effect.type === 'poof') {
      drawPoofEffect(ctx, cx, cy, cellSize, progress);
    } else {
      drawBurnEffect(ctx, cx, cy, cellSize, progress);
    }
  }
}

function drawPoofEffect(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cellSize: number,
  progress: number
): void {
  for (let i = 0; i < 3; i++) {
    const delay = i * 0.15;
    const cp = Math.max(
      0,
      Math.min(1, (progress - delay) / (1 - delay))
    );
    if (cp > 0) {
      const r = cellSize * 0.15 * (1 + cp * 2);
      const a = (1 - cp) * 0.8;
      ctx.fillStyle = `rgba(255, 215, 0, ${String(a)})`;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2;
    const dist = cellSize * 0.3 * progress;
    const sx = cx + Math.cos(angle) * dist;
    const sy = cy + Math.sin(angle) * dist;
    const a = 1 - progress;
    ctx.fillStyle = `rgba(255, 255, 255, ${String(a)})`;
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawBurnEffect(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cellSize: number,
  progress: number
): void {
  for (let i = 0; i < 8; i++) {
    const angle =
      (i / 8) * Math.PI * 2 + progress * Math.PI;
    const xOff = Math.cos(angle) * cellSize * 0.15;
    const yOff = -progress * cellSize * 0.5;
    const px = cx + xOff;
    const py = cy + yOff;
    const a = (1 - progress) * 0.9;

    const grad = ctx.createRadialGradient(
      px, py, 0, px, py, 4
    );
    grad.addColorStop(0, `rgba(220, 38, 38, ${String(a)})`);
    grad.addColorStop(
      0.5,
      `rgba(245, 158, 11, ${String(a)})`
    );
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    const r = 3 * (1 - progress * 0.5);
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const smokeA = (1 - progress) * 0.3;
  ctx.fillStyle = `rgba(64, 64, 64, ${String(smokeA)})`;
  ctx.beginPath();
  const smokeR = cellSize * 0.2 * (1 + progress);
  ctx.arc(
    cx,
    cy - progress * cellSize * 0.2,
    smokeR,
    0,
    Math.PI * 2
  );
  ctx.fill();
}
