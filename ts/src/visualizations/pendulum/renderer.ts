// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { PendulumState } from '../../pendulum/types';

export const CANVAS_SIZE = 400;

const PIVOT_X = CANVAS_SIZE / 2;
const PIVOT_Y = CANVAS_SIZE / 2;
const ROD_LENGTH = 140;
const TIP_RADIUS = 14;
const PIVOT_RADIUS = 6;

const SKY_TOP = '#b3d9ff';
const SKY_BOTTOM = '#e6f0ff';
const ROD_UPRIGHT_COLOR = '#2d6a4f';
const ROD_MID_COLOR = '#c2873a';
const ROD_DOWN_COLOR = '#dc2626';
const TIP_UPRIGHT_COLOR = '#40916c';
const TIP_MID_COLOR = '#d4a843';
const TIP_DOWN_COLOR = '#ef4444';

function lerpColor(a: string, b: string, t: number): string {
  const parse = (hex: string): [number, number, number] => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16)
  ];
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${String(r)}, ${String(g)}, ${String(bl)})`;
}

function normalizeAngle(theta: number): number {
  const twoPi = 2 * Math.PI;
  const wrapped = (theta + Math.PI) % twoPi;
  return (wrapped < 0 ? wrapped + twoPi : wrapped) - Math.PI;
}

function angleColor(theta: number, component: 'rod' | 'tip'): string {
  // |theta|=0 → upright (green), |theta|=pi → hanging down (red)
  const ratio = Math.abs(normalizeAngle(theta)) / Math.PI;
  const upright = component === 'rod' ? ROD_UPRIGHT_COLOR : TIP_UPRIGHT_COLOR;
  const mid = component === 'rod' ? ROD_MID_COLOR : TIP_MID_COLOR;
  const down = component === 'rod' ? ROD_DOWN_COLOR : TIP_DOWN_COLOR;
  if (ratio < 0.5) {
    return lerpColor(upright, mid, ratio * 2);
  }
  return lerpColor(mid, down, (ratio - 0.5) * 2);
}

export function renderPendulum(
  canvas: HTMLCanvasElement,
  state: Readonly<PendulumState>
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) { return; }

  const { theta, thetaDot } = state;

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Background gradient
  const bg = ctx.createRadialGradient(PIVOT_X, PIVOT_Y, 0, PIVOT_X, PIVOT_Y, CANVAS_SIZE * 0.7);
  bg.addColorStop(0, SKY_BOTTOM);
  bg.addColorStop(1, SKY_TOP);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Faint angle guide circles
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y, ROD_LENGTH, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Upright reference line (faint dashed)
  ctx.strokeStyle = 'rgba(0,0,0,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(PIVOT_X, PIVOT_Y - ROD_LENGTH - 20);
  ctx.lineTo(PIVOT_X, PIVOT_Y + ROD_LENGTH + 20);
  ctx.stroke();
  ctx.setLineDash([]);

  // Rod
  const tipX = PIVOT_X + Math.sin(theta) * ROD_LENGTH;
  const tipY = PIVOT_Y - Math.cos(theta) * ROD_LENGTH;

  const rodColor = angleColor(theta, 'rod');
  ctx.strokeStyle = rodColor;
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(PIVOT_X, PIVOT_Y);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();
  ctx.lineCap = 'butt';

  // Tip circle
  const tipColor = angleColor(theta, 'tip');
  ctx.fillStyle = tipColor;
  ctx.beginPath();
  ctx.arc(tipX, tipY, TIP_RADIUS, 0, Math.PI * 2);
  ctx.fill();

  // Speed arc indicator (thin arc showing angular velocity direction)
  const speedRatio = Math.abs(thetaDot) / 8;
  if (speedRatio > 0.05) {
    const arcRadius = ROD_LENGTH + 22;
    const arcSpan = Math.min(speedRatio * Math.PI * 0.6, Math.PI * 0.6);
    // angle=0 in canvas coords is right (3 o'clock); pendulum up = -PI/2
    const baseAngle = theta - Math.PI / 2; // convert from pendulum to canvas angle
    const startAngle = thetaDot > 0 ? baseAngle : baseAngle - arcSpan;
    const endAngle = thetaDot > 0 ? baseAngle + arcSpan : baseAngle;
    ctx.strokeStyle = 'rgba(37, 99, 235, 0.35)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(PIVOT_X, PIVOT_Y, arcRadius, startAngle, endAngle);
    ctx.stroke();
    ctx.lineCap = 'butt';
  }

  // Pivot
  ctx.fillStyle = '#4a5568';
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y, PIVOT_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#2d3748';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y, PIVOT_RADIUS, 0, Math.PI * 2);
  ctx.stroke();
}
