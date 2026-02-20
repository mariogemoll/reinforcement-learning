// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ANGLE_LIMIT, POSITION_LIMIT } from '../../cartpole/environment';
import type { CartPoleState } from '../../cartpole/types';

const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 300;

const GROUND_Y = CANVAS_HEIGHT * 0.78;
const SCALE = 100; // 1 world unit = 100px
const CART_WIDTH = 60;
const CART_HEIGHT = 30;
const POLE_LENGTH = 100;
const WHEEL_RADIUS = 6;

const SKY_TOP = '#b3d9ff';
const SKY_BOTTOM = '#e6f0ff';
const GROUND_COLOR = '#8B7355';
const TRACK_COLOR = '#555';
const CART_COLOR = '#2d3748';
const CART_HIGHLIGHT = '#4a5568';
const POLE_COLOR = '#c2873a';
const POLE_DANGER_COLOR = '#dc2626';
const LIMIT_ZONE_COLOR = 'rgba(220, 38, 38, 0.08)';
const LIMIT_LINE_COLOR = 'rgba(220, 38, 38, 0.3)';
const WHEEL_COLOR = '#1a202c';
const TICK_COLOR = '#777';

export function renderCartPole(
  canvas: HTMLCanvasElement,
  state: Readonly<CartPoleState>,
  limits: {
    angleLimit: number;
    positionLimit: number;
  } = {
    angleLimit: ANGLE_LIMIT,
    positionLimit: POSITION_LIMIT
  }
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  const [x, , theta] = state;
  const centerX = CANVAS_WIDTH / 2;

  ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Sky gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, GROUND_Y);
  skyGrad.addColorStop(0, SKY_TOP);
  skyGrad.addColorStop(1, SKY_BOTTOM);
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, CANVAS_WIDTH, GROUND_Y);

  // Ground
  ctx.fillStyle = GROUND_COLOR;
  ctx.fillRect(0, GROUND_Y, CANVAS_WIDTH, CANVAS_HEIGHT - GROUND_Y);

  // Angle limit zones (faint red at edges)
  const leftLimitX = centerX - limits.positionLimit * SCALE;
  const rightLimitX = centerX + limits.positionLimit * SCALE;

  ctx.fillStyle = LIMIT_ZONE_COLOR;
  ctx.fillRect(0, 0, leftLimitX, GROUND_Y);
  ctx.fillRect(rightLimitX, 0, CANVAS_WIDTH - rightLimitX, GROUND_Y);

  ctx.strokeStyle = LIMIT_LINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(leftLimitX, 0);
  ctx.lineTo(leftLimitX, GROUND_Y);
  ctx.moveTo(rightLimitX, 0);
  ctx.lineTo(rightLimitX, GROUND_Y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Track line
  ctx.strokeStyle = TRACK_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(leftLimitX - 20, GROUND_Y);
  ctx.lineTo(rightLimitX + 20, GROUND_Y);
  ctx.stroke();

  // Track tick marks
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  for (let pos = -limits.positionLimit; pos <= limits.positionLimit; pos += 0.5) {
    const tickX = centerX + pos * SCALE;
    const isMain = Math.abs(pos % 1) < 0.01;
    const tickHeight = isMain ? 6 : 3;
    ctx.strokeStyle = TICK_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tickX, GROUND_Y);
    ctx.lineTo(tickX, GROUND_Y + tickHeight);
    ctx.stroke();
    if (isMain) {
      ctx.fillText(pos.toFixed(0), tickX, GROUND_Y + tickHeight + 10);
    }
  }

  // Cart position in pixels
  const cartX = centerX + x * SCALE;
  const cartTop = GROUND_Y - CART_HEIGHT;

  // Cart body
  ctx.fillStyle = CART_COLOR;
  ctx.fillRect(cartX - CART_WIDTH / 2, cartTop, CART_WIDTH, CART_HEIGHT);

  // Cart highlight stripe
  ctx.fillStyle = CART_HIGHLIGHT;
  ctx.fillRect(cartX - CART_WIDTH / 2, cartTop, CART_WIDTH, 4);

  // Wheels
  ctx.fillStyle = WHEEL_COLOR;
  ctx.beginPath();
  ctx.arc(cartX - CART_WIDTH / 4, GROUND_Y, WHEEL_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cartX + CART_WIDTH / 4, GROUND_Y, WHEEL_RADIUS, 0, Math.PI * 2);
  ctx.fill();

  // Pole
  const poleBaseX = cartX;
  const poleBaseY = cartTop;
  const poleTipX = poleBaseX + Math.sin(theta) * POLE_LENGTH;
  const poleTipY = poleBaseY - Math.cos(theta) * POLE_LENGTH;

  const angleRatio = Math.abs(theta) / limits.angleLimit;
  const poleColor = angleRatio > 0.7
    ? lerpColor(POLE_COLOR, POLE_DANGER_COLOR, (angleRatio - 0.7) / 0.3)
    : POLE_COLOR;

  ctx.strokeStyle = poleColor;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(poleBaseX, poleBaseY);
  ctx.lineTo(poleTipX, poleTipY);
  ctx.stroke();

  // Pole tip circle
  ctx.fillStyle = poleColor;
  ctx.beginPath();
  ctx.arc(poleTipX, poleTipY, 5, 0, Math.PI * 2);
  ctx.fill();

  // Pivot point
  ctx.fillStyle = '#718096';
  ctx.beginPath();
  ctx.arc(poleBaseX, poleBaseY, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.lineCap = 'butt';
}

function lerpColor(a: string, b: string, t: number): string {
  const ar = parseInt(a.slice(1, 3), 16);
  const ag = parseInt(a.slice(3, 5), 16);
  const ab = parseInt(a.slice(5, 7), 16);
  const br = parseInt(b.slice(1, 3), 16);
  const bg = parseInt(b.slice(3, 5), 16);
  const bb = parseInt(b.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${String(r)}, ${String(g)}, ${String(bl)})`;
}

export { CANVAS_HEIGHT, CANVAS_WIDTH };
