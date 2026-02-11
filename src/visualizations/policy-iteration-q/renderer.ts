// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { CANVAS_COLORS } from '../../config/constants';
import type {
  Action,
  ActionValues,
  Grid,
  Policy,
  StateValues
} from '../../core/types';
import {
  directionOffsets,
  drawGoal,
  drawPolicyArrow,
  drawTrapCross as drawTrap
} from '../../rendering/cell-icons';
import { drawAgentCircle, drawEffects } from '../dp-shared/grid-effects';
import type { AgentPosition, Effect, ValueRange } from '../dp-shared/types';

interface PolicyTipOverlay {
  x: number;
  y: number;
  cellSize: number;
  action: Action;
  strokeColor: string;
  outlineColor: string;
}

interface ActionValueLabelOverlay {
  x: number;
  y: number;
  cellSize: number;
  actionMap: Map<Action, number> | undefined;
}

function lerpFloat(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function hslToRgbString(
  hue: number,
  saturation: number,
  lightness: number
): string {
  const h = ((hue % 360) + 360) % 360;
  const s = Math.max(0, Math.min(100, saturation)) / 100;
  const l = Math.max(0, Math.min(100, lightness)) / 100;

  const c = (1 - Math.abs((2 * l) - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;

  if (hp >= 0 && hp < 1) {
    r1 = c;
    g1 = x;
  } else if (hp >= 1 && hp < 2) {
    r1 = x;
    g1 = c;
  } else if (hp >= 2 && hp < 3) {
    g1 = c;
    b1 = x;
  } else if (hp >= 3 && hp < 4) {
    g1 = x;
    b1 = c;
  } else if (hp >= 4 && hp < 5) {
    r1 = x;
    b1 = c;
  } else {
    r1 = c;
    b1 = x;
  }

  const m = l - (c / 2);
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return `rgb(${String(r)}, ${String(g)}, ${String(b)})`;
}

function redYellowGreenHue(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  const lowHue = 0;
  const midHue = 45;
  const highHue = 142;

  if (c < 0.5) {
    return lerpFloat(lowHue, midHue, c * 2);
  }
  return lerpFloat(midHue, highHue, (c - 0.5) * 2);
}

function redYellowGreen(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const hue = redYellowGreenHue(c);
  if (c < 0.5) {
    const lightness = lerpFloat(42, 58, c * 2);
    return hslToRgbString(hue, 82, lightness);
  }
  const lightness = lerpFloat(58, 54, (c - 0.5) * 2);
  return hslToRgbString(hue, 78, lightness);
}

function redYellowGreenForeground(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const hue = redYellowGreenHue(c);
  if (c < 0.5) {
    const lightness = lerpFloat(49, 64, c * 2);
    return hslToRgbString(hue, 98, lightness);
  }
  const lightness = lerpFloat(64, 60, (c - 0.5) * 2);
  return hslToRgbString(hue, 94, lightness);
}

function redYellowGreenPolicyFill(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const hue = redYellowGreenHue(c);
  if (c < 0.5) {
    const lightness = lerpFloat(39, 50, c * 2);
    return hslToRgbString(hue, 100, lightness);
  }
  const lightness = lerpFloat(50, 46, (c - 0.5) * 2);
  return hslToRgbString(hue, 100, lightness);
}

function valueColor(
  value: number,
  minVal: number,
  maxVal: number
): string {
  const range = maxVal - minVal;
  const norm = range === 0
    ? 0.5
    : (value - minVal) / range;
  return redYellowGreen(Math.pow(norm, 0.4));
}

function getCellColor(
  cellType: Grid[number][number]
): string {
  switch (cellType) {
  case 'floor': return CANVAS_COLORS.cells.floor;
  case 'wall': return CANVAS_COLORS.cells.wall;
  case 'goal': return CANVAS_COLORS.cells.goal;
  case 'trap': return CANVAS_COLORS.cells.trap;
  default: return CANVAS_COLORS.cells.fallback;
  }
}

function computeValueRange(
  grid: Grid,
  stateValues: StateValues
): ValueRange {
  let minVal = Infinity;
  let maxVal = -Infinity;

  grid.forEach((row, ri) => {
    row.forEach((cellType, ci) => {
      if (
        cellType !== 'goal'
        && cellType !== 'trap'
        && cellType !== 'wall'
      ) {
        const key = `${String(ri)},${String(ci)}`;
        const v = stateValues.get(key);
        if (v !== undefined) {
          minVal = Math.min(minVal, v);
          maxVal = Math.max(maxVal, v);
        }
      }
    });
  });

  if (!isFinite(minVal)) {
    return { minVal: 0, maxVal: 0 };
  }
  return { minVal, maxVal };
}

function computeActionValueRange(
  actionValues: ActionValues
): ValueRange {
  let minVal = Infinity;
  let maxVal = -Infinity;

  for (const actionMap of actionValues.values()) {
    for (const value of actionMap.values()) {
      minVal = Math.min(minVal, value);
      maxVal = Math.max(maxVal, value);
    }
  }

  if (!isFinite(minVal) || !isFinite(maxVal)) {
    return { minVal: 0, maxVal: 0 };
  }

  return { minVal, maxVal };
}

export function renderGrid(
  canvas: HTMLCanvasElement,
  grid: Grid,
  cellSize: number,
  stateValues: StateValues,
  actionValues: ActionValues,
  policy: Policy,
  agents: AgentPosition[],
  effects: Effect[],
  showValueBg: boolean,
  showArrows: boolean,
  showNumbers: boolean,
  valueRange: ValueRange | null
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const { minVal, maxVal } = showValueBg
    ? (valueRange ?? computeValueRange(grid, stateValues))
    : { minVal: 0, maxVal: 0 };
  const actionRange = computeActionValueRange(actionValues);
  const policyOverlays: PolicyTipOverlay[] = [];
  const actionValueLabelOverlays: ActionValueLabelOverlay[] = [];

  grid.forEach((row, ri) => {
    row.forEach((cellType, ci) => {
      const x = ci * cellSize;
      const y = ri * cellSize;
      const key = `${String(ri)},${String(ci)}`;
      const v = stateValues.get(key);

      if (
        showValueBg
        && v !== undefined
        && cellType === 'floor'
      ) {
        ctx.fillStyle = valueColor(v, minVal, maxVal);
      } else {
        ctx.fillStyle = getCellColor(cellType);
      }
      ctx.fillRect(x, y, cellSize, cellSize);

      if (cellType === 'floor') {
        const overlay = drawActionValueTriangles(
          ctx,
          x,
          y,
          cellSize,
          actionValues.get(key),
          policy.get(key),
          actionRange
        );
        if (overlay !== null) {
          policyOverlays.push(overlay);
        }
      }

      ctx.strokeStyle = CANVAS_COLORS.gridLine;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);

      if (cellType === 'goal') {
        drawGoal(ctx, x, y, cellSize);
      } else if (cellType === 'trap') {
        drawTrap(ctx, x, y, cellSize);
      }

      if (showNumbers && cellType === 'floor') {
        actionValueLabelOverlays.push({
          x,
          y,
          cellSize,
          actionMap: actionValues.get(key)
        });
      }

      if (showArrows && cellType === 'floor') {
        const action = policy.get(key);
        if (action !== undefined) {
          drawPolicyArrow(ctx, x, y, cellSize, action);
        }
      }
    });
  });

  for (const pos of agents) {
    drawAgentCircle(ctx, pos.row, pos.col, cellSize);
  }

  drawEffects(ctx, effects, cellSize);

  // Render policy tips last so they are never cut by grid/borders/overlays.
  for (const overlay of policyOverlays) {
    drawDirectionalTriangle(
      ctx,
      overlay.x,
      overlay.y,
      overlay.cellSize,
      overlay.action,
      overlay.strokeColor,
      overlay.outlineColor,
      true
    );
  }

  for (const overlay of actionValueLabelOverlays) {
    drawActionValueLabels(
      ctx,
      overlay.x,
      overlay.y,
      overlay.cellSize,
      overlay.actionMap
    );
  }
}

function drawActionValueTriangles(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  actionMap: Map<Action, number> | undefined,
  policyAction: Action | undefined,
  actionRange: ValueRange
): PolicyTipOverlay | null {
  const entries: [Action, number][] = [
    ['up', actionMap?.get('up') ?? 0],
    ['down', actionMap?.get('down') ?? 0],
    ['left', actionMap?.get('left') ?? 0],
    ['right', actionMap?.get('right') ?? 0]
  ];
  const range = Math.max(1e-9, actionRange.maxVal - actionRange.minVal);
  const normalizedEntries = entries.map(([action, value]) => ({
    action,
    value,
    normalized: (value - actionRange.minVal) / range
  }));

  drawCellFlood(
    ctx,
    x,
    y,
    cellSize,
    normalizedEntries.map(item => ({
      action: item.action,
      color: redYellowGreen(item.normalized)
    }))
  );

  let policyOverlay: PolicyTipOverlay | null = null;
  for (const { action, normalized } of normalizedEntries) {
    const fillColor = redYellowGreen(normalized);
    const tipColor = redYellowGreenForeground(normalized);
    const outlineColor = 'rgb(31, 31, 31)';
    const policyOutlineColor = 'rgba(31, 31, 31, 0.6)';
    const policyTipColor = redYellowGreenPolicyFill(normalized);
    const fillAlpha = 0.46 + normalized * 0.32;
    drawInvertedTriangle(
      ctx,
      x,
      y,
      cellSize,
      action,
      fillColor,
      fillAlpha
    );

    const isPolicyAction = policyAction === action;
    if (isPolicyAction) {
      policyOverlay = {
        x,
        y,
        cellSize,
        action,
        strokeColor: policyTipColor,
        outlineColor: policyOutlineColor
      };
    } else {
      drawDirectionalTriangle(
        ctx,
        x,
        y,
        cellSize,
        action,
        tipColor,
        outlineColor,
        false
      );
    }
  }

  return policyOverlay;
}

function drawInvertedTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  action: Action,
  fillColor: string,
  fillAlpha: number
): void {
  const cx = x + cellSize / 2;
  const cy = y + cellSize / 2;

  ctx.save();
  ctx.beginPath();
  if (action === 'up') {
    ctx.moveTo(x, y);
    ctx.lineTo(x + cellSize, y);
  } else if (action === 'down') {
    ctx.moveTo(x + cellSize, y + cellSize);
    ctx.lineTo(x, y + cellSize);
  } else if (action === 'left') {
    ctx.moveTo(x, y + cellSize);
    ctx.lineTo(x, y);
  } else {
    ctx.moveTo(x + cellSize, y);
    ctx.lineTo(x + cellSize, y + cellSize);
  }
  ctx.lineTo(cx, cy);
  ctx.closePath();
  ctx.globalAlpha = fillAlpha;
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.restore();
}

function drawCellFlood(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  actionColors: { action: Action; color: string }[]
): void {
  const avgNormalized = actionColors.reduce((acc, item) => {
    const match = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(item.color);
    if (match === null) {
      return acc;
    }
    const r = Number(match[1]);
    const g = Number(match[2]);
    return acc + ((g - r + 255) / 510);
  }, 0) / Math.max(1, actionColors.length);

  const baseColor = redYellowGreen(avgNormalized);
  ctx.save();
  ctx.globalAlpha = 0.84;
  ctx.fillStyle = baseColor;
  ctx.fillRect(x, y, cellSize, cellSize);
  ctx.restore();

  // Keep base fill flat; directional structure comes from hard triangles.
}

function drawActionValueLabels(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  actionMap: Map<Action, number> | undefined
): void {
  const labels: [Action, number][] = [
    ['up', actionMap?.get('up') ?? 0],
    ['down', actionMap?.get('down') ?? 0],
    ['left', actionMap?.get('left') ?? 0],
    ['right', actionMap?.get('right') ?? 0]
  ];
  const fontSize = Math.max(8, cellSize * 0.15);
  const radialOffset = cellSize * 0.35;

  ctx.save();
  ctx.font = `600 ${String(fontSize)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';

  for (const [action, value] of labels) {
    const { dx, dy } = directionOffsets[action];
    const centerX = x + cellSize / 2;
    const centerY = y + cellSize / 2;
    const textX = centerX + dx * radialOffset;
    const textY = centerY + dy * radialOffset;
    const text = value.toFixed(1);
    ctx.fillText(text, textX, textY);
  }
  ctx.restore();
}

function drawDirectionalTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellSize: number,
  action: Action,
  strokeColor: string,
  outlineColor: string,
  isPolicyAction: boolean
): void {
  const centerX = x + cellSize / 2;
  const centerY = y + cellSize / 2;
  const { dx, dy } = directionOffsets[action];

  const radialOffset = (cellSize * 0.35)
    - (isPolicyAction ? 1.5 : 0);
  const tipLength = cellSize * 0.11;
  const halfWidth = cellSize
    * (isPolicyAction ? 0.165 : 0.135);
  const apexX = centerX + dx * radialOffset;
  const apexY = centerY + dy * radialOffset;
  const baseCenterX = apexX - dx * tipLength;
  const baseCenterY = apexY - dy * tipLength;
  const px = -dy;
  const py = dx;

  const leftX = baseCenterX + px * halfWidth;
  const leftY = baseCenterY + py * halfWidth;
  const rightX = baseCenterX - px * halfWidth;
  const rightY = baseCenterY - py * halfWidth;

  if (isPolicyAction) {
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 11.8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(leftX, leftY);
    ctx.lineTo(apexX, apexY);
    ctx.lineTo(rightX, rightY);
    ctx.stroke();
  }

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = isPolicyAction ? 7.2 : 4.2;
  ctx.lineCap = isPolicyAction ? 'round' : 'butt';
  ctx.lineJoin = isPolicyAction ? 'round' : 'miter';
  ctx.miterLimit = 3;
  ctx.beginPath();
  ctx.moveTo(leftX, leftY);
  ctx.lineTo(apexX, apexY);
  ctx.lineTo(rightX, rightY);
  ctx.stroke();
}
