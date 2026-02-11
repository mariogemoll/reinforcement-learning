// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { CANVAS_COLORS } from '../../config/constants';
import type { Grid, Policy, StateValues } from '../../core/types';
import {
  drawGoal,
  drawPolicyArrow,
  drawTrapCross as drawTrap
} from '../../rendering/cell-icons';
import { drawAgentCircle, drawEffects } from '../dp-shared/grid-effects';
import type { AgentPosition, Effect, ValueRange } from '../dp-shared/types';

function redYellowGreen(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  if (c < 0.5) {
    return `rgb(255, ${String(Math.floor(255 * c * 2))}, 0)`;
  }
  const r = Math.floor(255 * (1 - (c - 0.5) * 2));
  return `rgb(${String(r)}, 255, 0)`;
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

export function renderGrid(
  canvas: HTMLCanvasElement,
  grid: Grid,
  cellSize: number,
  stateValues: StateValues,
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

      ctx.strokeStyle = CANVAS_COLORS.gridLine;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);

      if (cellType === 'goal') {
        drawGoal(ctx, x, y, cellSize);
      } else if (cellType === 'trap') {
        drawTrap(ctx, x, y, cellSize);
      }

      if (
        showNumbers
        && v !== undefined
        && cellType === 'floor'
      ) {
        ctx.fillStyle = '#000000';
        const fontSize = Math.max(10, cellSize * 0.22);
        ctx.font = `bold ${String(fontSize)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(
          v.toFixed(1),
          x + cellSize / 2,
          y + cellSize - 2
        );
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
}
