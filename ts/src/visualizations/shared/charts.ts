// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { Grid, Policy, StateValues } from '../../core/types';
import type { BaseSnapshot, ValueRange } from './types';

const CHART_PAD = { top: 6, right: 20, bottom: 18, left: 60 };

interface DeltaPolicySnapshot {
  delta: number;
  phase: 'evaluation' | 'improvement';
  policy: Policy;
}

export function configureCanvas(
  canvas: HTMLCanvasElement
): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
} | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }

  const dpr = window.devicePixelRatio || 1;
  const displayW = Math.max(1, canvas.clientWidth);
  const displayH = Math.max(1, canvas.clientHeight);
  const targetW = Math.round(displayW * dpr);
  const targetH = Math.round(displayH * dpr);

  if (
    canvas.width !== targetW
    || canvas.height !== targetH
  ) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, displayW, displayH);

  return { ctx, w: displayW, h: displayH };
}

export function chartXForIndex(
  index: number,
  totalSnapshots: number,
  width: number
): number {
  const cw = width - CHART_PAD.left - CHART_PAD.right;
  return CHART_PAD.left
    + (cw * index) / Math.max(1, totalSnapshots - 1);
}

export function chartXToNearestIndex(
  x: number,
  width: number,
  totalSnapshots: number
): number {
  if (totalSnapshots <= 1) {
    return 0;
  }
  const cw = width - CHART_PAD.left - CHART_PAD.right;
  const norm = (x - CHART_PAD.left) / Math.max(1, cw);
  const raw = Math.round(norm * (totalSnapshots - 1));
  return Math.max(0, Math.min(raw, totalSnapshots - 1));
}

export function drawXAxis(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number
): void {
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(CHART_PAD.left, CHART_PAD.top);
  ctx.lineTo(CHART_PAD.left, h - CHART_PAD.bottom);
  ctx.lineTo(w - CHART_PAD.right, h - CHART_PAD.bottom);
  ctx.stroke();

}

export function drawYAxisLabel(
  ctx: CanvasRenderingContext2D,
  h: number,
  text: string
): void {
  // Keep rotated y-axis labels readable even when chart height is reduced.
  let fontSize = 12;
  const minFontSize = 9;
  const maxLabelLength = Math.max(1, h - 8);
  while (fontSize > minFontSize) {
    ctx.font = `${String(fontSize)}px sans-serif`;
    if (ctx.measureText(text).width <= maxLabelLength) {
      break;
    }
    fontSize--;
  }

  ctx.save();
  ctx.font = `${String(fontSize)}px sans-serif`;
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

export function drawCurrentTimestepGuide(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  totalSnapshots: number,
  currentIndex: number
): void {
  const x = chartXForIndex(currentIndex, totalSnapshots, w);
  ctx.strokeStyle = 'rgba(37, 99, 235, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, CHART_PAD.top);
  ctx.lineTo(x, h - CHART_PAD.bottom);
  ctx.stroke();
}

function formatLogTick(value: number): string {
  if (value >= 1) {
    return value.toFixed(0);
  }
  if (value >= 0.1) {
    return value.toFixed(1);
  }
  if (value >= 0.01) {
    return value.toFixed(2);
  }
  return value.toExponential(0);
}

function countPolicyChanges(
  previous: Policy,
  next: Policy
): number {
  let changed = 0;
  for (const [key, action] of previous) {
    if (next.get(key) !== action) {
      changed++;
    }
  }
  return changed;
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

export function computeRoundValueRanges(
  grid: Grid,
  snapshots: BaseSnapshot[]
): ValueRange[] {
  const ranges: ValueRange[] = [];
  let currentRound = 0;

  for (const snap of snapshots) {
    const range = computeValueRange(grid, snap.stateValues);
    if (currentRound >= ranges.length) {
      ranges[currentRound] = range;
    } else {
      ranges[currentRound] = {
        minVal: Math.min(
          ranges[currentRound].minVal,
          range.minVal
        ),
        maxVal: Math.max(
          ranges[currentRound].maxVal,
          range.maxVal
        )
      };
    }

    if (snap.phase === 'improvement') {
      currentRound++;
    }
  }

  return ranges;
}

export function renderChart(
  chartCanvas: HTMLCanvasElement,
  snapshots: DeltaPolicySnapshot[],
  currentIndex: number
): void {
  const configured = configureCanvas(chartCanvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const ch = h - CHART_PAD.top - CHART_PAD.bottom;

  const points = snapshots
    .map((s, i) => ({ index: i, value: s.delta }))
    .filter(p => p.value > 0);
  if (points.length === 0) {
    return;
  }

  const values = points.map(p => p.value);
  const maxDelta = Math.max(...values);
  const minDelta = Math.min(...values);
  const logMax = Math.log10(maxDelta);
  const logMin = Math.log10(minDelta);
  const logRange = Math.max(1e-9, logMax - logMin);

  const valueToY = (value: number): number => {
    const norm = (Math.log10(value) - logMin) / logRange;
    return CHART_PAD.top + ch * (1 - norm);
  };

  drawXAxis(ctx, w, h);

  // Y-axis labels and grid
  ctx.fillStyle = '#555';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const logVal = logMax - (logRange * i) / yTicks;
    const val = 10 ** logVal;
    const y = valueToY(val);
    ctx.fillText(formatLogTick(val), CHART_PAD.left - 10, y);

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CHART_PAD.left, y);
    ctx.lineTo(w - CHART_PAD.right, y);
    ctx.stroke();
  }

  drawYAxisLabel(ctx, h, '\u0394 values');

  const visiblePoints = points.filter(p => p.index <= currentIndex);
  if (visiblePoints.length >= 2) {
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();
    visiblePoints.forEach((point, i) => {
      const x = chartXForIndex(point.index, snapshots.length, w);
      const y = valueToY(point.value);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  drawCurrentTimestepGuide(
    ctx, w, h, snapshots.length, currentIndex
  );
}

export function renderPolicyChangeChart(
  chartCanvas: HTMLCanvasElement,
  snapshots: DeltaPolicySnapshot[],
  currentIndex: number
): void {
  const configured = configureCanvas(chartCanvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const ch = h - CHART_PAD.top - CHART_PAD.bottom;

  const hasImprovementPhase = snapshots.some(
    s => s.phase === 'improvement'
  );
  const points = snapshots
    .map((snap, index) => ({ snap, index }))
    .filter(({ snap, index }) =>
      index > 0
      && (hasImprovementPhase
        ? snap.phase === 'improvement'
        : true)
    )
    .map(({ index }) => ({
      index,
      value: countPolicyChanges(
        snapshots[index - 1].policy, snapshots[index].policy
      )
    }));
  if (points.length === 0) {
    return;
  }

  const maxChange = Math.max(1, ...points.map(p => p.value));
  drawXAxis(ctx, w, h);

  // Y-axis labels and grid
  ctx.fillStyle = '#555';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const val = (maxChange * (yTicks - i)) / yTicks;
    const y = CHART_PAD.top + (ch * i) / yTicks;
    ctx.fillText(String(Math.round(val)), CHART_PAD.left - 10, y);

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CHART_PAD.left, y);
    ctx.lineTo(w - CHART_PAD.right, y);
    ctx.stroke();
  }

  drawYAxisLabel(ctx, h, '# changed actions');

  const visiblePoints = points.filter(p => p.index <= currentIndex);
  if (visiblePoints.length >= 2) {
    ctx.strokeStyle = '#0891b2';
    ctx.lineWidth = 2;
    ctx.beginPath();
    visiblePoints.forEach((point, i) => {
      const x = chartXForIndex(point.index, snapshots.length, w);
      const y = CHART_PAD.top + ch * (1 - point.value / maxChange);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  ctx.fillStyle = '#0891b2';
  visiblePoints.forEach(point => {
    const x = chartXForIndex(point.index, snapshots.length, w);
    const y = CHART_PAD.top + ch * (1 - point.value / maxChange);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  drawCurrentTimestepGuide(
    ctx, w, h, snapshots.length, currentIndex
  );
}

export function renderTimelineChart(
  chartCanvas: HTMLCanvasElement,
  totalSnapshots: number,
  currentIndex: number
): void {
  const configured = configureCanvas(chartCanvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const centerY = Math.round(h / 2);
  const leftX = CHART_PAD.left;
  const rightX = w - CHART_PAD.right;
  const currentX = chartXForIndex(currentIndex, totalSnapshots, w);

  ctx.strokeStyle = '#d1d5db';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(leftX, centerY);
  ctx.lineTo(rightX, centerY);
  ctx.stroke();

  ctx.strokeStyle = '#2563eb';
  ctx.beginPath();
  ctx.moveTo(leftX, centerY);
  ctx.lineTo(currentX, centerY);
  ctx.stroke();

  ctx.fillStyle = '#1d4ed8';
  ctx.beginPath();
  ctx.arc(currentX, centerY, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(currentX, centerY, 6, 0, Math.PI * 2);
  ctx.stroke();

}
