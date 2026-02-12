// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  chartXForIndex,
  configureCanvas,
  drawCurrentTimestepGuide,
  drawXAxis,
  drawYAxisLabel
} from '../shared/charts';

interface DataPoint {
  index: number;
  value: number;
}

const CHART_PAD = { top: 6, right: 20, bottom: 18, left: 60 };

function computeAxisMax(values: number[]): number {
  const peak = Math.max(0, ...values);
  if (peak <= 1) {
    return 1;
  }
  const padded = peak * 1.1;
  const magnitude = 10 ** Math.floor(Math.log10(padded));
  return Math.ceil(padded / magnitude) * magnitude;
}

export function renderValueRMSEChart(
  canvas: HTMLCanvasElement,
  dataPoints: DataPoint[],
  totalSnapshots: number,
  currentIndex: number
): void {
  const configured = configureCanvas(canvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const ch = h - CHART_PAD.top - CHART_PAD.bottom;

  const visiblePoints = dataPoints.filter(
    p => p.index <= currentIndex
  );
  const maxRMSE = computeAxisMax(
    visiblePoints.map(point => point.value)
  );

  drawXAxis(ctx, w, h);

  // Y-axis labels and grid
  ctx.fillStyle = '#555';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const val = (maxRMSE * (yTicks - i)) / yTicks;
    const y = CHART_PAD.top + (ch * i) / yTicks;
    ctx.fillText(
      String(Math.round(val)),
      CHART_PAD.left - 10,
      y
    );

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CHART_PAD.left, y);
    ctx.lineTo(w - CHART_PAD.right, y);
    ctx.stroke();
  }

  drawYAxisLabel(ctx, h, 'RMSE vs optimal');

  if (visiblePoints.length >= 2) {
    ctx.strokeStyle = '#d946ef';
    ctx.lineWidth = 2;
    ctx.beginPath();
    visiblePoints.forEach((point, i) => {
      const x = chartXForIndex(
        point.index, totalSnapshots, w
      );
      const clamped = Math.min(maxRMSE, point.value);
      const y = CHART_PAD.top + ch * (1 - clamped / maxRMSE);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  drawCurrentTimestepGuide(
    ctx, w, h, totalSnapshots, currentIndex
  );
}

export function renderPolicyAgreementChart(
  canvas: HTMLCanvasElement,
  dataPoints: DataPoint[],
  totalSnapshots: number,
  currentIndex: number
): void {
  const configured = configureCanvas(canvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const ch = h - CHART_PAD.top - CHART_PAD.bottom;

  const visiblePoints = dataPoints.filter(
    p => p.index <= currentIndex
  );

  drawXAxis(ctx, w, h);

  // Y-axis labels and grid (0-100%)
  ctx.fillStyle = '#555';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const val = 100 * (yTicks - i) / yTicks;
    const y = CHART_PAD.top + (ch * i) / yTicks;
    ctx.fillText(`${String(Math.round(val))}%`, CHART_PAD.left - 10, y);

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CHART_PAD.left, y);
    ctx.lineTo(w - CHART_PAD.right, y);
    ctx.stroke();
  }

  drawYAxisLabel(ctx, h, 'Policy agreement');

  if (visiblePoints.length >= 2) {
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    visiblePoints.forEach((point, i) => {
      const x = chartXForIndex(
        point.index, totalSnapshots, w
      );
      const y = CHART_PAD.top
        + ch * (1 - point.value / 100);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  drawCurrentTimestepGuide(
    ctx, w, h, totalSnapshots, currentIndex
  );
}
