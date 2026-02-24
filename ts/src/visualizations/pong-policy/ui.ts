// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { PongPolicyVizDom, QBarEls } from './types';

function makeQBarRow(label: string, color: string): { row: HTMLDivElement; els: QBarEls } {
  const row = document.createElement('div');
  row.className = 'pong-policy-qbar-row';

  const labelEl = document.createElement('span');
  labelEl.className = 'pong-policy-qbar-label';
  labelEl.textContent = label;

  const track = document.createElement('div');
  track.className = 'pong-policy-qbar-track';

  const bar = document.createElement('div');
  bar.className = 'pong-policy-qbar-fill';
  bar.style.background = color;
  bar.style.width = '0%';

  const value = document.createElement('span');
  value.className = 'pong-policy-qbar-value';
  value.textContent = '0.00';

  track.appendChild(bar);
  row.append(labelEl, track, value);
  return { row, els: { bar, value } };
}

export function createPongPolicyVizDom(): PongPolicyVizDom {
  const container = document.createElement('div');
  container.className = 'visualization pong-policy-viz-container';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'pong-viz-canvas-wrap';

  const canvas = document.createElement('canvas');
  canvas.className = 'pong-viz-canvas';

  const overlay = document.createElement('div');
  overlay.className = 'pong-viz-overlay';
  overlay.hidden = true;

  const overlayPanel = document.createElement('div');
  overlayPanel.className = 'pong-viz-overlay-panel';

  const overlayTitle = document.createElement('div');
  overlayTitle.className = 'pong-viz-overlay-title';

  const playAgainBtn = document.createElement('button');
  playAgainBtn.type = 'button';
  playAgainBtn.textContent = 'Play again';

  overlayPanel.append(overlayTitle, playAgainBtn);
  overlay.appendChild(overlayPanel);
  canvasWrap.append(canvas, overlay);

  // Q-value panel
  const qPanel = document.createElement('div');
  qPanel.className = 'pong-policy-qpanel';

  const qTitle = document.createElement('div');
  qTitle.className = 'pong-policy-qpanel-title';
  qTitle.textContent = 'Q-values (NN)';

  const { row: noopRow, els: noopEls } = makeQBarRow('Noop', '#64748b');
  const { row: upRow,   els: upEls   } = makeQBarRow('Up',   '#3b9eff');
  const { row: downRow, els: downEls } = makeQBarRow('Down', '#a78bfa');

  qPanel.append(qTitle, noopRow, upRow, downRow);

  // Footer
  const footer = document.createElement('div');
  footer.className = 'pong-viz-footer';

  const nnLabel = document.createElement('span');
  nnLabel.className = 'pong-viz-player-label';
  nnLabel.style.color = '#3b9eff';
  nnLabel.textContent = 'NN';

  const hint = document.createElement('span');
  hint.className = 'pong-viz-hint';
  hint.textContent = 'NN vs CPU';

  const cpuLabel = document.createElement('span');
  cpuLabel.className = 'pong-viz-cpu-label';
  cpuLabel.textContent = 'CPU';

  footer.append(nnLabel, hint, cpuLabel);
  container.append(canvasWrap, qPanel, footer);

  return {
    container,
    canvas,
    overlay,
    overlayTitle,
    playAgainBtn,
    qBars: { noop: noopEls, up: upEls, down: downEls }
  };
}
