// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { MinAtarBreakoutVizDom } from './types';

export function createMinAtarBreakoutVizDom(): MinAtarBreakoutVizDom {
  const container = document.createElement('div');
  container.className = 'visualization pong-viz-container';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'pong-viz-canvas-wrap';

  const canvas = document.createElement('canvas');
  canvas.className = 'pong-viz-canvas';
  canvas.tabIndex = 0;

  const overlay = document.createElement('div');
  overlay.className = 'pong-viz-overlay';
  overlay.hidden = true;

  const overlayPanel = document.createElement('div');
  overlayPanel.className = 'pong-viz-overlay-panel';

  const overlayTitle = document.createElement('div');
  overlayTitle.className = 'pong-viz-overlay-title';

  const newGameBtn = document.createElement('button');
  newGameBtn.type = 'button';
  newGameBtn.textContent = 'New game';

  overlayPanel.append(overlayTitle, newGameBtn);
  overlay.appendChild(overlayPanel);
  canvasWrap.append(canvas, overlay);

  const footer = document.createElement('div');
  footer.className = 'pong-viz-footer';

  const scoreLabel = document.createElement('span');
  scoreLabel.className = 'pong-viz-player-label';
  scoreLabel.textContent = 'SCORE ';

  const scoreValue = document.createElement('span');
  scoreValue.textContent = '0';
  scoreLabel.append(scoreValue);

  const hint = document.createElement('span');
  hint.className = 'pong-viz-hint';
  hint.textContent =
    'Click to focus   A / \u2190 left   D / \u2192 right';

  const stepLabel = document.createElement('span');
  stepLabel.className = 'pong-viz-cpu-label';
  stepLabel.textContent = 'STEP ';

  const stepValue = document.createElement('span');
  stepValue.textContent = '0';
  stepLabel.append(stepValue);

  footer.append(scoreLabel, hint, stepLabel);
  container.append(canvasWrap, footer);

  return {
    container,
    canvas,
    overlay,
    overlayTitle,
    newGameBtn,
    scoreValue,
    stepValue
  };
}
