// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { PixelPongVizDom } from './types';

export function createPixelPongVizDom(): PixelPongVizDom {
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

  // Footer
  const footer = document.createElement('div');
  footer.className = 'pong-viz-footer';

  const playerLabel = document.createElement('span');
  playerLabel.className = 'pong-viz-player-label';
  playerLabel.textContent = 'YOU';

  const hint = document.createElement('span');
  hint.className = 'pong-viz-hint';
  hint.textContent =
    'Click to focus\u2003\u2003W\u202f/\u202f\u2191\u2002up\u2003\u2003' +
    'S\u202f/\u202f\u2193\u2002down';

  const cpuLabel = document.createElement('span');
  cpuLabel.className = 'pong-viz-cpu-label';
  cpuLabel.textContent = 'CPU';

  footer.append(playerLabel, hint, cpuLabel);
  container.append(canvasWrap, footer);

  return { container, canvas, overlay, overlayTitle, newGameBtn };
}
