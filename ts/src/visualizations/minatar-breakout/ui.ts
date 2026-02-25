// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { MinAtarBreakoutVizDom } from './types';

interface QBarEls {
  bar: HTMLDivElement;
  value: HTMLSpanElement;
}

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

export function createMinAtarBreakoutVizDom(): MinAtarBreakoutVizDom {
  const container = document.createElement('div');
  container.className = 'visualization minatar-breakout-viz-container';

  const layout = document.createElement('div');
  layout.className = 'minatar-breakout-layout';

  const controls = document.createElement('div');
  controls.className = 'minatar-breakout-controls';

  const modeSection = document.createElement('div');
  modeSection.className = 'section';

  const modeTitle = document.createElement('div');
  modeTitle.className = 'label';
  modeTitle.textContent = 'Mode';

  const modeOptions = document.createElement('div');
  modeOptions.className = 'pi-viz-radio-options';

  const radioGroupName = 'minatar-breakout-mode';

  const userModeLabel = document.createElement('label');
  userModeLabel.className = 'pi-viz-radio-option';
  const userModeRadio = document.createElement('input');
  userModeRadio.type = 'radio';
  userModeRadio.name = radioGroupName;
  userModeRadio.value = 'user';
  userModeRadio.checked = false;
  const userModeText = document.createElement('span');
  userModeText.textContent = 'User';
  userModeLabel.append(userModeRadio, userModeText);

  const policyModeLabel = document.createElement('label');
  policyModeLabel.className = 'pi-viz-radio-option';
  const policyModeRadio = document.createElement('input');
  policyModeRadio.type = 'radio';
  policyModeRadio.name = radioGroupName;
  policyModeRadio.value = 'policy';
  policyModeRadio.checked = true;
  const policyModeText = document.createElement('span');
  policyModeText.textContent = 'Policy demo';
  policyModeLabel.append(policyModeRadio, policyModeText);

  modeOptions.append(userModeLabel, policyModeLabel);
  modeSection.append(modeTitle, modeOptions);

  const restartSection = document.createElement('div');
  restartSection.className = 'section';
  const restartBtn = document.createElement('button');
  restartBtn.className = 'control-reset';
  restartBtn.type = 'button';
  restartBtn.textContent = 'Start';
  restartSection.appendChild(restartBtn);

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

  const overlayRestartBtn = document.createElement('button');
  overlayRestartBtn.type = 'button';
  overlayRestartBtn.textContent = 'Start';

  overlayPanel.append(overlayTitle, overlayRestartBtn);
  overlay.appendChild(overlayPanel);
  canvasWrap.append(canvas, overlay);

  const qPanel = document.createElement('div');
  qPanel.className = 'pong-policy-qpanel';
  qPanel.hidden = true;

  const qTitle = document.createElement('div');
  qTitle.className = 'pong-policy-qpanel-title';
  qTitle.textContent = 'Q-values (NN)';

  const barColor = '#9ca3af';
  const { row: noopRow, els: noopEls } = makeQBarRow('Noop', barColor);
  const { row: leftRow, els: leftEls } = makeQBarRow('Left', barColor);
  const { row: rightRow, els: rightEls } = makeQBarRow('Right', barColor);
  qPanel.append(qTitle, noopRow, leftRow, rightRow);

  const stats = document.createElement('div');
  stats.className = 'section';

  const statsGrid = document.createElement('div');
  statsGrid.className = 'stat-grid';

  const scoreRow = document.createElement('div');
  scoreRow.className = 'stat-card';

  const scoreLabel = document.createElement('span');
  scoreLabel.className = 'label';
  scoreLabel.textContent = 'Score';
  const scoreValue = document.createElement('span');
  scoreValue.className = 'minatar-breakout-stat-value';
  scoreValue.textContent = '0';
  scoreRow.append(scoreLabel, scoreValue);

  const stepRow = document.createElement('div');
  stepRow.className = 'stat-card';

  const hint = document.createElement('span');
  hint.className = 'minatar-breakout-hint';
  hint.textContent =
    'Click to focus   A / \u2190 left   D / \u2192 right';

  const stepLabel = document.createElement('span');
  stepLabel.className = 'label';
  stepLabel.textContent = 'Step';

  const stepValue = document.createElement('span');
  stepValue.className = 'minatar-breakout-stat-value';
  stepValue.textContent = '0';
  stepRow.append(stepLabel, stepValue);

  statsGrid.append(scoreRow, stepRow);
  stats.appendChild(statsGrid);

  const stage = document.createElement('div');
  stage.className = 'minatar-breakout-stage';
  stage.appendChild(canvasWrap);

  const hintSection = document.createElement('div');
  hintSection.className = 'section minatar-breakout-hint-section';
  hintSection.appendChild(hint);

  controls.append(modeSection, restartSection, stats, qPanel, hintSection);

  layout.append(stage, controls);
  container.appendChild(layout);

  return {
    container,
    canvas,
    userModeRadio,
    policyModeRadio,
    policyModeText,
    restartBtn,
    hint,
    qPanel,
    overlay,
    overlayTitle,
    overlayRestartBtn,
    scoreValue,
    stepValue,
    qBars: { noop: noopEls, left: leftEls, right: rightEls }
  };
}
