// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { CartpolePolicyVizDom } from './types';

const DEFAULT_SPEED = 1;

export function createCartpolePolicyVizDom(): CartpolePolicyVizDom {
  const container = document.createElement('div');
  container.className = 'visualization';

  const canvas = document.createElement('canvas');
  canvas.className = 'cartpole-viz-canvas';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'cartpole-viz-canvas-wrap';

  const terminalOverlay = document.createElement('div');
  terminalOverlay.className = 'cartpole-viz-terminal-overlay';
  terminalOverlay.hidden = true;
  terminalOverlay.setAttribute('aria-live', 'polite');

  const terminalPanel = document.createElement('div');
  terminalPanel.className = 'cartpole-viz-terminal-panel';
  const terminalTitle = document.createElement('div');
  terminalTitle.className = 'terminal-title';
  const terminalSummary = document.createElement('div');
  terminalSummary.className = 'terminal-summary';
  terminalPanel.append(terminalTitle, terminalSummary);
  terminalOverlay.appendChild(terminalPanel);

  const sidePanel = document.createElement('div');
  sidePanel.className = 'cartpole-viz-side-panel';

  // Episode + steps stat cards
  const statsSection = document.createElement('div');
  statsSection.className = 'section';
  const statGrid = document.createElement('div');
  statGrid.className = 'stat-grid';

  const episodesCard = document.createElement('div');
  episodesCard.className = 'stat-card';
  const episodesLabel = document.createElement('span');
  episodesLabel.className = 'label';
  episodesLabel.textContent = 'Episodes';
  const episodesValue = document.createElement('strong');
  episodesValue.textContent = '0';
  episodesCard.append(episodesLabel, episodesValue);

  const stepsCard = document.createElement('div');
  stepsCard.className = 'stat-card';
  const stepsLabel = document.createElement('span');
  stepsLabel.className = 'label';
  stepsLabel.textContent = 'Steps';
  const stepsValue = document.createElement('strong');
  stepsValue.textContent = '0';
  stepsCard.append(stepsLabel, stepsValue);

  statGrid.append(episodesCard, stepsCard);
  statsSection.appendChild(statGrid);

  // State section
  const stateSection = document.createElement('div');
  stateSection.className = 'section';
  const stateGrid = document.createElement('div');
  stateGrid.className = 'stat-grid';

  const positionCard = document.createElement('div');
  positionCard.className = 'stat-card';
  const positionLabel = document.createElement('span');
  positionLabel.className = 'label';
  positionLabel.textContent = 'Position';
  const positionValue = document.createElement('strong');
  positionValue.textContent = '0.00';
  positionCard.append(positionLabel, positionValue);

  const velocityCard = document.createElement('div');
  velocityCard.className = 'stat-card';
  const velocityLabel = document.createElement('span');
  velocityLabel.className = 'label';
  velocityLabel.textContent = 'Velocity';
  const velocityValue = document.createElement('strong');
  velocityValue.textContent = '0.00';
  velocityCard.append(velocityLabel, velocityValue);

  const angleCard = document.createElement('div');
  angleCard.className = 'stat-card';
  const angleLabel = document.createElement('span');
  angleLabel.className = 'label';
  angleLabel.textContent = 'Angle';
  const angleValue = document.createElement('strong');
  angleValue.textContent = '0.00\u00b0';
  angleCard.append(angleLabel, angleValue);

  const angVelCard = document.createElement('div');
  angVelCard.className = 'stat-card';
  const angVelLabel = document.createElement('span');
  angVelLabel.className = 'label';
  angVelLabel.textContent = 'Ang. Vel.';
  const angularVelocityValue = document.createElement('strong');
  angularVelocityValue.textContent = '0.00';
  angVelCard.append(angVelLabel, angularVelocityValue);

  stateGrid.append(positionCard, velocityCard, angleCard, angVelCard);
  stateSection.appendChild(stateGrid);

  // Q-values section
  const qSection = document.createElement('div');
  qSection.className = 'section';

  const qLabel = document.createElement('div');
  qLabel.className = 'label';
  qLabel.style.marginBottom = '4px';
  qLabel.textContent = 'Q-values';

  const qLeftRow = document.createElement('div');
  qLeftRow.className = 'cp-policy-q-row';
  const qLeftLabel = document.createElement('span');
  qLeftLabel.className = 'cp-policy-q-action-label';
  qLeftLabel.textContent = '\u2190 Left';
  const qLeftBarWrap = document.createElement('div');
  qLeftBarWrap.className = 'cp-policy-q-bar-wrap';
  const qLeftBar = document.createElement('div');
  qLeftBar.className = 'cp-policy-q-bar';
  qLeftBarWrap.appendChild(qLeftBar);
  const qLeftValue = document.createElement('span');
  qLeftValue.className = 'cp-policy-q-value mono-value';
  qLeftValue.textContent = '0.00';
  qLeftRow.append(qLeftLabel, qLeftBarWrap, qLeftValue);

  const qRightRow = document.createElement('div');
  qRightRow.className = 'cp-policy-q-row';
  const qRightLabel = document.createElement('span');
  qRightLabel.className = 'cp-policy-q-action-label';
  qRightLabel.textContent = 'Right \u2192';
  const qRightBarWrap = document.createElement('div');
  qRightBarWrap.className = 'cp-policy-q-bar-wrap';
  const qRightBar = document.createElement('div');
  qRightBar.className = 'cp-policy-q-bar';
  qRightBarWrap.appendChild(qRightBar);
  const qRightValue = document.createElement('span');
  qRightValue.className = 'cp-policy-q-value mono-value';
  qRightValue.textContent = '0.00';
  qRightRow.append(qRightLabel, qRightBarWrap, qRightValue);

  const actionRow = document.createElement('div');
  actionRow.className = 'cp-policy-action-row';
  const actionLabel = document.createElement('span');
  actionLabel.className = 'label';
  actionLabel.textContent = 'Action';
  const actionValue = document.createElement('strong');
  actionValue.className = 'cp-policy-action-value';
  actionValue.textContent = '\u2192';
  actionRow.append(actionLabel, actionValue);

  qSection.append(qLabel, qLeftRow, qRightRow, actionRow);

  // Controls: speed slider + pause + reset
  const controlsSection = document.createElement('div');
  controlsSection.className = 'section controls-stack';

  const sliderWrap = document.createElement('div');
  sliderWrap.className = 'slider-wrap';
  const sliderLabel = document.createElement('div');
  sliderLabel.className = 'slider-label';
  const sliderText = document.createElement('span');
  sliderText.textContent = 'Speed';
  const speedValueEl = document.createElement('strong');
  speedValueEl.className = 'mono-value';
  speedValueEl.textContent = `${String(DEFAULT_SPEED)}x`;
  sliderLabel.append(sliderText, speedValueEl);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '8';
  speedSlider.step = '1';
  speedSlider.value = String(DEFAULT_SPEED);
  speedSlider.setAttribute('aria-label', 'Simulation speed');
  sliderWrap.append(sliderLabel, speedSlider);

  const btnRow = document.createElement('div');
  btnRow.className = 'cp-policy-btn-row';

  const pauseBtn = document.createElement('button');
  pauseBtn.className = 'cp-policy-pause-btn';
  pauseBtn.type = 'button';
  pauseBtn.textContent = 'Pause';

  const resetBtn = document.createElement('button');
  resetBtn.className = 'control-reset cp-policy-reset-btn';
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset';

  btnRow.append(pauseBtn, resetBtn);
  controlsSection.append(sliderWrap, btnRow);

  sidePanel.append(statsSection, stateSection, qSection, controlsSection);
  canvasWrap.append(canvas, terminalOverlay);
  container.append(canvasWrap, sidePanel);

  return {
    container,
    canvas,
    terminalOverlay,
    terminalTitle,
    terminalSummary,
    episodesValue,
    stepsValue,
    positionValue,
    velocityValue,
    angleValue,
    angularVelocityValue,
    qLeftValue,
    qRightValue,
    qLeftBar,
    qRightBar,
    qLeftRow,
    qRightRow,
    actionValue,
    pauseBtn,
    resetBtn,
    speedSlider,
    speedValueEl
  };
}
