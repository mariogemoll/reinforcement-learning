// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createTimelineStepControls } from '../shared/ui';
import type { CartPoleVizDom } from './types';

const DEFAULT_SPEED = 1;
const TRAJECTORY_CHART_WIDTH = 1000;
const SPEED_SLIDER_MIN = 0;
const SPEED_SLIDER_MAX = 100;
const SPEED_SLIDER_DEFAULT = 50;

export function createCartPoleVizDom(): CartPoleVizDom {
  const container = document.createElement('div');
  container.className = 'visualization';
  const topRow = document.createElement('div');
  topRow.className = 'cartpole-viz-top-row';
  const bottomPanel = document.createElement('div');
  bottomPanel.className = 'cartpole-viz-bottom-panel';

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

  // Row 1: episode + steps
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

  // Row 2: position + velocity + angle + angular velocity
  const stateSection = document.createElement('div');
  stateSection.className = 'section';
  const stateGrid = document.createElement('div');
  stateGrid.className = 'stat-grid cartpole-viz-state-grid';

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

  // Action chart + timeline (wide row under top section)
  const chartSection = document.createElement('div');
  chartSection.className = 'section cp-policy-chart-stack';

  const actionChartCanvas = document.createElement('canvas');
  actionChartCanvas.className = 'pi-viz-chart pi-viz-scrub-chart cp-policy-action-chart';
  actionChartCanvas.width = TRAJECTORY_CHART_WIDTH;
  actionChartCanvas.height = 50;

  const timelineCanvas = document.createElement('canvas');
  timelineCanvas.className = 'pi-viz-chart pi-viz-timeline-chart cp-policy-timeline-chart';
  timelineCanvas.width = TRAJECTORY_CHART_WIDTH;
  timelineCanvas.height = 26;

  chartSection.append(actionChartCanvas, timelineCanvas);

  // Controls: timeline navigation
  const controlsSection = document.createElement('div');
  controlsSection.className = 'section controls-stack';

  const {
    stepRow,
    goToStartBtn,
    stepBackBtn,
    playBtn,
    stepForwardBtn,
    stepCounterEl
  } = createTimelineStepControls();

  // Row 3: speed slider
  const speedSection = document.createElement('div');
  speedSection.className = 'section';
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
  speedSlider.min = String(SPEED_SLIDER_MIN);
  speedSlider.max = String(SPEED_SLIDER_MAX);
  speedSlider.step = '1';
  speedSlider.value = String(SPEED_SLIDER_DEFAULT);
  speedSlider.setAttribute('aria-label', 'Simulation speed');
  sliderWrap.append(sliderLabel, speedSlider);
  speedSection.appendChild(sliderWrap);

  // Row 4: new trajectory button
  const resetSection = document.createElement('div');
  resetSection.className = 'section';
  const resetBtn = document.createElement('button');
  resetBtn.className = 'control-reset';
  resetBtn.type = 'button';
  resetBtn.textContent = 'New trajectory';
  resetSection.appendChild(resetBtn);

  controlsSection.append(stepRow);

  sidePanel.append(
    statsSection,
    stateSection,
    speedSection,
    resetSection
  );
  canvasWrap.append(canvas, terminalOverlay);
  topRow.append(canvasWrap, sidePanel);
  bottomPanel.append(chartSection, controlsSection);
  container.append(topRow, bottomPanel);

  return {
    container,
    canvas,
    actionChartCanvas,
    timelineCanvas,
    terminalOverlay,
    terminalTitle,
    terminalSummary,
    episodesValue,
    stepsValue,
    positionValue,
    velocityValue,
    angleValue,
    angularVelocityValue,
    goToStartBtn,
    stepBackBtn,
    playBtn,
    stepForwardBtn,
    stepCounterEl,
    resetBtn,
    speedSlider,
    speedValueEl
  };
}
