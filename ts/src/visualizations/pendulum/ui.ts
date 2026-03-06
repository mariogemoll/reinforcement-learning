// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createTimelineStepControls } from '../shared/ui';
import type { PendulumVizDom } from './types';

const TRAJECTORY_CHART_WIDTH = 1000;
const SPEED_SLIDER_MIN = 0;
const SPEED_SLIDER_MAX = 100;
const SPEED_SLIDER_DEFAULT = 50;

export function createPendulumVizDom(): PendulumVizDom {
  const container = document.createElement('div');
  container.className = 'visualization';

  const topRow = document.createElement('div');
  topRow.className = 'pendulum-viz-top-row';

  const bottomPanel = document.createElement('div');
  bottomPanel.className = 'pendulum-viz-bottom-panel';

  // --- Canvas ---
  const canvas = document.createElement('canvas');
  canvas.className = 'pendulum-viz-canvas';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'pendulum-viz-canvas-wrap';

  const terminalOverlay = document.createElement('div');
  terminalOverlay.className = 'pendulum-viz-terminal-overlay';
  terminalOverlay.hidden = true;
  terminalOverlay.setAttribute('aria-live', 'polite');

  const terminalPanel = document.createElement('div');
  terminalPanel.className = 'pendulum-viz-terminal-panel';
  const terminalTitle = document.createElement('div');
  terminalTitle.className = 'terminal-title';
  const terminalSummary = document.createElement('div');
  terminalSummary.className = 'terminal-summary';
  terminalPanel.append(terminalTitle, terminalSummary);
  terminalOverlay.appendChild(terminalPanel);

  canvasWrap.append(canvas, terminalOverlay);

  // --- Side panel ---
  const sidePanel = document.createElement('div');
  sidePanel.className = 'pendulum-viz-side-panel';

  // Stats: episodes + steps
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

  // State: angle, angular velocity, torque, return
  const stateSection = document.createElement('div');
  stateSection.className = 'section';
  const stateGrid = document.createElement('div');
  stateGrid.className = 'stat-grid pendulum-viz-state-grid';

  const angleSection = document.createElement('div');
  angleSection.className = 'stat-card';
  const angleLbl = document.createElement('span');
  angleLbl.className = 'label';
  angleLbl.textContent = 'Angle';
  const angleValue = document.createElement('strong');
  angleValue.textContent = '0.0\u00b0';
  angleSection.append(angleLbl, angleValue);

  const angVelSection = document.createElement('div');
  angVelSection.className = 'stat-card';
  const angVelLbl = document.createElement('span');
  angVelLbl.className = 'label';
  angVelLbl.textContent = 'Ang. Vel.';
  const angVelValue = document.createElement('strong');
  angVelValue.textContent = '0.00';
  angVelSection.append(angVelLbl, angVelValue);

  const torqueSection = document.createElement('div');
  torqueSection.className = 'stat-card';
  const torqueLbl = document.createElement('span');
  torqueLbl.className = 'label';
  torqueLbl.textContent = 'Torque';
  const torqueValue = document.createElement('strong');
  torqueValue.textContent = '0.00';
  torqueSection.append(torqueLbl, torqueValue);

  const returnSection = document.createElement('div');
  returnSection.className = 'stat-card';
  const returnLbl = document.createElement('span');
  returnLbl.className = 'label';
  returnLbl.textContent = 'Return';
  const returnValue = document.createElement('strong');
  returnValue.textContent = '0.0';
  returnSection.append(returnLbl, returnValue);

  stateGrid.append(angleSection, angVelSection, torqueSection, returnSection);
  stateSection.appendChild(stateGrid);

  // Speed slider
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
  speedValueEl.textContent = '1x';
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

  // New trajectory button
  const resetSection = document.createElement('div');
  resetSection.className = 'section';
  const resetBtn = document.createElement('button');
  resetBtn.className = 'control-reset';
  resetBtn.type = 'button';
  resetBtn.textContent = 'New trajectory';
  resetSection.appendChild(resetBtn);

  sidePanel.append(statsSection, stateSection, speedSection, resetSection);

  // --- Bottom panel: torque chart + timeline + step controls ---
  const chartSection = document.createElement('div');
  chartSection.className = 'section pendulum-viz-chart-stack';

  const torqueChartCanvas = document.createElement('canvas');
  torqueChartCanvas.className = 'pi-viz-chart pi-viz-scrub-chart pendulum-viz-torque-chart';
  torqueChartCanvas.width = TRAJECTORY_CHART_WIDTH;
  torqueChartCanvas.height = 50;

  const timelineCanvas = document.createElement('canvas');
  timelineCanvas.className = 'pi-viz-chart pi-viz-timeline-chart pendulum-viz-timeline-chart';
  timelineCanvas.width = TRAJECTORY_CHART_WIDTH;
  timelineCanvas.height = 26;

  chartSection.append(torqueChartCanvas, timelineCanvas);

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

  controlsSection.append(stepRow);

  bottomPanel.append(chartSection, controlsSection);
  topRow.append(canvasWrap, sidePanel);
  container.append(topRow, bottomPanel);

  return {
    container,
    canvas,
    torqueChartCanvas,
    timelineCanvas,
    terminalOverlay,
    terminalTitle,
    terminalSummary,
    episodesValue,
    stepsValue,
    angleValue,
    angVelValue,
    torqueValue,
    returnValue,
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
