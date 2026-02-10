import type { Action } from '../../core/types';

const ACTION_ARROWS: Record<Action, string> = {
  up: '\u2191', down: '\u2193', left: '\u2190', right: '\u2192'
};

export interface GridworldVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  terminalOverlay: HTMLDivElement;
  terminalTitle: HTMLElement;
  terminalSummary: HTMLElement;
  stepsValue: HTMLElement;
  rewardValue: HTMLElement;
  intendedMoveValue: HTMLElement;
  actualMoveValue: HTMLElement;
  resetBtn: HTMLButtonElement;
  slipperinessSlider: HTMLInputElement;
  slipperinessValueEl: HTMLElement;
  dpadButtons: HTMLButtonElement[];
}

export function createGridworldVizDom(initialSlipperiness: number): GridworldVizDom {
  const container = document.createElement('div');
  container.className = 'visualization';

  const canvas = document.createElement('canvas');
  canvas.className = 'gridworld-viz-canvas';
  canvas.tabIndex = 0;

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'gridworld-viz-canvas-wrap';

  const terminalOverlay = document.createElement('div');
  terminalOverlay.className = 'gridworld-viz-terminal-overlay';
  terminalOverlay.hidden = true;
  terminalOverlay.setAttribute('aria-live', 'polite');

  const terminalPanel = document.createElement('div');
  terminalPanel.className = 'gridworld-viz-terminal-panel';
  const terminalTitle = document.createElement('div');
  terminalTitle.className = 'terminal-title';
  const terminalSummary = document.createElement('div');
  terminalSummary.className = 'terminal-summary';
  terminalPanel.append(terminalTitle, terminalSummary);
  terminalOverlay.appendChild(terminalPanel);

  const sidePanel = document.createElement('div');
  sidePanel.className = 'gridworld-viz-side-panel';

  const statsSection = document.createElement('div');
  statsSection.className = 'section';
  const statGrid = document.createElement('div');
  statGrid.className = 'stat-grid';

  const stepsCard = document.createElement('div');
  stepsCard.className = 'stat-card';
  const stepsLabel = document.createElement('span');
  stepsLabel.className = 'label';
  stepsLabel.textContent = 'Steps';
  const stepsValue = document.createElement('strong');
  stepsValue.textContent = '0';
  stepsCard.append(stepsLabel, stepsValue);

  const rewardCard = document.createElement('div');
  rewardCard.className = 'stat-card';
  const rewardLabel = document.createElement('span');
  rewardLabel.className = 'label';
  rewardLabel.textContent = 'Reward';
  const rewardValue = document.createElement('strong');
  rewardValue.textContent = '0.0';
  rewardCard.append(rewardLabel, rewardValue);

  statGrid.append(stepsCard, rewardCard);
  statsSection.appendChild(statGrid);

  const statusSection = document.createElement('div');
  statusSection.className = 'section';
  const statusText = document.createElement('div');
  statusText.className = 'status-text';
  const statusGrid = document.createElement('div');
  statusGrid.className = 'status-grid';

  const intendedCard = document.createElement('div');
  intendedCard.className = 'status-card';
  const intendedLabel = document.createElement('span');
  intendedLabel.className = 'label';
  intendedLabel.textContent = 'Last move: intended';
  const intendedMoveValue = document.createElement('strong');
  intendedMoveValue.className = 'move-pending';
  intendedMoveValue.textContent = '-';
  intendedCard.append(intendedLabel, intendedMoveValue);

  const actualCard = document.createElement('div');
  actualCard.className = 'status-card';
  const actualLabel = document.createElement('span');
  actualLabel.className = 'label';
  actualLabel.textContent = 'Last move: actual';
  const actualMoveValue = document.createElement('strong');
  actualMoveValue.className = 'move-pending';
  actualMoveValue.textContent = '-';
  actualCard.append(actualLabel, actualMoveValue);

  statusGrid.append(intendedCard, actualCard);
  statusText.appendChild(statusGrid);
  statusSection.appendChild(statusText);

  const gamepadSection = document.createElement('div');
  gamepadSection.className = 'section gridworld-viz-gamepad-section';
  const dpad = document.createElement('div');
  dpad.className = 'dpad';
  const dpadButtons: HTMLButtonElement[] = [];
  const dpadActions: Action[] = ['up', 'left', 'right', 'down'];
  dpadActions.forEach((action) => {
    const btn = document.createElement('button');
    btn.className = 'dpad-btn';
    btn.dataset.action = action;
    btn.title = action[0].toUpperCase() + action.slice(1);
    btn.type = 'button';
    btn.textContent = ACTION_ARROWS[action];
    dpadButtons.push(btn);
    dpad.appendChild(btn);
  });
  gamepadSection.appendChild(dpad);

  const controlsSection = document.createElement('div');
  controlsSection.className = 'section controls-stack';
  const sliderWrap = document.createElement('div');
  sliderWrap.className = 'slider-wrap';
  const sliderLabel = document.createElement('div');
  sliderLabel.className = 'slider-label';
  const sliderText = document.createElement('span');
  sliderText.textContent = 'Slipperiness';
  const slipperinessValueEl = document.createElement('strong');
  slipperinessValueEl.className = 'mono-value';
  slipperinessValueEl.textContent = `${String(Math.round(initialSlipperiness * 100))}%`;
  sliderLabel.append(sliderText, slipperinessValueEl);

  const slipperinessSlider = document.createElement('input');
  slipperinessSlider.type = 'range';
  slipperinessSlider.min = '0';
  slipperinessSlider.max = '1';
  slipperinessSlider.step = '0.05';
  slipperinessSlider.value = initialSlipperiness.toFixed(2);
  slipperinessSlider.setAttribute('aria-label', 'Slipperiness');

  sliderWrap.append(sliderLabel, slipperinessSlider);

  const resetBtn = document.createElement('button');
  resetBtn.className = 'control-reset';
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset';

  controlsSection.append(sliderWrap, resetBtn);

  sidePanel.append(statsSection, statusSection, gamepadSection, controlsSection);
  canvasWrap.append(canvas, terminalOverlay);
  container.append(canvasWrap, sidePanel);

  return {
    container,
    canvas,
    terminalOverlay,
    terminalTitle,
    terminalSummary,
    stepsValue,
    rewardValue,
    intendedMoveValue,
    actualMoveValue,
    resetBtn,
    slipperinessSlider,
    slipperinessValueEl,
    dpadButtons
  };
}
