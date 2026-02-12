// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { ExtraCheckbox } from './types';

export interface DPVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  chartCanvas: HTMLCanvasElement;
  policyChartCanvas: HTMLCanvasElement;
  timelineCanvas: HTMLCanvasElement;
  phaseExplanationEl: HTMLParagraphElement;
  slipperinessSlider: HTMLInputElement;
  slipperinessValueEl: HTMLElement;
  gammaSlider: HTMLInputElement;
  gammaValueEl: HTMLElement;
  thetaSlider: HTMLInputElement;
  thetaValueEl: HTMLElement;
  initialValuesZeroRadio: HTMLInputElement;
  initialValuesRandomRadio: HTMLInputElement;
  extraCheckboxInputs: HTMLInputElement[];
  goToStartBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  stepBackBtn: HTMLButtonElement;
  playBtn: HTMLButtonElement;
  stepForwardBtn: HTMLButtonElement;
  stepCounterEl: HTMLElement;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (className !== undefined) {
    element.className = className;
  }
  return element;
}

export function createDPVizDom(
  initialSlipperiness: number,
  initialGamma: number,
  initialTheta: number,
  initialValueMode: 'zero' | 'random',
  initialValuesLabel: string,
  radioGroupName: string,
  extraCheckboxes: ExtraCheckbox[]
): DPVizDom {
  const container = el('div', 'visualization');
  const layout = el('div', 'pi-viz-layout');

  const canvas = document.createElement('canvas');
  canvas.className = 'pi-viz-canvas';
  canvas.tabIndex = 0;

  const canvasWrap = el('div', 'pi-viz-canvas-wrap');
  canvasWrap.appendChild(canvas);

  const leftCol = el('div', 'pi-viz-left-col');
  const middleCol = el('div', 'pi-viz-middle-col');
  const rightCol = el('div', 'pi-viz-right-col');

  leftCol.append(canvasWrap);

  const chartStack = el('div', 'pi-viz-chart-stack');

  const chartCol = el('div', 'pi-viz-col pi-viz-chart-col');
  const chartCanvas = document.createElement('canvas');
  chartCanvas.className = 'pi-viz-chart pi-viz-scrub-chart';
  chartCanvas.width = 800;
  chartCanvas.height = 102;
  chartCol.append(chartCanvas);

  const policyChartCol = el('div', 'pi-viz-col pi-viz-chart-col');
  const policyChartCanvas = document.createElement('canvas');
  policyChartCanvas.className = 'pi-viz-chart pi-viz-scrub-chart';
  policyChartCanvas.width = 800;
  policyChartCanvas.height = 102;
  policyChartCol.append(policyChartCanvas);

  const timelineChartCol = el('div', 'pi-viz-col pi-viz-chart-col');
  const timelineCanvas = document.createElement('canvas');
  timelineCanvas.className = 'pi-viz-chart pi-viz-timeline-chart';
  timelineCanvas.width = 800;
  timelineCanvas.height = 52;
  timelineChartCol.append(timelineCanvas);

  chartStack.append(chartCol, policyChartCol, timelineChartCol);

  const controlsCol = el('div', 'pi-viz-col pi-viz-controls-col');
  const phaseExplanationEl = document.createElement('p');
  phaseExplanationEl.className = 'pi-viz-phase-explanation';
  const paramsStack = el('div', 'pi-viz-param-stack');

  const slipperinessWrap = el('div', 'slider-wrap');
  const slipperinessLabelEl = el('div', 'slider-label');
  const slipperinessText = document.createElement('span');
  slipperinessText.textContent = 'Slipperiness';
  const slipperinessValueEl = document.createElement('strong');
  slipperinessValueEl.className = 'mono-value';
  slipperinessValueEl.textContent = `${String(Math.round(initialSlipperiness * 100))}%`;
  slipperinessLabelEl.append(slipperinessText, slipperinessValueEl);
  const slipperinessSlider = document.createElement('input');
  slipperinessSlider.type = 'range';
  slipperinessSlider.min = '0';
  slipperinessSlider.max = '1';
  slipperinessSlider.step = '0.01';
  slipperinessSlider.value = initialSlipperiness.toFixed(2);
  slipperinessSlider.setAttribute('aria-label', 'Slipperiness');
  slipperinessWrap.append(slipperinessLabelEl, slipperinessSlider);

  const gammaWrap = el('div', 'slider-wrap');
  const gammaLabelEl = el('div', 'slider-label');
  const gammaText = document.createElement('span');
  gammaText.textContent = 'Gamma';
  const gammaValueEl = document.createElement('strong');
  gammaValueEl.className = 'mono-value';
  gammaValueEl.textContent = initialGamma.toFixed(2);
  gammaLabelEl.append(gammaText, gammaValueEl);
  const gammaSlider = document.createElement('input');
  gammaSlider.type = 'range';
  gammaSlider.min = '0';
  gammaSlider.max = '0.99';
  gammaSlider.step = '0.01';
  gammaSlider.value = initialGamma.toFixed(2);
  gammaSlider.setAttribute('aria-label', 'Gamma');
  gammaWrap.append(gammaLabelEl, gammaSlider);

  const thetaWrap = el('div', 'slider-wrap');
  const thetaLabelEl = el('div', 'slider-label');
  const thetaText = document.createElement('span');
  thetaText.textContent = 'Theta';
  const thetaValueEl = document.createElement('strong');
  thetaValueEl.className = 'mono-value';
  thetaValueEl.textContent = initialTheta.toFixed(3);
  thetaLabelEl.append(thetaText, thetaValueEl);
  const thetaSlider = document.createElement('input');
  thetaSlider.type = 'range';
  thetaSlider.min = '0.001';
  thetaSlider.max = '0.1';
  thetaSlider.step = '0.001';
  thetaSlider.value = initialTheta.toFixed(3);
  thetaSlider.setAttribute('aria-label', 'Theta');
  thetaWrap.append(thetaLabelEl, thetaSlider);

  const initialValuesWrap = el('div', 'pi-viz-initial-values');
  const initialValuesLabelEl = el('div', 'slider-label');
  const initialValuesText = document.createElement('span');
  initialValuesText.textContent = initialValuesLabel;
  initialValuesLabelEl.append(initialValuesText);

  const initialValuesOptions = el(
    'div',
    'pi-viz-radio-options'
  );
  const initialValuesZeroLabel = el(
    'label',
    'pi-viz-radio-option'
  );
  const initialValuesZeroRadio = document.createElement(
    'input'
  );
  initialValuesZeroRadio.type = 'radio';
  initialValuesZeroRadio.name = radioGroupName;
  initialValuesZeroRadio.value = 'zero';
  initialValuesZeroRadio.checked = initialValueMode === 'zero';
  const initialValuesZeroText = document.createElement('span');
  initialValuesZeroText.textContent = '0';
  initialValuesZeroLabel.append(
    initialValuesZeroRadio,
    initialValuesZeroText
  );

  const initialValuesRandomLabel = el(
    'label',
    'pi-viz-radio-option'
  );
  const initialValuesRandomRadio = document.createElement(
    'input'
  );
  initialValuesRandomRadio.type = 'radio';
  initialValuesRandomRadio.name = radioGroupName;
  initialValuesRandomRadio.value = 'random';
  initialValuesRandomRadio.checked =
    initialValueMode === 'random';
  const initialValuesRandomText = document.createElement(
    'span'
  );
  initialValuesRandomText.textContent = 'random';
  initialValuesRandomLabel.append(
    initialValuesRandomRadio,
    initialValuesRandomText
  );

  initialValuesOptions.append(
    initialValuesZeroLabel,
    initialValuesRandomLabel
  );
  initialValuesWrap.append(
    initialValuesLabelEl,
    initialValuesOptions
  );

  const extraCheckboxInputs: HTMLInputElement[] = [];
  const extraCheckboxWraps: HTMLDivElement[] = [];
  for (const cb of extraCheckboxes) {
    const wrap = el('div', 'pi-viz-initial-values');
    const label = el('label', 'pi-viz-radio-option');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = cb.initialChecked;
    checkbox.setAttribute('aria-label', cb.ariaLabel);
    const text = document.createElement('span');
    text.textContent = cb.label;
    label.append(checkbox, text);
    wrap.append(label);
    extraCheckboxInputs.push(checkbox);
    extraCheckboxWraps.push(wrap);
  }

  paramsStack.append(
    slipperinessWrap,
    gammaWrap,
    thetaWrap,
    initialValuesWrap,
    ...extraCheckboxWraps
  );

  const goToStartBtn = document.createElement('button');
  goToStartBtn.type = 'button';
  goToStartBtn.className = 'pi-viz-step-btn';
  goToStartBtn.textContent = '\u23EE';
  goToStartBtn.title = 'Go to start';

  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'pi-viz-step-btn pi-viz-reset-btn';
  resetBtn.textContent = 'Reset';
  resetBtn.title = 'Reset';
  initialValuesOptions.append(resetBtn);

  const stepBackBtn = document.createElement('button');
  stepBackBtn.type = 'button';
  stepBackBtn.className = 'pi-viz-step-btn';
  stepBackBtn.textContent = '<';
  stepBackBtn.title = 'Step back';

  const playBtn = document.createElement('button');
  playBtn.type = 'button';
  playBtn.className = 'pi-viz-step-btn pi-viz-play-btn';
  playBtn.textContent = '\u25B6';
  playBtn.title = 'Play';

  const stepForwardBtn = document.createElement('button');
  stepForwardBtn.type = 'button';
  stepForwardBtn.className = 'pi-viz-step-btn';
  stepForwardBtn.textContent = '>';
  stepForwardBtn.title = 'Step forward';

  const stepRow = el('div', 'pi-viz-step-row');
  const stepCounterEl = el('span', 'pi-viz-step-counter');
  stepCounterEl.textContent = '1/1';
  stepCounterEl.setAttribute('aria-label', 'Current step');
  stepRow.append(
    goToStartBtn,
    stepBackBtn,
    playBtn,
    stepForwardBtn,
    stepCounterEl
  );

  controlsCol.append(stepRow);
  middleCol.append(paramsStack);
  rightCol.append(chartStack, controlsCol);
  layout.append(leftCol, rightCol, middleCol);
  container.append(layout, phaseExplanationEl);

  return {
    container,
    canvas,
    chartCanvas,
    policyChartCanvas,
    timelineCanvas,
    phaseExplanationEl,
    slipperinessSlider,
    slipperinessValueEl,
    gammaSlider,
    gammaValueEl,
    thetaSlider,
    thetaValueEl,
    initialValuesZeroRadio,
    initialValuesRandomRadio,
    extraCheckboxInputs,
    goToStartBtn,
    resetBtn,
    stepBackBtn,
    playBtn,
    stepForwardBtn,
    stepCounterEl
  };
}
