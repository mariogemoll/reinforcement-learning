// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface MonteCarloVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  chartCanvas: HTMLCanvasElement;
  policyChartCanvas: HTMLCanvasElement;
  valueRMSECanvas: HTMLCanvasElement;
  policyAgreementCanvas: HTMLCanvasElement;
  timelineCanvas: HTMLCanvasElement;
  phaseExplanationEl: HTMLParagraphElement;
  slipperinessSlider: HTMLInputElement;
  slipperinessValueEl: HTMLElement;
  gammaSlider: HTMLInputElement;
  gammaValueEl: HTMLElement;
  epsilonSlider: HTMLInputElement;
  epsilonValueEl: HTMLElement;
  episodesPerBatchSlider: HTMLInputElement;
  episodesPerBatchValueEl: HTMLElement;
  maxStepsSlider: HTMLInputElement;
  maxStepsValueEl: HTMLElement;
  visitModeFirstRadio: HTMLInputElement;
  visitModeEveryRadio: HTMLInputElement;
  startModeFixedRadio: HTMLInputElement;
  startModeExploringRadio: HTMLInputElement;
  colorModeValueRadio: HTMLInputElement;
  colorModeDisagreementRadio: HTMLInputElement;
  showDisagreementCheckbox: HTMLInputElement;
  showValuesCheckbox: HTMLInputElement;
  resampleBtn: HTMLButtonElement;
  goToStartBtn: HTMLButtonElement;
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

export function createMonteCarloVizDom(
  radioNamePrefix: string,
  initialSlipperiness: number,
  initialGamma: number,
  initialEpsilon: number,
  initialEpisodesPerBatch: number,
  initialMaxSteps: number
): MonteCarloVizDom {
  const container = el('div', 'visualization');
  const layout = el('div', 'mc-viz-grid');

  const canvas = document.createElement('canvas');
  canvas.className = 'pi-viz-canvas';
  canvas.tabIndex = 0;

  const canvasWrap = el('div', 'pi-viz-canvas-wrap');
  canvasWrap.appendChild(canvas);

  const gridCell = el('div', 'mc-viz-cell-grid');

  const chartCol = el(
    'div', 'pi-viz-col pi-viz-chart-col'
  );
  const chartCanvas = document.createElement('canvas');
  chartCanvas.className = 'pi-viz-chart pi-viz-scrub-chart';
  chartCanvas.width = 800;
  chartCanvas.height = 102;
  chartCol.append(chartCanvas);

  const policyChartCol = el(
    'div', 'pi-viz-col pi-viz-chart-col'
  );
  const policyChartCanvas = document.createElement('canvas');
  policyChartCanvas.className =
    'pi-viz-chart pi-viz-scrub-chart';
  policyChartCanvas.width = 800;
  policyChartCanvas.height = 102;
  policyChartCol.append(policyChartCanvas);

  const valueRMSECol = el(
    'div', 'pi-viz-col pi-viz-chart-col'
  );
  const valueRMSECanvas = document.createElement('canvas');
  valueRMSECanvas.className =
    'pi-viz-chart pi-viz-scrub-chart';
  valueRMSECanvas.width = 800;
  valueRMSECanvas.height = 102;
  valueRMSECol.append(valueRMSECanvas);

  const policyAgreementCol = el(
    'div', 'pi-viz-col pi-viz-chart-col'
  );
  const policyAgreementCanvas =
    document.createElement('canvas');
  policyAgreementCanvas.className =
    'pi-viz-chart pi-viz-scrub-chart';
  policyAgreementCanvas.width = 800;
  policyAgreementCanvas.height = 102;
  policyAgreementCol.append(policyAgreementCanvas);

  const timelineChartCol = el(
    'div', 'pi-viz-col pi-viz-chart-col'
  );
  const timelineCanvas = document.createElement('canvas');
  timelineCanvas.className =
    'pi-viz-chart pi-viz-timeline-chart';
  timelineCanvas.width = 800;
  timelineCanvas.height = 52;
  timelineChartCol.append(timelineCanvas);

  const chartsLeftCell = el('div', 'mc-viz-cell-charts-left');
  chartsLeftCell.append(chartCol, policyChartCol);

  const chartsRightCell = el('div', 'mc-viz-cell-charts-right');
  chartsRightCell.append(valueRMSECol, policyAgreementCol);

  const timelineCell = el('div', 'mc-viz-cell-timeline');
  timelineCell.append(timelineChartCol);

  const controlsCol = el(
    'div', 'pi-viz-col pi-viz-controls-col'
  );
  const phaseExplanationEl = document.createElement('p');
  phaseExplanationEl.className = 'pi-viz-phase-explanation';
  const paramsStack = el('div', 'pi-viz-param-stack');

  // Slipperiness slider
  const slipperinessWrap = el('div', 'slider-wrap');
  const slipperinessLabelEl = el('div', 'slider-label');
  const slipperinessText = document.createElement('span');
  slipperinessText.textContent = 'Slipperiness';
  const slipperinessValueEl = document.createElement('strong');
  slipperinessValueEl.className = 'mono-value';
  slipperinessValueEl.textContent =
    `${String(Math.round(initialSlipperiness * 100))}%`;
  slipperinessLabelEl.append(
    slipperinessText, slipperinessValueEl
  );
  const slipperinessSlider = document.createElement('input');
  slipperinessSlider.type = 'range';
  slipperinessSlider.min = '0';
  slipperinessSlider.max = '1';
  slipperinessSlider.step = '0.01';
  slipperinessSlider.value = initialSlipperiness.toFixed(2);
  slipperinessSlider.setAttribute(
    'aria-label', 'Slipperiness'
  );
  slipperinessWrap.append(
    slipperinessLabelEl, slipperinessSlider
  );

  // Gamma slider
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

  // Epsilon slider
  const epsilonWrap = el('div', 'slider-wrap');
  const epsilonLabelEl = el('div', 'slider-label');
  const epsilonText = document.createElement('span');
  epsilonText.textContent = 'Epsilon';
  const epsilonValueEl = document.createElement('strong');
  epsilonValueEl.className = 'mono-value';
  epsilonValueEl.textContent = initialEpsilon.toFixed(2);
  epsilonLabelEl.append(epsilonText, epsilonValueEl);
  const epsilonSlider = document.createElement('input');
  epsilonSlider.type = 'range';
  epsilonSlider.min = '0';
  epsilonSlider.max = '1';
  epsilonSlider.step = '0.01';
  epsilonSlider.value = initialEpsilon.toFixed(2);
  epsilonSlider.setAttribute('aria-label', 'Epsilon');
  epsilonWrap.append(epsilonLabelEl, epsilonSlider);

  // Episodes per batch slider
  const episodesWrap = el('div', 'slider-wrap');
  const episodesLabelEl = el('div', 'slider-label');
  const episodesText = document.createElement('span');
  episodesText.textContent = 'Episodes per batch';
  const episodesPerBatchValueEl =
    document.createElement('strong');
  episodesPerBatchValueEl.className = 'mono-value';
  episodesPerBatchValueEl.textContent =
    String(initialEpisodesPerBatch);
  episodesLabelEl.append(
    episodesText, episodesPerBatchValueEl
  );
  const episodesPerBatchSlider =
    document.createElement('input');
  episodesPerBatchSlider.type = 'range';
  episodesPerBatchSlider.min = '1';
  episodesPerBatchSlider.max = '50';
  episodesPerBatchSlider.step = '1';
  episodesPerBatchSlider.value =
    String(initialEpisodesPerBatch);
  episodesPerBatchSlider.setAttribute(
    'aria-label', 'Episodes per batch'
  );
  episodesWrap.append(episodesLabelEl, episodesPerBatchSlider);

  // Max steps per episode slider
  const maxStepsWrap = el('div', 'slider-wrap');
  const maxStepsLabelEl = el('div', 'slider-label');
  const maxStepsText = document.createElement('span');
  maxStepsText.textContent = 'Max steps per episode';
  const maxStepsValueEl = document.createElement('strong');
  maxStepsValueEl.className = 'mono-value';
  maxStepsValueEl.textContent = String(initialMaxSteps);
  maxStepsLabelEl.append(maxStepsText, maxStepsValueEl);
  const maxStepsSlider = document.createElement('input');
  maxStepsSlider.type = 'range';
  maxStepsSlider.min = '10';
  maxStepsSlider.max = '1000';
  maxStepsSlider.step = '10';
  maxStepsSlider.value = String(initialMaxSteps);
  maxStepsSlider.setAttribute(
    'aria-label', 'Max steps per episode'
  );
  maxStepsWrap.append(maxStepsLabelEl, maxStepsSlider);

  // Options grid: radio groups + checkboxes in a compact 2-column layout
  const optionsGrid = el('div', 'mc-viz-options-grid');

  // Visit mode radio buttons
  const visitModeFirstLabel = document.createElement('label');
  visitModeFirstLabel.className = 'mc-viz-radio-label';
  const visitModeFirstRadio = document.createElement('input');
  visitModeFirstRadio.type = 'radio';
  visitModeFirstRadio.name = `${radioNamePrefix}-visit-mode`;
  visitModeFirstRadio.value = 'first';
  visitModeFirstRadio.checked = true;
  visitModeFirstLabel.append(
    visitModeFirstRadio, document.createTextNode(' First-visit')
  );

  const visitModeEveryLabel = document.createElement('label');
  visitModeEveryLabel.className = 'mc-viz-radio-label';
  const visitModeEveryRadio = document.createElement('input');
  visitModeEveryRadio.type = 'radio';
  visitModeEveryRadio.name = `${radioNamePrefix}-visit-mode`;
  visitModeEveryRadio.value = 'every';
  visitModeEveryLabel.append(
    visitModeEveryRadio, document.createTextNode(' Every-visit')
  );

  // Start mode radio buttons
  const startModeFixedLabel = document.createElement('label');
  startModeFixedLabel.className = 'mc-viz-radio-label';
  const startModeFixedRadio = document.createElement('input');
  startModeFixedRadio.type = 'radio';
  startModeFixedRadio.name = `${radioNamePrefix}-start-mode`;
  startModeFixedRadio.value = 'fixed';
  startModeFixedRadio.checked = true;
  startModeFixedLabel.append(
    startModeFixedRadio, document.createTextNode(' Fixed start')
  );

  const startModeExploringLabel = document.createElement('label');
  startModeExploringLabel.className = 'mc-viz-radio-label';
  const startModeExploringRadio = document.createElement('input');
  startModeExploringRadio.type = 'radio';
  startModeExploringRadio.name = `${radioNamePrefix}-start-mode`;
  startModeExploringRadio.value = 'exploring';
  startModeExploringLabel.append(
    startModeExploringRadio,
    document.createTextNode(' Exploring starts')
  );

  // Color mode radio buttons
  const colorModeValueLabel = document.createElement('label');
  colorModeValueLabel.className = 'mc-viz-radio-label';
  const colorModeValueRadio = document.createElement('input');
  colorModeValueRadio.type = 'radio';
  colorModeValueRadio.name = `${radioNamePrefix}-color-mode`;
  colorModeValueRadio.value = 'value';
  colorModeValueRadio.checked = true;
  colorModeValueLabel.append(
    colorModeValueRadio, document.createTextNode(' Show values')
  );

  const colorModeDisagreementLabel = document.createElement('label');
  colorModeDisagreementLabel.className = 'mc-viz-radio-label';
  const colorModeDisagreementRadio = document.createElement('input');
  colorModeDisagreementRadio.type = 'radio';
  colorModeDisagreementRadio.name = `${radioNamePrefix}-color-mode`;
  colorModeDisagreementRadio.value = 'disagreement';
  colorModeDisagreementLabel.append(
    colorModeDisagreementRadio,
    document.createTextNode(' Show error')
  );

  optionsGrid.append(
    visitModeFirstLabel, visitModeEveryLabel,
    startModeFixedLabel, startModeExploringLabel,
    colorModeValueLabel, colorModeDisagreementLabel
  );

  // Checkboxes below the grid canvas
  const checkboxRow = el('div', 'mc-viz-checkbox-row');

  const disagreementLabel = document.createElement('label');
  disagreementLabel.className = 'mc-viz-checkbox-label';
  const showDisagreementCheckbox =
    document.createElement('input');
  showDisagreementCheckbox.type = 'checkbox';
  showDisagreementCheckbox.setAttribute(
    'aria-label', 'Show policy disagreement'
  );
  disagreementLabel.append(
    showDisagreementCheckbox,
    document.createTextNode(' Show policy disagreement')
  );

  const showValuesLabel = document.createElement('label');
  showValuesLabel.className = 'mc-viz-checkbox-label';
  const showValuesCheckbox =
    document.createElement('input');
  showValuesCheckbox.type = 'checkbox';
  showValuesCheckbox.setAttribute(
    'aria-label', 'Show values on grid'
  );
  showValuesLabel.append(
    showValuesCheckbox,
    document.createTextNode(' Show values')
  );

  checkboxRow.append(disagreementLabel, showValuesLabel);

  const resampleBtn = document.createElement('button');
  resampleBtn.type = 'button';
  resampleBtn.className = 'mc-viz-resample-btn';
  resampleBtn.textContent = 'Resample randomness';

  paramsStack.append(
    slipperinessWrap,
    gammaWrap,
    epsilonWrap,
    episodesWrap,
    maxStepsWrap,
    optionsGrid,
    resampleBtn
  );

  // Step controls
  const goToStartBtn = document.createElement('button');
  goToStartBtn.type = 'button';
  goToStartBtn.className = 'pi-viz-step-btn';
  goToStartBtn.textContent = '\u23EE';
  goToStartBtn.title = 'Go to start';

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
  timelineCell.append(controlsCol);

  const slidersCell = el('div', 'mc-viz-cell-sliders');
  slidersCell.append(paramsStack);

  gridCell.append(canvasWrap, checkboxRow);

  layout.append(
    gridCell, chartsLeftCell, chartsRightCell,
    timelineCell, slidersCell
  );
  container.append(layout, phaseExplanationEl);

  return {
    container,
    canvas,
    chartCanvas,
    policyChartCanvas,
    valueRMSECanvas,
    policyAgreementCanvas,
    timelineCanvas,
    phaseExplanationEl,
    slipperinessSlider,
    slipperinessValueEl,
    gammaSlider,
    gammaValueEl,
    epsilonSlider,
    epsilonValueEl,
    episodesPerBatchSlider,
    episodesPerBatchValueEl,
    maxStepsSlider,
    maxStepsValueEl,
    visitModeFirstRadio,
    visitModeEveryRadio,
    startModeFixedRadio,
    startModeExploringRadio,
    colorModeValueRadio,
    colorModeDisagreementRadio,
    showDisagreementCheckbox,
    showValuesCheckbox,
    resampleBtn,
    goToStartBtn,
    stepBackBtn,
    playBtn,
    stepForwardBtn,
    stepCounterEl
  };
}
