// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import { createGridFromLayout } from '../../core/grid';
import { ACTIONS, buildTransitionTable } from '../../core/mdp';
import type {
  ActionValues,
  Grid,
  Policy,
  StateValues
} from '../../core/types';
import { initializeActionValues } from '../../dp/shared';
import { runValueIterationQ } from '../../dp/value-iteration-q';
import {
  DEFAULT_MAX_STEPS,
  generateRandomPool,
  type MonteCarloSnapshot,
  runMonteCarlo,
  TOTAL_BATCHES
} from '../../mc/monte-carlo';
import {
  createAnimationLoop,
  getAgentPosition,
  precalculatePath,
  updateAgents
} from '../shared/animation';
import {
  chartXToNearestIndex,
  renderChart,
  renderPolicyChangeChart,
  renderTimelineChart
} from '../shared/charts';
import { wireEvents } from '../shared/event-wiring';
import {
  createPlaybackController,
  createRepeatButton
} from '../shared/playback';
import { renderGrid } from '../shared/q-renderer';
import type {
  Agent,
  AgentPosition,
  Effect,
  ValueRange
} from '../shared/types';
import {
  renderPolicyAgreementChart,
  renderValueRMSEChart
} from './agreement-charts';
import type {
  ColorMode,
  MonteCarloBaseSnapshot,
  MonteCarloVisualization,
  MonteCarloVisualizationConfig
} from './types';
import { createMonteCarloVizDom } from './ui';

const MONTE_CARLO_AGENT_STEP_MS = 15;

export type { MonteCarloSnapshot, MonteCarloVisualization } from './types';

function initMonteCarloCoreVisualization<
  TSnapshot extends MonteCarloBaseSnapshot
>(
  parent: HTMLElement,
  config: MonteCarloVisualizationConfig<TSnapshot>
): MonteCarloVisualization {
  const layout = MEDIUM_GRID;
  const cellSize = 50;
  const grid = createGridFromLayout(layout);
  const startRow = 0;
  const startCol = 0;

  let currentSuccessProb: number = DP_DEFAULTS.successProb;
  let currentGamma: number = DP_DEFAULTS.gamma;
  let currentEpsilon = 0.1;
  let currentEpisodesPerBatch = 20;
  let currentMaxSteps: number = DEFAULT_MAX_STEPS;
  let exploringStarts = false;
  let firstVisit = true;
  let randomPool = generateRandomPool();
  let table = buildTransitionTable(
    grid, currentSuccessProb, REWARDS
  );
  let snapshots: TSnapshot[] = [];
  let globalQValueRange: ValueRange | null = null;
  let optimalActionValues: ActionValues = new Map();
  let optimalPolicy: Policy = new Map();
  let valueRMSEPoints: { index: number; value: number }[] = [];
  let policyAgreementPoints: { index: number; value: number }[] = [];

  function getFloorKeys(): string[] {
    const keys: string[] = [];
    grid.forEach((row, ri) => {
      row.forEach((cellType, ci) => {
        if (cellType === 'floor') {
          keys.push(`${String(ri)},${String(ci)}`);
        }
      });
    });
    return keys;
  }

  function deriveCellValuesFromActionValues(
    actionValues: ActionValues
  ): StateValues {
    const cellValues: StateValues = new Map();
    for (const [key, actionMap] of actionValues) {
      let maxVal = -Infinity;
      for (const value of actionMap.values()) {
        if (value > maxVal) {
          maxVal = value;
        }
      }
      cellValues.set(key, isFinite(maxVal) ? maxVal : 0);
    }
    return cellValues;
  }

  function computeRangeFromCellValues(
    cellValues: StateValues
  ): ValueRange {
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const value of cellValues.values()) {
      minVal = Math.min(minVal, value);
      maxVal = Math.max(maxVal, value);
    }
    return isFinite(minVal)
      ? { minVal, maxVal }
      : { minVal: 0, maxVal: 0 };
  }

  function computeAgreementData(): void {
    const floorKeys = getFloorKeys();
    const rmsePoints: { index: number; value: number }[] = [];
    const agreementPoints: { index: number; value: number }[] = [];
    const optimalityTolerance = 1e-9;

    for (let i = 0; i < snapshots.length; i++) {
      const snap = snapshots[i];

      // RMSE over action-values: sqrt(mean((Q_mc - Q_opt)^2))
      let sumSqErr = 0;
      let rmseCount = 0;
      for (const key of floorKeys) {
        const mcActionMap = snap.actionValues.get(key);
        const optActionMap = optimalActionValues.get(key);
        if (!mcActionMap || !optActionMap) {
          continue;
        }
        for (const action of ACTIONS) {
          const mcQ = mcActionMap.get(action) ?? 0;
          const optQ = optActionMap.get(action) ?? 0;
          sumSqErr += (mcQ - optQ) ** 2;
          rmseCount++;
        }
      }
      const rmse = rmseCount > 0 ? Math.sqrt(sumSqErr / rmseCount) : 0;
      rmsePoints.push({ index: i, value: rmse });

      // Policy agreement %
      let matching = 0;
      let agreementCount = 0;
      for (const key of floorKeys) {
        const policyAction = snap.policy.get(key);
        const optActionMap = optimalActionValues.get(key);
        if (!policyAction || !optActionMap) {
          continue;
        }
        let bestQ = -Infinity;
        for (const action of ACTIONS) {
          const q = optActionMap.get(action) ?? -Infinity;
          if (q > bestQ) {
            bestQ = q;
          }
        }
        const chosenQ = optActionMap.get(policyAction) ?? -Infinity;
        if (bestQ - chosenQ <= optimalityTolerance) {
          matching++;
        }
        agreementCount++;
      }
      const pct = agreementCount > 0
        ? (matching / agreementCount) * 100
        : 0;
      agreementPoints.push({ index: i, value: pct });
    }

    valueRMSEPoints = rmsePoints;
    policyAgreementPoints = agreementPoints;
  }

  function recomputeSnapshots(): void {
    table = buildTransitionTable(
      grid, currentSuccessProb, REWARDS
    );
    snapshots = config.computeSnapshots(
      table,
      grid,
      startRow,
      startCol,
      currentGamma,
      currentEpsilon,
      currentEpisodesPerBatch,
      TOTAL_BATCHES,
      exploringStarts,
      firstVisit,
      currentMaxSteps,
      randomPool
    );
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const snap of snapshots) {
      const cellRange = computeRangeFromCellValues(
        deriveCellValuesFromActionValues(snap.actionValues)
      );
      minVal = Math.min(minVal, cellRange.minVal);
      maxVal = Math.max(maxVal, cellRange.maxVal);
    }
    globalQValueRange = isFinite(minVal)
      ? { minVal, maxVal }
      : null;

    // Compute optimal baseline in Q-space
    const viResult = runValueIterationQ(
      table,
      initializeActionValues(grid),
      currentGamma,
      1e-8
    );
    optimalActionValues = viResult.finalActionValues;
    optimalPolicy = viResult.finalPolicy;

    computeAgreementData();
  }

  recomputeSnapshots();

  // Build DOM
  const initialSlipperiness = 1 - currentSuccessProb;
  const dom = createMonteCarloVizDom(
    config.radioNamePrefix,
    initialSlipperiness,
    currentGamma,
    currentEpsilon,
    currentEpisodesPerBatch,
    currentMaxSteps
  );
  const {
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
  } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(container);
  } else {
    parent.appendChild(container);
  }

  canvas.width = layout.cols * cellSize;
  canvas.height = layout.rows * cellSize;

  // State
  let currentIndex = 0;
  let recomputeFrameId: number | null = null;
  let isTimelineDragging = false;
  let isChartDragging = false;
  const agents: Agent[] = [];
  const effects: Effect[] = [];
  let nextAgentId = 1;
  let pendingSnapshotIndex: number | null = null;
  let colorMode: ColorMode = 'value';
  let showPolicyDisagreement = false;
  let showValues = false;

  function currentSnapshot(): TSnapshot {
    return snapshots[currentIndex];
  }

  function pruneExpiredEffects(now: number): void {
    for (let i = effects.length - 1; i >= 0; i--) {
      if (now - effects[i].startTime >= effects[i].duration) {
        effects.splice(i, 1);
      }
    }
  }

  function currentAgentPositions(timestamp: number): {
    row: number;
    col: number;
  }[] {
    return agents.flatMap((agent) => {
      const pos = getAgentPosition(
        agent, timestamp, MONTE_CARLO_AGENT_STEP_MS
      );
      return pos.finished ? [] : [{ row: pos.row, col: pos.col }];
    });
  }

  function computeActionErrorValues(
    actionValues: ActionValues
  ): ActionValues {
    const errorValues: ActionValues = new Map();
    for (const [key, actionMap] of actionValues) {
      const optimalActionMap = optimalActionValues.get(key);
      const errorMap = new Map(actionMap);
      for (const action of ACTIONS) {
        const currentQ = actionMap.get(action) ?? 0;
        const optimalQ = optimalActionMap?.get(action) ?? 0;
        errorMap.set(action, Math.abs(currentQ - optimalQ));
      }
      errorValues.set(key, errorMap);
    }
    return errorValues;
  }

  function render(timestamp: number = Date.now()): void {
    const snap = currentSnapshot();
    pruneExpiredEffects(timestamp);
    const displaySnap = pendingSnapshotIndex !== null
      ? snapshots[pendingSnapshotIndex]
      : snap;
    let displayActionValues: ActionValues;
    let displayCellValues: StateValues;
    let displayRange: ValueRange | null;

    if (colorMode === 'disagreement') {
      displayActionValues = computeActionErrorValues(
        displaySnap.actionValues
      );
      displayCellValues = deriveCellValuesFromActionValues(
        displayActionValues
      );
      displayRange = computeRangeFromCellValues(displayCellValues);
    } else {
      displayActionValues = displaySnap.actionValues;
      displayCellValues = deriveCellValuesFromActionValues(
        displayActionValues
      );
      displayRange = globalQValueRange;
    }

    config.renderGrid(
      canvas,
      grid,
      cellSize,
      displaySnap,
      currentAgentPositions(timestamp),
      effects,
      displayCellValues,
      displayActionValues,
      displayRange,
      showPolicyDisagreement ? optimalPolicy : null,
      showValues
    );

    renderChart(chartCanvas, snapshots, currentIndex);
    renderPolicyChangeChart(
      policyChartCanvas, snapshots, currentIndex
    );
    renderValueRMSEChart(
      valueRMSECanvas,
      valueRMSEPoints,
      snapshots.length,
      currentIndex
    );
    renderPolicyAgreementChart(
      policyAgreementCanvas,
      policyAgreementPoints,
      snapshots.length,
      currentIndex
    );
    renderTimelineChart(
      timelineCanvas, snapshots.length, currentIndex
    );
  }

  const animationLoop = createAnimationLoop(() => {
    const now = Date.now();
    updateAgents(agents, effects, now, MONTE_CARLO_AGENT_STEP_MS);
    pruneExpiredEffects(now);
    const shouldContinue =
      agents.length > 0 || effects.length > 0;
    if (!shouldContinue) {
      pendingSnapshotIndex = null;
    }
    render(now);
    return shouldContinue;
  });

  function clearAgentsAndEffects(): void {
    agents.length = 0;
    effects.length = 0;
    pendingSnapshotIndex = null;
    animationLoop.stop();
  }

  function spawnAgentsForBatch(snap: TSnapshot): void {
    const now = Date.now();
    for (const episode of snap.episodes) {
      if (episode.path.length < 2) {
        continue;
      }
      agents.push({
        id: nextAgentId++,
        path: episode.path,
        spawnTime: now,
        terminalType: episode.terminalType
      });
    }
    if (agents.length > 0) {
      animationLoop.start();
    }
  }

  function batchNumber(): number {
    return Math.ceil(currentIndex / 2);
  }

  function countPolicyChanges(): number {
    if (currentIndex === 0) {
      return 0;
    }
    const prev = snapshots[currentIndex - 1].policy;
    const curr = currentSnapshot().policy;
    let changed = 0;
    for (const [key, action] of prev) {
      if (curr.get(key) !== action) {
        changed++;
      }
    }
    return changed;
  }

  function getPhaseExplanation(): string {
    return config.getPhaseExplanation(
      currentSnapshot(),
      currentIndex,
      snapshots.length,
      batchNumber(),
      currentEpisodesPerBatch,
      currentEpsilon,
      exploringStarts,
      firstVisit,
      countPolicyChanges()
    );
  }

  function updateInfo(): void {
    const atEnd = currentIndex === snapshots.length - 1;
    const totalSteps = Math.max(1, snapshots.length);
    const currentStep = Math.min(
      totalSteps, currentIndex + 1
    );
    goToStartBtn.disabled = currentIndex === 0;
    stepBackBtn.disabled = currentIndex === 0;
    stepForwardBtn.disabled = atEnd;
    playBtn.textContent = playback.isPlaying()
      ? '\u23F8'
      : '\u25B6';
    playBtn.title = playback.isPlaying() ? 'Pause' : 'Play';
    stepCounterEl.textContent =
      `${String(currentStep)}/${String(totalSteps)}`;
    phaseExplanationEl.classList.toggle(
      'pi-viz-phase-explanation-hidden',
      playback.isPlaying()
    );
    phaseExplanationEl.setAttribute(
      'aria-hidden',
      playback.isPlaying() ? 'true' : 'false'
    );
    if (!playback.isPlaying()) {
      phaseExplanationEl.textContent = getPhaseExplanation();
    }
  }

  function goToIndex(index: number): void {
    clearAgentsAndEffects();
    currentIndex = Math.max(
      0, Math.min(index, snapshots.length - 1)
    );
    updateInfo();
    render();
  }

  // Playback controller
  const playback = createPlaybackController({
    getSnapshotCount: () => snapshots.length,
    getCurrentIndex: () => currentIndex,
    setCurrentIndex(i: number): void {
      currentIndex = i;
    },
    onUpdate(): void {
      updateInfo();
      render();
    }
  });

  function clamp(
    value: number, min: number, max: number
  ): number {
    return Math.min(max, Math.max(min, value));
  }

  function updateParameterLabels(): void {
    const slipperiness = 1 - currentSuccessProb;
    slipperinessValueEl.textContent =
      `${String(Math.round(slipperiness * 100))}%`;
    gammaValueEl.textContent = currentGamma.toFixed(2);
    epsilonValueEl.textContent = currentEpsilon.toFixed(2);
    episodesPerBatchValueEl.textContent =
      String(currentEpisodesPerBatch);
    maxStepsValueEl.textContent =
      String(currentMaxSteps);
  }

  function applyParameterChange(): void {
    playback.stop();
    clearAgentsAndEffects();
    const previousIndex = currentIndex;
    const wasAtEnd = previousIndex >= snapshots.length - 1;
    recomputeSnapshots();
    if (wasAtEnd) {
      currentIndex = Math.max(0, snapshots.length - 1);
    } else {
      currentIndex = Math.max(
        0, Math.min(previousIndex, snapshots.length - 1)
      );
    }
    updateInfo();
    render();
  }

  function cancelScheduledParameterRecompute(): void {
    if (recomputeFrameId !== null) {
      cancelAnimationFrame(recomputeFrameId);
      recomputeFrameId = null;
    }
  }

  function scheduleParameterRecompute(): void {
    if (recomputeFrameId !== null) {
      return;
    }
    recomputeFrameId = window.requestAnimationFrame(() => {
      recomputeFrameId = null;
      applyParameterChange();
    });
  }

  // --- Event handlers ---

  const handleGoToStart = (): void => {
    playback.stop();
    goToIndex(0);
  };

  const handleStepBack = (): void => {
    playback.stop();
    if (currentIndex > 0) {
      goToIndex(currentIndex - 1);
    }
  };

  const handlePlay = (): void => {
    playback.toggle();
    updateInfo();
    render();
  };

  const handleStepForward = (): void => {
    playback.stop();
    if (currentIndex < snapshots.length - 1) {
      clearAgentsAndEffects();
      currentIndex++;
      const snap = currentSnapshot();
      if (snap.phase === 'evaluation' && snap.episodes.length > 0) {
        pendingSnapshotIndex = currentIndex - 1;
        spawnAgentsForBatch(snap);
      }
      updateInfo();
      render(Date.now());
    }
  };

  const handleCanvasClick = (e: MouseEvent): void => {
    playback.stop();

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    if (
      row < 0 || row >= layout.rows
      || col < 0 || col >= layout.cols
      || grid[row][col] !== 'floor'
    ) {
      return;
    }

    const { path, terminalType } = precalculatePath(
      row, col, currentSnapshot().policy, table, grid
    );
    if (path.length < 2) {
      return;
    }

    agents.push({
      id: nextAgentId++,
      path,
      spawnTime: Date.now(),
      terminalType
    });
    animationLoop.start();
    render(Date.now());
  };

  const stepBackRepeat = createRepeatButton(
    stepBackBtn, handleStepBack
  );
  const stepForwardRepeat = createRepeatButton(
    stepForwardBtn, handleStepForward
  );

  const handleSlipperinessInput = (): void => {
    const slipperiness = clamp(
      Number(slipperinessSlider.value), 0, 1
    );
    slipperinessSlider.value = slipperiness.toFixed(2);
    currentSuccessProb = 1 - slipperiness;
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleGammaInput = (): void => {
    currentGamma = clamp(
      Number(gammaSlider.value), 0, 0.99
    );
    gammaSlider.value = currentGamma.toFixed(2);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleEpsilonInput = (): void => {
    currentEpsilon = clamp(
      Number(epsilonSlider.value), 0, 1
    );
    epsilonSlider.value = currentEpsilon.toFixed(2);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleEpisodesInput = (): void => {
    currentEpisodesPerBatch = clamp(
      Math.round(Number(episodesPerBatchSlider.value)), 1, 50
    );
    episodesPerBatchSlider.value =
      String(currentEpisodesPerBatch);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleMaxStepsInput = (): void => {
    currentMaxSteps = clamp(
      Math.round(Number(maxStepsSlider.value) / 10) * 10,
      10, 1000
    );
    maxStepsSlider.value = String(currentMaxSteps);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleStartModeChange = (): void => {
    exploringStarts = startModeExploringRadio.checked;
    scheduleParameterRecompute();
  };

  const handleVisitModeChange = (): void => {
    firstVisit = visitModeFirstRadio.checked;
    scheduleParameterRecompute();
  };

  const handleResample = (): void => {
    randomPool = generateRandomPool();
    applyParameterChange();
  };

  const handleColorModeChange = (): void => {
    colorMode = colorModeDisagreementRadio.checked
      ? 'disagreement'
      : 'value';
    render();
  };

  const handleDisagreementChange = (): void => {
    showPolicyDisagreement = showDisagreementCheckbox.checked;
    render();
  };

  const handleShowValuesChange = (): void => {
    showValues = showValuesCheckbox.checked;
    render();
  };

  const scrubTimeline = (clientX: number): void => {
    const rect = timelineCanvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const index = chartXToNearestIndex(
      localX, timelineCanvas.clientWidth, snapshots.length
    );
    goToIndex(index);
  };

  const scrubChart = (
    chart: HTMLCanvasElement, clientX: number
  ): void => {
    const rect = chart.getBoundingClientRect();
    const localX = clientX - rect.left;
    const index = chartXToNearestIndex(
      localX, chart.clientWidth, snapshots.length
    );
    goToIndex(index);
  };

  const handleTimelinePointerDown = (
    e: PointerEvent
  ): void => {
    playback.stop();
    isTimelineDragging = true;
    timelineCanvas.setPointerCapture(e.pointerId);
    scrubTimeline(e.clientX);
  };

  const handleTimelinePointerMove = (
    e: PointerEvent
  ): void => {
    if (!isTimelineDragging) {
      return;
    }
    scrubTimeline(e.clientX);
  };

  const handleTimelinePointerUp = (
    e: PointerEvent
  ): void => {
    if (!isTimelineDragging) {
      return;
    }
    isTimelineDragging = false;
    if (timelineCanvas.hasPointerCapture(e.pointerId)) {
      timelineCanvas.releasePointerCapture(e.pointerId);
    }
  };

  const handleChartPointerDown = (
    e: PointerEvent
  ): void => {
    playback.stop();
    isChartDragging = true;
    const chart = e.currentTarget as HTMLCanvasElement;
    chart.setPointerCapture(e.pointerId);
    scrubChart(chart, e.clientX);
  };

  const handleChartPointerMove = (
    e: PointerEvent
  ): void => {
    if (!isChartDragging) {
      return;
    }
    const chart = e.currentTarget as HTMLCanvasElement;
    scrubChart(chart, e.clientX);
  };

  const handleChartPointerUp = (
    e: PointerEvent
  ): void => {
    if (!isChartDragging) {
      return;
    }
    isChartDragging = false;
    const chart = e.currentTarget as HTMLCanvasElement;
    if (chart.hasPointerCapture(e.pointerId)) {
      chart.releasePointerCapture(e.pointerId);
    }
  };

  // Wire up all events
  const teardownEvents = wireEvents([
    [resampleBtn, 'click', handleResample],
    [goToStartBtn, 'click', handleGoToStart],
    [stepBackBtn, 'pointerdown', stepBackRepeat.handlePointerDown as EventListener],
    [stepBackBtn, 'pointerup', stepBackRepeat.handlePointerUp as EventListener],
    [stepBackBtn, 'pointercancel', stepBackRepeat.handlePointerUp as EventListener],
    [stepBackBtn, 'click', stepBackRepeat.handleClick],
    [playBtn, 'click', handlePlay],
    [stepForwardBtn, 'pointerdown', stepForwardRepeat.handlePointerDown as EventListener],
    [stepForwardBtn, 'pointerup', stepForwardRepeat.handlePointerUp as EventListener],
    [stepForwardBtn, 'pointercancel', stepForwardRepeat.handlePointerUp as EventListener],
    [stepForwardBtn, 'click', stepForwardRepeat.handleClick],
    [slipperinessSlider, 'input', handleSlipperinessInput],
    [gammaSlider, 'input', handleGammaInput],
    [epsilonSlider, 'input', handleEpsilonInput],
    [episodesPerBatchSlider, 'input', handleEpisodesInput],
    [maxStepsSlider, 'input', handleMaxStepsInput],
    [visitModeFirstRadio, 'change', handleVisitModeChange],
    [visitModeEveryRadio, 'change', handleVisitModeChange],
    [startModeFixedRadio, 'change', handleStartModeChange],
    [startModeExploringRadio, 'change', handleStartModeChange],
    [colorModeValueRadio, 'change', handleColorModeChange],
    [colorModeDisagreementRadio, 'change', handleColorModeChange],
    [showDisagreementCheckbox, 'change', handleDisagreementChange],
    [showValuesCheckbox, 'change', handleShowValuesChange],
    [timelineCanvas, 'pointerdown', handleTimelinePointerDown as EventListener],
    [timelineCanvas, 'pointermove', handleTimelinePointerMove as EventListener],
    [timelineCanvas, 'pointerup', handleTimelinePointerUp as EventListener],
    [timelineCanvas, 'pointercancel', handleTimelinePointerUp as EventListener],
    [chartCanvas, 'pointerdown', handleChartPointerDown as EventListener],
    [chartCanvas, 'pointermove', handleChartPointerMove as EventListener],
    [chartCanvas, 'pointerup', handleChartPointerUp as EventListener],
    [chartCanvas, 'pointercancel', handleChartPointerUp as EventListener],
    [policyChartCanvas, 'pointerdown', handleChartPointerDown as EventListener],
    [policyChartCanvas, 'pointermove', handleChartPointerMove as EventListener],
    [policyChartCanvas, 'pointerup', handleChartPointerUp as EventListener],
    [policyChartCanvas, 'pointercancel', handleChartPointerUp as EventListener],
    [valueRMSECanvas, 'pointerdown', handleChartPointerDown as EventListener],
    [valueRMSECanvas, 'pointermove', handleChartPointerMove as EventListener],
    [valueRMSECanvas, 'pointerup', handleChartPointerUp as EventListener],
    [valueRMSECanvas, 'pointercancel', handleChartPointerUp as EventListener],
    [policyAgreementCanvas, 'pointerdown', handleChartPointerDown as EventListener],
    [policyAgreementCanvas, 'pointermove', handleChartPointerMove as EventListener],
    [policyAgreementCanvas, 'pointerup', handleChartPointerUp as EventListener],
    [policyAgreementCanvas, 'pointercancel', handleChartPointerUp as EventListener],
    [canvas, 'click', handleCanvasClick as EventListener]
  ]);

  // Initial render
  updateParameterLabels();
  updateInfo();
  render();

  return {
    destroy(): void {
      playback.destroy();
      cancelScheduledParameterRecompute();
      clearAgentsAndEffects();
      stepBackRepeat.destroy();
      stepForwardRepeat.destroy();
      teardownEvents();
      container.remove();
    }
  };
}

export function initMonteCarloVisualization(
  parent: HTMLElement
): MonteCarloVisualization {
  return initMonteCarloCoreVisualization<MonteCarloSnapshot>(parent, {
    radioNamePrefix: 'mc',

    computeSnapshots: runMonteCarlo,

    renderGrid(
      canvas: HTMLCanvasElement,
      grid: Grid,
      cellSize: number,
      snapshot: MonteCarloSnapshot,
      agents: AgentPosition[],
      effects: Effect[],
      displayCellValues: StateValues,
      displayActionValues: ActionValues,
      displayRange: ValueRange | null,
      optimalPolicy: Policy | null,
      showValues: boolean
    ): void {
      renderGrid(
        canvas,
        grid,
        cellSize,
        displayCellValues,
        displayActionValues,
        snapshot.policy,
        agents,
        effects,
        true,
        false,
        showValues,
        displayRange,
        optimalPolicy
      );
    },

    getPhaseExplanation(
      snapshot: MonteCarloSnapshot,
      currentIndex: number,
      totalSnapshots: number,
      batchNumber: number,
      episodesPerBatch: number,
      epsilon: number,
      exploringStarts: boolean,
      firstVisit: boolean,
      policyChanges: number
    ): string {
      const atStart = currentIndex === 0;
      const atEnd = currentIndex === totalSnapshots - 1;

      if (atStart) {
        return [
          'Initial state: All Q-values = 0, policy is arbitrary.',
          'Step forward to generate the first batch of',
          String(episodesPerBatch),
          `episodes following an \u03B5-greedy policy (\u03B5=${epsilon.toFixed(2)}).`
        ].join(' ');
      }

      if (atEnd) {
        return [
          `After ${String(snapshot.totalEpisodes)} total episodes,`,
          'Q-values have stabilized.',
          'The greedy policy is derived as',
          'argmax_a Q(s,a) \u2014 fully model-free.'
        ].join(' ');
      }

      if (snapshot.phase === 'evaluation') {
        const startDesc = exploringStarts
          ? 'random states'
          : '(0,0)';
        const visitDesc = firstVisit
          ? 'first-visit'
          : 'every-visit';
        return [
          `Batch ${String(batchNumber)} \u2014 Evaluation:`,
          `Generated ${String(snapshot.episodes.length)}`,
          `\u03B5-greedy episodes from ${startDesc},`,
          `updated Q(s,a) using ${visitDesc} returns.`,
          `Max Q-value change: ${snapshot.delta.toFixed(4)}.`
        ].join(' ');
      }

      return [
        `Batch ${String(batchNumber)} \u2014 Improvement:`,
        'Derived greedy policy as argmax_a Q(s,a).',
        `${String(policyChanges)} action${policyChanges === 1 ? '' : 's'} changed.`
      ].join(' ');
    }
  });
}
