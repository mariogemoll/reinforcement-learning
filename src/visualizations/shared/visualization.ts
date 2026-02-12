// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ANIMATION } from '../../config/constants';
import { createGridFromLayout } from '../../core/grid';
import { buildTransitionTable } from '../../core/mdp';
import { generateRandomPolicy } from '../../core/policy';
import {
  createAnimationLoop,
  getAgentPosition,
  precalculatePath,
  updateAgents
} from './animation';
import {
  chartXToNearestIndex,
  computeRoundValueRanges,
  renderChart,
  renderPolicyChangeChart,
  renderTimelineChart
} from './charts';
import { wireEvents } from './event-wiring';
import { createPlaybackController, createRepeatButton } from './playback';
import type {
  Agent,
  BaseSnapshot,
  DPVisualization,
  DPVisualizationConfig,
  Effect,
  InitParams,
  ValueRange
} from './types';
import { createDPVizDom } from './ui';

export function initializeDPVisualization<
  TSnapshot extends BaseSnapshot
>(
  config: DPVisualizationConfig<TSnapshot>,
  params: InitParams
): DPVisualization {
  const {
    parent,
    layout,
    cellSize,
    successProb: initialSuccessProb,
    rewardModel,
    gamma: initialGamma,
    theta: initialTheta
  } = params;

  const grid = createGridFromLayout(layout);
  const initialPolicy = generateRandomPolicy(grid);
  let initialValueMode: 'zero' | 'random' = 'zero';
  config.createZeroValues(grid);
  let currentSuccessProb = initialSuccessProb;
  let currentGamma = initialGamma;
  let currentTheta = initialTheta;
  let table = buildTransitionTable(
    grid, currentSuccessProb, rewardModel
  );
  let snapshots: TSnapshot[] = [];
  let roundValueRanges = computeRoundValueRanges(grid, snapshots);
  let snapshotRoundIndex: number[] = [];

  function recomputeSnapshots(): void {
    table = buildTransitionTable(
      grid,
      currentSuccessProb,
      rewardModel
    );

    snapshots = config.computeSnapshots(
      table,
      initialPolicy,
      currentGamma,
      currentTheta
    );

    roundValueRanges = computeRoundValueRanges(
      grid,
      snapshots
    );
    snapshotRoundIndex = [];
    let round = 0;
    for (const snap of snapshots) {
      snapshotRoundIndex.push(round);
      if (snap.phase === 'improvement') {
        round++;
      }
    }
  }

  recomputeSnapshots();

  // Build DOM
  const initialSlipperiness = 1 - currentSuccessProb;
  const dom = createDPVizDom(
    initialSlipperiness,
    currentGamma,
    currentTheta,
    initialValueMode,
    config.initialValuesLabel,
    config.radioGroupName,
    config.extraCheckboxes
  );
  const {
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
        agent,
        timestamp,
        ANIMATION.agentStepMs
      );
      return pos.finished ? [] : [{ row: pos.row, col: pos.col }];
    });
  }

  function render(timestamp: number = Date.now()): void {
    const snap = currentSnapshot();
    const roundIndex = snapshotRoundIndex[currentIndex] ?? 0;
    const valueRange: ValueRange | null =
      roundValueRanges[roundIndex] ?? null;
    pruneExpiredEffects(timestamp);

    config.renderGrid(
      canvas,
      grid,
      cellSize,
      snap,
      currentAgentPositions(timestamp),
      effects,
      valueRange
    );

    renderChart(chartCanvas, snapshots, currentIndex);
    renderPolicyChangeChart(
      policyChartCanvas, snapshots, currentIndex
    );
    renderTimelineChart(
      timelineCanvas, snapshots.length, currentIndex
    );
  }

  const animationLoop = createAnimationLoop(() => {
    const now = Date.now();
    updateAgents(
      agents,
      effects,
      now,
      ANIMATION.agentStepMs
    );
    pruneExpiredEffects(now);
    render(now);
    return agents.length > 0 || effects.length > 0;
  });

  function clearAgentsAndEffects(): void {
    agents.length = 0;
    effects.length = 0;
    animationLoop.stop();
  }

  function updateInfo(): void {
    const atEnd =
      currentIndex === snapshots.length - 1;
    const totalSteps = Math.max(1, snapshots.length);
    const currentStep = Math.min(
      totalSteps,
      currentIndex + 1
    );
    goToStartBtn.disabled = currentIndex === 0;
    resetBtn.disabled = false;
    stepBackBtn.disabled = currentIndex === 0;
    stepForwardBtn.disabled = atEnd;
    playBtn.textContent = playback.isPlaying()
      ? '\u23F8'
      : '\u25B6';
    playBtn.title = playback.isPlaying() ? 'Pause' : 'Play';
    stepCounterEl.textContent = `${String(currentStep)}/${String(totalSteps)}`;
    phaseExplanationEl.classList.toggle(
      'pi-viz-phase-explanation-hidden',
      playback.isPlaying()
    );
    phaseExplanationEl.setAttribute(
      'aria-hidden',
      playback.isPlaying() ? 'true' : 'false'
    );
    if (!playback.isPlaying()) {
      phaseExplanationEl.textContent =
        getPhaseExplanationText();
    }
  }

  function getPhaseExplanationText(): string {
    const snap = currentSnapshot();
    const hasNext = currentIndex + 1 < snapshots.length;
    const nextPhase = hasNext
      ? snapshots[currentIndex + 1].phase
      : null;
    const atStart = currentIndex === 0;
    const atEnd = currentIndex === snapshots.length - 1;
    return config.getPhaseExplanation(
      snap, nextPhase, atStart, atEnd, initialValueMode
    );
  }

  function goToIndex(index: number): void {
    clearAgentsAndEffects();
    currentIndex = Math.max(
      0,
      Math.min(index, snapshots.length - 1)
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

  function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  function updateParameterLabels(): void {
    const slipperiness = 1 - currentSuccessProb;
    slipperinessValueEl.textContent = `${String(Math.round(slipperiness * 100))}%`;
    gammaValueEl.textContent = currentGamma.toFixed(2);
    thetaValueEl.textContent = currentTheta.toFixed(3);
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
        0,
        Math.min(previousIndex, snapshots.length - 1)
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

  function hardReset(): void {
    cancelScheduledParameterRecompute();
    playback.stop();
    clearAgentsAndEffects();
    const previousIndex = currentIndex;
    const wasAtEnd = previousIndex >= snapshots.length - 1;
    if (initialValueMode === 'zero') {
      config.createZeroValues(grid);
    } else {
      config.randomizeValues();
    }
    recomputeSnapshots();
    if (wasAtEnd) {
      currentIndex = Math.max(0, snapshots.length - 1);
    } else {
      currentIndex = Math.max(
        0,
        Math.min(previousIndex, snapshots.length - 1)
      );
    }
    updateInfo();
    render();
  }

  // --- Event handlers ---

  const handleReset = (): void => {
    hardReset();
  };

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
      goToIndex(currentIndex + 1);
    }
  };

  // Repeat buttons
  const stepBackRepeat = createRepeatButton(
    stepBackBtn, handleStepBack
  );
  const stepForwardRepeat = createRepeatButton(
    stepForwardBtn, handleStepForward
  );

  const handleSlipperinessInput = (): void => {
    const slipperiness = clamp(
      Number(slipperinessSlider.value),
      0,
      1
    );
    slipperinessSlider.value = slipperiness.toFixed(2);
    currentSuccessProb = 1 - slipperiness;
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleGammaInput = (): void => {
    currentGamma = clamp(
      Number(gammaSlider.value),
      0,
      0.99
    );
    gammaSlider.value = currentGamma.toFixed(2);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleThetaInput = (): void => {
    currentTheta = clamp(
      Number(thetaSlider.value),
      0.001,
      0.1
    );
    thetaSlider.value = currentTheta.toFixed(3);
    updateParameterLabels();
    scheduleParameterRecompute();
  };

  const handleInitialValuesChange = (): void => {
    const nextMode = initialValuesRandomRadio.checked
      ? 'random'
      : 'zero';
    if (nextMode === initialValueMode) {
      return;
    }
    initialValueMode = nextMode;
    initialValuesZeroRadio.checked =
      initialValueMode === 'zero';
    initialValuesRandomRadio.checked =
      initialValueMode === 'random';
    hardReset();
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
      row,
      col,
      currentSnapshot().policy,
      table,
      grid
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

  const scrubTimeline = (clientX: number): void => {
    const rect = timelineCanvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const index = chartXToNearestIndex(
      localX, timelineCanvas.clientWidth, snapshots.length
    );
    goToIndex(index);
  };

  const scrubChart = (
    chart: HTMLCanvasElement,
    clientX: number
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

  // Wire extra checkbox change handlers
  const extraCheckboxHandlers: (() => void)[] = [];
  for (let i = 0; i < config.extraCheckboxes.length; i++) {
    const cb = config.extraCheckboxes[i];
    const input = extraCheckboxInputs[i];
    const handler = (): void => {
      cb.onChange(input.checked);
      render();
    };
    extraCheckboxHandlers.push(handler);
  }

  // Wire up all events
  const extraCheckboxEntries: [EventTarget, string, EventListener][] =
    extraCheckboxInputs.map((input, i) =>
      [input, 'change', extraCheckboxHandlers[i]]
    );

  const teardownEvents = wireEvents([
    [goToStartBtn, 'click', handleGoToStart],
    [resetBtn, 'click', handleReset],
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
    [thetaSlider, 'input', handleThetaInput],
    [initialValuesZeroRadio, 'change', handleInitialValuesChange],
    [initialValuesRandomRadio, 'change', handleInitialValuesChange],
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
    [canvas, 'click', handleCanvasClick as EventListener],
    ...extraCheckboxEntries
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
