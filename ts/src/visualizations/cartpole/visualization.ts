// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { DqnPolicy } from '../../cartpole/dqn-policy';
import {
  ANGLE_LIMIT,
  createCartPoleEnvironment,
  POSITION_LIMIT } from '../../cartpole/environment';
import type { CartPoleAction, CartPoleState } from '../../cartpole/types';
import {
  chartXForIndex,
  chartXToNearestIndex,
  configureCanvas,
  drawCurrentTimestepGuide,
  renderTimelineChart
} from '../shared/charts';
import { wireEvents } from '../shared/event-wiring';
import { createRepeatButton } from '../shared/playback';
import { CANVAS_HEIGHT, CANVAS_WIDTH, renderCartPole } from './renderer';
import { createCartPoleVizDom } from './ui';

export interface CartPoleVisualization {
  destroy(): void;
}

// How many ticks to step per interval at speed=1
const BASE_TICK_MS = 40;
const ACTION_CHART_PAD = { top: 2, right: 20, bottom: 6, left: 60 };
const ACTION_BAND_TOP = 0.08;
const ACTION_BAND_BOTTOM = 0.92;
const MAX_TRAJECTORY_STEPS = 2000;
const MIN_SPEED = 0.125;
const MAX_SPEED = 8;
const SLIDER_MAX = 100;

interface TrajectoryFrame {
  state: CartPoleState;
  action: CartPoleAction;
  terminated: boolean;
  truncated: boolean;
}

function cloneState(
  state: Readonly<CartPoleState>
): CartPoleState {
  return [state[0], state[1], state[2], state[3]];
}

function renderActionChart(
  chartCanvas: HTMLCanvasElement,
  frames: readonly TrajectoryFrame[],
  currentIndex: number
): void {
  const configured = configureCanvas(chartCanvas);
  if (!configured) {
    return;
  }
  const { ctx, w, h } = configured;
  const ch = h - ACTION_CHART_PAD.top - ACTION_CHART_PAD.bottom;
  const centerY = ACTION_CHART_PAD.top + ch / 2;
  const leftY = ACTION_CHART_PAD.top + ch * ACTION_BAND_TOP;
  const rightY = ACTION_CHART_PAD.top + ch * ACTION_BAND_BOTTOM;
  const leftLabelY = (centerY + leftY) / 2;
  const rightLabelY = (centerY + rightY) / 2;

  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ACTION_CHART_PAD.left, leftY);
  ctx.lineTo(w - ACTION_CHART_PAD.right, leftY);
  ctx.moveTo(ACTION_CHART_PAD.left, rightY);
  ctx.lineTo(w - ACTION_CHART_PAD.right, rightY);
  ctx.stroke();

  ctx.fillStyle = '#555';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('Left', ACTION_CHART_PAD.left - 10, leftLabelY);
  ctx.fillText('Right', ACTION_CHART_PAD.left - 10, rightLabelY);

  const yForAction = (action: CartPoleAction): number =>
    action === 0 ? leftY : rightY;

  const visibleCount = Math.min(
    frames.length,
    currentIndex + 1
  );

  for (let i = 0; i < visibleCount; i++) {
    const frame = frames[i];
    const x = chartXForIndex(i, frames.length, w);
    const prevX = i > 0
      ? chartXForIndex(i - 1, frames.length, w)
      : ACTION_CHART_PAD.left;
    const nextX = i < frames.length - 1
      ? chartXForIndex(i + 1, frames.length, w)
      : w - ACTION_CHART_PAD.right;
    const leftBound = i === 0
      ? ACTION_CHART_PAD.left
      : (prevX + x) / 2;
    const rightBound = i === frames.length - 1
      ? w - ACTION_CHART_PAD.right
      : (x + nextX) / 2;
    const x0 = Math.floor(leftBound);
    const x1 = Math.ceil(rightBound);
    const y = yForAction(frame.action);
    const top = Math.min(centerY, y);
    const height = Math.abs(y - centerY);
    ctx.fillStyle = '#d1d5db';
    ctx.fillRect(x0, top, Math.max(1, x1 - x0), height);
  }

  drawCurrentTimestepGuide(
    ctx,
    w,
    h,
    frames.length,
    currentIndex
  );
}

function sliderToSpeed(
  sliderValue: number
): number {
  const t = Math.max(0, Math.min(1, sliderValue / SLIDER_MAX));
  const minLog = Math.log(MIN_SPEED);
  const maxLog = Math.log(MAX_SPEED);
  return Math.exp(minLog + t * (maxLog - minLog));
}

function formatSpeed(speed: number): string {
  if (speed >= 1) {
    return `${speed.toFixed(1).replace(/\.0$/, '')}x`;
  }
  return `${speed.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')}x`;
}

export function initializeCartPoleVisualization(
  parent: HTMLElement,
  runDqnPolicy: DqnPolicy
): CartPoleVisualization {
  const env = createCartPoleEnvironment({
    angleLimit: ANGLE_LIMIT,
    positionLimit: POSITION_LIMIT
  });
  const dom = createCartPoleVizDom();
  const {
    container,
    canvas,
    actionChartCanvas,
    timelineCanvas,
    resetBtn,
    speedSlider,
    speedValueEl,
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

  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let episodes = 0;
  let trajectory: TrajectoryFrame[] = [];
  let currentIndex = 0;
  let playing = false;
  let playbackTimerId: number | null = null;
  let isTimelineDragging = false;
  let isChartDragging = false;
  let speed = 1;

  const currentFrame = (): TrajectoryFrame =>
    trajectory[currentIndex];

  const atEnd = (): boolean =>
    currentIndex >= trajectory.length - 1;

  const stepDelayMs = (): number =>
    Math.max(12, BASE_TICK_MS / Math.max(0.001, speed));

  const stopPlayback = (): void => {
    if (playbackTimerId !== null) {
      window.clearTimeout(playbackTimerId);
      playbackTimerId = null;
    }
    playing = false;
  };

  const updateStats = (): void => {
    const state = currentFrame().state;
    dom.episodesValue.textContent = String(episodes);
    dom.stepsValue.textContent = String(currentIndex);
    dom.positionValue.textContent = state[0].toFixed(2);
    dom.velocityValue.textContent = state[1].toFixed(2);
    dom.angleValue.textContent = (state[2] * 180 / Math.PI).toFixed(1) + '\u00b0';
    dom.angularVelocityValue.textContent = state[3].toFixed(2);
  };

  const updateOverlay = (): void => {
    if (!atEnd()) {
      dom.terminalOverlay.hidden = true;
      dom.terminalOverlay.classList.remove('is-success', 'is-failure');
      return;
    }
    const frame = currentFrame();
    const success = frame.truncated && !frame.terminated;
    dom.terminalOverlay.hidden = false;
    dom.terminalOverlay.classList.toggle('is-success', success);
    dom.terminalOverlay.classList.toggle('is-failure', !success);
    dom.terminalTitle.textContent = success
      ? 'Trajectory complete'
      : 'Trajectory ended';
    dom.terminalSummary.textContent = `Length: ${String(trajectory.length - 1)} steps`;
  };

  const updateControls = (): void => {
    goToStartBtn.disabled = currentIndex === 0;
    stepBackBtn.disabled = currentIndex === 0;
    stepForwardBtn.disabled = atEnd();
    playBtn.textContent = playing ? '\u23F8' : '\u25B6';
    playBtn.title = playing ? 'Pause' : 'Play';
    stepCounterEl.textContent = `${String(currentIndex + 1)}/${String(trajectory.length)}`;
  };

  const render = (): void => {
    updateStats();
    updateOverlay();
    renderCartPole(
      canvas,
      currentFrame().state
    );
    renderActionChart(
      actionChartCanvas,
      trajectory,
      currentIndex
    );
    renderTimelineChart(
      timelineCanvas,
      trajectory.length,
      currentIndex
    );
  };

  const goToIndex = (index: number): void => {
    currentIndex = Math.max(
      0,
      Math.min(index, trajectory.length - 1)
    );
    updateControls();
    render();
  };

  const schedulePlaybackStep = (): void => {
    if (!playing) {
      return;
    }
    if (atEnd()) {
      generateTrajectory(true);
      return;
    }

    playbackTimerId = window.setTimeout(() => {
      playbackTimerId = null;
      if (!playing) {
        return;
      }
      if (!atEnd()) {
        currentIndex++;
        updateControls();
        render();
        schedulePlaybackStep();
      } else {
        stopPlayback();
        updateControls();
        render();
      }
    }, stepDelayMs());
  };

  const startPlayback = (): void => {
    if (playing) {
      return;
    }
    if (atEnd()) {
      currentIndex = 0;
    }
    playing = true;
    updateControls();
    render();
    schedulePlaybackStep();
  };

  const togglePlayback = (): void => {
    if (playing) {
      stopPlayback();
      updateControls();
      render();
      return;
    }
    startPlayback();
  };

  const buildTrajectory = (): TrajectoryFrame[] => {
    const frames: TrajectoryFrame[] = [];
    env.reset();
    let action = runDqnPolicy(env.getState()).action;
    frames.push({
      state: cloneState(env.getState()),
      action,
      terminated: false,
      truncated: false
    });

    for (let i = 0; i < MAX_TRAJECTORY_STEPS; i++) {
      const result = env.step(action);
      action = runDqnPolicy(result.state).action;
      frames.push({
        state: cloneState(result.state),
        action,
        terminated: result.terminated,
        truncated: result.truncated
      });
      if (result.terminated || result.truncated) {
        break;
      }
    }

    return frames;
  };

  const generateTrajectory = (
    autoplay: boolean
  ): void => {
    stopPlayback();
    trajectory = buildTrajectory();
    episodes++;
    currentIndex = 0;
    if (autoplay) {
      playing = true;
      schedulePlaybackStep();
    }
    updateControls();
    render();
  };

  const handleGoToStart = (): void => {
    stopPlayback();
    goToIndex(0);
  };

  const handleStepBack = (): void => {
    stopPlayback();
    if (currentIndex > 0) {
      goToIndex(currentIndex - 1);
    }
  };

  const handlePlay = (): void => {
    togglePlayback();
  };

  const handleStepForward = (): void => {
    stopPlayback();
    if (!atEnd()) {
      goToIndex(currentIndex + 1);
    }
  };

  const handleReset = (): void => {
    generateTrajectory(true);
  };

  const handleSpeedInput = (): void => {
    speed = sliderToSpeed(Number(speedSlider.value));
    speedValueEl.textContent = formatSpeed(speed);
    if (playing) {
      if (playbackTimerId !== null) {
        window.clearTimeout(playbackTimerId);
        playbackTimerId = null;
      }
      schedulePlaybackStep();
    }
  };

  const scrubTimeline = (clientX: number): void => {
    const rect = timelineCanvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const index = chartXToNearestIndex(
      localX,
      timelineCanvas.clientWidth,
      trajectory.length
    );
    goToIndex(index);
  };

  const scrubActionChart = (clientX: number): void => {
    const rect = actionChartCanvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const index = chartXToNearestIndex(
      localX,
      actionChartCanvas.clientWidth,
      trajectory.length
    );
    goToIndex(index);
  };

  const handleTimelinePointerDown = (
    e: PointerEvent
  ): void => {
    stopPlayback();
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

  const handleActionChartPointerDown = (
    e: PointerEvent
  ): void => {
    stopPlayback();
    isChartDragging = true;
    actionChartCanvas.setPointerCapture(e.pointerId);
    scrubActionChart(e.clientX);
  };

  const handleActionChartPointerMove = (
    e: PointerEvent
  ): void => {
    if (!isChartDragging) {
      return;
    }
    scrubActionChart(e.clientX);
  };

  const handleActionChartPointerUp = (
    e: PointerEvent
  ): void => {
    if (!isChartDragging) {
      return;
    }
    isChartDragging = false;
    if (actionChartCanvas.hasPointerCapture(e.pointerId)) {
      actionChartCanvas.releasePointerCapture(e.pointerId);
    }
  };

  const stepBackRepeat = createRepeatButton(
    stepBackBtn,
    handleStepBack
  );
  const stepForwardRepeat = createRepeatButton(
    stepForwardBtn,
    handleStepForward
  );

  const teardownEvents = wireEvents([
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
    [resetBtn, 'click', handleReset],
    [speedSlider, 'input', handleSpeedInput],
    [timelineCanvas, 'pointerdown', handleTimelinePointerDown as EventListener],
    [timelineCanvas, 'pointermove', handleTimelinePointerMove as EventListener],
    [timelineCanvas, 'pointerup', handleTimelinePointerUp as EventListener],
    [timelineCanvas, 'pointercancel', handleTimelinePointerUp as EventListener],
    [actionChartCanvas, 'pointerdown', handleActionChartPointerDown as EventListener],
    [actionChartCanvas, 'pointermove', handleActionChartPointerMove as EventListener],
    [actionChartCanvas, 'pointerup', handleActionChartPointerUp as EventListener],
    [actionChartCanvas, 'pointercancel', handleActionChartPointerUp as EventListener]
  ]);

  // Initial render
  speed = sliderToSpeed(Number(speedSlider.value));
  speedValueEl.textContent = formatSpeed(speed);
  generateTrajectory(true);

  return {
    destroy(): void {
      stopPlayback();
      stepBackRepeat.destroy();
      stepForwardRepeat.destroy();
      teardownEvents();
      container.remove();
    }
  };
}
