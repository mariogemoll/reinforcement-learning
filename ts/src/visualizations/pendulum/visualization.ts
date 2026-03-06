// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createPendulumEnvironment } from '../../pendulum/environment';
import type { PendulumPolicy } from '../../pendulum/policy';
import type { PendulumObs, PendulumState } from '../../pendulum/types';
import {
  chartXToNearestIndex,
  configureCanvas,
  drawCurrentTimestepGuide,
  renderTimelineChart
} from '../shared/charts';
import { wireEvents } from '../shared/event-wiring';
import { createRepeatButton } from '../shared/playback';
import { CANVAS_SIZE, renderPendulum } from './renderer';
import { createPendulumVizDom } from './ui';

export interface PendulumVisualization {
  destroy(): void;
}

const BASE_TICK_MS = 50; // Pendulum DT=0.05s → 50ms at 1x
const TORQUE_CHART_PAD = { top: 4, right: 20, bottom: 4, left: 44 };
const MIN_SPEED = 0.125;
const MAX_SPEED = 8;
const SLIDER_MAX = 100;

interface TrajectoryFrame {
  state: PendulumState;
  obs: PendulumObs;
  torque: number;
  reward: number;
  cumReturn: number;
  truncated: boolean;
}

function renderTorqueChart(
  chartCanvas: HTMLCanvasElement,
  frames: readonly TrajectoryFrame[],
  currentIndex: number
): void {
  const configured = configureCanvas(chartCanvas);
  if (!configured) { return; }
  const { ctx, w, h } = configured;
  const pad = TORQUE_CHART_PAD;
  const ch = h - pad.top - pad.bottom;
  const cw = w - pad.left - pad.right;
  const centerY = pad.top + ch / 2;

  // Y-axis labels
  ctx.fillStyle = '#555';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('+2', pad.left - 4, pad.top);
  ctx.fillText('0', pad.left - 4, centerY);
  ctx.fillText('\u22122', pad.left - 4, pad.top + ch);

  // Zero line
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, centerY);
  ctx.lineTo(w - pad.right, centerY);
  ctx.stroke();

  const torqueToY = (t: number): number =>
    centerY - (t / 2) * (ch / 2);

  const visibleCount = Math.min(frames.length, currentIndex + 1);
  if (visibleCount < 1) { return; }

  // Filled area
  ctx.fillStyle = 'rgba(37, 99, 235, 0.15)';
  ctx.beginPath();
  for (let i = 0; i < visibleCount; i++) {
    const x = pad.left + (cw * i) / Math.max(1, frames.length - 1);
    const y = torqueToY(frames[i].torque);
    if (i === 0) { ctx.moveTo(x, centerY); ctx.lineTo(x, y); }
    else { ctx.lineTo(x, y); }
  }
  const lastX = pad.left + (cw * (visibleCount - 1)) / Math.max(1, frames.length - 1);
  ctx.lineTo(lastX, centerY);
  ctx.closePath();
  ctx.fill();

  // Line
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < visibleCount; i++) {
    const x = pad.left + (cw * i) / Math.max(1, frames.length - 1);
    const y = torqueToY(frames[i].torque);
    if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
  }
  ctx.stroke();

  drawCurrentTimestepGuide(ctx, w, h, frames.length, currentIndex);
}

function sliderToSpeed(sliderValue: number): number {
  const t = Math.max(0, Math.min(1, sliderValue / SLIDER_MAX));
  return Math.exp(Math.log(MIN_SPEED) + t * (Math.log(MAX_SPEED) - Math.log(MIN_SPEED)));
}

function formatSpeed(speed: number): string {
  if (speed >= 1) { return `${speed.toFixed(1).replace(/\.0$/, '')}x`; }
  return `${speed.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')}x`;
}

export function initializePendulumVisualization(
  parent: HTMLElement,
  policy: PendulumPolicy
): PendulumVisualization {
  const env = createPendulumEnvironment();
  const dom = createPendulumVizDom();
  const {
    container, canvas, torqueChartCanvas, timelineCanvas,
    resetBtn, speedSlider, speedValueEl,
    goToStartBtn, stepBackBtn, playBtn, stepForwardBtn, stepCounterEl
  } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) { placeholder.replaceWith(container); }
  else { parent.appendChild(container); }

  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;

  let episodes = 0;
  let trajectory: TrajectoryFrame[] = [];
  let currentIndex = 0;
  let playing = false;
  let playbackTimerId: number | null = null;
  let isTimelineDragging = false;
  let isChartDragging = false;
  let speed = 1;

  const currentFrame = (): TrajectoryFrame => trajectory[currentIndex];
  const atEnd = (): boolean => currentIndex >= trajectory.length - 1;
  const stepDelayMs = (): number => Math.max(12, BASE_TICK_MS / Math.max(0.001, speed));

  const stopPlayback = (): void => {
    if (playbackTimerId !== null) { window.clearTimeout(playbackTimerId); playbackTimerId = null; }
    playing = false;
  };

  const updateStats = (): void => {
    const frame = currentFrame();
    const thetaDeg = (frame.state.theta * 180 / Math.PI).toFixed(1);
    dom.episodesValue.textContent = String(episodes);
    dom.stepsValue.textContent = String(currentIndex);
    dom.angleValue.textContent = `${thetaDeg}\u00b0`;
    dom.angVelValue.textContent = frame.state.thetaDot.toFixed(2);
    dom.torqueValue.textContent = frame.torque.toFixed(2);
    dom.returnValue.textContent = frame.cumReturn.toFixed(1);
  };

  const updateOverlay = (): void => {
    if (!atEnd()) { dom.terminalOverlay.hidden = true; return; }
    const frame = currentFrame();
    dom.terminalOverlay.hidden = false;
    dom.terminalOverlay.classList.toggle('is-success', frame.truncated);
    dom.terminalOverlay.classList.toggle('is-failure', !frame.truncated);
    dom.terminalTitle.textContent = 'Trajectory complete';
    dom.terminalSummary.textContent = `Return: ${frame.cumReturn.toFixed(1)}`;
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
    renderPendulum(canvas, currentFrame().state);
    renderTorqueChart(torqueChartCanvas, trajectory, currentIndex);
    renderTimelineChart(timelineCanvas, trajectory.length, currentIndex);
  };

  const goToIndex = (index: number): void => {
    currentIndex = Math.max(0, Math.min(index, trajectory.length - 1));
    updateControls();
    render();
  };

  const schedulePlaybackStep = (): void => {
    if (!playing) { return; }
    if (atEnd()) { generateTrajectory(true); return; }
    playbackTimerId = window.setTimeout(() => {
      playbackTimerId = null;
      if (!playing) { return; }
      if (!atEnd()) { currentIndex++; updateControls(); render(); schedulePlaybackStep(); }
      else { stopPlayback(); updateControls(); render(); }
    }, stepDelayMs());
  };

  const startPlayback = (): void => {
    if (playing) { return; }
    if (atEnd()) { currentIndex = 0; }
    playing = true;
    updateControls();
    render();
    schedulePlaybackStep();
  };

  const togglePlayback = (): void => {
    if (playing) { stopPlayback(); updateControls(); render(); return; }
    startPlayback();
  };

  const buildTrajectory = (): TrajectoryFrame[] => {
    const frames: TrajectoryFrame[] = [];
    const obs = env.reset();
    const state = { ...env.getState() };
    let torque = policy(obs).torque;
    let cumReturn = 0;
    frames.push({
      state,
      obs: [...obs] as PendulumObs,
      torque,
      reward: 0,
      cumReturn,
      truncated: false
    });

    for (;;) {
      const result = env.step(torque);
      cumReturn += result.reward;
      const nextState = { ...env.getState() };
      torque = policy(result.obs).torque;
      frames.push({
        state: nextState,
        obs: [...result.obs] as PendulumObs,
        torque,
        reward: result.reward,
        cumReturn,
        truncated: result.truncated
      });
      if (result.truncated) { break; }
    }
    return frames;
  };

  const generateTrajectory = (autoplay: boolean): void => {
    stopPlayback();
    trajectory = buildTrajectory();
    episodes++;
    currentIndex = 0;
    if (autoplay) { playing = true; schedulePlaybackStep(); }
    updateControls();
    render();
  };

  const scrubTimeline = (clientX: number): void => {
    const rect = timelineCanvas.getBoundingClientRect();
    goToIndex(
      chartXToNearestIndex(
        clientX - rect.left,
        timelineCanvas.clientWidth,
        trajectory.length
      )
    );
  };

  const scrubTorqueChart = (clientX: number): void => {
    const rect = torqueChartCanvas.getBoundingClientRect();
    goToIndex(
      chartXToNearestIndex(
        clientX - rect.left,
        torqueChartCanvas.clientWidth,
        trajectory.length
      )
    );
  };

  const handleTimelinePointerDown = (e: PointerEvent): void => {
    stopPlayback(); isTimelineDragging = true;
    timelineCanvas.setPointerCapture(e.pointerId); scrubTimeline(e.clientX);
  };
  const handleTimelinePointerMove = (e: PointerEvent): void => {
    if (isTimelineDragging) { scrubTimeline(e.clientX); }
  };
  const handleTimelinePointerUp = (e: PointerEvent): void => {
    if (!isTimelineDragging) { return; }
    isTimelineDragging = false;
    if (timelineCanvas.hasPointerCapture(e.pointerId)) {
      timelineCanvas.releasePointerCapture(e.pointerId);
    }
  };

  const handleChartPointerDown = (e: PointerEvent): void => {
    stopPlayback(); isChartDragging = true;
    torqueChartCanvas.setPointerCapture(e.pointerId); scrubTorqueChart(e.clientX);
  };
  const handleChartPointerMove = (e: PointerEvent): void => {
    if (isChartDragging) { scrubTorqueChart(e.clientX); }
  };
  const handleChartPointerUp = (e: PointerEvent): void => {
    if (!isChartDragging) { return; }
    isChartDragging = false;
    if (torqueChartCanvas.hasPointerCapture(e.pointerId)) {
      torqueChartCanvas.releasePointerCapture(e.pointerId);
    }
  };

  const handleSpeedInput = (): void => {
    speed = sliderToSpeed(Number(speedSlider.value));
    speedValueEl.textContent = formatSpeed(speed);
    if (playing && playbackTimerId !== null) {
      window.clearTimeout(playbackTimerId); playbackTimerId = null; schedulePlaybackStep();
    }
  };

  const stepBackRepeat = createRepeatButton(stepBackBtn, () => {
    stopPlayback(); if (currentIndex > 0) { goToIndex(currentIndex - 1); }
  });
  const stepForwardRepeat = createRepeatButton(stepForwardBtn, () => {
    stopPlayback(); if (!atEnd()) { goToIndex(currentIndex + 1); }
  });
  const handleGoToStartClick = (): void => {
    stopPlayback();
    goToIndex(0);
  };
  const handlePlayClick = (): void => {
    togglePlayback();
  };
  const handleResetClick = (): void => {
    generateTrajectory(true);
  };

  const teardownEvents = wireEvents([
    [goToStartBtn, 'click', handleGoToStartClick],
    [stepBackBtn, 'pointerdown', stepBackRepeat.handlePointerDown as EventListener],
    [stepBackBtn, 'pointerup', stepBackRepeat.handlePointerUp as EventListener],
    [stepBackBtn, 'pointercancel', stepBackRepeat.handlePointerUp as EventListener],
    [stepBackBtn, 'click', stepBackRepeat.handleClick],
    [playBtn, 'click', handlePlayClick],
    [stepForwardBtn, 'pointerdown', stepForwardRepeat.handlePointerDown as EventListener],
    [stepForwardBtn, 'pointerup', stepForwardRepeat.handlePointerUp as EventListener],
    [stepForwardBtn, 'pointercancel', stepForwardRepeat.handlePointerUp as EventListener],
    [stepForwardBtn, 'click', stepForwardRepeat.handleClick],
    [resetBtn, 'click', handleResetClick],
    [speedSlider, 'input', handleSpeedInput],
    [timelineCanvas, 'pointerdown', handleTimelinePointerDown as EventListener],
    [timelineCanvas, 'pointermove', handleTimelinePointerMove as EventListener],
    [timelineCanvas, 'pointerup', handleTimelinePointerUp as EventListener],
    [timelineCanvas, 'pointercancel', handleTimelinePointerUp as EventListener],
    [torqueChartCanvas, 'pointerdown', handleChartPointerDown as EventListener],
    [torqueChartCanvas, 'pointermove', handleChartPointerMove as EventListener],
    [torqueChartCanvas, 'pointerup', handleChartPointerUp as EventListener],
    [torqueChartCanvas, 'pointercancel', handleChartPointerUp as EventListener]
  ]);

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
