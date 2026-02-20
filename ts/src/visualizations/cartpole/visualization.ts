// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { DqnPolicy } from '../../cartpole/dqn-policy';
import {
  ANGLE_LIMIT,
  createCartPoleEnvironment,
  POSITION_LIMIT } from '../../cartpole/environment';
import { CANVAS_HEIGHT, CANVAS_WIDTH, renderCartPole } from './renderer';
import { createCartPoleVizDom } from './ui';

export interface CartPoleVisualization {
  destroy(): void;
}

// How many ticks to step per interval at speed=1
const BASE_TICK_MS = 40;

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
    container, canvas, resetBtn, speedSlider, speedValueEl,
    pauseBtn
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
  let steps = 0;
  let paused = false;
  let speed = 1;

  const updateQDisplay = (): void => {
    const { qLeft, qRight, action } = runDqnPolicy(env.getState());

    // Normalize bars relative to each other in a [0,1] range
    const minQ = Math.min(qLeft, qRight);
    const maxQ = Math.max(qLeft, qRight);
    const range = maxQ - minQ;

    const leftFrac = range > 0 ? (qLeft - minQ) / range : 0.5;
    const rightFrac = range > 0 ? (qRight - minQ) / range : 0.5;

    dom.qLeftBar.style.width = `${String(Math.round(leftFrac * 100))}%`;
    dom.qRightBar.style.width = `${String(Math.round(rightFrac * 100))}%`;
    dom.qLeftValue.textContent = qLeft.toFixed(2);
    dom.qRightValue.textContent = qRight.toFixed(2);

    dom.qLeftRow.classList.toggle('is-chosen', action === 0);
    dom.qRightRow.classList.toggle('is-chosen', action === 1);
    dom.actionValue.textContent = action === 0 ? '\u2190 Left' : 'Right \u2192';
    dom.actionValue.className = `cp-policy-action-value ${action === 0 ? 'is-left' : 'is-right'}`;
  };

  const updateStats = (): void => {
    const state = env.getState();
    dom.episodesValue.textContent = String(episodes);
    dom.stepsValue.textContent = String(steps);
    dom.positionValue.textContent = state[0].toFixed(2);
    dom.velocityValue.textContent = state[1].toFixed(2);
    dom.angleValue.textContent = (state[2] * 180 / Math.PI).toFixed(1) + '\u00b0';
    dom.angularVelocityValue.textContent = state[3].toFixed(2);
  };

  const render = (): void => {
    updateQDisplay();
    updateStats();
    renderCartPole(canvas, env.getState());
  };

  // Run the policy one step
  const tick = (): void => {
    if (paused) {return;}

    const { action } = runDqnPolicy(env.getState());
    const result = env.step(action);
    steps++;

    if (result.terminated || result.truncated) {
      episodes++;
      env.reset();
      steps = 0;
    }

    render();
  };

  // Interval that runs multiple ticks per fire depending on speed
  const intervalId = window.setInterval(() => {
    for (let i = 0; i < speed; i++) {
      tick();
    }
  }, BASE_TICK_MS);

  const resetEpisode = (): void => {
    env.reset();
    steps = 0;
    render();
  };

  const handlePause = (): void => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  };

  const handleReset = (): void => {
    resetEpisode();
  };

  const handleSpeedInput = (): void => {
    speed = Number(speedSlider.value);
    speedValueEl.textContent = `${String(speed)}x`;
  };

  pauseBtn.addEventListener('click', handlePause);
  resetBtn.addEventListener('click', handleReset);
  speedSlider.addEventListener('input', handleSpeedInput);

  // Initial render
  render();

  return {
    destroy(): void {
      window.clearInterval(intervalId);
      pauseBtn.removeEventListener('click', handlePause);
      resetBtn.removeEventListener('click', handleReset);
      speedSlider.removeEventListener('input', handleSpeedInput);
      container.remove();
    }
  };
}
