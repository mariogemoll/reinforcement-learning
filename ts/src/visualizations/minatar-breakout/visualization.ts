// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  BREAKOUT_MAX_STEPS,
  type BreakoutAction,
  type BreakoutState,
  getBreakoutObservation,
  resetBreakoutState,
  stepBreakoutState
} from '../../minatar/breakout';
import { wireEvents } from '../shared/event-wiring';
import {
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  renderMinAtarBreakout
} from './renderer';
import { createMinAtarBreakoutVizDom } from './ui';

export interface MinAtarBreakoutVisualization {
  destroy(): void;
}

const STEP_INTERVAL_MS = 160;

export function initializeMinAtarBreakoutVisualization(
  parent: HTMLElement
): MinAtarBreakoutVisualization {
  const dom = createMinAtarBreakoutVizDom();
  const {
    container,
    canvas,
    overlay,
    overlayTitle,
    newGameBtn,
    scoreValue,
    stepValue
  } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder !== null) {
    placeholder.replaceWith(container);
  } else {
    parent.appendChild(container);
  }

  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let state: BreakoutState = resetBreakoutState();
  let observation = getBreakoutObservation(state);
  let action: BreakoutAction = 0;
  let score = 0;
  let intervalId: number | null = null;

  const render = (): void => {
    renderMinAtarBreakout(canvas, observation);
    scoreValue.textContent = String(score);
    stepValue.textContent = String(state.time);
  };

  const stopTimer = (): void => {
    if (intervalId !== null) {
      window.clearInterval(intervalId);
      intervalId = null;
    }
  };

  const tick = (): void => {
    const result = stepBreakoutState(state, action);
    state = result.state;
    observation = result.observation;
    score += result.reward;
    render();

    if (result.done) {
      stopTimer();
      overlayTitle.textContent = state.time >= BREAKOUT_MAX_STEPS
        ? 'Time limit reached'
        : 'Game over';
      overlay.hidden = false;
      action = 0;
    }
  };

  const startTimer = (): void => {
    stopTimer();
    intervalId = window.setInterval(tick, STEP_INTERVAL_MS);
  };

  const handleNewGame = (): void => {
    state = resetBreakoutState();
    observation = getBreakoutObservation(state);
    action = 0;
    score = 0;
    overlay.hidden = true;
    render();
    startTimer();
    canvas.focus();
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') {
      event.preventDefault();
      action = 1;
    } else if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') {
      event.preventDefault();
      action = 2;
    } else if (event.key === ' ' || event.key === 'Spacebar') {
      event.preventDefault();
      action = 0;
    }
  };

  const handleKeyUp = (event: KeyboardEvent): void => {
    if (
      event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A' ||
      event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D'
    ) {
      action = 0;
    }
  };

  const handleBlur = (): void => {
    action = 0;
  };

  const teardownEvents = wireEvents([
    [newGameBtn, 'click', handleNewGame],
    [canvas, 'keydown', handleKeyDown as EventListener],
    [canvas, 'keyup', handleKeyUp as EventListener],
    [canvas, 'blur', handleBlur]
  ]);

  overlayTitle.textContent = 'MinAtar Breakout';
  overlay.hidden = false;
  render();

  return {
    destroy(): void {
      stopTimer();
      teardownEvents();
      container.remove();
    }
  };
}
