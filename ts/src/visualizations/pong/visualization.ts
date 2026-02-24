// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  computerPongPlayer,
  ENV_WIDTH,
  resetPongState,
  stepPongState
} from '../../pong/environment';
import type { PongAction, PongState } from '../../pong/types';
import { wireEvents } from '../shared/event-wiring';
import { CANVAS_HEIGHT, CANVAS_WIDTH, renderPong } from './renderer';
import { createPongVizDom } from './ui';

export interface PongVisualization {
  destroy(): void;
}

const STEP_INTERVAL_MS = 100;
const WINNING_SCORE = 7;

function randomInitVY(): 1 | -1 {
  return Math.random() < 0.5 ? 1 : -1;
}

export function initializePongVisualization(parent: HTMLElement): PongVisualization {
  const dom = createPongVizDom();
  const { container, canvas, overlay, overlayTitle, newGameBtn } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder !== null) {
    placeholder.replaceWith(container);
  } else {
    parent.appendChild(container);
  }

  canvas.width  = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let state: PongState = resetPongState(randomInitVY());
  let playerScore = 0;
  let aiScore = 0;
  let action: PongAction = 0;
  let intervalId: number | null = null;

  const render = (): void => {
    renderPong(canvas, state, playerScore, aiScore, '#28eb58', '#eb3528');
  };

  const startEpisode = (): void => {
    state = resetPongState(randomInitVY());
    render();
  };

  const stopTimer = (): void => {
    if (intervalId !== null) {
      window.clearInterval(intervalId);
      intervalId = null;
    }
  };

  const tick = (): void => {
    const { state: nextState, done } = stepPongState(state, action, computerPongPlayer(state));

    if (done) {
      if (nextState.ballCol >= ENV_WIDTH) {
        playerScore++;
      } else {
        aiScore++;
      }

      if (playerScore >= WINNING_SCORE || aiScore >= WINNING_SCORE) {
        stopTimer();
        state = nextState;
        render();
        overlayTitle.textContent = playerScore >= WINNING_SCORE ? 'You win!' : 'CPU wins';
        overlay.hidden = false;
      } else {
        startEpisode();
      }
    } else {
      state = nextState;
      render();
    }
  };

  const startTimer = (): void => {
    stopTimer();
    intervalId = window.setInterval(tick, STEP_INTERVAL_MS);
  };

  const handleNewGame = (): void => {
    playerScore = 0;
    aiScore = 0;
    action = 0;
    overlay.hidden = true;
    startEpisode();
    startTimer();
    canvas.focus();
  };

  const handleKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {
      e.preventDefault();
      action = 1;
    } else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') {
      e.preventDefault();
      action = 2;
    }
  };

  const handleKeyUp = (e: KeyboardEvent): void => {
    if (
      e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W' ||
      e.key === 'ArrowDown' || e.key === 's' || e.key === 'S'
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

  overlayTitle.textContent = 'Pong';
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
