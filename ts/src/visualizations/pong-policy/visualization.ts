// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  computerPongPlayer,
  ENV_WIDTH,
  resetPongState,
  stepPongState
} from '../../pong/environment';
import type { PongNNPolicy, PongQValues } from '../../pong/nn-policy';
import type { PongState } from '../../pong/types';
import { CANVAS_HEIGHT, CANVAS_WIDTH, renderPong } from '../pong/renderer';
import { wireEvents } from '../shared/event-wiring';
import type { PongPolicyVizDom } from './types';
import { createPongPolicyVizDom } from './ui';

export interface PongPolicyVisualization {
  destroy(): void;
}

const STEP_INTERVAL_MS = 100;
const WINNING_SCORE = 7;
const NN_COLOR = '#3b9eff';
const CPU_COLOR = '#eb3528';

function randomInitVY(): 1 | -1 {
  return Math.random() < 0.5 ? 1 : -1;
}

function updateQBars(
  dom: PongPolicyVizDom,
  qValues: PongQValues,
  chosenAction: 0 | 1 | 2
): void {
  const vals = [qValues.noop, qValues.up, qValues.down];
  const min = Math.min(...vals);
  const shifted = vals.map(v => v - min);
  const max = Math.max(...shifted, 1e-6);

  const keys: ('noop' | 'up' | 'down')[] = ['noop', 'up', 'down'];
  const actionIdx = [0, 1, 2];
  for (const [i, key] of keys.entries()) {
    const pct = (shifted[i] / max) * 100;
    const els = dom.qBars[key];
    els.bar.style.width = `${pct.toFixed(1)}%`;
    els.value.textContent = vals[i].toFixed(2);
    const track = els.bar.parentElement;
    if (track === null) {
      continue;
    }
    track.classList.toggle(
      'pong-policy-qbar-track-active',
      actionIdx[i] === chosenAction
    );
  }
}

export function initializePongPolicyVisualization(
  parent: HTMLElement,
  policy: PongNNPolicy
): PongPolicyVisualization {
  const dom = createPongPolicyVizDom();
  const { container, canvas, overlay, overlayTitle, playAgainBtn } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder !== null) {
    placeholder.replaceWith(container);
  } else {
    parent.appendChild(container);
  }

  canvas.width  = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let state: PongState = resetPongState(randomInitVY());
  let nnScore = 0;
  let cpuScore = 0;
  let intervalId: number | null = null;

  const render = (): void => {
    renderPong(canvas, state, nnScore, cpuScore, NN_COLOR, CPU_COLOR);
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
    const { qValues, action: nnAction } = policy(state);
    updateQBars(dom, qValues, nnAction);

    const cpuAction = computerPongPlayer(state);
    const { state: nextState, done } = stepPongState(state, nnAction, cpuAction);

    if (done) {
      if (nextState.ballCol >= ENV_WIDTH) {
        nnScore++;
      } else {
        cpuScore++;
      }

      if (nnScore >= WINNING_SCORE || cpuScore >= WINNING_SCORE) {
        stopTimer();
        state = nextState;
        render();
        overlayTitle.textContent = nnScore >= WINNING_SCORE ? 'NN wins!' : 'CPU wins';
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

  const handlePlayAgain = (): void => {
    nnScore = 0;
    cpuScore = 0;
    overlay.hidden = true;
    startEpisode();
    startTimer();
  };

  const teardownEvents = wireEvents([
    [playAgainBtn, 'click', handlePlayAgain]
  ]);

  startEpisode();
  startTimer();

  return {
    destroy(): void {
      stopTimer();
      teardownEvents();
      container.remove();
    }
  };
}
