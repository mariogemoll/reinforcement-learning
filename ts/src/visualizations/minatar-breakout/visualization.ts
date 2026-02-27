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
import { createBreakoutCNNPolicy } from '../../minatar/breakout-cnn-policy';
import { createBreakoutMLPPolicy } from '../../minatar/breakout-mlp-policy';
import {
  type BreakoutPolicy,
  type BreakoutQValues } from '../../minatar/breakout-policy-types';
import { createBreakoutUserPlayer } from '../../minatar/breakout-user-player';
import { loadSafetensors } from '../../shared/safetensors';
import { wireEvents } from '../shared/event-wiring';
import {
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  renderMinAtarBreakout
} from './renderer';
import type { MinAtarBreakoutVizDom } from './types';
import { createMinAtarBreakoutVizDom } from './ui';

export interface MinAtarBreakoutVisualization {
  destroy(): void;
}

type BreakoutMode = 'user' | 'policy';

const USER_STEP_INTERVAL_MS = 160;
const POLICY_STEP_INTERVAL_MS = 120;

function updateQBars(
  qBars: MinAtarBreakoutVizDom['qBars'],
  qValues: BreakoutQValues,
  chosenAction: BreakoutAction
): void {
  const vals = [qValues.noop, qValues.left, qValues.right];
  const min = Math.min(...vals);
  const shifted = vals.map(v => v - min);
  const max = Math.max(...shifted, 1e-6);
  const keys: ('noop' | 'left' | 'right')[] = ['noop', 'left', 'right'];
  const actionIdx = [0, 1, 2];

  for (const [i, key] of keys.entries()) {
    const pct = (shifted[i] / max) * 100;
    const els = qBars[key];
    els.bar.style.width = `${pct.toFixed(1)}%`;
    els.value.textContent = vals[i].toFixed(2);
    const track = els.bar.parentElement;
    if (track === null) {
      continue;
    }
    track.classList.toggle('pong-policy-qbar-track-active', actionIdx[i] === chosenAction);
  }
}

function clearQBars(qBars: MinAtarBreakoutVizDom['qBars']): void {
  for (const key of ['noop', 'left', 'right'] as const) {
    qBars[key].bar.style.width = '0%';
    qBars[key].value.textContent = '0.00';
    const track = qBars[key].bar.parentElement;
    if (track !== null) {
      track.classList.remove('pong-policy-qbar-track-active');
    }
  }
}

export function initializeMinAtarBreakoutVisualization(
  parent: HTMLElement,
  policyWeightsUrl?: string
): MinAtarBreakoutVisualization {
  const dom = createMinAtarBreakoutVizDom();
  const {
    container,
    canvas,
    userModeRadio,
    policyModeRadio,
    policyModeText,
    restartBtn,
    overlayRestartBtn,
    hint,
    qPanel,
    overlay,
    overlayTitle,
    scoreValue,
    stepValue,
    qBars
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
  const userPlayer = createBreakoutUserPlayer();
  let score = 0;
  let intervalId: number | null = null;
  let mode: BreakoutMode = policyModeRadio.checked ? 'policy' : 'user';
  let policy: BreakoutPolicy | null = null;
  let policyLoadPromise: Promise<BreakoutPolicy> | null = null;
  const isPolicyMode = (): boolean => mode === 'policy';

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
    let action: BreakoutAction = userPlayer.getAction();
    if (mode === 'policy' && policy !== null) {
      const { qValues, action: policyAction } = policy(state);
      updateQBars(qBars, qValues, policyAction);
      action = policyAction;
    }

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
      userPlayer.reset();
    }
  };

  const startTimer = (): void => {
    stopTimer();
    const stepMs = mode === 'policy' ? POLICY_STEP_INTERVAL_MS : USER_STEP_INTERVAL_MS;
    intervalId = window.setInterval(tick, stepMs);
  };

  const syncModeUi = (): void => {
    const isPolicy = mode === 'policy';
    qPanel.hidden = !isPolicy;
    hint.textContent = isPolicy
      ? 'Policy plays automatically'
      : 'Click to focus   A / \u2190 left   D / \u2192 right';
    restartBtn.textContent = isPolicy ? 'Restart demo' : 'New game';
    overlayRestartBtn.textContent = isPolicy ? 'Start demo' : 'Start';
    if (!isPolicy) {
      clearQBars(qBars);
    }
  };

  const resetEpisode = (): void => {
    state = resetBreakoutState();
    observation = getBreakoutObservation(state);
    userPlayer.reset();
    score = 0;
    if (mode !== 'policy') {
      clearQBars(qBars);
    }
  };

  const ensurePolicyLoaded = async(): Promise<boolean> => {
    if (policy !== null) {
      return true;
    }
    if (policyWeightsUrl === undefined) {
      policyModeText.textContent = 'Policy demo (weights missing)';
      policyModeRadio.disabled = true;
      return false;
    }
    if (policyLoadPromise === null) {
      policyModeText.textContent = 'Policy demo (loading...)';
      policyLoadPromise = loadSafetensors(policyWeightsUrl).then(tensors => {
        if (
          Object.hasOwn(tensors, 'dense_out.weight') &&
          Object.hasOwn(tensors, 'dense_out.bias')
        ) {
          return createBreakoutMLPPolicy(tensors);
        }
        if (
          Object.hasOwn(tensors, 'out_layer.kernel') &&
          Object.hasOwn(tensors, 'out_layer.bias')
        ) {
          return createBreakoutCNNPolicy(tensors);
        }
        throw new Error('Unrecognized breakout checkpoint format');
      }).then(loadedPolicy => {
        policy = loadedPolicy;
        policyModeText.textContent = 'Policy demo';
        policyModeRadio.disabled = false;
        return loadedPolicy;
      }).catch((error: unknown) => {
        policyModeText.textContent = 'Policy demo (weights missing)';
        policyModeRadio.disabled = true;
        policyLoadPromise = null;
        throw error;
      });
    }
    try {
      await policyLoadPromise;
      return true;
    } catch {
      return false;
    }
  };

  const handleRestart = async(): Promise<void> => {
    if (mode === 'policy' && policy === null) {
      overlay.hidden = false;
      overlayTitle.textContent = 'Loading policy...';
      restartBtn.disabled = true;
      overlayRestartBtn.disabled = true;
      const loaded = await ensurePolicyLoaded();
      restartBtn.disabled = false;
      overlayRestartBtn.disabled = false;
      if (!loaded || !isPolicyMode()) {
        if (isPolicyMode()) {
          mode = 'user';
          userModeRadio.checked = true;
          syncModeUi();
        }
        stopTimer();
        resetEpisode();
        render();
        overlay.hidden = false;
        overlayTitle.textContent = 'Breakout';
        return;
      }
    }
    resetEpisode();
    overlay.hidden = true;
    render();
    startTimer();
    canvas.focus();
  };

  const handleModeChange = (): void => {
    mode = policyModeRadio.checked ? 'policy' : 'user';
    syncModeUi();
    if (mode === 'policy') {
      stopTimer();
      resetEpisode();
      render();
      overlay.hidden = false;
      overlayTitle.textContent = 'Policy demo';
      return;
    }
    stopTimer();
    resetEpisode();
    render();
    overlay.hidden = false;
    overlayTitle.textContent = 'Breakout';
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    if (mode !== 'user') {
      return;
    }
    userPlayer.onKeyDown(event);
  };

  const handleKeyUp = (event: KeyboardEvent): void => {
    if (mode !== 'user') {
      return;
    }
    userPlayer.onKeyUp(event);
  };

  const handleBlur = (): void => {
    userPlayer.onBlur();
  };

  const teardownEvents = wireEvents([
    [restartBtn, 'click', (): void => { void handleRestart(); }],
    [overlayRestartBtn, 'click', (): void => { void handleRestart(); }],
    [userModeRadio, 'change', handleModeChange],
    [policyModeRadio, 'change', handleModeChange],
    [canvas, 'keydown', handleKeyDown as EventListener],
    [canvas, 'keyup', handleKeyUp as EventListener],
    [canvas, 'blur', handleBlur]
  ]);

  overlayTitle.textContent = mode === 'policy' ? 'Policy demo' : 'Breakout';
  overlay.hidden = false;
  syncModeUi();
  render();

  if (policyWeightsUrl === undefined) {
    policyModeText.textContent = 'Policy demo (weights missing)';
    policyModeRadio.disabled = true;
    if (mode === 'policy') {
      mode = 'user';
      userModeRadio.checked = true;
      syncModeUi();
      overlayTitle.textContent = 'Breakout';
    }
  }

  return {
    destroy(): void {
      stopTimer();
      teardownEvents();
      container.remove();
    }
  };
}
