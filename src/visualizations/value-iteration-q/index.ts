// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import type { ActionValues } from '../../core/types';
import {
  cloneActionValues,
  initializeActionValues
} from '../../dp/shared';
import { runValueIterationQ } from '../../dp/value-iteration-q';
import { renderGrid } from '../dp-shared/q-renderer';
import type {
  DPVisualization,
  InitParams
} from '../dp-shared/types';
import { initializeDPVisualization } from '../dp-shared/visualization';

export type ValueIterationQVisualization = DPVisualization;

function initializeValueIterationQVisualization(
  params: InitParams
): ValueIterationQVisualization {
  let initialActionValues: ActionValues = new Map();
  let showQLabels = false;

  return initializeDPVisualization({
    computeSnapshots(table, _initialPolicy, gamma, theta) {
      return runValueIterationQ(
        table,
        cloneActionValues(initialActionValues),
        gamma,
        theta
      ).snapshots;
    },

    createZeroValues(grid) {
      initialActionValues = initializeActionValues(grid);
    },

    randomizeValues() {
      for (const actionMap of initialActionValues.values()) {
        for (const [action] of actionMap) {
          actionMap.set(action, (Math.random() * 20) - 10);
        }
      }
    },

    renderGrid(canvas, grid, cellSize, snapshot, agents, effects, valueRange) {
      renderGrid(
        canvas,
        grid,
        cellSize,
        snapshot.stateValues,
        snapshot.actionValues,
        snapshot.policy,
        agents,
        effects,
        false,
        false,
        showQLabels,
        valueRange
      );
    },

    getPhaseExplanation(snapshot, _nextPhase, atStart, atEnd, initialValueMode) {
      const initialActionValuesText = initialValueMode === 'zero'
        ? 'all action-values = 0'
        : 'random action-values';

      if (atStart) {
        return [
          'Initial state: Action-values are'
          + ` arbitrary (here: ${initialActionValuesText}).`,
          'The first step performs a Bellman optimality backup in Q-space:',
          "for each (s,a) we set Q(s,a) = R(s) + gamma * sum P(s'|s,a) * max_a' Q(s',a')."
        ].join(' ');
      }

      if (atEnd) {
        return [
          "The action-values don't change substantially any more, we have",
          'reached convergence. The greedy policy with respect to these',
          'Q-values is optimal.'
        ].join(' ');
      }

      return [
        `Value iteration sweep (delta: ${snapshot.delta.toFixed(4)}):`,
        'For each state-action pair, we set Q(s,a) to the expected reward',
        'plus gamma times the max Q-value of the next state. This combines',
        'evaluation and improvement into a single step. We repeat until',
        'convergence.'
      ].join(' ');
    },

    initialValuesLabel: 'Initial values',
    radioGroupName: 'vi-viz-q-initial-values',
    extraCheckboxes: [{
      label: 'Show values',
      ariaLabel: 'Show values on grid',
      initialChecked: false,
      onChange(checked: boolean): void {
        showQLabels = checked;
      }
    }]
  }, params);
}

export function initValueIterationQVisualization(
  parent: HTMLElement
): ValueIterationQVisualization {
  return initializeValueIterationQVisualization({
    parent,
    layout: MEDIUM_GRID,
    cellSize: 50,
    successProb: DP_DEFAULTS.successProb,
    rewardModel: REWARDS,
    gamma: DP_DEFAULTS.gamma,
    theta: DP_DEFAULTS.theta
  });
}
