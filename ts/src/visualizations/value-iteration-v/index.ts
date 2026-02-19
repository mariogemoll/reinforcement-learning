// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import { initializeStateValues } from '../../core/policy';
import type { StateValues } from '../../core/types';
import { runValueIterationV } from '../../dp/value-iteration-v';
import type {
  DPVisualization,
  InitParams
} from '../shared/types';
import { renderGrid } from '../shared/v-renderer';
import { initializeDPVisualization } from '../shared/visualization';

export type ValueIterationVVisualization = DPVisualization;

function initializeValueIterationVVisualization(
  params: InitParams
): ValueIterationVVisualization {
  let initialValues: StateValues = new Map();

  return initializeDPVisualization({
    computeSnapshots(table, _initialPolicy, gamma, theta) {
      return runValueIterationV(
        table,
        new Map(initialValues),
        gamma,
        theta
      ).snapshots;
    },

    createZeroValues(grid) {
      initialValues = initializeStateValues(grid);
    },

    randomizeValues() {
      for (const key of initialValues.keys()) {
        initialValues.set(key, (Math.random() * 20) - 10);
      }
    },

    renderGrid(canvas, grid, cellSize, snapshot, agents, effects, valueRange) {
      renderGrid(
        canvas,
        grid,
        cellSize,
        snapshot.stateValues,
        snapshot.policy,
        agents,
        effects,
        true,
        true,
        true,
        valueRange,
        false
      );
    },

    getPhaseExplanation(snapshot, _nextPhase, atStart, atEnd, initialValueMode) {
      const initialValuesText = initialValueMode === 'zero'
        ? 'all values = 0'
        : 'random values';

      if (atStart) {
        return [
          `Initial state: Values are arbitrary (here: ${initialValuesText}).`,
          'The first step will perform a Bellman optimality backup:',
          'for each state, we compute the value of every possible action',
          'and set the state value to the maximum.'
        ].join(' ');
      }

      if (atEnd) {
        return [
          "The values don't change substantially any more, we have",
          'reached convergence. The greedy policy with respect to these',
          'values is optimal.'
        ].join(' ');
      }

      return [
        `Value iteration sweep (delta: ${snapshot.delta.toFixed(4)}):`,
        'For each state, we compute the expected return of every action',
        'and set the state value to the maximum. This combines evaluation',
        'and improvement into a single step. We repeat until the values',
        "don't change any more (convergence)."
      ].join(' ');
    },

    initialValuesLabel: 'Initial values',
    radioGroupName: 'vi-viz-v-initial-values',
    extraCheckboxes: []
  }, params);
}

export function initValueIterationVVisualization(
  parent: HTMLElement
): ValueIterationVVisualization {
  return initializeValueIterationVVisualization({
    parent,
    layout: MEDIUM_GRID,
    cellSize: 50,
    successProb: DP_DEFAULTS.successProb,
    rewardModel: REWARDS,
    gamma: DP_DEFAULTS.gamma,
    theta: DP_DEFAULTS.theta
  });
}
