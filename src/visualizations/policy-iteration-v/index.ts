// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import { initializeStateValues } from '../../core/policy';
import type { StateValues } from '../../core/types';
import { runPolicyIterationV } from '../../dp/policy-iteration-v';
import type {
  DPVisualization,
  InitParams
} from '../dp-shared/types';
import { initializeDPVisualization } from '../dp-shared/visualization';
import { renderGrid } from './renderer';

export type PolicyIterationVVisualization = DPVisualization;

function initializePolicyIterationVVisualization(
  params: InitParams
): PolicyIterationVVisualization {
  // initialValues is set by createZeroValues (called by
  // initializeDPVisualization before computeSnapshots).
  let initialValues: StateValues = new Map();

  return initializeDPVisualization({
    computeSnapshots(table, initialPolicy, gamma, theta) {
      return runPolicyIterationV(
        table,
        new Map(initialPolicy),
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
        valueRange
      );
    },

    getPhaseExplanation(snapshot, nextPhase, atStart, atEnd, initialValueMode) {
      const initialValuesText = initialValueMode === 'zero'
        ? 'all values = 0'
        : 'random values';

      if (atStart) {
        return [
          `Initial state: Policy and values are arbitrary (here: ${initialValuesText}).`,
          'The first step will kick off the first round of policy evaluation:',
          'we set the value of each state to the reward of taking the action',
          'prescribed by the policy plus the weighted sum of the values of the',
          'next states, weighted by the probabilities of landing in each state',
          'after taking the action.'
        ].join(' ');
      }

      if (
        snapshot.phase === 'evaluation'
        && nextPhase === 'improvement'
      ) {
        return [
          "The values don't change substantially any more, so we have reached",
          'convergence. The next step will be policy improvement: we update',
          'the policy to be optimal for the current values.'
        ].join(' ');
      }

      if (snapshot.phase === 'evaluation') {
        return [
          'Policy evaluation: In each step we set the value of each state to',
          'the reward of taking the action prescribed by the policy plus the',
          'weighted sum of the values of the next states, weighted by the',
          'probabilities of landing in each state after taking the action.',
          "We do this until the values don't change any more (convergence)."
        ].join(' ');
      }

      if (atEnd) {
        return [
          "The values don't change substantially any more, we have reached",
          'convergence, and policy improvement no longer changes the policy.',
          'We have found the optimal policy and are done.'
        ].join(' ');
      }

      return [
        'After policy iteration: we have set a new policy. The next step',
        'starts the next round of policy evaluation, where we set each',
        "state's value to the reward of the policy action plus the weighted",
        'sum of successor-state values, weighted by transition probabilities.'
      ].join(' ');
    },

    initialValuesLabel: 'Initial values',
    radioGroupName: 'pi-viz-v-initial-values',
    extraCheckboxes: []
  }, params);
}

export function initPolicyIterationVVisualization(
  parent: HTMLElement
): PolicyIterationVVisualization {
  return initializePolicyIterationVVisualization({
    parent,
    layout: MEDIUM_GRID,
    cellSize: 50,
    successProb: DP_DEFAULTS.successProb,
    rewardModel: REWARDS,
    gamma: DP_DEFAULTS.gamma,
    theta: DP_DEFAULTS.theta
  });
}
