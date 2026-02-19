// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import type { ActionValues } from '../../core/types';
import { runPolicyIterationQ } from '../../dp/policy-iteration-q';
import {
  cloneActionValues,
  initializeActionValues
} from '../../dp/shared';
import { renderGrid } from '../shared/q-renderer';
import type {
  DPVisualization,
  InitParams
} from '../shared/types';
import { initializeDPVisualization } from '../shared/visualization';

export type PolicyIterationQVisualization = DPVisualization;

function initializePolicyIterationQVisualization(
  params: InitParams
): PolicyIterationQVisualization {
  // initialActionValues is set by createZeroValues (called by
  // initializeDPVisualization before computeSnapshots).
  let initialActionValues: ActionValues = new Map();
  let showQLabels = false;

  return initializeDPVisualization({
    computeSnapshots(table, initialPolicy, gamma, theta) {
      return runPolicyIterationQ(
        table,
        new Map(initialPolicy),
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
        valueRange,
        null
      );
    },

    getPhaseExplanation(snapshot, nextPhase, atStart, atEnd, initialValueMode) {
      const initialActionValuesText = initialValueMode === 'zero'
        ? 'all action-values = 0'
        : 'random action-values';

      if (atStart) {
        return [
          'Initial state: Policy and action-values are'
          + ` arbitrary (here: ${initialActionValuesText}).`,
          'The first step starts policy evaluation in Q-space:',
          'for each state-action pair we update Q(s,a) to expected',
          "reward plus gamma times Q(s', pi(s')) under the current policy."
        ].join(' ');
      }

      if (
        snapshot.phase === 'evaluation'
        && nextPhase === 'improvement'
      ) {
        return [
          "The action-values don't change substantially any more, so we have reached",
          'convergence. The next step will be policy improvement: we update',
          'the policy greedily from the current action-values.'
        ].join(' ');
      }

      if (snapshot.phase === 'evaluation') {
        return [
          'Policy evaluation in Q-space: for each (s,a), update Q(s,a) as',
          "expected reward plus gamma times Q(s', pi(s')) under the current policy.",
          'Triangles show Q(s,a) for up/down/left/right within each floor cell.',
          'We sweep until Q changes are below theta.'
        ].join(' ');
      }

      if (atEnd) {
        return [
          "The action-values don't change substantially any more, we have reached",
          'convergence, and policy improvement no longer changes the policy.',
          'We have found the optimal policy and are done.'
        ].join(' ');
      }

      return [
        'After policy iteration: we have set a new policy. The next step',
        'starts the next round of policy evaluation in Q-space, where we',
        'update each Q(s,a) under the fixed improved policy.'
      ].join(' ');
    },

    initialValuesLabel: 'Initial values',
    radioGroupName: 'pi-viz-q-initial-values',
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

export function initPolicyIterationQVisualization(
  parent: HTMLElement
): PolicyIterationQVisualization {
  return initializePolicyIterationQVisualization({
    parent,
    layout: MEDIUM_GRID,
    cellSize: 50,
    successProb: DP_DEFAULTS.successProb,
    rewardModel: REWARDS,
    gamma: DP_DEFAULTS.gamma,
    theta: DP_DEFAULTS.theta
  });
}
