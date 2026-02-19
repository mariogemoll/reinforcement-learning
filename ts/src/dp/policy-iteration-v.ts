// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { actionToIndex, type TransitionTable } from '../core/mdp';
import type { Policy, StateValues } from '../core/types';
import {
  computeBellmanValue,
  greedyPolicyFromValues,
  policiesEqual,
  stateKey
} from './shared';

export interface VDPSnapshot {
  stateValues: StateValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
}

export interface VDPResult {
  snapshots: VDPSnapshot[];
  finalPolicy: Policy;
  finalValues: StateValues;
}

function evaluationSweep(
  table: TransitionTable,
  policy: Policy,
  currentValues: StateValues,
  gamma: number
): { values: StateValues; delta: number } {
  const newValues: StateValues = new Map();
  let delta = 0;
  const size = table.rows * table.cols;

  for (let s = 0; s < size; s++) {
    if (table.cellTypes[s] === 0) {
      continue;
    }

    const key = stateKey(table, s);

    if (table.isTerminal[s] === 1) {
      newValues.set(key, table.rewards[s]);
      continue;
    }

    const action = policy.get(key);
    if (action === undefined) {
      newValues.set(key, 0);
      continue;
    }

    const actionIndex = actionToIndex(action);
    const value = computeBellmanValue(
      table, s, actionIndex, currentValues, gamma
    );
    newValues.set(key, value);

    const oldValue = currentValues.get(key) ?? 0;
    delta = Math.max(delta, Math.abs(value - oldValue));
  }

  return { values: newValues, delta };
}

export function runPolicyIterationV(
  table: TransitionTable,
  initialPolicy: Policy,
  initialValues: StateValues,
  gamma: number,
  theta: number
): VDPResult {
  const snapshots: VDPSnapshot[] = [];
  let currentPolicy = new Map(initialPolicy);
  let currentValues = new Map(initialValues);
  const maxOuterIterations = 50;

  snapshots.push({
    stateValues: new Map(currentValues),
    policy: new Map(currentPolicy),
    phase: 'evaluation',
    delta: 0
  });

  for (let outer = 0; outer < maxOuterIterations; outer++) {
    // Policy evaluation: sweep until convergence
    const maxEvalSweeps = 200;
    for (let sweep = 0; sweep < maxEvalSweeps; sweep++) {
      const result = evaluationSweep(
        table, currentPolicy, currentValues, gamma
      );
      currentValues = result.values;

      snapshots.push({
        stateValues: new Map(currentValues),
        policy: new Map(currentPolicy),
        phase: 'evaluation',
        delta: result.delta
      });

      if (result.delta < theta) {
        break;
      }
    }

    // Policy improvement
    const oldPolicy = currentPolicy;
    currentPolicy = greedyPolicyFromValues(
      table, currentValues, gamma
    );

    snapshots.push({
      stateValues: new Map(currentValues),
      policy: new Map(currentPolicy),
      phase: 'improvement',
      delta: 0
    });

    if (policiesEqual(oldPolicy, currentPolicy)) {
      break;
    }
  }

  return {
    snapshots,
    finalPolicy: currentPolicy,
    finalValues: currentValues
  };
}
