// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  ACTIONS,
  actionToIndex,
  type TransitionTable
} from '../core/mdp';
import type { Policy, StateValues } from '../core/types';
import {
  computeBellmanValue,
  greedyPolicyFromValues,
  stateKey
} from './shared';

export interface VIDPSnapshot {
  stateValues: StateValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
}

export interface VIDPResult {
  snapshots: VIDPSnapshot[];
  finalPolicy: Policy;
  finalValues: StateValues;
}

export function runValueIterationV(
  table: TransitionTable,
  initialValues: StateValues,
  gamma: number,
  theta: number
): VIDPResult {
  const snapshots: VIDPSnapshot[] = [];
  let currentValues = new Map(initialValues);
  const maxSweeps = 500;

  const initialPolicy = greedyPolicyFromValues(
    table, currentValues, gamma
  );
  snapshots.push({
    stateValues: new Map(currentValues),
    policy: new Map(initialPolicy),
    phase: 'evaluation',
    delta: 0
  });

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
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

      let bestValue = -Infinity;
      for (const action of ACTIONS) {
        const actionIndex = actionToIndex(action);
        const value = computeBellmanValue(
          table, s, actionIndex, currentValues, gamma
        );
        if (value > bestValue) {
          bestValue = value;
        }
      }

      newValues.set(key, bestValue);

      const oldValue = currentValues.get(key) ?? 0;
      delta = Math.max(delta, Math.abs(bestValue - oldValue));
    }

    currentValues = newValues;
    const policy = greedyPolicyFromValues(
      table, currentValues, gamma
    );

    snapshots.push({
      stateValues: new Map(currentValues),
      policy: new Map(policy),
      phase: 'evaluation',
      delta
    });

    if (delta < theta) {
      break;
    }
  }

  const lastSnapshot = snapshots[snapshots.length - 1];
  return {
    snapshots,
    finalPolicy: lastSnapshot.policy,
    finalValues: lastSnapshot.stateValues
  };
}
