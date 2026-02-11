// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  ACTIONS,
  actionToIndex,
  forEachTransition,
  type TransitionTable
} from '../core/mdp';
import type {
  Action,
  ActionValues,
  Policy,
  StateValues
} from '../core/types';
import { cloneActionValues, stateKey } from './shared';

export interface QVIDPSnapshot {
  actionValues: ActionValues;
  stateValues: StateValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
}

export interface QVIDPResult {
  snapshots: QVIDPSnapshot[];
  finalPolicy: Policy;
  finalActionValues: ActionValues;
  finalStateValues: StateValues;
}

function maxQValue(
  actionValues: ActionValues,
  key: string
): number {
  const actionMap = actionValues.get(key);
  if (actionMap === undefined) {
    return 0;
  }
  let best = -Infinity;
  for (const value of actionMap.values()) {
    if (value > best) {
      best = value;
    }
  }
  return isFinite(best) ? best : 0;
}

function deriveStateValues(
  table: TransitionTable,
  actionValues: ActionValues
): StateValues {
  const values: StateValues = new Map();
  const size = table.rows * table.cols;

  for (let s = 0; s < size; s++) {
    if (table.cellTypes[s] === 0) {
      continue;
    }

    const key = stateKey(table, s);
    if (table.isTerminal[s] === 1) {
      values.set(key, table.rewards[s]);
      continue;
    }

    values.set(key, maxQValue(actionValues, key));
  }

  return values;
}

function greedyPolicy(
  table: TransitionTable,
  actionValues: ActionValues
): Policy {
  const policy: Policy = new Map();
  const size = table.rows * table.cols;

  for (let s = 0; s < size; s++) {
    if (table.cellTypes[s] !== 1) {
      continue;
    }
    if (table.isTerminal[s] === 1) {
      continue;
    }

    const key = stateKey(table, s);
    const actionMap = actionValues.get(key);
    if (actionMap === undefined) {
      continue;
    }

    let bestAction: Action = ACTIONS[0];
    let bestValue = -Infinity;
    for (const [action, value] of actionMap) {
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    policy.set(key, bestAction);
  }

  return policy;
}

export function runValueIterationQ(
  table: TransitionTable,
  initialActionValues: ActionValues,
  gamma: number,
  theta: number
): QVIDPResult {
  const snapshots: QVIDPSnapshot[] = [];
  let currentActionValues = cloneActionValues(
    initialActionValues
  );
  const maxSweeps = 500;

  const initialPolicy = greedyPolicy(
    table, currentActionValues
  );
  const initialStateValues = deriveStateValues(
    table, currentActionValues
  );
  snapshots.push({
    actionValues: cloneActionValues(currentActionValues),
    stateValues: new Map(initialStateValues),
    policy: new Map(initialPolicy),
    phase: 'evaluation',
    delta: 0
  });

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    const newActionValues: ActionValues = new Map();
    let delta = 0;
    const size = table.rows * table.cols;

    for (let s = 0; s < size; s++) {
      if (table.cellTypes[s] === 0) {
        continue;
      }

      const key = stateKey(table, s);
      const newActionMap = new Map<Action, number>();

      if (table.isTerminal[s] === 1) {
        for (const action of ACTIONS) {
          newActionMap.set(action, table.rewards[s]);
        }
        newActionValues.set(key, newActionMap);
        continue;
      }

      for (const action of ACTIONS) {
        const actionIndex = actionToIndex(action);
        const reward = table.rewards[s];
        let value = 0;

        forEachTransition(
          table,
          s,
          actionIndex,
          (nextState, probability) => {
            const nextKey = stateKey(table, nextState);
            const nextMax = maxQValue(
              currentActionValues, nextKey
            );
            value += probability * (reward + gamma * nextMax);
          }
        );

        newActionMap.set(action, value);

        const oldValue = currentActionValues.get(key)
          ?.get(action) ?? 0;
        delta = Math.max(delta, Math.abs(value - oldValue));
      }

      newActionValues.set(key, newActionMap);
    }

    currentActionValues = newActionValues;
    const policy = greedyPolicy(table, currentActionValues);
    const stateValues = deriveStateValues(
      table, currentActionValues
    );

    snapshots.push({
      actionValues: cloneActionValues(currentActionValues),
      stateValues: new Map(stateValues),
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
    finalActionValues: lastSnapshot.actionValues,
    finalStateValues: lastSnapshot.stateValues
  };
}
