// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  ACTIONS,
  actionToIndex,
  forEachTransition,
  type TransitionTable } from '../core/mdp';
import type { Action, Policy, StateValues } from '../core/types';

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

function stateKey(
  table: TransitionTable,
  index: number
): string {
  const row = Math.floor(index / table.cols);
  const col = index % table.cols;
  return `${String(row)},${String(col)}`;
}

function computeBellmanValue(
  table: TransitionTable,
  stateIndex: number,
  actionIndex: number,
  values: StateValues,
  gamma: number
): number {
  const reward = table.rewards[stateIndex];
  let value = 0;
  forEachTransition(
    table,
    stateIndex,
    actionIndex,
    (nextState, probability) => {
      const nextVal = values.get(stateKey(table, nextState))
        ?? 0;
      value += probability * (reward + gamma * nextVal);
    }
  );
  return value;
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

function improvePolicy(
  table: TransitionTable,
  values: StateValues,
  gamma: number
): Policy {
  const newPolicy: Policy = new Map();
  const size = table.rows * table.cols;

  for (let s = 0; s < size; s++) {
    if (table.cellTypes[s] !== 1) {
      continue;
    }
    if (table.isTerminal[s] === 1) {
      continue;
    }

    const key = stateKey(table, s);
    let bestAction: Action = 'up';
    let bestValue = -Infinity;

    for (const action of ACTIONS) {
      const actionIndex = actionToIndex(action);
      const value = computeBellmanValue(
        table, s, actionIndex, values, gamma
      );
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    newPolicy.set(key, bestAction);
  }

  return newPolicy;
}

function policiesEqual(a: Policy, b: Policy): boolean {
  if (a.size !== b.size) {
    return false;
  }
  for (const [key, action] of a) {
    if (b.get(key) !== action) {
      return false;
    }
  }
  return true;
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
    currentPolicy = improvePolicy(table, currentValues, gamma);

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
