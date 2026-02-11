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

export interface QDPSnapshot {
  actionValues: ActionValues;
  stateValues: StateValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
}

export interface QDPResult {
  snapshots: QDPSnapshot[];
  finalPolicy: Policy;
  finalActionValues: ActionValues;
  finalStateValues: StateValues;
}

function stateKey(
  table: TransitionTable,
  index: number
): string {
  const row = Math.floor(index / table.cols);
  const col = index % table.cols;
  return `${String(row)},${String(col)}`;
}

function getStateActionValue(
  actionValues: ActionValues,
  key: string,
  action: Action,
  fallback = 0
): number {
  return actionValues.get(key)?.get(action) ?? fallback;
}

function getPolicyValue(
  table: TransitionTable,
  stateIndex: number,
  policy: Policy,
  actionValues: ActionValues
): number {
  const key = stateKey(table, stateIndex);
  const policyAction = policy.get(key);
  if (policyAction === undefined) {
    return table.rewards[stateIndex];
  }
  return getStateActionValue(
    actionValues,
    key,
    policyAction,
    table.rewards[stateIndex]
  );
}

function deriveStateValues(
  table: TransitionTable,
  policy: Policy,
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

    values.set(
      key,
      getPolicyValue(table, s, policy, actionValues)
    );
  }

  return values;
}

function computeActionValue(
  table: TransitionTable,
  stateIndex: number,
  actionIndex: number,
  policy: Policy,
  actionValues: ActionValues,
  gamma: number
): number {
  const reward = table.rewards[stateIndex];
  let value = 0;

  forEachTransition(
    table,
    stateIndex,
    actionIndex,
    (nextState, probability) => {
      const nextV = getPolicyValue(
        table,
        nextState,
        policy,
        actionValues
      );
      value += probability * (reward + gamma * nextV);
    }
  );

  return value;
}

function evaluationSweep(
  table: TransitionTable,
  policy: Policy,
  currentActionValues: ActionValues,
  gamma: number
): { actionValues: ActionValues; delta: number } {
  const newActionValues: ActionValues = new Map();
  let delta = 0;
  const size = table.rows * table.cols;

  for (let s = 0; s < size; s++) {
    if (table.cellTypes[s] === 0) {
      continue;
    }

    const key = stateKey(table, s);
    const nextActionMap = new Map<Action, number>();

    if (table.isTerminal[s] === 1) {
      for (const action of ACTIONS) {
        nextActionMap.set(action, table.rewards[s]);
      }
      newActionValues.set(key, nextActionMap);
      continue;
    }

    for (const action of ACTIONS) {
      const actionIndex = actionToIndex(action);
      const value = computeActionValue(
        table,
        s,
        actionIndex,
        policy,
        currentActionValues,
        gamma
      );
      nextActionMap.set(action, value);
      const oldValue = getStateActionValue(
        currentActionValues,
        key,
        action,
        0
      );
      delta = Math.max(delta, Math.abs(value - oldValue));
    }

    newActionValues.set(key, nextActionMap);
  }

  return { actionValues: newActionValues, delta };
}

function improvePolicy(
  table: TransitionTable,
  actionValues: ActionValues
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
    let bestAction: Action = ACTIONS[0];
    let bestValue = -Infinity;

    for (const action of ACTIONS) {
      const value = getStateActionValue(
        actionValues,
        key,
        action,
        -Infinity
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

export function runPolicyIterationQ(
  table: TransitionTable,
  initialPolicy: Policy,
  initialActionValues: ActionValues,
  gamma: number,
  theta: number
): QDPResult {
  const snapshots: QDPSnapshot[] = [];
  let currentPolicy = new Map(initialPolicy);
  let currentActionValues: ActionValues = new Map(
    initialActionValues
  );

  const cloneActionValues = (
    values: ActionValues
  ): ActionValues => {
    const copy: ActionValues = new Map();
    for (const [state, actionMap] of values) {
      copy.set(state, new Map(actionMap));
    }
    return copy;
  };

  const maxOuterIterations = 50;
  let currentStateValues = deriveStateValues(
    table,
    currentPolicy,
    currentActionValues
  );

  snapshots.push({
    actionValues: cloneActionValues(currentActionValues),
    stateValues: new Map(currentStateValues),
    policy: new Map(currentPolicy),
    phase: 'evaluation',
    delta: 0
  });

  for (let outer = 0; outer < maxOuterIterations; outer++) {
    // Policy evaluation in Q-space under fixed pi.
    const maxEvalSweeps = 200;
    for (let sweep = 0; sweep < maxEvalSweeps; sweep++) {
      const result = evaluationSweep(
        table,
        currentPolicy,
        currentActionValues,
        gamma
      );
      currentActionValues = result.actionValues;
      currentStateValues = deriveStateValues(
        table,
        currentPolicy,
        currentActionValues
      );

      snapshots.push({
        actionValues: cloneActionValues(currentActionValues),
        stateValues: new Map(currentStateValues),
        policy: new Map(currentPolicy),
        phase: 'evaluation',
        delta: result.delta
      });

      if (result.delta < theta) {
        break;
      }
    }

    // Greedy policy improvement from current Q-values.
    const oldPolicy = currentPolicy;
    currentPolicy = improvePolicy(table, currentActionValues);
    currentStateValues = deriveStateValues(
      table,
      currentPolicy,
      currentActionValues
    );

    snapshots.push({
      actionValues: cloneActionValues(currentActionValues),
      stateValues: new Map(currentStateValues),
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
    finalActionValues: currentActionValues,
    finalStateValues: currentStateValues
  };
}
