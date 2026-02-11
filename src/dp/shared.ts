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
  Grid,
  Policy,
  StateValues
} from '../core/types';

export function stateKey(
  table: TransitionTable,
  index: number
): string {
  const row = Math.floor(index / table.cols);
  const col = index % table.cols;
  return `${String(row)},${String(col)}`;
}

export function computeBellmanValue(
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

export function greedyPolicyFromValues(
  table: TransitionTable,
  values: StateValues,
  gamma: number
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

    policy.set(key, bestAction);
  }

  return policy;
}

export function policiesEqual(a: Policy, b: Policy): boolean {
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

export function cloneActionValues(
  values: ActionValues
): ActionValues {
  const copy: ActionValues = new Map();
  for (const [state, actionMap] of values) {
    copy.set(state, new Map(actionMap));
  }
  return copy;
}

export function initializeActionValues(grid: Grid): ActionValues {
  const actionValues: ActionValues = new Map();
  const actions: Action[] = ['up', 'down', 'left', 'right'];

  grid.forEach((row, rowIndex) => {
    row.forEach((cellType, colIndex) => {
      if (cellType === 'wall') {
        return;
      }
      const key = `${String(rowIndex)},${String(colIndex)}`;
      const actionMap = new Map<Action, number>();
      for (const action of actions) {
        actionMap.set(action, 0);
      }
      actionValues.set(key, actionMap);
    });
  });

  return actionValues;
}
