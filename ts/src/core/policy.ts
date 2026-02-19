// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { Action, Grid, Policy, StateValues } from './types';

const ACTIONS: Action[] = ['up', 'down', 'left', 'right'];

export function generateRandomPolicy(grid: Grid): Policy {
  const policy: Policy = new Map();

  grid.forEach((row, rowIndex) => {
    row.forEach((cellType, colIndex) => {
      if (cellType === 'floor') {
        const action = ACTIONS[
          Math.floor(Math.random() * ACTIONS.length)
        ];
        policy.set(`${String(rowIndex)},${String(colIndex)}`, action);
      }
    });
  });

  return policy;
}

export function initializeStateValues(grid: Grid): StateValues {
  const stateValues: StateValues = new Map();

  grid.forEach((row, rowIndex) => {
    row.forEach((cellType, colIndex) => {
      if (cellType !== 'wall') {
        stateValues.set(
          `${String(rowIndex)},${String(colIndex)}`,
          0
        );
      }
    });
  });

  return stateValues;
}
