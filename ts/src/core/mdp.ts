// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { Action, Grid } from './types';

const ACTION_COUNT = 4;

export const ACTIONS: Action[] = ['up', 'down', 'left', 'right'];

export const CELL_CODE = {
  wall: 0,
  floor: 1,
  goal: 2,
  trap: 3
} as const;

export interface RewardModel {
  goal: number;
  trap: number;
  step: number;
}

export function actionToIndex(action: Action): number {
  switch (action) {
  case 'up':
    return 0;
  case 'down':
    return 1;
  case 'left':
    return 2;
  case 'right':
    return 3;
  }
}

export function indexToAction(index: number): Action {
  return ACTIONS[index] ?? 'up';
}

function getPerpActionIndices(actionIndex: number): [number, number] {
  if (actionIndex === 0 || actionIndex === 1) {
    return [2, 3];
  }
  return [0, 1];
}

export interface TransitionTable {
  rows: number;
  cols: number;
  cellTypes: Uint8Array;
  isTerminal: Uint8Array;
  rewards: Float64Array;
  transitions: Int32Array;
  successProb: number;
  perpProb: number;
}

export function buildTransitionTable(
  grid: Grid,
  successProb: number,
  rewardModel: RewardModel
): TransitionTable {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const size = rows * cols;
  const cellTypes = new Uint8Array(size);
  const isTerminal = new Uint8Array(size);
  const rewards = new Float64Array(size);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const cell = grid[row][col];
      const index = row * cols + col;

      if (cell === 'wall') {
        rewards[index] = 0;
        continue;
      }

      if (cell === 'goal') {
        cellTypes[index] = CELL_CODE.goal;
        isTerminal[index] = 1;
        rewards[index] = rewardModel.goal;
        continue;
      }

      if (cell === 'trap') {
        cellTypes[index] = CELL_CODE.trap;
        isTerminal[index] = 1;
        rewards[index] = rewardModel.trap;
        continue;
      }

      cellTypes[index] = CELL_CODE.floor;
      rewards[index] = rewardModel.step;
    }
  }

  const transitions = new Int32Array(size * ACTION_COUNT * 3);
  const perpProb = (1 - successProb) / 2;

  const isValid = (row: number, col: number): boolean => {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      return false;
    }
    return grid[row][col] !== 'wall';
  };

  const move = (row: number, col: number, actionIndex: number): [number, number] => {
    switch (actionIndex) {
    case 0:
      return [row - 1, col];
    case 1:
      return [row + 1, col];
    case 2:
      return [row, col - 1];
    case 3:
      return [row, col + 1];
    default:
      return [row, col];
    }
  };

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const stateIndex = row * cols + col;
      if (cellTypes[stateIndex] === CELL_CODE.wall) {
        continue;
      }

      for (let actionIndex = 0; actionIndex < ACTION_COUNT; actionIndex++) {
        const base = (stateIndex * ACTION_COUNT + actionIndex) * 3;
        const [nextRow, nextCol] = move(row, col, actionIndex);
        transitions[base] = isValid(nextRow, nextCol) ? nextRow * cols + nextCol : stateIndex;

        const [perp1, perp2] = getPerpActionIndices(actionIndex);
        const [p1Row, p1Col] = move(row, col, perp1);
        const [p2Row, p2Col] = move(row, col, perp2);
        transitions[base + 1] = isValid(p1Row, p1Col) ? p1Row * cols + p1Col : stateIndex;
        transitions[base + 2] = isValid(p2Row, p2Col) ? p2Row * cols + p2Col : stateIndex;
      }
    }
  }

  return {
    rows,
    cols,
    cellTypes,
    isTerminal,
    rewards,
    transitions,
    successProb,
    perpProb
  };
}

export function forEachTransition(
  table: TransitionTable,
  stateIndex: number,
  actionIndex: number,
  visitor: (nextState: number, probability: number) => void
): void {
  const base = (stateIndex * ACTION_COUNT + actionIndex) * 3;
  visitor(table.transitions[base], table.successProb);
  visitor(table.transitions[base + 1], table.perpProb);
  visitor(table.transitions[base + 2], table.perpProb);
}

export function sampleNextState(
  table: TransitionTable,
  stateIndex: number,
  actionIndex: number,
  rand: number = Math.random()
): number {
  const base = (stateIndex * ACTION_COUNT + actionIndex) * 3;
  if (rand <= table.successProb) {
    return table.transitions[base];
  }
  if (rand <= table.successProb + table.perpProb) {
    return table.transitions[base + 1];
  }
  return table.transitions[base + 2];
}
