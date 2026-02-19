// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  actionToIndex,
  buildTransitionTable,
  CELL_CODE,
  type RewardModel,
  sampleNextState
} from '../../core/mdp';
import type { Action, Grid } from '../../core/types';
import type { GridworldEpisodeState } from './types';

interface CreateGridworldModelParams {
  grid: Grid;
  startRow: number;
  startCol: number;
  successProb: number;
  rewardModel: RewardModel;
}

export interface GridworldModel {
  getState(): Readonly<GridworldEpisodeState>;
  step(intended: Action): void;
  reset(): void;
  setSlipperiness(slipperiness: number): number;
}

export function createGridworldModel(params: CreateGridworldModelParams): GridworldModel {
  const { grid, startRow, startCol, rewardModel } = params;
  let successProb = params.successProb;
  let table = buildTransitionTable(grid, successProb, rewardModel);

  const state: GridworldEpisodeState = {
    row: startRow,
    col: startCol,
    steps: 0,
    cumulativeReward: 0,
    status: 'playing',
    trail: [{ row: startRow, col: startCol }],
    lastMove: null
  };

  const reset = (): void => {
    state.row = startRow;
    state.col = startCol;
    state.steps = 0;
    state.cumulativeReward = 0;
    state.status = 'playing';
    state.trail = [{ row: startRow, col: startCol }];
    state.lastMove = null;
  };

  const step = (intended: Action): void => {
    if (state.status !== 'playing') {
      return;
    }

    const stateIndex = state.row * table.cols + state.col;
    const actionIndex = actionToIndex(intended);
    const nextIndex = sampleNextState(table, stateIndex, actionIndex);
    const nextRow = Math.floor(nextIndex / table.cols);
    const nextCol = nextIndex % table.cols;

    const dr = nextRow - state.row;
    const dc = nextCol - state.col;
    let actual: Action = intended;
    if (dr === -1 && dc === 0) {
      actual = 'up';
    } else if (dr === 1 && dc === 0) {
      actual = 'down';
    } else if (dr === 0 && dc === -1) {
      actual = 'left';
    } else if (dr === 0 && dc === 1) {
      actual = 'right';
    }

    state.lastMove = { intended, actual, slipped: actual !== intended };
    state.row = nextRow;
    state.col = nextCol;
    state.steps++;

    const reward = table.rewards[nextIndex];
    state.cumulativeReward += reward;
    state.trail.push({ row: nextRow, col: nextCol });

    if (table.isTerminal[nextIndex]) {
      const cellType = table.cellTypes[nextIndex];
      state.status = cellType === CELL_CODE.goal ? 'reached-goal' : 'reached-trap';
    }
  };

  const setSlipperiness = (slipperinessInput: number): number => {
    const slipperiness = Math.min(1, Math.max(0, slipperinessInput));
    successProb = 1 - slipperiness;
    table = buildTransitionTable(grid, successProb, rewardModel);
    return slipperiness;
  };

  return {
    getState(): Readonly<GridworldEpisodeState> {
      return state;
    },
    step,
    reset,
    setSlipperiness
  };
}
