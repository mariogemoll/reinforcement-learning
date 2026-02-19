// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  ACTIONS,
  actionToIndex,
  indexToAction,
  sampleNextState,
  type TransitionTable
} from '../core/mdp';
import type {
  Action,
  ActionValues,
  Grid,
  Policy
} from '../core/types';
import { initializeActionValues } from '../dp/shared';

export const DEFAULT_MAX_STEPS = 200;
export const MAX_EPISODES_PER_BATCH = 50;
export const TOTAL_BATCHES = 30;
export const SEED_POOL_SIZE = TOTAL_BATCHES * MAX_EPISODES_PER_BATCH;

export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateRandomPool(): Uint32Array {
  const pool = new Uint32Array(SEED_POOL_SIZE);
  for (let i = 0; i < SEED_POOL_SIZE; i++) {
    pool[i] = (Math.random() * 0xFFFFFFFF) >>> 0;
  }
  return pool;
}

export interface MCEpisode {
  path: { row: number; col: number }[];
  actions: Action[];
  terminalType: 'goal' | 'trap' | null;
}

function generateEpisode(
  table: TransitionTable,
  startRow: number,
  startCol: number,
  maxSteps: number,
  policy: Policy,
  epsilon: number,
  nextRandom: () => number
): MCEpisode {
  const path: { row: number; col: number }[] = [
    { row: startRow, col: startCol }
  ];
  const actions: Action[] = [];
  let stateIndex = startRow * table.cols + startCol;

  for (let step = 0; step < maxSteps; step++) {
    if (table.isTerminal[stateIndex] === 1) {
      const cellType = table.cellTypes[stateIndex];
      return {
        path,
        actions,
        terminalType: cellType === 2 ? 'goal' : 'trap'
      };
    }

    const epsilonCheck = nextRandom();
    const actionRand = nextRandom();
    const transitionRand = nextRandom();

    // Epsilon-greedy action selection
    let actionIndex: number;
    const row = Math.floor(stateIndex / table.cols);
    const col = stateIndex % table.cols;
    const key = `${String(row)},${String(col)}`;
    const greedyAction = policy.get(key);
    if (
      greedyAction !== undefined
      && epsilonCheck >= epsilon
    ) {
      actionIndex = actionToIndex(greedyAction);
    } else {
      actionIndex = Math.floor(actionRand * 4);
    }

    actions.push(indexToAction(actionIndex));

    const nextIndex = sampleNextState(
      table, stateIndex, actionIndex, transitionRand
    );
    const nextRow = Math.floor(nextIndex / table.cols);
    const nextCol = nextIndex % table.cols;
    path.push({ row: nextRow, col: nextCol });
    stateIndex = nextIndex;
  }

  return { path, actions, terminalType: null };
}

export interface MonteCarloSnapshot {
  actionValues: ActionValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
  episodes: MCEpisode[];
  totalEpisodes: number;
}

function stateKey(
  table: TransitionTable,
  index: number
): string {
  const row = Math.floor(index / table.cols);
  const col = index % table.cols;
  return `${String(row)},${String(col)}`;
}

function derivePolicy(actionValues: ActionValues): Policy {
  const policy: Policy = new Map();
  for (const [key, actionMap] of actionValues) {
    let bestAction: Action = 'up';
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

function cloneActionValues(
  values: ActionValues
): ActionValues {
  const copy: ActionValues = new Map();
  for (const [state, actionMap] of values) {
    copy.set(state, new Map(actionMap));
  }
  return copy;
}

export function runMonteCarlo(
  table: TransitionTable,
  grid: Grid,
  startRow: number,
  startCol: number,
  gamma: number,
  epsilon: number,
  episodesPerBatch: number,
  totalBatches: number,
  exploringStarts: boolean,
  firstVisit: boolean,
  maxStepsPerEpisode: number,
  seedPool: Uint32Array
): MonteCarloSnapshot[] {
  const qValues = initializeActionValues(grid);

  // Fix terminal state Q-values to their rewards
  const size = table.rows * table.cols;
  for (let s = 0; s < size; s++) {
    if (table.isTerminal[s] === 1) {
      const key = stateKey(table, s);
      const actionMap = qValues.get(key);
      if (actionMap) {
        for (const action of ACTIONS) {
          actionMap.set(action, table.rewards[s]);
        }
      }
    }
  }

  // Precompute non-terminal floor cells for exploring starts
  const floorCells: { row: number; col: number }[] = [];
  if (exploringStarts) {
    for (let r = 0; r < grid.length; r++) {
      for (let c = 0; c < grid[r].length; c++) {
        const si = r * table.cols + c;
        if (
          grid[r][c] === 'floor'
          && table.isTerminal[si] === 0
        ) {
          floorCells.push({ row: r, col: c });
        }
      }
    }
  }

  // Visit counts per (state, action)
  const visitCounts = new Map<string, number>();

  const snapshots: MonteCarloSnapshot[] = [];
  let totalEps = 0;

  // Initial snapshot
  let currentPolicy = derivePolicy(qValues);
  snapshots.push({
    actionValues: cloneActionValues(qValues),
    policy: new Map(currentPolicy),
    episodes: [],
    phase: 'evaluation',
    delta: 0,
    totalEpisodes: 0
  });

  for (let batch = 0; batch < totalBatches; batch++) {
    const episodes: MCEpisode[] = [];
    let maxDelta = 0;

    for (let ep = 0; ep < episodesPerBatch; ep++) {
      const globalEpisodeIndex =
        batch * MAX_EPISODES_PER_BATCH + ep;
      const rng = mulberry32(seedPool[globalEpisodeIndex]);

      let epStartRow = startRow;
      let epStartCol = startCol;
      if (exploringStarts && floorCells.length > 0) {
        const cell = floorCells[
          Math.floor(rng() * floorCells.length)
        ];
        epStartRow = cell.row;
        epStartCol = cell.col;
      }
      const episode = generateEpisode(
        table, epStartRow, epStartCol, maxStepsPerEpisode,
        currentPolicy, epsilon, rng
      );
      episodes.push(episode);
      totalEps++;

      // Backward pass: compute returns
      const returns = new Float64Array(episode.path.length);
      const lastIdx = episode.path.length - 1;
      const lastState =
        episode.path[lastIdx].row * table.cols
        + episode.path[lastIdx].col;
      returns[lastIdx] = table.rewards[lastState];

      for (let t = lastIdx - 1; t >= 0; t--) {
        const si =
          episode.path[t].row * table.cols
          + episode.path[t].col;
        returns[t] = table.rewards[si] + gamma * returns[t + 1];
      }

      // Forward pass: incremental mean update on Q(s,a)
      const visited = new Set<string>();
      for (let t = 0; t < episode.actions.length; t++) {
        const si =
          episode.path[t].row * table.cols
          + episode.path[t].col;
        if (table.isTerminal[si] === 1) {
          continue;
        }

        const key = stateKey(table, si);
        const action = episode.actions[t];
        const saKey = `${key}:${action}`;
        if (firstVisit && visited.has(saKey)) {
          continue;
        }
        visited.add(saKey);

        const count = (visitCounts.get(saKey) ?? 0) + 1;
        visitCounts.set(saKey, count);

        const actionMap = qValues.get(key);
        if (!actionMap) {
          continue;
        }
        const oldValue = actionMap.get(action) ?? 0;
        const newValue =
          oldValue + (returns[t] - oldValue) / count;
        actionMap.set(action, newValue);

        maxDelta = Math.max(
          maxDelta,
          Math.abs(newValue - oldValue)
        );
      }
    }

    // Evaluation snapshot: updated Q-values, old policy
    snapshots.push({
      actionValues: cloneActionValues(qValues),
      policy: new Map(currentPolicy),
      episodes,
      phase: 'evaluation',
      delta: maxDelta,
      totalEpisodes: totalEps
    });

    // Improvement: derive greedy policy from Q-values
    currentPolicy = derivePolicy(qValues);
    snapshots.push({
      actionValues: cloneActionValues(qValues),
      policy: new Map(currentPolicy),
      episodes: [],
      phase: 'improvement',
      delta: 0,
      totalEpisodes: totalEps
    });
  }

  return snapshots;
}
