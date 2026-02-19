// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ANIMATION } from '../../config/constants';
import {
  actionToIndex,
  sampleNextState,
  type TransitionTable } from '../../core/mdp';
import type { Grid, Policy } from '../../core/types';
import type { Agent, Effect, PathStep } from './types';

export interface AnimationLoop {
  readonly running: boolean;
  start(): void;
  stop(): void;
}

export function createAnimationLoop(
  onFrame: (timestamp: number) => boolean
): AnimationLoop {
  let frameId: number | null = null;
  let isRunning = false;

  function loop(timestamp: number): void {
    if (!isRunning) {
      return;
    }
    const shouldContinue = onFrame(timestamp);
    if (shouldContinue) {
      frameId = requestAnimationFrame(loop);
    } else {
      isRunning = false;
      frameId = null;
    }
  }

  return {
    start(): void {
      if (isRunning) {
        return;
      }
      isRunning = true;
      frameId = requestAnimationFrame(loop);
    },
    stop(): void {
      isRunning = false;
      if (frameId !== null) {
        cancelAnimationFrame(frameId);
        frameId = null;
      }
    },
    get running(): boolean {
      return isRunning;
    }
  };
}

export function precalculatePath(
  startRow: number,
  startCol: number,
  policy: Policy,
  table: TransitionTable,
  grid: Grid
): { path: PathStep[]; terminalType: 'goal' | 'trap' | null } {
  const path: PathStep[] = [{ row: startRow, col: startCol }];
  let row = startRow;
  let col = startCol;
  const maxSteps = 1000;

  for (let step = 0; step < maxSteps; step++) {
    const cell = grid[row]?.[col];
    if (cell === 'goal' || cell === 'trap') {
      return {
        path,
        terminalType: cell === 'goal' ? 'goal' : 'trap'
      };
    }

    const key = `${String(row)},${String(col)}`;
    const action = policy.get(key);
    if (action === undefined) {
      break;
    }

    const stateIndex = row * table.cols + col;
    const actionIndex = actionToIndex(action);
    const nextIndex = sampleNextState(
      table, stateIndex, actionIndex
    );
    row = Math.floor(nextIndex / table.cols);
    col = nextIndex % table.cols;

    path.push({ row, col });
  }

  return { path, terminalType: null };
}

export function getAgentPosition(
  agent: Agent,
  timestamp: number,
  stepDuration: number
): { row: number; col: number; finished: boolean } {
  const elapsed = timestamp - agent.spawnTime;
  if (elapsed < 0) {
    return {
      row: agent.path[0].row,
      col: agent.path[0].col,
      finished: false
    };
  }

  const stepIndex = elapsed / stepDuration;
  const currentStep = Math.floor(stepIndex);
  const progress = stepIndex - currentStep;

  if (currentStep >= agent.path.length - 1) {
    const last = agent.path[agent.path.length - 1];
    return { row: last.row, col: last.col, finished: true };
  }

  const from = agent.path[currentStep];
  const to = agent.path[currentStep + 1];

  const eased = progress < 0.5
    ? 2 * progress * progress
    : 1 - Math.pow(-2 * progress + 2, 2) / 2;

  return {
    row: from.row + (to.row - from.row) * eased,
    col: from.col + (to.col - from.col) * eased,
    finished: false
  };
}

export function updateAgents(
  agents: Agent[],
  effects: Effect[],
  timestamp: number,
  stepDuration: number
): void {
  const toRemove: number[] = [];

  agents.forEach((agent, index) => {
    const pos = getAgentPosition(
      agent, timestamp, stepDuration
    );
    if (pos.finished) {
      if (agent.terminalType !== null) {
        effects.push({
          type: agent.terminalType === 'goal'
            ? 'poof'
            : 'burn',
          row: pos.row,
          col: pos.col,
          startTime: Date.now(),
          duration: ANIMATION.effectDurationMs
        });
      }
      toRemove.push(index);
    }
  });

  for (let i = toRemove.length - 1; i >= 0; i--) {
    agents.splice(toRemove[i], 1);
  }
}
