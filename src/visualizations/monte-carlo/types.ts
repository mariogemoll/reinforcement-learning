// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { TransitionTable } from '../../core/mdp';
import type {
  ActionValues,
  Grid,
  Policy,
  StateValues
} from '../../core/types';
import type { MCEpisode } from '../../mc/monte-carlo';
import type { AgentPosition, Effect, ValueRange } from '../shared/types';

export interface MonteCarloBaseSnapshot {
  actionValues: ActionValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
  episodes: MCEpisode[];
  totalEpisodes: number;
}

export interface MonteCarloVisualization {
  destroy(): void;
}

export type ColorMode = 'value' | 'disagreement';

export interface MonteCarloVisualizationConfig<TSnapshot extends MonteCarloBaseSnapshot> {
  radioNamePrefix: string;

  computeSnapshots(
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
  ): TSnapshot[];

  renderGrid(
    canvas: HTMLCanvasElement,
    grid: Grid,
    cellSize: number,
    snapshot: TSnapshot,
    agents: AgentPosition[],
    effects: Effect[],
    displayCellValues: StateValues,
    displayActionValues: ActionValues,
    displayRange: ValueRange | null,
    optimalPolicy: Policy | null,
    showValues: boolean
  ): void;

  getPhaseExplanation(
    snapshot: TSnapshot,
    currentIndex: number,
    totalSnapshots: number,
    batchNumber: number,
    episodesPerBatch: number,
    epsilon: number,
    exploringStarts: boolean,
    firstVisit: boolean,
    countPolicyChanges: number
  ): string;
}

export type { MonteCarloSnapshot } from '../../mc/monte-carlo';
