// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { RewardModel, TransitionTable } from '../../core/mdp';
import type { Grid, GridLayout, Policy, StateValues } from '../../core/types';

export interface PathStep {
  row: number;
  col: number;
}

export interface Effect {
  type: 'poof' | 'burn';
  row: number;
  col: number;
  startTime: number;
  duration: number;
}

export interface Agent {
  id: number;
  path: PathStep[];
  spawnTime: number;
  terminalType: 'goal' | 'trap' | null;
}

export interface BaseSnapshot {
  stateValues: StateValues;
  policy: Policy;
  phase: 'evaluation' | 'improvement';
  delta: number;
}

export interface ValueRange {
  minVal: number;
  maxVal: number;
}

export interface AgentPosition {
  row: number;
  col: number;
}

export interface ExtraCheckbox {
  label: string;
  ariaLabel: string;
  initialChecked: boolean;
  onChange(checked: boolean): void;
}

export interface DPVisualizationConfig<TSnapshot extends BaseSnapshot> {
  initialValuesLabel: string;
  radioGroupName: string;
  extraCheckboxes: ExtraCheckbox[];

  computeSnapshots(
    table: TransitionTable,
    initialPolicy: Policy,
    gamma: number,
    theta: number
  ): TSnapshot[];

  createZeroValues(grid: Grid): void;
  randomizeValues(): void;

  renderGrid(
    canvas: HTMLCanvasElement,
    grid: Grid,
    cellSize: number,
    snapshot: TSnapshot,
    agents: AgentPosition[],
    effects: Effect[],
    valueRange: ValueRange | null
  ): void;

  getPhaseExplanation(
    snapshot: TSnapshot,
    nextPhase: 'evaluation' | 'improvement' | null,
    atStart: boolean,
    atEnd: boolean,
    initialValueMode: 'zero' | 'random'
  ): string;
}

export interface InitParams {
  parent: HTMLElement;
  layout: GridLayout;
  cellSize: number;
  successProb: number;
  rewardModel: RewardModel;
  gamma: number;
  theta: number;
}

export interface DPVisualization {
  destroy(): void;
}
