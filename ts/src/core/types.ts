// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type CellType = 'floor' | 'wall' | 'goal' | 'trap';

export type Action = 'up' | 'down' | 'left' | 'right';

export type Grid = CellType[][];

export interface GridLayout {
  rows: number;
  cols: number;
  walls: [number, number][];
  goals: [number, number][];
  traps: [number, number][];
  agentStart?: [number, number];
}

export type Policy = Map<string, Action>;

export type StateValues = Map<string, number>;

export type ActionValues = Map<string, Map<Action, number>>;
