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
