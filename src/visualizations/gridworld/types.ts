import type { Action } from '../../core/types';

export type EpisodeStatus = 'playing' | 'reached-goal' | 'reached-trap';

export interface Position {
  row: number;
  col: number;
}

export interface LastMove {
  intended: Action;
  actual: Action;
  slipped: boolean;
}

export interface GridworldEpisodeState {
  row: number;
  col: number;
  steps: number;
  cumulativeReward: number;
  status: EpisodeStatus;
  trail: Position[];
  lastMove: LastMove | null;
}
