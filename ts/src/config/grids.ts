// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { GridLayout } from '../core/types';

export const MEDIUM_GRID: GridLayout = {
  rows: 6,
  cols: 8,
  walls: [
    [0, 3],
    [1, 1],
    [1, 3],
    [1, 5],
    [1, 6],
    [3, 1],
    [3, 2],
    [3, 4],
    [3, 5],
    [5, 1],
    [5, 2],
    [5, 4]
  ],
  goals: [[5, 7]],
  traps: [[3, 7]],
  agentStart: [0, 0]
};
