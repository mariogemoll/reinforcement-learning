// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type PongAction = 0 | 1 | 2; // noop, up, down

export interface PongState {
  ballRow: number;
  ballCol: number;
  ballVRow: number;
  ballVCol: number;
  p1Center: number; // player paddle (left, col = 0)
  p2Center: number; // AI paddle    (right, col = WIDTH-1)
  time: number;
}

export interface PongStepResult {
  state: PongState;
  done: boolean;
}
