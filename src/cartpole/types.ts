// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type CartPoleState = [number, number, number, number];

export type CartPoleAction = 0 | 1;

export interface CartPoleStepResult {
  state: CartPoleState;
  reward: number;
  terminated: boolean;
  truncated: boolean;
}
