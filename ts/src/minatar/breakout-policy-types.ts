// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { BreakoutAction, BreakoutState } from './breakout';

export interface BreakoutQValues {
  noop: number;
  left: number;
  right: number;
}

export interface BreakoutPolicyOutput {
  qValues: BreakoutQValues;
  action: BreakoutAction;
}

export type BreakoutPolicy = (state: BreakoutState) => BreakoutPolicyOutput;
