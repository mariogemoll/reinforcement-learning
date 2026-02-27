// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { BreakoutAction } from './breakout';

export interface BreakoutUserPlayer {
  getAction(): BreakoutAction;
  reset(): void;
  onKeyDown(event: KeyboardEvent): void;
  onKeyUp(event: KeyboardEvent): void;
  onBlur(): void;
}

export function createBreakoutUserPlayer(): BreakoutUserPlayer {
  let action: BreakoutAction = 0;

  return {
    getAction(): BreakoutAction {
      return action;
    },

    reset(): void {
      action = 0;
    },

    onKeyDown(event: KeyboardEvent): void {
      if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') {
        event.preventDefault();
        action = 1;
      } else if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') {
        event.preventDefault();
        action = 2;
      } else if (event.key === ' ' || event.key === 'Spacebar') {
        event.preventDefault();
        action = 0;
      }
    },

    onKeyUp(event: KeyboardEvent): void {
      if (
        event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A' ||
        event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D'
      ) {
        action = 0;
      }
    },

    onBlur(): void {
      action = 0;
    }
  };
}
