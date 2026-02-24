// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { PongVisualization } from './visualization';
import {
  initializePongVisualization,
  type PongVisualization
} from './visualization';

export function initPongVisualization(parent: HTMLElement): PongVisualization {
  return initializePongVisualization(parent);
}
