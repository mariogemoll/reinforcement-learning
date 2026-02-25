// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { MinAtarBreakoutVisualization } from './visualization';
import {
  initializeMinAtarBreakoutVisualization,
  type MinAtarBreakoutVisualization
} from './visualization';

export function initMinAtarBreakoutVisualization(
  parent: HTMLElement,
  weightsUrl?: string
): MinAtarBreakoutVisualization {
  return initializeMinAtarBreakoutVisualization(parent, weightsUrl);
}
