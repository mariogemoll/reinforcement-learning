// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { PixelPongVisualization } from './visualization';
import {
  initializePixelPongVisualization,
  type PixelPongVisualization
} from './visualization';

export function initPixelPongVisualization(parent: HTMLElement): PixelPongVisualization {
  return initializePixelPongVisualization(parent);
}
