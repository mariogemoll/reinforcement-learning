// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { DP_DEFAULTS, REWARDS } from '../../config/constants';
import { MEDIUM_GRID } from '../../config/grids';
import { type GridworldVisualization,initializeGridworldVisualization } from './visualization';

export type { GridworldVisualization } from './visualization';

export function initGridworldVisualization(parent: HTMLElement): GridworldVisualization {
  return initializeGridworldVisualization({
    parent,
    layout: MEDIUM_GRID,
    cellSize: 50,
    successProb: DP_DEFAULTS.successProb,
    rewardModel: REWARDS
  });
}
