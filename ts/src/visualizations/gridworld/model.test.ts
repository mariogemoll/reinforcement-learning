// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it, vi } from 'vitest';

import type { Grid } from '../../core/types';
import { createGridworldModel } from './model';

const rewardModel = {
  goal: 10,
  trap: -10,
  step: -0.1
};

describe('createGridworldModel', () => {
  it('initializes with start position and zeroed episode stats', () => {
    const grid: Grid = [
      ['floor', 'floor'],
      ['floor', 'goal']
    ];

    const model = createGridworldModel({
      grid,
      startRow: 0,
      startCol: 0,
      successProb: 1,
      rewardModel
    });

    const state = model.getState();
    expect(state.row).toBe(0);
    expect(state.col).toBe(0);
    expect(state.steps).toBe(0);
    expect(state.cumulativeReward).toBe(0);
    expect(state.status).toBe('playing');
    expect(state.trail).toEqual([{ row: 0, col: 0 }]);
    expect(state.lastMove).toBeNull();
  });

  it('applies deterministic successful moves and step reward', () => {
    const grid: Grid = [['floor', 'floor', 'floor']];

    const model = createGridworldModel({
      grid,
      startRow: 0,
      startCol: 0,
      successProb: 1,
      rewardModel
    });

    model.step('right');
    const state = model.getState();

    expect(state.row).toBe(0);
    expect(state.col).toBe(1);
    expect(state.steps).toBe(1);
    expect(state.cumulativeReward).toBe(-0.1);
    expect(state.lastMove).toEqual({
      intended: 'right',
      actual: 'right',
      slipped: false
    });
  });

  it('marks terminal goal and ignores subsequent moves', () => {
    const grid: Grid = [['floor', 'goal']];

    const model = createGridworldModel({
      grid,
      startRow: 0,
      startCol: 0,
      successProb: 1,
      rewardModel
    });

    model.step('right');
    const afterGoal = model.getState();
    expect(afterGoal.status).toBe('reached-goal');
    expect(afterGoal.col).toBe(1);
    expect(afterGoal.steps).toBe(1);
    expect(afterGoal.cumulativeReward).toBe(10);

    model.step('left');
    const afterIgnoredMove = model.getState();
    expect(afterIgnoredMove.col).toBe(1);
    expect(afterIgnoredMove.steps).toBe(1);
    expect(afterIgnoredMove.cumulativeReward).toBe(10);
  });

  it('resets state after progress', () => {
    const grid: Grid = [['floor', 'goal']];

    const model = createGridworldModel({
      grid,
      startRow: 0,
      startCol: 0,
      successProb: 1,
      rewardModel
    });

    model.step('right');
    model.reset();

    const state = model.getState();
    expect(state.row).toBe(0);
    expect(state.col).toBe(0);
    expect(state.steps).toBe(0);
    expect(state.cumulativeReward).toBe(0);
    expect(state.status).toBe('playing');
    expect(state.trail).toEqual([{ row: 0, col: 0 }]);
    expect(state.lastMove).toBeNull();
  });

  it('uses slipperiness to force perpendicular movement and marks slip', () => {
    const grid: Grid = [
      ['floor', 'floor', 'floor'],
      ['floor', 'floor', 'floor'],
      ['floor', 'floor', 'floor']
    ];
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.25);

    try {
      const model = createGridworldModel({
        grid,
        startRow: 1,
        startCol: 1,
        successProb: 1,
        rewardModel
      });

      const clamped = model.setSlipperiness(5);
      expect(clamped).toBe(1);

      model.step('up');
      const state = model.getState();
      expect(state.row).toBe(1);
      expect(state.col).toBe(0);
      expect(state.lastMove).toEqual({
        intended: 'up',
        actual: 'left',
        slipped: true
      });
    } finally {
      randomSpy.mockRestore();
    }
  });
});
