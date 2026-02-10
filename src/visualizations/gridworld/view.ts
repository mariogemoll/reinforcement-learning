// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { CANVAS_COLORS } from '../../config/constants';
import type { RewardModel } from '../../core/mdp';
import type { Action, Grid } from '../../core/types';
import { drawAgent, drawIntendedArrow } from '../../rendering/agent';
import { drawGoal, drawTrapCross as drawTrap } from '../../rendering/cell-icons';
import type { GridworldEpisodeState, Position } from './types';
import type { GridworldVizDom } from './ui';

const ACTION_ARROWS: Record<Action, string> = {
  up: '\u2191', down: '\u2193', left: '\u2190', right: '\u2192'
};

export function updateStats(dom: GridworldVizDom, state: Readonly<GridworldEpisodeState>): void {
  dom.stepsValue.textContent = String(state.steps);
  dom.rewardValue.textContent = state.cumulativeReward.toFixed(1);

  if (state.lastMove) {
    const { intended, actual, slipped } = state.lastMove;
    const moveClass = slipped ? 'move-slip' : 'move-ok';
    dom.intendedMoveValue.className = '';
    dom.intendedMoveValue.textContent = `${ACTION_ARROWS[intended]} ${intended}`;
    dom.actualMoveValue.className = moveClass;
    dom.actualMoveValue.textContent = `${ACTION_ARROWS[actual]} ${actual}`;
  } else {
    dom.intendedMoveValue.className = 'move-pending';
    dom.intendedMoveValue.textContent = '-';
    dom.actualMoveValue.className = 'move-pending';
    dom.actualMoveValue.textContent = '-';
  }
}

export function renderGridworld(
  canvas: HTMLCanvasElement,
  grid: Grid,
  cellSize: number,
  state: Readonly<GridworldEpisodeState>
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawGridCells(ctx, grid, cellSize);
  drawTrail(ctx, state.trail, cellSize);
  drawAgent(ctx, state.col * cellSize, state.row * cellSize, cellSize);

  if (state.lastMove) {
    const prevPos = state.trail[state.trail.length - 2];
    const px = prevPos.col * cellSize + cellSize / 2;
    const py = prevPos.row * cellSize + cellSize / 2;
    drawIntendedArrow(ctx, px, py, cellSize, state.lastMove.intended, state.lastMove.slipped);
  }

}

export function updateTerminalOverlay(
  dom: GridworldVizDom,
  state: Readonly<GridworldEpisodeState>,
  rewardModel: RewardModel
): void {
  if (state.status === 'playing') {
    dom.terminalOverlay.hidden = true;
    dom.terminalOverlay.classList.remove('is-goal', 'is-trap');
    return;
  }

  const isGoal = state.status === 'reached-goal';
  dom.terminalOverlay.hidden = false;
  dom.terminalOverlay.classList.toggle('is-goal', isGoal);
  dom.terminalOverlay.classList.toggle('is-trap', !isGoal);
  dom.terminalTitle.textContent = isGoal
    ? `Goal reached! +${String(rewardModel.goal)}`
    : `Fell in trap! ${String(rewardModel.trap)}`;
  const stepsText = String(state.steps);
  const totalRewardText = state.cumulativeReward.toFixed(1);
  dom.terminalSummary.textContent = `${stepsText} steps, total reward: ${totalRewardText}`;
}

function drawGridCells(
  ctx: CanvasRenderingContext2D,
  grid: Grid,
  cellSize: number
): void {
  grid.forEach((row, rowIndex) => {
    row.forEach((cellType, colIndex) => {
      const x = colIndex * cellSize;
      const y = rowIndex * cellSize;

      ctx.fillStyle = getCellColor(cellType);
      ctx.fillRect(x, y, cellSize, cellSize);

      ctx.strokeStyle = CANVAS_COLORS.gridLine;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);

      if (cellType === 'goal') {drawGoal(ctx, x, y, cellSize);}
      else if (cellType === 'trap') {drawTrap(ctx, x, y, cellSize);}
    });
  });
}

function drawTrail(
  ctx: CanvasRenderingContext2D,
  trail: Position[],
  cellSize: number
): void {
  for (let i = 0; i < trail.length - 1; i++) {
    const { row, col } = trail[i];
    const x = col * cellSize;
    const y = row * cellSize;
    const alpha = 0.15 + 0.25 * (i / Math.max(1, trail.length - 1));
    ctx.fillStyle = `rgba(${CANVAS_COLORS.trail.cellRgb}, ${String(alpha)})`;
    ctx.fillRect(x, y, cellSize, cellSize);

    ctx.fillStyle = `rgba(${CANVAS_COLORS.trail.dotRgb}, ${String(alpha + 0.1)})`;
    ctx.beginPath();
    ctx.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.1, 0, Math.PI * 2);
    ctx.fill();
  }

  if (trail.length > 1) {
    ctx.strokeStyle = CANVAS_COLORS.trail.path;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const first = trail[0];
    ctx.moveTo(first.col * cellSize + cellSize / 2, first.row * cellSize + cellSize / 2);
    for (let i = 1; i < trail.length; i++) {
      const { row, col } = trail[i];
      ctx.lineTo(col * cellSize + cellSize / 2, row * cellSize + cellSize / 2);
    }
    ctx.stroke();
  }
}


function getCellColor(cellType: Grid[number][number]): string {
  switch (cellType) {
  case 'floor': return CANVAS_COLORS.cells.floor;
  case 'wall': return CANVAS_COLORS.cells.wall;
  case 'goal': return CANVAS_COLORS.cells.goal;
  case 'trap': return CANVAS_COLORS.cells.trap;
  default: return CANVAS_COLORS.cells.fallback;
  }
}
