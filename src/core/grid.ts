import type { CellType, Grid, GridLayout } from './types';

export function createGridFromLayout(layout: GridLayout): Grid {
  const grid: Grid = Array.from({ length: layout.rows }, () =>
    Array.from({ length: layout.cols }, () => 'floor' as CellType)
  );

  for (const [row, col] of layout.walls) {
    setCell(grid, row, col, 'wall');
  }
  for (const [row, col] of layout.goals) {
    setCell(grid, row, col, 'goal');
  }
  for (const [row, col] of layout.traps) {
    setCell(grid, row, col, 'trap');
  }

  return grid;
}

function setCell(grid: Grid, row: number, col: number, type: CellType): void {
  if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length) {
    return;
  }
  grid[row][col] = type;
}
