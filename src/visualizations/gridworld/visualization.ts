import { createGridFromLayout } from '../../core/grid';
import type { RewardModel } from '../../core/mdp';
import type { Action, GridLayout } from '../../core/types';
import { createGridworldModel } from './model';
import { createGridworldVizDom } from './ui';
import { renderGridworld, updateStats, updateTerminalOverlay } from './view';

interface InitializeGridworldVisualizationParams {
  parent: HTMLElement;
  layout: GridLayout;
  cellSize: number;
  successProb: number;
  rewardModel: RewardModel;
}

export interface GridworldVisualization {
  destroy(): void;
}

export function initializeGridworldVisualization(
  params: InitializeGridworldVisualizationParams
): GridworldVisualization {
  const { parent, layout, cellSize, successProb, rewardModel } = params;
  const grid = createGridFromLayout(layout);
  const startRow = layout.agentStart?.[0] ?? 0;
  const startCol = layout.agentStart?.[1] ?? 0;
  const initialSlipperiness = 1 - successProb;

  const model = createGridworldModel({
    grid,
    startRow,
    startCol,
    successProb,
    rewardModel
  });

  const dom = createGridworldVizDom(initialSlipperiness);
  const { container, canvas, resetBtn, slipperinessSlider, slipperinessValueEl, dpadButtons } = dom;

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(container);
  } else {
    parent.appendChild(container);
  }

  canvas.width = layout.cols * cellSize;
  canvas.height = layout.rows * cellSize;
  container.style.setProperty(
    '--gridworld-viz-grid-height',
    String(canvas.height)
  );

  const render = (): void => {
    const state = model.getState();
    updateStats(dom, state);
    renderGridworld(canvas, grid, cellSize, state);
    updateTerminalOverlay(dom, state, rewardModel);
  };

  const move = (action: Action): void => {
    model.step(action);
    render();
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    let action: Action | null = null;
    switch (event.key) {
    case 'ArrowUp':
    case 'w':
    case 'W':
      action = 'up';
      break;
    case 'ArrowDown':
    case 's':
    case 'S':
      action = 'down';
      break;
    case 'ArrowLeft':
    case 'a':
    case 'A':
      action = 'left';
      break;
    case 'ArrowRight':
    case 'd':
    case 'D':
      action = 'right';
      break;
    }

    if (action) {
      event.preventDefault();
      move(action);
    }
  };

  const handleCanvasClick = (): void => {
    canvas.focus();
  };

  const handleResetClick = (): void => {
    model.reset();
    render();
    canvas.focus();
  };

  const handleSlipperinessInput = (): void => {
    const slipperiness = model.setSlipperiness(Number(slipperinessSlider.value));
    slipperinessValueEl.textContent = `${String(Math.round(slipperiness * 100))}%`;
  };

  const dpadHandlers = dpadButtons.map((button) => {
    const handler = (): void => {
      const action = button.dataset.action as Action | undefined;
      if (!action) {
        return;
      }
      move(action);
    };
    button.addEventListener('click', handler);
    return { button, handler };
  });

  canvas.addEventListener('keydown', handleKeyDown);
  canvas.addEventListener('click', handleCanvasClick);
  resetBtn.addEventListener('click', handleResetClick);
  slipperinessSlider.addEventListener('input', handleSlipperinessInput);

  render();

  return {
    destroy(): void {
      canvas.removeEventListener('keydown', handleKeyDown);
      canvas.removeEventListener('click', handleCanvasClick);
      resetBtn.removeEventListener('click', handleResetClick);
      slipperinessSlider.removeEventListener('input', handleSlipperinessInput);
      dpadHandlers.forEach(({ button, handler }) => {
        button.removeEventListener('click', handler);
      });
      container.remove();
    }
  };
}
