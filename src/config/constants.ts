// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export const DP_DEFAULTS = {
  successProb: 0.8,
  gamma: 0.9,
  theta: 0.01
} as const;

export const ANIMATION = {
  agentStepMs: 50,
  effectDurationMs: 600,
  spawnDurationMs: 3000
} as const;

export const REWARDS = {
  goal: 10,
  trap: -10,
  step: -0.1
} as const;

export const CANVAS_COLORS = {
  cells: {
    floor: '#90EE90',
    wall: '#5A2D0C',
    goal: '#22C55E',
    trap: '#DC143C',
    fallback: '#CCCCCC'
  },
  gridLine: '#FFFFFF',
  agent: {
    body: '#1E3A8A',
    highlight: '#93C5FD'
  },
  arrow: {
    success: 'rgba(22, 163, 74, 0.5)',
    slipped: 'rgba(234, 88, 12, 0.6)'
  },
  icons: {
    goal: '#166534',
    trap: '#8b0000'
  },
  terminalOverlay: {
    goalTint: 'rgba(34, 197, 94, 0.3)',
    trapTint: 'rgba(220, 20, 60, 0.3)',
    panelBackground: 'rgba(255, 255, 255, 0.85)',
    goalText: '#15803d',
    trapText: '#b91c1c',
    summaryText: '#555'
  },
  trail: {
    cellRgb: '65, 105, 225',
    dotRgb: '30, 58, 138',
    path: 'rgba(30, 58, 138, 0.3)'
  }
} as const;
