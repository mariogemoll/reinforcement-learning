// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { type GridworldVisualization,initGridworldVisualization } from './visualizations/gridworld';
import {
  initPolicyIterationQVisualization,
  type PolicyIterationQVisualization
} from './visualizations/policy-iteration-q';
import {
  initPolicyIterationVVisualization,
  type PolicyIterationVVisualization } from './visualizations/policy-iteration-v';

let gridworldVisualization: GridworldVisualization | null = null;
let policyIterationVVisualization: PolicyIterationVVisualization | null = null;
let policyIterationQVisualization: PolicyIterationQVisualization | null = null;

function initialize(): void {
  const gridworldPanel = document.getElementById(
    'gridworld-visualization'
  );
  if (gridworldPanel) {
    gridworldVisualization?.destroy();
    gridworldVisualization =
      initGridworldVisualization(gridworldPanel);
  }

  const piVPanel = document.getElementById(
    'policy-iteration-v-visualization'
  );
  if (piVPanel) {
    policyIterationVVisualization?.destroy();
    policyIterationVVisualization =
      initPolicyIterationVVisualization(piVPanel);
  }

  const piQPanel = document.getElementById(
    'policy-iteration-q-visualization'
  );
  if (piQPanel) {
    policyIterationQVisualization?.destroy();
    policyIterationQVisualization =
      initPolicyIterationQVisualization(piQPanel);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
