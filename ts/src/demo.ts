// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  type CartpolePolicyVisualization,
  initCartpolePolicyVisualization
} from './visualizations/cartpole';
import { type GridworldVisualization,initGridworldVisualization } from './visualizations/gridworld';
import {
  initMonteCarloVisualization,
  type MonteCarloVisualization
} from './visualizations/monte-carlo';
import {
  initPolicyIterationQVisualization,
  type PolicyIterationQVisualization
} from './visualizations/policy-iteration-q';
import {
  initPolicyIterationVVisualization,
  type PolicyIterationVVisualization } from './visualizations/policy-iteration-v';
import {
  initValueIterationQVisualization,
  type ValueIterationQVisualization
} from './visualizations/value-iteration-q';
import {
  initValueIterationVVisualization,
  type ValueIterationVVisualization
} from './visualizations/value-iteration-v';

let cartpolePolicyVisualization: CartpolePolicyVisualization | null = null;
let gridworldVisualization: GridworldVisualization | null = null;
let policyIterationVVisualization: PolicyIterationVVisualization | null = null;
let policyIterationQVisualization: PolicyIterationQVisualization | null = null;
let valueIterationVVisualization: ValueIterationVVisualization | null = null;
let valueIterationQVisualization: ValueIterationQVisualization | null = null;
let monteCarloVisualization: MonteCarloVisualization | null = null;

function initialize(): void {
  const cartpolePolicyPanel = document.getElementById(
    'cartpole-visualization'
  );
  if (cartpolePolicyPanel) {
    const panel = cartpolePolicyPanel;
    cartpolePolicyVisualization?.destroy();
    cartpolePolicyVisualization = null;
    void initCartpolePolicyVisualization(panel, '/public/dqn-weights.safetensors').then(viz => {
      cartpolePolicyVisualization = viz;
    });
  }

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

  const viVPanel = document.getElementById(
    'value-iteration-v-visualization'
  );
  if (viVPanel) {
    valueIterationVVisualization?.destroy();
    valueIterationVVisualization =
      initValueIterationVVisualization(viVPanel);
  }

  const viQPanel = document.getElementById(
    'value-iteration-q-visualization'
  );
  if (viQPanel) {
    valueIterationQVisualization?.destroy();
    valueIterationQVisualization =
      initValueIterationQVisualization(viQPanel);
  }

  const monteCarloPanel = document.getElementById(
    'monte-carlo-visualization'
  );
  if (monteCarloPanel) {
    monteCarloVisualization?.destroy();
    monteCarloVisualization =
      initMonteCarloVisualization(monteCarloPanel);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
