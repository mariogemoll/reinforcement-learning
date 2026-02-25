// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  type CartPoleVisualization,
  initCartPoleVisualization
} from './visualizations/cartpole';
import {
  type GridworldVisualization,
  initGridworldVisualization
} from './visualizations/gridworld';
import {
  initMinAtarBreakoutVisualization,
  type MinAtarBreakoutVisualization
} from './visualizations/minatar-breakout';
import {
  initMonteCarloVisualization,
  type MonteCarloVisualization
} from './visualizations/monte-carlo';
import {
  initPixelPongVisualization,
  type PixelPongVisualization
} from './visualizations/pixel-pong';
import {
  initPixelPongPolicyVisualization,
  type PixelPongPolicyVisualization
} from './visualizations/pixel-pong-policy';
import {
  initPolicyIterationQVisualization,
  type PolicyIterationQVisualization
} from './visualizations/policy-iteration-q';
import {
  initPolicyIterationVVisualization,
  type PolicyIterationVVisualization
} from './visualizations/policy-iteration-v';
import { initPongVisualization, type PongVisualization } from './visualizations/pong';
import {
  initPongPolicyVisualization,
  type PongPolicyVisualization
} from './visualizations/pong-policy';
import {
  initValueIterationQVisualization,
  type ValueIterationQVisualization
} from './visualizations/value-iteration-q';
import {
  initValueIterationVVisualization,
  type ValueIterationVVisualization
} from './visualizations/value-iteration-v';

let cartpoleVisualization: CartPoleVisualization | null = null;
let pongVisualization: PongVisualization | null = null;
let pongPolicyVisualization: PongPolicyVisualization | null = null;
let pixelPongVisualization: PixelPongVisualization | null = null;
let pixelPongPolicyVisualization: PixelPongPolicyVisualization | null = null;
let gridworldVisualization: GridworldVisualization | null = null;
let policyIterationVVisualization: PolicyIterationVVisualization | null = null;
let policyIterationQVisualization: PolicyIterationQVisualization | null = null;
let valueIterationVVisualization: ValueIterationVVisualization | null = null;
let valueIterationQVisualization: ValueIterationQVisualization | null = null;
let monteCarloVisualization: MonteCarloVisualization | null = null;
let minAtarBreakoutVisualization: MinAtarBreakoutVisualization | null = null;

function initialize(): void {
  const pongPanel = document.getElementById('pong-visualization');
  if (pongPanel) {
    pongVisualization?.destroy();
    pongVisualization = initPongVisualization(pongPanel);
  }

  const pongPolicyPanel = document.getElementById('pong-policy-visualization');
  if (pongPolicyPanel) {
    pongPolicyVisualization?.destroy();
    pongPolicyVisualization = null;
    void initPongPolicyVisualization(
      pongPolicyPanel,
      '/public/pong-weights.safetensors'
    ).then(viz => {
      pongPolicyVisualization = viz;
    }).catch(() => {
      const placeholder = pongPolicyPanel.querySelector('.placeholder');
      if (placeholder) {
        placeholder.textContent =
          'Pong policy: weights not found (place pong-weights.safetensors in ts/public/)';
      }
    });
  }

  const pixelPongPanel = document.getElementById('pixel-pong-visualization');
  if (pixelPongPanel) {
    pixelPongVisualization?.destroy();
    pixelPongVisualization = initPixelPongVisualization(pixelPongPanel);
  }

  const pixelPongPolicyPanel = document.getElementById('pixel-pong-policy-visualization');
  if (pixelPongPolicyPanel) {
    pixelPongPolicyVisualization?.destroy();
    pixelPongPolicyVisualization = null;
    void initPixelPongPolicyVisualization(
      pixelPongPolicyPanel,
      '/public/pixel-pong-weights.safetensors'
    ).then(viz => {
      pixelPongPolicyVisualization = viz;
    }).catch(() => {
      const placeholder = pixelPongPolicyPanel.querySelector('.placeholder');
      if (placeholder) {
        placeholder.textContent =
          'Pixel Pong policy: weights not found ' +
          '(place pixel-pong-weights.safetensors in ts/public/)';
      }
    });
  }

  const cartpolePolicyPanel = document.getElementById(
    'cartpole-visualization'
  );
  if (cartpolePolicyPanel) {
    const panel = cartpolePolicyPanel;
    cartpoleVisualization?.destroy();
    cartpoleVisualization = null;
    void initCartPoleVisualization(panel, '/public/dqn-weights.safetensors').then(viz => {
      cartpoleVisualization = viz;
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

  const minAtarBreakoutPanel = document.getElementById(
    'minatar-breakout-visualization'
  );
  if (minAtarBreakoutPanel) {
    minAtarBreakoutVisualization?.destroy();
    minAtarBreakoutVisualization =
      initMinAtarBreakoutVisualization(
        minAtarBreakoutPanel,
        '/public/dqn-minatar-breakout-weights.safetensors'
      );
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
