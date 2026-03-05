// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { CartPoleVisualization } from '../visualizations/cartpole/visualization';
import {
  type CartPoleVisualization,
  initializeCartPoleVisualization
} from '../visualizations/cartpole/visualization';
import { reinforceToDqnAdapter } from './policy-adapter';
import { loadReinforcePolicy, type ReinforcePolicy } from './reinforce-policy';

export async function initCartPoleReinforceVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<CartPoleVisualization> {
  const policy: ReinforcePolicy = await loadReinforcePolicy(weightsUrl);
  const dqnPolicy = reinforceToDqnAdapter(policy);
  return initializeCartPoleVisualization(parent, dqnPolicy);
}
