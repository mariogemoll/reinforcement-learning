// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { CartPoleVisualization } from './visualization';
import { loadDqnPolicy } from '../../cartpole/dqn-policy';
import {
  type CartPoleVisualization,
  initializeCartPoleVisualization
} from './visualization';

export async function initCartPoleVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<CartPoleVisualization> {
  const policy = await loadDqnPolicy(weightsUrl);
  return initializeCartPoleVisualization(parent, policy);
}
