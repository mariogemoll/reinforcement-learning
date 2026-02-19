// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { CartpolePolicyVisualization } from './visualization';
import { loadDqnPolicy } from '../../cartpole/dqn-policy';
import {
  type CartpolePolicyVisualization,
  initializeCartpolePolicyVisualization
} from './visualization';

export async function initCartpolePolicyVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<CartpolePolicyVisualization> {
  const policy = await loadDqnPolicy(weightsUrl);
  return initializeCartpolePolicyVisualization(parent, policy);
}
