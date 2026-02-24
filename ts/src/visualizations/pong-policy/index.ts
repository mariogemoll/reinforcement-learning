// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { PongPolicyVisualization } from './visualization';
import { loadPongNNPolicy } from '../../pong/nn-policy';
import {
  initializePongPolicyVisualization,
  type PongPolicyVisualization
} from './visualization';

export async function initPongPolicyVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<PongPolicyVisualization> {
  const policy = await loadPongNNPolicy(weightsUrl);
  return initializePongPolicyVisualization(parent, policy);
}
