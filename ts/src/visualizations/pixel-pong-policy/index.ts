// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { PixelPongPolicyVisualization } from './visualization';
import { loadPixelPongNNPolicy } from '../../pong/pixel-nn-policy';
import {
  initializePixelPongPolicyVisualization,
  type PixelPongPolicyVisualization
} from './visualization';

export async function initPixelPongPolicyVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<PixelPongPolicyVisualization> {
  const policy = await loadPixelPongNNPolicy(weightsUrl);
  return initializePixelPongPolicyVisualization(parent, policy);
}
