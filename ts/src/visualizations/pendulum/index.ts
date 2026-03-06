// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type { PendulumVisualization } from './visualization';
import { loadPendulumPolicy } from '../../pendulum/policy';
import {
  initializePendulumVisualization,
  type PendulumVisualization } from './visualization';

export async function initPendulumVisualization(
  parent: HTMLElement,
  weightsUrl: string
): Promise<PendulumVisualization> {
  const policy = await loadPendulumPolicy(weightsUrl);
  return initializePendulumVisualization(parent, policy);
}
