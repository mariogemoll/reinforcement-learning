// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { DqnOutput, DqnPolicy } from './dqn-policy';
import type { ReinforcePolicy } from './reinforce-policy';

/**
 * Adapt a REINFORCE policy to work with the DQN visualization.
 * The visualization expects qLeft/qRight values, but for REINFORCE
 * we use the action probabilities instead.
 */
export function reinforceToDqnAdapter(reinforcePolicy: ReinforcePolicy): DqnPolicy {
  return (state): DqnOutput => {
    const result = reinforcePolicy(state);
    // Use probabilities in place of Q-values for visualization purposes
    return {
      qLeft: result.probLeft,
      qRight: result.probRight,
      action: result.action
    };
  };
}
