// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Internal simulation state.
export interface PendulumState {
  theta: number;    // angle in radians; 0 = upright
  thetaDot: number; // angular velocity rad/s
}

// Observation vector fed to the policy: [cos(theta), sin(theta), thetaDot].
export type PendulumObs = [number, number, number];
