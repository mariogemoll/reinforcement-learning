// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { PendulumObs, PendulumState } from './types';

const G = 10.0;
const M = 1.0;
const L = 1.0;
const DT = 0.05;
const MAX_SPEED = 8.0;
const MAX_TORQUE = 2.0;
const MAX_STEPS = 200;

function angleNormalize(x: number): number {
  const twoPi = 2 * Math.PI;
  const wrapped = (x + Math.PI) % twoPi;
  return (wrapped < 0 ? wrapped + twoPi : wrapped) - Math.PI;
}

export interface PendulumStepResult {
  obs: PendulumObs;
  reward: number;
  truncated: boolean;
}

export interface PendulumStepStateResult extends PendulumStepResult {
  state: PendulumState;
  steps: number;
}

export interface PendulumEnvironment {
  getState(): Readonly<PendulumState>;
  step(torque: number): PendulumStepResult;
  reset(): PendulumObs;
}

export function stepPendulumState(
  state: Readonly<PendulumState>,
  steps: number,
  torque: number
): PendulumStepStateResult {
  const u = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, torque));
  const reward = -(angleNormalize(state.theta) ** 2 + 0.1 * state.thetaDot ** 2 + 0.001 * u ** 2);
  const thetaAcc = 3 * G / (2 * L) * Math.sin(state.theta) + 3 / (M * L * L) * u;
  const thetaDot = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, state.thetaDot + thetaAcc * DT));
  const theta = state.theta + thetaDot * DT;
  const nextSteps = steps + 1;
  return {
    state: { theta, thetaDot },
    obs: [Math.cos(theta), Math.sin(theta), thetaDot],
    reward,
    truncated: nextSteps >= MAX_STEPS,
    steps: nextSteps
  };
}

export function createPendulumEnvironment(): PendulumEnvironment {
  let theta = 0;
  let thetaDot = 0;
  let steps = 0;

  const getObs = (): PendulumObs => [Math.cos(theta), Math.sin(theta), thetaDot];

  const reset = (): PendulumObs => {
    theta = Math.random() * (2 * Math.PI) - Math.PI;
    thetaDot = Math.random() * 2 - 1;
    steps = 0;
    return getObs();
  };

  const step = (torque: number): PendulumStepResult => {
    const result = stepPendulumState({ theta, thetaDot }, steps, torque);
    theta = result.state.theta;
    thetaDot = result.state.thetaDot;
    steps = result.steps;
    return { obs: result.obs, reward: result.reward, truncated: result.truncated };
  };

  reset();

  return {
    getState: () => ({ theta, thetaDot }),
    step,
    reset
  };
}

export { MAX_STEPS, MAX_TORQUE };
