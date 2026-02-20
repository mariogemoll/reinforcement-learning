// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { CartPoleAction, CartPoleState, CartPoleStepResult } from './types';

const GRAVITY = 9.8;
const CART_MASS = 1.0;
const POLE_MASS = 0.1;
const TOTAL_MASS = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH = 0.5;
const POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH;
const DEFAULT_FORCE_MAG = 10.0;
const DT = 0.02;

const ANGLE_LIMIT = 12 * 2 * Math.PI / 360;
const POSITION_LIMIT = 2.4;
const MAX_STEPS = 500;

export interface CartPoleEnvironmentOptions {
  angleLimit?: number;
  positionLimit?: number;
  maxSteps?: number;
}

export interface CartPoleEnvironment {
  getState(): Readonly<CartPoleState>;
  step(action: CartPoleAction): CartPoleStepResult;
  reset(): CartPoleState;
  setForceMag(force: number): void;
  getForceMag(): number;
}

export function createCartPoleEnvironment(
  options?: CartPoleEnvironmentOptions
): CartPoleEnvironment {
  const angleLimit = options?.angleLimit ?? ANGLE_LIMIT;
  const positionLimit = options?.positionLimit ?? POSITION_LIMIT;
  const maxSteps = options?.maxSteps ?? MAX_STEPS;
  let forceMag = DEFAULT_FORCE_MAG;
  let steps = 0;
  const state: CartPoleState = [0, 0, 0, 0];

  const initializeRandom = (): void => {
    state[0] = (Math.random() * 0.1) - 0.05;
    state[1] = (Math.random() * 0.1) - 0.05;
    state[2] = (Math.random() * 0.1) - 0.05;
    state[3] = (Math.random() * 0.1) - 0.05;
  };

  initializeRandom();

  const reset = (): CartPoleState => {
    steps = 0;
    initializeRandom();
    return [...state];
  };

  const step = (action: CartPoleAction): CartPoleStepResult => {
    const [x, xDot, theta, thetaDot] = state;

    const force = action === 1 ? forceMag : -forceMag;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
    const thetaAcc = (GRAVITY * sinTheta - cosTheta * temp)
      / (POLE_HALF_LENGTH * (4 / 3 - POLE_MASS * cosTheta * cosTheta / TOTAL_MASS));
    const xAcc = temp - POLE_MASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

    state[0] = x + DT * xDot;
    state[1] = xDot + DT * xAcc;
    state[2] = theta + DT * thetaDot;
    state[3] = thetaDot + DT * thetaAcc;

    steps++;

    const terminated = Math.abs(state[2]) > angleLimit
      || Math.abs(state[0]) > positionLimit;
    const truncated = steps >= maxSteps;

    return {
      state: [...state],
      reward: 1,
      terminated,
      truncated
    };
  };

  return {
    getState(): Readonly<CartPoleState> {
      return state;
    },
    step,
    reset,
    setForceMag(force: number): void {
      forceMag = force;
    },
    getForceMag(): number {
      return forceMag;
    }
  };
}

export { ANGLE_LIMIT, MAX_STEPS, POSITION_LIMIT };
