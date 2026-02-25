// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export type BreakoutAction = 0 | 1 | 2; // noop, left, right

export interface BreakoutState {
  ballY: number;
  ballX: number;
  ballDir: number; // 0=UL, 1=UR, 2=DR, 3=DL
  pos: number; // paddle x
  brickMap: Uint8Array; // 10x10 row-major
  strike: boolean;
  lastY: number;
  lastX: number;
  time: number;
  terminal: boolean;
}

export interface BreakoutStepResult {
  observation: Float32Array;
  state: BreakoutState;
  reward: number;
  done: boolean;
}

export const BREAKOUT_GRID_SIZE = 10;
export const BREAKOUT_CHANNELS = 4;
export const BREAKOUT_MAX_STEPS = 1000;
const BREAKOUT_ACTIONS = [0, 1, 3] as const;

// Direction tables matching MinAtar's canonical implementation.
const X_REFLECT = [1, 0, 3, 2];
const Y_REFLECT = [3, 2, 1, 0];
const DIAG_REFLECT = [2, 3, 0, 1];

const DX = [-1, 1, 1, -1];
const DY = [-1, -1, 1, 1];

function makeBrickMap(): Uint8Array {
  const map = new Uint8Array(BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE);
  for (let row = 1; row <= 3; row++) {
    for (let col = 0; col < BREAKOUT_GRID_SIZE; col++) {
      map[row * BREAKOUT_GRID_SIZE + col] = 1;
    }
  }
  return map;
}

function brickAt(map: Uint8Array, row: number, col: number): number {
  return map[row * BREAKOUT_GRID_SIZE + col];
}

function brickSet(
  map: Uint8Array,
  row: number,
  col: number,
  value: number
): Uint8Array {
  const copy = new Uint8Array(map);
  copy[row * BREAKOUT_GRID_SIZE + col] = value;
  return copy;
}

function brickCount(map: Uint8Array): number {
  let count = 0;
  for (const brick of map) {
    if (brick !== 0) {
      count++;
    }
  }
  return count;
}

function randomStartSide(): 0 | 1 {
  return Math.random() < 0.5 ? 0 : 1;
}

export function resetBreakoutState(startSide: 0 | 1 = randomStartSide()): BreakoutState {
  const ballX = startSide === 0 ? 0 : (BREAKOUT_GRID_SIZE - 1);
  const ballDir = startSide === 0 ? 2 : 3;

  return {
    ballY: 3,
    ballX,
    ballDir,
    pos: 4,
    brickMap: makeBrickMap(),
    strike: false,
    lastY: 3,
    lastX: ballX,
    time: 0,
    terminal: false
  };
}

export function getBreakoutObservation(state: BreakoutState): Float32Array {
  const obs = new Float32Array(
    BREAKOUT_GRID_SIZE * BREAKOUT_GRID_SIZE * BREAKOUT_CHANNELS
  );

  // ch0: paddle
  obs[((9 * BREAKOUT_GRID_SIZE) + state.pos) * BREAKOUT_CHANNELS] = 1;
  // ch1: ball
  obs[(((state.ballY * BREAKOUT_GRID_SIZE) + state.ballX) * BREAKOUT_CHANNELS) + 1] = 1;
  // ch2: trail
  obs[(((state.lastY * BREAKOUT_GRID_SIZE) + state.lastX) * BREAKOUT_CHANNELS) + 2] = 1;
  // ch3: bricks
  for (let i = 0; i < state.brickMap.length; i++) {
    if (state.brickMap[i] !== 0) {
      obs[(i * BREAKOUT_CHANNELS) + 3] = 1;
    }
  }

  return obs;
}

export function stepBreakoutState(
  state: BreakoutState,
  action: BreakoutAction
): BreakoutStepResult {
  const mappedAction = BREAKOUT_ACTIONS[action];
  let { pos, ballDir } = state;
  const { ballX, ballY } = state;
  const lastX = ballX;
  const lastY = ballY;

  if (mappedAction === 1) {
    pos = Math.max(0, pos - 1);
  } else if (mappedAction === 3) {
    pos = Math.min(BREAKOUT_GRID_SIZE - 1, pos + 1);
  }

  let newX = ballX + DX[ballDir];
  let newY = ballY + DY[ballDir];

  if (newX < 0 || newX >= BREAKOUT_GRID_SIZE) {
    newX = Math.max(0, Math.min(BREAKOUT_GRID_SIZE - 1, newX));
    ballDir = X_REFLECT[ballDir];
  }

  let reward = 0;
  let terminal = false;

  const borderCondY = newY < 0;
  if (borderCondY) {
    newY = 0;
    ballDir = Y_REFLECT[ballDir];
  }

  const strikeToggle = !borderCondY && brickAt(state.brickMap, newY, newX) === 1;
  const strikeBool = !state.strike && strikeToggle;

  let newBrickMap = state.brickMap;
  if (strikeBool) {
    reward = 1;
    newBrickMap = brickSet(state.brickMap, newY, newX, 0);
    newY = lastY;
    ballDir = Y_REFLECT[ballDir];
  }

  const brickCond = !strikeToggle && newY === BREAKOUT_GRID_SIZE - 1;
  if (brickCond) {
    if (brickCount(newBrickMap) === 0) {
      newBrickMap = makeBrickMap();
    }

    if (state.ballX === pos) {
      ballDir = Y_REFLECT[ballDir];
      newY = lastY;
    } else if (newX === pos) {
      ballDir = DIAG_REFLECT[ballDir];
      newY = lastY;
    } else {
      terminal = true;
    }
  }

  const newTime = state.time + 1;
  const done = terminal || newTime >= BREAKOUT_MAX_STEPS;

  const nextState: BreakoutState = {
    ballY: newY,
    ballX: newX,
    ballDir,
    pos,
    brickMap: newBrickMap,
    strike: strikeToggle,
    lastY,
    lastX,
    time: newTime,
    terminal: done
  };

  return {
    observation: getBreakoutObservation(nextState),
    state: nextState,
    reward,
    done
  };
}
