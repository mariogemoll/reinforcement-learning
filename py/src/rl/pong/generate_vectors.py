# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate golden test vectors from pong_env.py.

Writes two binary files consumed by the TypeScript vitest suite:
  ts/src/pong/environment.step-vectors.bin     — N × 17 float32
  ts/src/pong/environment.computer-vectors.bin — M × 8  float32

Step record layout (17 float32):
  input:     ball_row, ball_col, ball_vrow, ball_vcol, p1, p2, time  (7)
  actions:   p1_action, p2_action                                    (2)
  output:    ball_row, ball_col, ball_vrow, ball_vcol, p1, p2, time  (7)
  done:      0.0 or 1.0                                              (1)

Computer record layout (8 float32):
  input:     ball_row, ball_col, ball_vrow, ball_vcol, p1, p2, time  (7)
  action:    0.0, 1.0, or 2.0                                        (1)

Run with:
    .venv/bin/python generate_pong_vectors.py
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rl.pong.env import (
    HEIGHT,
    MAX_STEPS,
    PADDLE_HALF,
    WIDTH,
    PongState,
    computer_player,
    step,
)

STEP_STRIDE = 17
COMPUTER_STRIDE = 8


def state_to_row(s: PongState) -> list[float]:
    return [
        float(s.ball_row), float(s.ball_col),
        float(s.ball_vrow), float(s.ball_vcol),
        float(s.p1), float(s.p2),
        float(s.time),
    ]


def make_state(
    ball_row=15.0, ball_col=20.0, ball_vrow=0.0, ball_vcol=0.0,
    p1=15.0, p2=15.0, time=0,
) -> PongState:
    return PongState(
        ball_row=jnp.float32(ball_row),
        ball_col=jnp.float32(ball_col),
        ball_vrow=jnp.float32(ball_vrow),
        ball_vcol=jnp.float32(ball_vcol),
        p1=jnp.float32(p1),
        p2=jnp.float32(p2),
        time=jnp.int32(time),
    )


step_rows: list[list[float]] = []
computer_rows: list[list[float]] = []


def add_step(state: PongState, p1_action: int, p2_action: int) -> None:
    new_state, done = step(state, jnp.int32(p1_action), jnp.int32(p2_action))
    step_rows.append(
        state_to_row(state)
        + [float(p1_action), float(p2_action)]
        + state_to_row(new_state)
        + [1.0 if bool(done) else 0.0]
    )


def add_computer(state: PongState) -> None:
    computer_rows.append(
        state_to_row(state) + [float(int(computer_player(state)))]
    )


# ── Systematic cases ──────────────────────────────────────────────────────────

# Basic movement
for vrow, vcol in [(0.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (2.0, 1.0), (-2.0, -1.0)]:
    add_step(make_state(ball_row=15.0, ball_col=20.0, ball_vrow=vrow, ball_vcol=vcol), 0, 0)

# Paddle movement — all action × boundary combinations
for action in [0, 1, 2]:
    for p1 in [float(PADDLE_HALF), 15.0, float(HEIGHT - PADDLE_HALF - 1)]:
        add_step(make_state(p1=p1), action, 0)
    for p2 in [float(PADDLE_HALF), 15.0, float(HEIGHT - PADDLE_HALF - 1)]:
        add_step(make_state(p2=p2), 0, action)

# Bottom wall reflection
add_step(make_state(ball_row=28.0, ball_vrow=2.0,  ball_vcol=1.0), 0, 0)
add_step(make_state(ball_row=27.0, ball_vrow=3.0,  ball_vcol=1.0), 0, 0)
add_step(make_state(ball_row=29.0, ball_vrow=1.0,  ball_vcol=1.0), 0, 0)

# Top wall reflection
add_step(make_state(ball_row=1.0,  ball_vrow=-2.0, ball_vcol=1.0), 0, 0)
add_step(make_state(ball_row=0.0,  ball_vrow=-1.0, ball_vcol=1.0), 0, 0)
add_step(make_state(ball_row=0.0,  ball_vrow=-3.0, ball_vcol=1.0), 0, 0)

# Left-paddle reflections — all in-range distances
for dist in [-2, -1, 0, 1, 2]:
    add_step(make_state(ball_row=15.0 + dist, ball_col=2.0,  ball_vcol=-1.0, p1=15.0), 0, 0)

# Right-paddle reflections — all in-range distances
for dist in [-2, -1, 0, 1, 2]:
    add_step(make_state(ball_row=15.0 + dist, ball_col=38.0, ball_vcol=1.0,  p2=15.0), 0, 0)

# y-speed nudge clamped at MAX_SPEED
add_step(make_state(ball_row=17.0, ball_col=2.0,  ball_vcol=-1.0, ball_vrow=3.0,  p1=15.0), 0, 0)
add_step(make_state(ball_row=13.0, ball_col=2.0,  ball_vcol=-1.0, ball_vrow=-3.0, p1=15.0), 0, 0)

# Done conditions
add_step(make_state(ball_col=2.0,  ball_vcol=-3.0, ball_row=0.0, p1=15.0), 0, 0)
add_step(make_state(ball_col=39.0, ball_vcol=1.0,  ball_row=0.0, p2=15.0), 0, 0)
add_step(make_state(time=MAX_STEPS - 1), 0, 0)

# ── Random cases ──────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
for _ in range(200):
    row   = float(rng.uniform(1, HEIGHT - 1))
    col   = float(rng.uniform(1, WIDTH - 1))
    vrow  = float(rng.choice([-3, -2, -1, 0, 1, 2, 3]))
    vcol  = float(rng.choice([-1.0, 1.0]))
    p1    = float(rng.uniform(PADDLE_HALF, HEIGHT - PADDLE_HALF - 1))
    p2    = float(rng.uniform(PADDLE_HALF, HEIGHT - PADDLE_HALF - 1))
    t     = int(rng.integers(0, 100))
    add_step(
        make_state(ball_row=row, ball_col=col, ball_vrow=vrow, ball_vcol=vcol,
                   p1=p1, p2=p2, time=t),
        int(rng.integers(0, 3)),
        int(rng.integers(0, 3)),
    )

# ── computer_player cases ─────────────────────────────────────────────────────

for ball_row in range(0, HEIGHT, 2):
    for p2_pos in range(PADDLE_HALF, HEIGHT - PADDLE_HALF, 2):
        add_computer(make_state(ball_row=float(ball_row), p2=float(p2_pos)))

# ── Write binary files ────────────────────────────────────────────────────────

out_dir = Path(__file__).resolve().parents[4] / "ts" / "src" / "pong"

step_arr = np.array(step_rows, dtype=np.float32)
assert step_arr.shape[1] == STEP_STRIDE
step_path = out_dir / "environment.step-vectors.bin"
step_arr.tofile(step_path)
print(f"Wrote {len(step_rows)} step vectors to {step_path}")

computer_arr = np.array(computer_rows, dtype=np.float32)
assert computer_arr.shape[1] == COMPUTER_STRIDE
computer_path = out_dir / "environment.computer-vectors.bin"
computer_arr.tofile(computer_path)
print(f"Wrote {len(computer_rows)} computer vectors to {computer_path}")
