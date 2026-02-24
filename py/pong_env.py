# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""JAX Pong engine — exact Python mirror of ts/src/pong/environment.ts.

The game engine (step/reset) is kept separate from the AI opponent
(computer_player) so training code can substitute its own p2 policy.

Physics order: move paddles → move ball → reflect walls → reflect paddles.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── Constants (match TypeScript exactly) ──────────────────────────────────────
WIDTH = 40
HEIGHT = 30
PADDLE_HALF = 2
MAX_SPEED = 3
PADDLE_SPEED = 1
BALL_X_SPEED = 1
MAX_STEPS = 1000


# ── State ─────────────────────────────────────────────────────────────────────
class PongState(NamedTuple):
    ball_row: jax.Array   # float32  ball y-position
    ball_col: jax.Array   # float32  ball x-position
    ball_vrow: jax.Array  # float32  ball y-velocity
    ball_vcol: jax.Array  # float32  ball x-velocity
    p1: jax.Array         # float32  player-1 (left)  paddle centre y
    p2: jax.Array         # float32  player-2 (right) paddle centre y
    time: jax.Array       # int32    step counter


# ── Engine ────────────────────────────────────────────────────────────────────
def reset(key: jax.Array) -> PongState:
    """Return an initial state with the ball moving right and random vy ±1."""
    init_vy = jax.random.choice(key, jnp.array([-1.0, 1.0], dtype=jnp.float32))
    return PongState(
        ball_row=jnp.float32(HEIGHT / 2),
        ball_col=jnp.float32(WIDTH / 2),
        ball_vrow=init_vy,
        ball_vcol=jnp.float32(BALL_X_SPEED),
        p1=jnp.float32(HEIGHT / 2),
        p2=jnp.float32(HEIGHT / 2),
        time=jnp.int32(0),
    )


def step(
    state: PongState,
    p1_action: jax.Array,
    p2_action: jax.Array,
) -> tuple[PongState, jax.Array]:
    """One deterministic Pong step.  Returns (new_state, done).

    Actions: 0 = noop, 1 = up, 2 = down.
    Mirrors stepPongState in ts/src/pong/environment.ts exactly.
    """
    # ── Move paddles ──────────────────────────────────────────────────────────
    def _move(center: jax.Array, action: jax.Array) -> jax.Array:
        delta = jnp.equal(action, 2).astype(jnp.float32) - jnp.equal(action, 1).astype(jnp.float32)
        return jnp.clip(center + delta * PADDLE_SPEED, PADDLE_HALF, HEIGHT - PADDLE_HALF - 1)

    new_p1 = _move(state.p1, p1_action)
    new_p2 = _move(state.p2, p2_action)

    # ── Move ball ─────────────────────────────────────────────────────────────
    new_row = state.ball_row + state.ball_vrow
    new_col = state.ball_col + state.ball_vcol
    new_vrow = state.ball_vrow
    new_vcol = state.ball_vcol

    # ── Reflect off walls ─────────────────────────────────────────────────────
    # Bottom wall
    hit_bottom = new_row >= HEIGHT
    new_row  = jnp.where(hit_bottom, 2.0 * (HEIGHT - 1) - new_row, new_row)
    new_vrow = jnp.where(hit_bottom, -new_vrow, new_vrow)

    # Top wall (missing in gymnax — present in the TS engine)
    hit_top = new_row < 0.0
    new_row  = jnp.where(hit_top, -new_row, new_row)
    new_vrow = jnp.where(hit_top, -new_vrow, new_vrow)

    # ── Reflect off paddles ───────────────────────────────────────────────────
    left_refl  = 2.0 * 1.0 - new_col           # reflection axis at col 1
    right_refl = 2.0 * (WIDTH - 2) - new_col   # reflection axis at col 38
    dist_p1 = new_row - new_p1
    dist_p2 = new_row - new_p2

    hit_left = (left_refl >= 1.0) & (jnp.abs(dist_p1) <= PADDLE_HALF)
    new_col  = jnp.where(hit_left, left_refl, new_col)
    new_vcol = jnp.where(hit_left, -new_vcol, new_vcol)
    new_vrow = jnp.where(
        hit_left,
        jnp.clip(new_vrow + jnp.trunc(dist_p1 / PADDLE_HALF), -MAX_SPEED, MAX_SPEED),
        new_vrow,
    )

    hit_right = (right_refl < WIDTH - 2) & (jnp.abs(dist_p2) < PADDLE_HALF + 1)
    new_col  = jnp.where(hit_right, right_refl, new_col)
    new_vcol = jnp.where(hit_right, -new_vcol, new_vcol)
    new_vrow = jnp.where(
        hit_right,
        jnp.clip(new_vrow + jnp.trunc(dist_p2 / PADDLE_HALF), -MAX_SPEED, MAX_SPEED),
        new_vrow,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    new_time = state.time + 1
    done = (new_col < 0.0) | (new_col >= WIDTH) | (new_time >= MAX_STEPS)

    return PongState(
        ball_row=new_row,
        ball_col=new_col,
        ball_vrow=new_vrow,
        ball_vcol=new_vcol,
        p1=new_p1,
        p2=new_p2,
        time=new_time,
    ), done


# ── AI opponent ───────────────────────────────────────────────────────────────
def computer_player(state: PongState) -> jax.Array:
    """Greedy AI: always move p2 toward the ball.

    Returns 1 (up) or 2 (down) — never noop, matching computerPongPlayer in TS.
    """
    dist_down = jnp.abs(
        state.ball_row - jnp.clip(state.p2 + PADDLE_SPEED, PADDLE_HALF, HEIGHT - PADDLE_HALF - 1)
    )
    dist_up = jnp.abs(
        state.ball_row - jnp.clip(state.p2 - PADDLE_SPEED, PADDLE_HALF, HEIGHT - PADDLE_HALF - 1)
    )
    return jnp.where(dist_up < dist_down, jnp.int32(1), jnp.int32(2))
