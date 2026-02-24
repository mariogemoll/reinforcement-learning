# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pixel-observation wrapper around the JAX Pong engine.

Returns (HEIGHT, WIDTH) float32 grayscale frames as observations instead
of the raw PongState.  Background is 0, paddles and ball are 1.
The underlying engine in pong_env is unchanged.
"""

import jax
import jax.numpy as jnp

import pong_env

# Re-export constants so callers don't need to import both modules.
WIDTH = pong_env.WIDTH
HEIGHT = pong_env.HEIGHT


def render(state: pong_env.PongState) -> jax.Array:
    """Render a PongState to a (HEIGHT, WIDTH) float32 array.  0=bg, 1=object."""
    frame = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32)

    # Ball
    br = jnp.clip(state.ball_row, 0, HEIGHT - 1).astype(jnp.int32)
    bc = jnp.clip(state.ball_col, 0, WIDTH - 1).astype(jnp.int32)
    frame = frame.at[br, bc].set(1.0)

    # Player paddle (left, col 0)
    p1_center = jnp.round(state.p1).astype(jnp.int32)
    for offset in range(-pong_env.PADDLE_HALF, pong_env.PADDLE_HALF + 1):
        r = p1_center + offset
        frame = jnp.where(
            (r >= 0) & (r < HEIGHT),
            frame.at[r, 0].set(1.0),
            frame,
        )

    # AI paddle (right, col WIDTH-1)
    p2_center = jnp.round(state.p2).astype(jnp.int32)
    for offset in range(-pong_env.PADDLE_HALF, pong_env.PADDLE_HALF + 1):
        r = p2_center + offset
        frame = jnp.where(
            (r >= 0) & (r < HEIGHT),
            frame.at[r, WIDTH - 1].set(1.0),
            frame,
        )

    return frame


def reset(key: jax.Array) -> tuple[pong_env.PongState, jax.Array]:
    """Reset the environment.  Returns (state, pixels)."""
    state = pong_env.reset(key)
    return state, render(state)


def step(
    state: pong_env.PongState,
    p1_action: jax.Array,
    p2_action: jax.Array,
) -> tuple[pong_env.PongState, jax.Array, jax.Array]:
    """One step.  Returns (new_state, pixels, done).

    The caller keeps the opaque state for the next step call; the pixels
    array (HEIGHT, WIDTH) is the observation.
    """
    new_state, done = pong_env.step(state, p1_action, p2_action)
    return new_state, render(new_state), done
