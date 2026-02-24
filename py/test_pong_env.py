# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests verifying pong_env.py matches ts/src/pong/environment.ts exactly.

Each test cites the relevant TS formula so mismatches are easy to trace.
"""

import jax
import jax.numpy as jnp

from pong_env import (
    HEIGHT,
    MAX_STEPS,
    PADDLE_HALF,
    WIDTH,
    PongState,
    computer_player,
    reset,
    step,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(
    ball_row=15.0,
    ball_col=20.0,
    ball_vrow=0.0,
    ball_vcol=0.0,
    p1=15.0,
    p2=15.0,
    time=0,
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


def f(arr) -> float:
    return float(arr)


NOOP = jnp.int32(0)
UP   = jnp.int32(1)
DOWN = jnp.int32(2)


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_ball_at_center():
    state = reset(jax.random.PRNGKey(0))
    assert f(state.ball_row) == HEIGHT / 2   # 15.0
    assert f(state.ball_col) == WIDTH  / 2   # 20.0

def test_reset_paddles_at_center():
    state = reset(jax.random.PRNGKey(0))
    assert f(state.p1) == HEIGHT / 2
    assert f(state.p2) == HEIGHT / 2

def test_reset_ball_moves_right():
    # TS: ballVCol: BALL_X_SPEED (=1)
    state = reset(jax.random.PRNGKey(0))
    assert f(state.ball_vcol) == 1.0

def test_reset_time_is_zero():
    state = reset(jax.random.PRNGKey(0))
    assert int(state.time) == 0

def test_reset_init_vy_is_plus_or_minus_one():
    # TS resetPongState takes initVY: 1 | -1
    for seed in range(20):
        state = reset(jax.random.PRNGKey(seed))
        assert f(state.ball_vrow) in (1.0, -1.0)


# ── Basic movement ────────────────────────────────────────────────────────────

def test_ball_moves_by_velocity():
    # TS: newRow = ballRow + ballVRow; newCol = ballCol + ballVCol
    s, done = step(make_state(ball_row=10.0, ball_col=10.0, ball_vrow=2.0, ball_vcol=3.0), NOOP, NOOP)
    assert f(s.ball_row) == 12.0
    assert f(s.ball_col) == 13.0
    assert not bool(done)

def test_time_increments():
    s, _ = step(make_state(time=5), NOOP, NOOP)
    assert int(s.time) == 6


# ── Paddle movement ───────────────────────────────────────────────────────────

def test_paddle_noop_stays():
    s, _ = step(make_state(p1=15.0), NOOP, NOOP)
    assert f(s.p1) == 15.0

def test_paddle_action_down_increases_y():
    # TS: p1Step = (action===2?1:0) - (action===1?1:0)  → +1 for down
    s, _ = step(make_state(p1=15.0), DOWN, NOOP)
    assert f(s.p1) == 16.0

def test_paddle_action_up_decreases_y():
    s, _ = step(make_state(p1=15.0), UP, NOOP)
    assert f(s.p1) == 14.0

def test_paddle_clamps_at_bottom():
    # TS: clamp(center + delta, PADDLE_HALF_HEIGHT, ENV_HEIGHT - PADDLE_HALF_HEIGHT - 1)
    bottom = float(HEIGHT - PADDLE_HALF - 1)  # 27.0
    s, _ = step(make_state(p1=bottom), DOWN, NOOP)
    assert f(s.p1) == bottom

def test_paddle_clamps_at_top():
    top = float(PADDLE_HALF)  # 2.0
    s, _ = step(make_state(p1=top), UP, NOOP)
    assert f(s.p1) == top


# ── Wall reflections ──────────────────────────────────────────────────────────

def test_bottom_wall_reflection():
    # TS: newRow = 2*(ENV_HEIGHT-1) - newRow  →  58 - 30 = 28
    # ball_row=28, ball_vrow=2 → new_row=30 ≥ 30 → reflects to 28, flips vrow
    s, _ = step(make_state(ball_row=28.0, ball_vrow=2.0), NOOP, NOOP)
    assert f(s.ball_row)  == 28.0
    assert f(s.ball_vrow) == -2.0

def test_bottom_wall_reflection_with_overshoot():
    # ball_row=27, ball_vrow=3 → new_row=30 → 58-30=28
    s, _ = step(make_state(ball_row=27.0, ball_vrow=3.0), NOOP, NOOP)
    assert f(s.ball_row)  == 28.0
    assert f(s.ball_vrow) == -3.0

def test_top_wall_reflection():
    # TS: newRow = -newRow  →  -(-1) = 1, flips vrow
    # ball_row=1, ball_vrow=-2 → new_row=-1 < 0 → reflects to 1
    s, _ = step(make_state(ball_row=1.0, ball_vrow=-2.0), NOOP, NOOP)
    assert f(s.ball_row)  == 1.0
    assert f(s.ball_vrow) == 2.0

def test_top_wall_reflection_exact_zero():
    # ball_row=0, ball_vrow=-1 → new_row=-1 → reflects to 1
    s, _ = step(make_state(ball_row=0.0, ball_vrow=-1.0), NOOP, NOOP)
    assert f(s.ball_row)  == 1.0
    assert f(s.ball_vrow) == 1.0


# ── Done conditions ───────────────────────────────────────────────────────────

def test_done_when_ball_exits_left():
    # ball_col=2, ball_vcol=-3 → new_col=-1; paddle at row 15, ball at row 0 → miss → done
    # TS: done = newCol < 0 || ...
    _, done = step(make_state(ball_col=2.0, ball_vcol=-3.0, ball_row=0.0, p1=15.0), NOOP, NOOP)
    assert bool(done)

def test_done_when_ball_exits_right():
    # ball_col=39, ball_vcol=1 → new_col=40; paddle at row 15, ball at row 0 → miss → done
    # TS: done = ... || newCol >= ENV_WIDTH || ...
    _, done = step(make_state(ball_col=39.0, ball_vcol=1.0, ball_row=0.0, p2=15.0), NOOP, NOOP)
    assert bool(done)

def test_not_done_in_middle():
    _, done = step(make_state(ball_col=20.0, ball_vcol=1.0), NOOP, NOOP)
    assert not bool(done)

def test_done_at_max_steps():
    # TS: done = ... || newTime >= MAX_STEPS
    _, done = step(make_state(time=MAX_STEPS - 1), NOOP, NOOP)
    assert bool(done)

def test_not_done_one_before_max_steps():
    _, done = step(make_state(time=MAX_STEPS - 2), NOOP, NOOP)
    assert not bool(done)


# ── Paddle reflections ────────────────────────────────────────────────────────

def test_left_paddle_reflects_vcol():
    # ball at col=2 moving left (vcol=-1) hits left paddle at p1=15, ball_row=15
    # new_col=1; left_refl=2*1-1=1 ≥ 1 and |15-15|=0 ≤ 2 → HIT
    # TS: newCol = leftReflCol; newVCol = -newVCol
    s, _ = step(make_state(ball_row=15.0, ball_col=2.0, ball_vcol=-1.0, p1=15.0), NOOP, NOOP)
    assert f(s.ball_vcol) == 1.0
    assert f(s.ball_col)  == 1.0

def test_right_paddle_reflects_vcol():
    # ball at col=38 moving right (vcol=1) → new_col=39
    # right_refl=2*38-39=37 < 38 and |15-15|=0 < 3 → HIT
    s, _ = step(make_state(ball_row=15.0, ball_col=38.0, ball_vcol=1.0, p2=15.0), NOOP, NOOP)
    assert f(s.ball_vcol) == -1.0
    assert f(s.ball_col)  == 37.0

def test_left_paddle_miss_when_ball_far():
    # ball_row=0, p1=15 → dist=15, |15|>2 → no hit → ball exits
    _, done = step(make_state(ball_row=0.0, ball_col=2.0, ball_vcol=-3.0, p1=15.0), NOOP, NOOP)
    assert bool(done)

def test_right_paddle_miss_when_ball_far():
    _, done = step(make_state(ball_row=0.0, ball_col=39.0, ball_vcol=1.0, p2=15.0), NOOP, NOOP)
    assert bool(done)

def test_left_paddle_vrow_nudge_upward():
    # ball hits below paddle centre: dist_p1 = new_row - p1 = 17-15 = 2
    # TS: newVRow = clamp(newVRow + trunc(distP1 / PADDLE_HALF_HEIGHT), -3, 3)
    # trunc(2/2) = 1 → new_vrow = 0 + 1 = 1
    s, _ = step(make_state(ball_row=17.0, ball_col=2.0, ball_vcol=-1.0, ball_vrow=0.0, p1=15.0), NOOP, NOOP)
    assert f(s.ball_vrow) == 1.0

def test_left_paddle_vrow_nudge_downward():
    # dist_p1 = 13-15 = -2; trunc(-2/2) = -1 → new_vrow = 0 + (-1) = -1
    s, _ = step(make_state(ball_row=13.0, ball_col=2.0, ball_vcol=-1.0, ball_vrow=0.0, p1=15.0), NOOP, NOOP)
    assert f(s.ball_vrow) == -1.0

def test_right_paddle_vrow_nudge():
    # dist_p2 = 17-15=2; trunc(2/2)=1 → new_vrow = 0+1=1
    s, _ = step(make_state(ball_row=17.0, ball_col=38.0, ball_vcol=1.0, ball_vrow=0.0, p2=15.0), NOOP, NOOP)
    assert f(s.ball_vrow) == 1.0

def test_vrow_nudge_clamped_at_max_speed():
    # ball_vrow=3 (already at max), nudge +1 → clamped to 3
    s, _ = step(make_state(ball_row=17.0, ball_col=2.0, ball_vcol=-1.0, ball_vrow=3.0, p1=15.0), NOOP, NOOP)
    assert f(s.ball_vrow) == 3.0


# ── Computer player ───────────────────────────────────────────────────────────

def test_computer_moves_up_when_ball_above():
    # ball_row=5, p2=15 → moving up brings p2 to 14 (dist=9), down to 16 (dist=11)
    # TS: distIfUp < distIfDown ? 1 : 2
    action = computer_player(make_state(ball_row=5.0, p2=15.0))
    assert int(action) == 1  # up

def test_computer_moves_down_when_ball_below():
    action = computer_player(make_state(ball_row=25.0, p2=15.0))
    assert int(action) == 2  # down

def test_computer_never_returns_noop():
    # TS computerPongPlayer never returns 0
    for row in range(0, HEIGHT):
        for p2_pos in range(PADDLE_HALF, HEIGHT - PADDLE_HALF):
            action = computer_player(make_state(ball_row=float(row), p2=float(p2_pos)))
            assert int(action) in (1, 2)

def test_computer_clamps_paddle_at_bottom():
    # p2 already at bottom clamp — moving down stays there, distance same or worse
    bottom = float(HEIGHT - PADDLE_HALF - 1)
    action = computer_player(make_state(ball_row=bottom + 5.0, p2=bottom))
    assert int(action) == 2  # still tries to go down (but clamps)

def test_computer_clamps_paddle_at_top():
    top = float(PADDLE_HALF)
    action = computer_player(make_state(ball_row=0.0, p2=top))
    assert int(action) == 1  # still tries to go up (but clamps)


# ── Multi-step sequence ───────────────────────────────────────────────────────

def test_rally_stays_alive_with_perfect_ai():
    """A serve straight across should never exit left or right when both paddles
    track the ball perfectly via computer_player."""
    state = reset(jax.random.PRNGKey(42))
    # Override to ensure flat horizontal trajectory
    state = PongState(
        ball_row=jnp.float32(15.0),
        ball_col=jnp.float32(20.0),
        ball_vrow=jnp.float32(0.0),
        ball_vcol=jnp.float32(1.0),
        p1=jnp.float32(15.0),
        p2=jnp.float32(15.0),
        time=jnp.int32(0),
    )
    for _ in range(200):
        p1_act = computer_player(PongState(
            state.ball_row, state.ball_col, state.ball_vrow, state.ball_vcol,
            state.p1, state.p2, state.time,
        ))
        p2_act = computer_player(state)
        state, done = step(state, p1_act, p2_act)
        assert not bool(done), f"Rally ended unexpectedly at time {int(state.time)}"
