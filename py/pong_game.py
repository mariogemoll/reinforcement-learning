# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Interactive Pong game using the gymnax Pong-misc environment.

Controls
--------
W / Up Arrow    move paddle up
S / Down Arrow  move paddle down
R / Space       restart after game ends
Q / Escape      quit
"""

import matplotlib

matplotlib.use("MacOSX")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import gymnax  # noqa: E402

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
STEP_INTERVAL_MS = 100  # ms per game step — lower is faster
WINNING_SCORE = 7

env, params = gymnax.make("Pong-misc")
params = params.replace(use_ai_policy=True)

jit_step = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

# Pre-warm JIT so the first frame isn't slow
_key = jax.random.PRNGKey(0)
_obs, _state = jit_reset(_key, params)
_key, _sk = jax.random.split(_key)
jit_step(_sk, _state, 0, params)
print("Ready.")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG_C = np.array([0.05, 0.05, 0.10])
PLAYER_C = np.array([0.25, 0.92, 0.35])  # green  — left paddle  (you)
AI_C = np.array([0.92, 0.25, 0.25])  # red    — right paddle (cpu)
BALL_C = np.array([1.00, 1.00, 1.00])  # white


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render_state(s) -> np.ndarray:
    frame = np.tile(BG_C, (env.height, env.width, 1)).copy()

    # Ball
    br = int(np.clip(s.ball_position[0], 0, env.height - 1))
    bc = int(np.clip(s.ball_position[1], 0, env.width - 1))
    frame[br, bc] = BALL_C

    # Player paddle — left edge (col 0)
    p1 = float(s.paddle_centers[0])
    for r in range(round(p1) - env.paddle_half_height, round(p1) + env.paddle_half_height + 1):
        if 0 <= r < env.height:
            frame[r, 0] = PLAYER_C

    # AI paddle — right edge (col width-1)
    p2 = float(s.paddle_centers[1])
    for r in range(round(p2) - env.paddle_half_height, round(p2) + env.paddle_half_height + 1):
        if 0 <= r < env.height:
            frame[r, env.width - 1] = AI_C

    return frame


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8), facecolor="black")
try:
    fig.canvas.manager.set_window_title("Pong — gymnax")
except Exception:
    pass

# Game canvas (90 % height)
ax_game = fig.add_axes([0.0, 0.10, 1.0, 0.90])
ax_game.set_facecolor("black")
ax_game.axis("off")

# Score bar (bottom 10 %)
ax_score = fig.add_axes([0.0, 0.00, 1.0, 0.10])
ax_score.set_facecolor("black")
ax_score.axis("off")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def start_episode() -> None:
    """Reset env and add a random initial y-velocity so the ball isn't flat."""
    global key, state
    key, sk = jax.random.split(key)
    _, state = jit_reset(sk, params)
    key, sk = jax.random.split(key)
    vy = float(jax.random.choice(sk, jnp.array([-1, 1])))
    state = state.replace(ball_velocity=state.ball_velocity.at[0].set(vy))


def fix_top_wall(s):
    """Gymnax Pong-misc omits the top-wall reflection — apply it manually."""
    escaped = s.ball_position[0] < 0
    row = jnp.where(escaped, -s.ball_position[0], s.ball_position[0])
    vy = jnp.where(escaped, -s.ball_velocity[0], s.ball_velocity[0])
    return s.replace(
        ball_position=s.ball_position.at[0].set(row),
        ball_velocity=s.ball_velocity.at[0].set(vy),
    )


# Initial state
key = jax.random.PRNGKey(42)
start_episode()

img = ax_game.imshow(render_state(state), interpolation="nearest", aspect="auto")

hint = ax_game.text(
    0.5,
    0.01,
    "W / ↑  up     S / ↓  down",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#44445a",
    transform=ax_game.transAxes,
)

overlay = ax_game.text(
    0.5,
    0.5,
    "",
    ha="center",
    va="center",
    fontsize=22,
    color="white",
    fontfamily="monospace",
    fontweight="bold",
    transform=ax_game.transAxes,
    alpha=0.0,
    linespacing=1.8,
)

ax_score.text(
    0.22,
    0.5,
    "YOU",
    ha="center",
    va="center",
    fontsize=13,
    color=PLAYER_C,
    fontfamily="monospace",
    fontweight="bold",
    transform=ax_score.transAxes,
)
score_text = ax_score.text(
    0.50,
    0.5,
    "0  :  0",
    ha="center",
    va="center",
    fontsize=20,
    color="white",
    fontfamily="monospace",
    fontweight="bold",
    transform=ax_score.transAxes,
)
ax_score.text(
    0.78,
    0.5,
    "CPU",
    ha="center",
    va="center",
    fontsize=13,
    color=AI_C,
    fontfamily="monospace",
    fontweight="bold",
    transform=ax_score.transAxes,
)

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
player_score = 0
ai_score = 0
action = 0
game_over = False


def _reset_game():
    global player_score, ai_score, game_over, action
    player_score = 0
    ai_score = 0
    game_over = False
    action = 0
    start_episode()
    overlay.set_alpha(0.0)
    score_text.set_text("0  :  0")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def on_key_press(event):
    global action, game_over
    if event.key in ("q", "escape"):
        plt.close(fig)
        return
    if game_over:
        if event.key in ("r", " ", "enter"):
            _reset_game()
        return
    if event.key in ("up", "w"):
        action = 1
    elif event.key in ("down", "s"):
        action = 2


def on_key_release(event):
    global action
    if event.key in ("up", "w", "down", "s"):
        action = 0


fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)


# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------
def update(_frame):
    global state, key, action, player_score, ai_score, game_over

    if game_over:
        return [img, score_text, overlay]

    key, sk = jax.random.split(key)
    _obs, new_state, _reward, done, _info = jit_step(sk, state, action, params)
    new_state = fix_top_wall(new_state)

    if bool(np.array(done)):
        ball_col = float(np.array(new_state.ball_position[1]))
        if ball_col >= env.width:  # right paddle (AI) missed
            player_score += 1
        else:  # left paddle (player) missed
            ai_score += 1
        score_text.set_text(f"{player_score}  :  {ai_score}")

        if player_score >= WINNING_SCORE or ai_score >= WINNING_SCORE:
            game_over = True
            winner = "YOU WIN!" if player_score >= WINNING_SCORE else "CPU WINS"
            overlay.set_text(f"{winner}\n\nR / Space to play again")
            overlay.set_alpha(1.0)
            state = new_state
        else:
            start_episode()
    else:
        state = new_state

    img.set_data(render_state(state))
    return [img, score_text, overlay]


ani = animation.FuncAnimation(
    fig, update, interval=STEP_INTERVAL_MS, blit=True, cache_frame_data=False
)

plt.show()
