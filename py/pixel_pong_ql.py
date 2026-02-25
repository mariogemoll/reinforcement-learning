# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Approximate Q-learning for PixelPong.

Observations are N_FRAMES stacked (HEIGHT × WIDTH) grayscale frames from
pong_pixel_env, flattened to an (N_FRAMES * HEIGHT * WIDTH)-dim float32 vector.
Frame stacking is handled here (on top of the env), not inside the env itself.
The network is a plain MLP over this stacked pixel vector.
"""

import functools
from pathlib import Path

import anywidget
import jax
import jax.numpy as jnp
import numpy as np
import traitlets
from flax import nnx

import pong_env
import pong_pixel_env


class PixelPongQLVisualization(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "dist" / "pixel-pong-visualization.js"
    _css = (Path(__file__).parent.parent / "ts" / "reinforcement-learning.css").read_text()
    weights_base64 = traitlets.Unicode("").tag(sync=True)


NUM_ACTIONS = 3
N_FRAMES = 4
SINGLE_FRAME_DIM = pong_pixel_env.HEIGHT * pong_pixel_env.WIDTH  # 1200
# CNN architecture (VALID padding): conv0 5×5/stride-2 → (13,18,16), conv1 3×3/stride-2 → (6,8,32)
CONV_OUT_DIM = 6 * 8 * 32  # 1536
START_EPSILON = 1.0
END_EPSILON = 0.05
GAMMA = 0.99


def pixels_obs(state: pong_env.PongState) -> jax.Array:
    """Render a PongState to a flat float32 pixel vector of length SINGLE_FRAME_DIM."""
    return jnp.ravel(pong_pixel_env.render(state))


def make_frame_stack(frame: jax.Array) -> jax.Array:
    """Initialize a frame stack by repeating a single frame N_FRAMES times.

    Returns shape (N_FRAMES, SINGLE_FRAME_DIM).
    """
    return jnp.repeat(frame[None], N_FRAMES, axis=0)


def push_frame(stack: jax.Array, frame: jax.Array) -> jax.Array:
    """Append a new frame to the stack, dropping the oldest.

    stack: (N_FRAMES, SINGLE_FRAME_DIM)
    frame: (SINGLE_FRAME_DIM,)
    Returns updated stack of the same shape.
    """
    return jnp.roll(stack, -1, axis=0).at[-1].set(frame)


class QNetwork(nnx.Module):
    def __init__(self, hidden_dim, num_layers, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(N_FRAMES, 16, kernel_size=(5, 5), strides=(2, 2), padding="VALID", rngs=rngs)
        self.conv1 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=(2, 2), padding="VALID", rngs=rngs)
        dims = [CONV_OUT_DIM] + [hidden_dim] * num_layers + [NUM_ACTIONS]
        self.num_dense = len(dims) - 1
        for i in range(self.num_dense):
            setattr(self, f"dense_{i}", nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

    def __call__(self, x):
        # x: (N_FRAMES, HEIGHT, WIDTH) → (HEIGHT, WIDTH, N_FRAMES) for channels-last Conv
        x = jnp.transpose(x, (1, 2, 0))
        x = nnx.relu(self.conv0(x))
        x = nnx.relu(self.conv1(x))
        x = x.ravel()
        for i in range(self.num_dense - 1):
            x = nnx.relu(getattr(self, f"dense_{i}")(x))
        return getattr(self, f"dense_{self.num_dense - 1}")(x)


@functools.lru_cache(maxsize=None)
def _get_graphdef(hidden_dim, num_layers):
    graphdef, _ = nnx.split(QNetwork(hidden_dim, num_layers, rngs=nnx.Rngs(0)))
    return graphdef


def forward(hidden_dim, num_layers, params, x):
    return nnx.merge(_get_graphdef(hidden_dim, num_layers), params)(x)


def make_model(hidden_dim, num_layers, params):
    return nnx.merge(_get_graphdef(hidden_dim, num_layers), params)


def fresh_params(hidden_dim, num_layers):
    _, p = nnx.split(QNetwork(hidden_dim, num_layers, rngs=nnx.Rngs(0)))
    return p


@functools.lru_cache(maxsize=None)
def _make_body(hidden_dim, num_layers):
    """Compile the per-step body for fori_loop.

    Uses carry["step_offset"] as the base so the same body works for both
    full runs and chunked vmap execution: global_step = i + step_offset.
    Set step_offset=0 for a plain (non-chunked) run.
    """
    graphdef = _get_graphdef(hidden_dim, num_layers)

    def _fwd(params, x):
        return nnx.merge(graphdef, params)(x)

    def _train_step(params, obs, action, reward, next_obs, done, lr):
        def _loss(p):
            curr_q = _fwd(p, obs)[action]
            next_q = jnp.max(_fwd(p, next_obs))
            target = reward + GAMMA * next_q * (1.0 - done)
            return 0.5 * (curr_q - jax.lax.stop_gradient(target)) ** 2

        loss, grads = jax.value_and_grad(_loss)(params)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return loss, new_params

    def body(i, carry):
        key = carry["key"]
        key, ka, kr, kre = jax.random.split(key, 4)

        global_step = i + carry["step_offset"]

        obs = carry["frame_stack"].reshape(N_FRAMES, pong_pixel_env.HEIGHT, pong_pixel_env.WIDTH)
        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (global_step / carry["decay_dur"]),
        )
        greedy = jnp.argmax(_fwd(carry["params"], obs))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=NUM_ACTIONS)
        action = jax.lax.select(jax.random.uniform(ka) > eps, greedy, rand_a)

        p2_action = pong_env.computer_player(carry["env_state"])
        next_env_state, done = pong_env.step(carry["env_state"], action, p2_action)
        next_frame = pixels_obs(next_env_state)
        next_stack = push_frame(carry["frame_stack"], next_frame)
        next_obs = next_stack.reshape(N_FRAMES, pong_pixel_env.HEIGHT, pong_pixel_env.WIDTH)
        reward = jnp.float32(1.0) - done.astype(jnp.float32)

        timeout = carry["ep_len"] + 1 >= pong_env.MAX_STEPS
        terminal = done & ~timeout

        loss, params = _train_step(
            carry["params"],
            obs,
            action,
            reward,
            next_obs,
            terminal.astype(jnp.float32),
            carry["lr"],
        )

        reset_env_state = pong_env.reset(kre)
        reset_stack = make_frame_stack(pixels_obs(reset_env_state))
        new_frame_stack = jnp.where(done, reset_stack, next_stack)
        new_env_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_env_state, next_env_state
        )
        ep_return = jnp.where(done, jnp.float32(0.0), carry["ep_return"] + reward)
        ep_len = jnp.where(done, jnp.int32(0), carry["ep_len"] + 1)

        new_ep_rets = jax.lax.cond(
            done,
            lambda: carry["ep_rets"].at[carry["ep_count"]].set(carry["ep_return"] + reward),
            lambda: carry["ep_rets"],
        )
        new_ep_count = carry["ep_count"] + done.astype(jnp.int32)

        return {
            **carry,
            "key": key,
            "frame_stack": new_frame_stack,
            "env_state": new_env_state,
            "params": params,
            "ep_return": ep_return,
            "ep_len": ep_len,
            "ep_rets": new_ep_rets,
            "ep_count": new_ep_count,
            "loss_arr": carry["loss_arr"].at[global_step].set(loss),
        }

    return body


@functools.lru_cache(maxsize=None)
def _make_runner(total_steps, hidden_dim, num_layers):
    body = _make_body(hidden_dim, num_layers)

    @jax.jit
    def run(init_carry):
        return jax.lax.fori_loop(0, total_steps, body, init_carry)

    return run


@functools.lru_cache(maxsize=None)
def _make_chunk_runner(chunk_steps, hidden_dim, num_layers):
    body = _make_body(hidden_dim, num_layers)

    @jax.jit
    def run_chunk(carry):
        return jax.lax.fori_loop(0, chunk_steps, body, carry)

    return run_chunk


def get_runner(total_steps, hidden_dim, num_layers):
    """Public accessor for the cached, JIT-compiled runner."""
    return _make_runner(total_steps, hidden_dim, num_layers)


def get_chunk_runner(chunk_steps, hidden_dim, num_layers):
    """Public accessor for the cached chunk runner (for vmap + progress)."""
    return _make_chunk_runner(chunk_steps, hidden_dim, num_layers)


def build_init_carry(cfg, total_steps, init_params, key):
    """Build the initial carry dict for one training run."""
    key, kr = jax.random.split(key)
    env_state = pong_env.reset(kr)
    frame_stack = make_frame_stack(pixels_obs(env_state))
    return {
        "key": key,
        "frame_stack": frame_stack,
        "env_state": env_state,
        "params": init_params,
        "ep_return": jnp.float32(0.0),
        "ep_len": jnp.int32(0),
        "lr": jnp.float32(cfg["lr"]),
        "decay_dur": jnp.float32(cfg["decay_dur"]),
        "loss_arr": jnp.zeros((total_steps,), dtype=jnp.float32),
        "ep_rets": jnp.zeros((total_steps,), dtype=jnp.float32),
        "ep_count": jnp.int32(0),
        "step_offset": jnp.int32(0),
    }


def run_config(cfg, total_steps, init_params):
    """Run one full approximate Q-learning training loop for PixelPong.

    Returns (ep_rets, losses, params).
    """
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]
    runner = _make_runner(total_steps, hidden_dim, num_layers)

    key = jax.random.key(0)
    key, kr = jax.random.split(key)
    env_state = pong_env.reset(kr)
    frame_stack = make_frame_stack(pixels_obs(env_state))

    fc = runner(
        {
            "key": key,
            "frame_stack": frame_stack,
            "env_state": env_state,
            "params": init_params,
            "ep_return": jnp.float32(0.0),
            "ep_len": jnp.int32(0),
            "lr": jnp.float32(cfg["lr"]),
            "decay_dur": jnp.float32(cfg["decay_dur"]),
            "loss_arr": jnp.zeros((total_steps,), dtype=jnp.float32),
            "ep_rets": jnp.zeros((total_steps,), dtype=jnp.float32),
            "ep_count": jnp.int32(0),
            "step_offset": jnp.int32(0),
        }
    )

    ep_count = int(fc["ep_count"])
    ep_rets = np.asarray(fc["ep_rets"])[:ep_count].tolist()
    losses = np.asarray(fc["loss_arr"]).tolist()

    return ep_rets, losses, fc["params"]
