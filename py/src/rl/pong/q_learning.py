# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import functools

import anywidget
import jax
import jax.numpy as jnp
import numpy as np
import traitlets
from flax import nnx

from rl.core.assets import dist_asset_path, shared_css_text
from rl.pong import env as pong_env


class PongVisualization(anywidget.AnyWidget):
    _esm = dist_asset_path("pong-visualization.js")
    _css = shared_css_text()

    weights_base64 = traitlets.Unicode("").tag(sync=True)

NUM_ACTIONS = 3
# Compact features derived from PongState — normalised to ~[-1, 1].
# Mirrors extractFeatures() in ts/src/pong/nn-policy.ts.
OBS_DIM = 6
START_EPSILON = 1.0
END_EPSILON = 0.05
GAMMA = 0.99


def extract_features(state: pong_env.PongState) -> jax.Array:
    """Convert a PongState into a 6-dim float32 feature vector.

    Features (all normalised to approx [-1, 1]):
      0: p1 (our paddle)
      1: p2 (opponent)
      2: ball row
      3: ball col
      4: ball vrow
      5: ball vcol
    """
    return jnp.stack([
        state.p1       / (pong_env.HEIGHT / 2) - 1.0,
        state.p2       / (pong_env.HEIGHT / 2) - 1.0,
        state.ball_row / (pong_env.HEIGHT / 2) - 1.0,
        state.ball_col / (pong_env.WIDTH  / 2) - 1.0,
        state.ball_vrow / pong_env.MAX_SPEED,
        state.ball_vcol,
    ]).astype(jnp.float32)


class QNetwork(nnx.Module):
    def __init__(self, hidden_dim, num_layers, rngs: nnx.Rngs):
        dims = [OBS_DIM] + [hidden_dim] * num_layers + [NUM_ACTIONS]
        self.num_layers = len(dims) - 1
        for i in range(self.num_layers):
            setattr(self, f"layer_{i}", nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = nnx.relu(getattr(self, f"layer_{i}")(x))
        return getattr(self, f"layer_{self.num_layers - 1}")(x)


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
def _make_runner(total_steps, hidden_dim, num_layers):
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

        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (i / carry["decay_dur"]),
        )
        greedy = jnp.argmax(_fwd(carry["params"], carry["obs"]))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=NUM_ACTIONS)
        action = jax.lax.select(jax.random.uniform(ka) > eps, greedy, rand_a)

        p2_action = pong_env.computer_player(carry["env_state"])
        next_env_state, done = pong_env.step(carry["env_state"], action, p2_action)
        next_obs = extract_features(next_env_state)
        reward = jnp.float32(1.0) - done.astype(jnp.float32)

        timeout = carry["ep_len"] + 1 >= pong_env.MAX_STEPS
        terminal = done & ~timeout

        loss, params = _train_step(
            carry["params"],
            carry["obs"],
            action,
            reward,
            next_obs,
            terminal.astype(jnp.float32),
            carry["lr"],
        )

        reset_env_state = pong_env.reset(kre)
        reset_obs = extract_features(reset_env_state)
        new_obs = jnp.where(done, reset_obs, next_obs)
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
            "obs": new_obs,
            "env_state": new_env_state,
            "params": params,
            "ep_return": ep_return,
            "ep_len": ep_len,
            "ep_rets": new_ep_rets,
            "ep_count": new_ep_count,
            "loss_arr": carry["loss_arr"].at[i].set(loss),
        }

    @jax.jit
    def run(init_carry):
        return jax.lax.fori_loop(0, total_steps, body, init_carry)

    return run


def run_config(cfg, total_steps, init_params):
    """Run one full approximate Q-learning training loop for Pong.

    Returns (ep_rets, losses, params).
    """
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]
    runner = _make_runner(total_steps, hidden_dim, num_layers)

    key = jax.random.key(0)
    key, kr = jax.random.split(key)
    env_state = pong_env.reset(kr)
    obs = extract_features(env_state)

    fc = runner(
        {
            "key": key,
            "obs": obs,
            "env_state": env_state,
            "params": init_params,
            "ep_return": jnp.float32(0.0),
            "ep_len": jnp.int32(0),
            "lr": jnp.float32(cfg["lr"]),
            "decay_dur": jnp.float32(cfg["decay_dur"]),
            "loss_arr": jnp.zeros((total_steps,), dtype=jnp.float32),
            "ep_rets": jnp.zeros((total_steps,), dtype=jnp.float32),
            "ep_count": jnp.int32(0),
        }
    )

    ep_count = int(fc["ep_count"])
    ep_rets = np.asarray(fc["ep_rets"])[:ep_count].tolist()
    losses = np.asarray(fc["loss_arr"]).tolist()

    return ep_rets, losses, fc["params"]
