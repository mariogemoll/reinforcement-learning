# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import functools
import warnings

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# gymnax Pong-misc stores ball_velocity as int32 but clips float32 expressions
# into it in reflect_on_paddle.  The implicit cast is safe (velocities are
# bounded integers) but triggers a JAX FutureWarning at JIT-trace time.
# Suppress it here until gymnax fixes the upstream dtype.
warnings.filterwarnings(
    "ignore",
    message="scatter inputs have incompatible types",
    category=FutureWarning,
)

ENV_NAME = "Pong-misc"
NUM_ACTIONS = 3
# Compact hand-crafted features from the EnvState rather than the raw 30×40×3
# pixel observation that gymnax returns. This keeps the QL network small and
# avoids the representational bottleneck that would make plain online QL
# essentially impossible on pixel inputs.
OBS_DIM = 6
START_EPSILON = 1.0
END_EPSILON = 0.05
GAMMA = 0.99

_HEIGHT = 30
_WIDTH = 40
_MAX_SPEED = 3  # max |ball_velocity[0]|


@functools.lru_cache(maxsize=1)
def _get_env_bundle():
    env, params = gymnax.make(ENV_NAME)
    params = params.replace(use_ai_policy=True)
    return env, params


def extract_features(state):
    """Convert a Pong EnvState into a compact 6-dim float32 vector.

    All values are normalised to approximately [-1, 1]:
      0: our paddle y   (left)
      1: opponent paddle y  (right, AI-controlled)
      2: ball row
      3: ball col
      4: ball vertical velocity
      5: ball horizontal velocity
    """
    our_paddle = state.paddle_centers[0] / (_HEIGHT / 2) - 1.0
    opp_paddle = state.paddle_centers[1] / (_HEIGHT / 2) - 1.0
    ball_row = state.ball_position[0] / (_HEIGHT / 2) - 1.0
    ball_col = state.ball_position[1] / (_WIDTH / 2) - 1.0
    ball_vy = state.ball_velocity[0].astype(jnp.float32) / _MAX_SPEED
    ball_vx = state.ball_velocity[1].astype(jnp.float32)
    return jnp.stack(
        [our_paddle, opp_paddle, ball_row, ball_col, ball_vy, ball_vx]
    ).astype(jnp.float32)


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
    env, env_params = _get_env_bundle()
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
        key, ka, kr, ks, kre = jax.random.split(key, 5)

        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (i / carry["decay_dur"]),
        )
        greedy = jnp.argmax(_fwd(carry["params"], carry["obs"]))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=NUM_ACTIONS)
        action = jax.lax.select(jax.random.uniform(ka) > eps, greedy, rand_a)

        _, next_env_state, reward, done, _ = env.step(
            ks, carry["env_state"], action, env_params
        )
        next_obs = extract_features(next_env_state)

        timeout = carry["ep_len"] + 1 >= env_params.max_steps_in_episode
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

        _, reset_env_state = env.reset(kre, env_params)
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
    env, env_params = _get_env_bundle()
    runner = _make_runner(total_steps, hidden_dim, num_layers)

    key = jax.random.key(0)
    key, kr = jax.random.split(key)
    _, env_state = env.reset(kr, env_params)
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


def __getattr__(name):
    if name == "env":
        return _get_env_bundle()[0]
    if name == "env_params":
        return _get_env_bundle()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
