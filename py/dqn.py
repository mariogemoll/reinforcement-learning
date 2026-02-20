# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import functools

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

ENV_NAME = "CartPole-v1"
NUM_ACTIONS = 2
OBS_DIM = 4
START_EPSILON = 1.0
END_EPSILON = 0.05


@functools.lru_cache(maxsize=1)
def _get_env_bundle():
    return gymnax.make(ENV_NAME)


class QNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(OBS_DIM, 64, rngs=rngs)
        self.layer2 = nnx.Linear(64, 64, rngs=rngs)
        self.output = nnx.Linear(64, NUM_ACTIONS, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer1(x))
        x = nnx.relu(self.layer2(x))
        return self.output(x)


@functools.lru_cache(maxsize=1)
def _get_graphdef():
    graphdef, _ = nnx.split(QNetwork(rngs=nnx.Rngs(0)))
    return graphdef


def forward(params, x):
    return nnx.merge(_get_graphdef(), params)(x)


def make_model(params):
    """Reconstruct a QNetwork from params (e.g. for weight export)."""
    return nnx.merge(_get_graphdef(), params)


def fresh_params():
    """Return (params, target_params) initialised from seed 0."""
    r = nnx.Rngs(0)
    m = QNetwork(rngs=r.fork())
    t = QNetwork(rngs=r.fork())
    nnx.update(t, nnx.state(m))
    _, p = nnx.split(m)
    _, tp = nnx.split(t)
    return p, tp


@functools.lru_cache(maxsize=1)
def _get_tx():
    return optax.chain(optax.clip_by_global_norm(10.0), optax.scale_by_adam())


@jax.jit
def _train_step(params, target_params, opt_state, batch, lr):
    def _loss(p, tp, b):
        s, a, r, ns, d = b
        q = forward(p, s)
        curr_q = jnp.take_along_axis(q, a[:, None], axis=1).squeeze()
        next_q = jnp.max(forward(tp, ns), axis=1)
        tgt = r + 0.99 * next_q * (1.0 - d)
        return jnp.mean(optax.huber_loss(curr_q - jax.lax.stop_gradient(tgt)))

    loss, grads = jax.value_and_grad(_loss)(params, target_params, batch)
    updates, new_o = _get_tx().update(grads, opt_state, params)
    new_p = optax.apply_updates(params, jax.tree.map(lambda u: -lr * u, updates))
    return loss, new_p, new_o


@functools.lru_cache(maxsize=None)
def _make_runner(buf_cap, batch_sz, total_steps, env_name=ENV_NAME):
    env, env_params = _get_env_bundle() if env_name == ENV_NAME else gymnax.make(env_name)

    def body(i, carry):
        key = carry["key"]
        key, ka, kr, ks, kre, ksamp = jax.random.split(key, 6)

        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (i / carry["decay_dur"]),
        )
        greedy = jnp.argmax(forward(carry["params"], carry["obs"]))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=NUM_ACTIONS)
        action = jax.lax.select(jax.random.uniform(ka) > eps, greedy, rand_a)

        next_obs, next_env_state, reward, done, _ = env.step(
            ks, carry["env_state"], action, env_params
        )

        timeout = carry["ep_len"] + 1 >= env_params.max_steps_in_episode
        terminal = done & ~timeout

        bi = carry["buf_idx"] % buf_cap
        bs = carry["buf_states"].at[bi].set(carry["obs"])
        ba = carry["buf_actions"].at[bi].set(action)
        br = carry["buf_rewards"].at[bi].set(reward)
        bns = carry["buf_next_states"].at[bi].set(next_obs)
        bd = carry["buf_dones"].at[bi].set(terminal.astype(jnp.float32))
        bsz = jnp.minimum(carry["buf_size"] + 1, buf_cap)

        def do_train(_):
            idx = jax.random.randint(ksamp, shape=(batch_sz,), minval=0, maxval=bsz)
            batch = (bs[idx], ba[idx], br[idx], bns[idx], bd[idx])
            loss, p, o = _train_step(
                carry["params"],
                carry["target_params"],
                carry["opt_state"],
                batch,
                carry["lr"],
            )
            return loss, p, o

        def skip_train(_):
            return jnp.float32(0.0), carry["params"], carry["opt_state"]

        loss, params, opt_state = jax.lax.cond(
            bsz >= carry["learn_start"], do_train, skip_train, None
        )
        target_params = jax.lax.cond(
            i % carry["upd_every"] == 0,
            lambda: params,
            lambda: carry["target_params"],
        )

        reset_obs, reset_env_state = env.reset(kre, env_params)
        new_obs = jnp.where(done, reset_obs, next_obs)
        new_env_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_env_state, next_env_state
        )
        ep_return = jnp.where(done, jnp.float32(0.0), carry["ep_return"] + reward)
        ep_len = jnp.where(done, jnp.int32(0), carry["ep_len"] + 1)

        # Best-checkpoint tracking: rolling 50-episode mean.
        ep_buf_slot = carry["ep_ret_buf_idx"] % 50
        new_ep_ret_buf = jnp.where(
            done,
            carry["ep_ret_buf"].at[ep_buf_slot].set(carry["ep_return"] + reward),
            carry["ep_ret_buf"],
        )
        new_ep_ret_buf_idx = carry["ep_ret_buf_idx"] + done.astype(jnp.int32)
        new_ep_ret_buf_size = jnp.minimum(carry["ep_ret_buf_size"] + done.astype(jnp.int32), 50)
        rolling = jnp.mean(new_ep_ret_buf)
        should_update_best = done & (new_ep_ret_buf_size >= 50) & (rolling > carry["best_score"])
        new_best_params = jax.lax.cond(
            should_update_best, lambda: params, lambda: carry["best_params"]
        )
        new_best_score = jnp.where(should_update_best, rolling, carry["best_score"])

        return {
            **carry,
            "key": key,
            "obs": new_obs,
            "env_state": new_env_state,
            "params": params,
            "target_params": target_params,
            "opt_state": opt_state,
            "buf_states": bs,
            "buf_actions": ba,
            "buf_rewards": br,
            "buf_next_states": bns,
            "buf_dones": bd,
            "buf_idx": carry["buf_idx"] + 1,
            "buf_size": bsz,
            "ep_return": ep_return,
            "ep_len": ep_len,
            "ep_ret_buf": new_ep_ret_buf,
            "ep_ret_buf_idx": new_ep_ret_buf_idx,
            "ep_ret_buf_size": new_ep_ret_buf_size,
            "best_params": new_best_params,
            "best_score": new_best_score,
            "loss_arr": carry["loss_arr"].at[i].set(loss),
            "step_rewards": carry["step_rewards"].at[i].set(reward),
            "step_dones": carry["step_dones"].at[i].set(done.astype(jnp.float32)),
        }

    @jax.jit
    def run(init_carry):
        return jax.lax.fori_loop(0, total_steps, body, init_carry)

    return run


def run_config(cfg, total_steps, init_params):
    """Run one full DQN training loop.

    Returns (ep_rets, losses, best_params) where best_params are the weights
    at the peak rolling-50-episode mean return.
    """
    p, tp = init_params
    opt_state = _get_tx().init(p)
    env, env_params = _get_env_bundle()
    runner = _make_runner(cfg["buf_cap"], cfg["batch_size"], total_steps, ENV_NAME)

    key = jax.random.key(0)
    key, kr = jax.random.split(key)
    obs, env_state = env.reset(kr, env_params)

    fc = runner(
        {
            "key": key,
            "obs": obs,
            "env_state": env_state,
            "params": p,
            "target_params": tp,
            "opt_state": opt_state,
            "buf_states": jnp.zeros((cfg["buf_cap"], OBS_DIM), dtype=jnp.float32),
            "buf_actions": jnp.zeros((cfg["buf_cap"],), dtype=jnp.int32),
            "buf_rewards": jnp.zeros((cfg["buf_cap"],), dtype=jnp.float32),
            "buf_next_states": jnp.zeros((cfg["buf_cap"], OBS_DIM), dtype=jnp.float32),
            "buf_dones": jnp.zeros((cfg["buf_cap"],), dtype=jnp.float32),
            "buf_idx": jnp.int32(0),
            "buf_size": jnp.int32(0),
            "ep_return": jnp.float32(0.0),
            "ep_len": jnp.int32(0),
            "ep_ret_buf": jnp.zeros((50,), dtype=jnp.float32),
            "ep_ret_buf_idx": jnp.int32(0),
            "ep_ret_buf_size": jnp.int32(0),
            "best_params": p,
            "best_score": jnp.float32(-jnp.inf),
            "lr": jnp.float32(cfg["lr"]),
            "decay_dur": jnp.float32(cfg["decay_dur"]),
            "learn_start": jnp.int32(cfg["learn_start"]),
            "upd_every": jnp.int32(cfg["upd_every"]),
            "loss_arr": jnp.zeros((total_steps,), dtype=jnp.float32),
            "step_rewards": jnp.zeros((total_steps,), dtype=jnp.float32),
            "step_dones": jnp.zeros((total_steps,), dtype=jnp.float32),
        }
    )

    sr = np.asarray(fc["step_rewards"])
    sd = np.asarray(fc["step_dones"])
    ep_rets, ep_r = [], 0.0
    for r, d in zip(sr, sd):
        ep_r += float(r)
        if d:
            ep_rets.append(ep_r)
            ep_r = 0.0

    losses = np.asarray(fc["loss_arr"])
    losses = losses[losses != 0.0].tolist()

    return ep_rets, losses, fc["best_params"]


def __getattr__(name):
    # Backward compatibility for notebooks importing dqn.env / dqn.env_params.
    if name == "env":
        return _get_env_bundle()[0]
    if name == "env_params":
        return _get_env_bundle()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
