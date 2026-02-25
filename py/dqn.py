# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import functools
import warnings

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

warnings.filterwarnings(
    "ignore",
    message=(
        r"scatter inputs have incompatible types: cannot safely cast value from dtype=.* "
        r"to dtype=bool with jax_numpy_dtype_promotion=standard\."
    ),
    category=FutureWarning,
)

ENV_NAME = "CartPole-v1"
HIDDEN_DIM = 16
NUM_LAYERS = 1
START_EPSILON = 1.0
END_EPSILON = 0.05


@functools.lru_cache(maxsize=None)
def _get_env_bundle(env_name=ENV_NAME):
    return gymnax.make(env_name)


@functools.lru_cache(maxsize=None)
def _get_env_spec(env_name=ENV_NAME):
    env, env_params = _get_env_bundle(env_name)
    obs_shape = tuple(int(d) for d in env.observation_space(env_params).shape)
    num_actions = int(env.action_space(env_params).n)
    obs_dim = int(np.prod(obs_shape))
    return env, env_params, obs_shape, obs_dim, num_actions


class QNetwork(nnx.Module):
    def __init__(self, in_dim, num_actions, hidden_dim, num_layers, obs_ndim, rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.obs_ndim = obs_ndim
        dims = [in_dim] + [hidden_dim] * num_layers + [num_actions]
        self.num_layers = len(dims) - 1
        for i in range(self.num_layers):
            setattr(self, f"layer_{i}", nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == self.obs_ndim:
            x = x.reshape((self.in_dim,))
        elif x.ndim == self.obs_ndim + 1:
            x = x.reshape((x.shape[0], self.in_dim))
        else:
            x = x.reshape((-1, self.in_dim))
        for i in range(self.num_layers - 1):
            x = nnx.relu(getattr(self, f"layer_{i}")(x))
        return getattr(self, f"layer_{self.num_layers - 1}")(x)


@functools.lru_cache(maxsize=None)
def _get_graphdef(in_dim, num_actions, hidden_dim, num_layers, obs_ndim):
    graphdef, _ = nnx.split(
        QNetwork(in_dim, num_actions, hidden_dim, num_layers, obs_ndim, rngs=nnx.Rngs(0))
    )
    return graphdef


def forward(
    params,
    x,
    env_name=ENV_NAME,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    _, _, obs_shape, obs_dim, num_actions = _get_env_spec(env_name)
    return nnx.merge(
        _get_graphdef(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape)), params
    )(x)


def make_model(
    params,
    env_name=ENV_NAME,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    """Reconstruct a QNetwork from params (e.g. for weight export)."""
    _, _, obs_shape, obs_dim, num_actions = _get_env_spec(env_name)
    return nnx.merge(
        _get_graphdef(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape)), params
    )


def fresh_params(
    env_name=ENV_NAME,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    """Return (params, target_params) initialised from seed 0 for a given env/arch."""
    _, _, obs_shape, obs_dim, num_actions = _get_env_spec(env_name)
    r = nnx.Rngs(0)
    m = QNetwork(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape), rngs=r.fork())
    t = QNetwork(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape), rngs=r.fork())
    nnx.update(t, nnx.state(m))
    _, p = nnx.split(m)
    _, tp = nnx.split(t)
    return p, tp


@functools.lru_cache(maxsize=1)
def _get_tx():
    return optax.chain(optax.clip_by_global_norm(10.0), optax.scale_by_adam())


@functools.lru_cache(maxsize=None)
def _make_runner(
    buf_cap,
    batch_sz,
    total_steps,
    env_name=ENV_NAME,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    env, env_params, obs_shape, obs_dim, num_actions = _get_env_spec(env_name)
    graphdef = _get_graphdef(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape))

    def _fwd(params, x):
        return nnx.merge(graphdef, params)(x)

    @jax.jit
    def _train_step(params, target_params, opt_state, batch, lr):
        def _loss(p, tp, b):
            s, a, r, ns, d = b
            q = _fwd(p, s)
            curr_q = jnp.take_along_axis(q, a[:, None], axis=1).squeeze()
            next_q = jnp.max(_fwd(tp, ns), axis=1)
            d = d.astype(jnp.float32)
            tgt = r + 0.99 * next_q * (1.0 - d)
            return jnp.mean(optax.huber_loss(curr_q - jax.lax.stop_gradient(tgt)))

        loss, grads = jax.value_and_grad(_loss)(params, target_params, batch)
        updates, new_o = _get_tx().update(grads, opt_state, params)
        new_p = optax.apply_updates(params, jax.tree.map(lambda u: -lr * u, updates))
        return loss, new_p, new_o

    def body(i, carry):
        key = carry["key"]
        key, ka, kr, ks, kre, ksamp = jax.random.split(key, 6)

        global_step = i + carry["step_offset"]

        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (global_step / carry["decay_dur"]),
        )
        greedy = jnp.argmax(_fwd(carry["params"], carry["obs"]))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=num_actions)
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
        bd = carry["buf_dones"].at[bi].set(terminal)
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
            return loss, p, o, jnp.bool_(True)

        def skip_train(_):
            return jnp.float32(0.0), carry["params"], carry["opt_state"], jnp.bool_(False)

        loss, params, opt_state, trained = jax.lax.cond(
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

        loss_slot = carry["loss_idx"] % carry["loss_arr"].shape[0]
        new_loss_arr = jax.lax.cond(
            trained,
            lambda: carry["loss_arr"].at[loss_slot].set(loss),
            lambda: carry["loss_arr"],
        )
        new_loss_idx = carry["loss_idx"] + trained.astype(jnp.int32)

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
            "loss_arr": new_loss_arr,
            "loss_idx": new_loss_idx,
            "ep_rets": new_ep_rets,
            "ep_count": new_ep_count,
            "step_offset": carry["step_offset"],
        }

    @jax.jit
    def run(init_carry):
        return jax.lax.fori_loop(0, total_steps, body, init_carry)

    return run


@functools.lru_cache(maxsize=None)
def _make_chunk_runner(
    buf_cap,
    batch_sz,
    chunk_steps,
    env_name=ENV_NAME,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    """Create a chunked runner for vmap + progress updates."""
    env, env_params, obs_shape, obs_dim, num_actions = _get_env_spec(env_name)
    graphdef = _get_graphdef(obs_dim, num_actions, hidden_dim, num_layers, len(obs_shape))

    def _fwd(params, x):
        return nnx.merge(graphdef, params)(x)

    @jax.jit
    def _train_step(params, target_params, opt_state, batch, lr):
        def _loss(p, tp, b):
            s, a, r, ns, d = b
            q = _fwd(p, s)
            curr_q = jnp.take_along_axis(q, a[:, None], axis=1).squeeze()
            next_q = jnp.max(_fwd(tp, ns), axis=1)
            d = d.astype(jnp.float32)
            tgt = r + 0.99 * next_q * (1.0 - d)
            return jnp.mean(optax.huber_loss(curr_q - jax.lax.stop_gradient(tgt)))

        loss, grads = jax.value_and_grad(_loss)(params, target_params, batch)
        updates, new_o = _get_tx().update(grads, opt_state, params)
        new_p = optax.apply_updates(params, jax.tree.map(lambda u: -lr * u, updates))
        return loss, new_p, new_o

    def body(i, carry):
        key = carry["key"]
        key, ka, kr, ks, kre, ksamp = jax.random.split(key, 6)

        global_step = i + carry["step_offset"]

        eps = jnp.maximum(
            END_EPSILON,
            START_EPSILON - (START_EPSILON - END_EPSILON) * (global_step / carry["decay_dur"]),
        )
        greedy = jnp.argmax(_fwd(carry["params"], carry["obs"]))
        rand_a = jax.random.randint(kr, shape=(), minval=0, maxval=num_actions)
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
        bd = carry["buf_dones"].at[bi].set(terminal)
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
            return loss, p, o, jnp.bool_(True)

        def skip_train(_):
            return jnp.float32(0.0), carry["params"], carry["opt_state"], jnp.bool_(False)

        loss, params, opt_state, trained = jax.lax.cond(
            bsz >= carry["learn_start"], do_train, skip_train, None
        )
        target_params = jax.lax.cond(
            global_step % carry["upd_every"] == 0,
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

        loss_slot = carry["loss_idx"] % carry["loss_arr"].shape[0]
        new_loss_arr = jax.lax.cond(
            trained,
            lambda: carry["loss_arr"].at[loss_slot].set(loss),
            lambda: carry["loss_arr"],
        )
        new_loss_idx = carry["loss_idx"] + trained.astype(jnp.int32)

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
            "loss_arr": new_loss_arr,
            "loss_idx": new_loss_idx,
            "ep_rets": new_ep_rets,
            "ep_count": new_ep_count,
            "step_offset": carry["step_offset"],
        }

    @jax.jit
    def run_chunk(init_carry):
        return jax.lax.fori_loop(0, chunk_steps, body, init_carry)

    return run_chunk


def get_chunk_runner(chunk_steps, cfg, env_name=ENV_NAME, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
    """Public accessor for the cached chunk runner (for vmap + progress)."""
    return _make_chunk_runner(
        cfg["buf_cap"],
        cfg["batch_size"],
        chunk_steps,
        env_name,
        hidden_dim,
        num_layers,
    )


def build_init_carry(cfg, total_steps, init_params, key):
    """Build the initial carry dict for one training run."""
    env_name = cfg.get("env_name", ENV_NAME)

    p, tp = init_params
    opt_state = _get_tx().init(p)
    env, env_params, obs_shape, _, _ = _get_env_spec(env_name)

    key, kr = jax.random.split(key)
    obs, env_state = env.reset(kr, env_params)

    return {
        "key": key,
        "obs": obs,
        "env_state": env_state,
        "params": p,
        "target_params": tp,
        "opt_state": opt_state,
        "buf_states": jnp.zeros((cfg["buf_cap"],) + obs_shape, dtype=jnp.float32),
        "buf_actions": jnp.zeros((cfg["buf_cap"],), dtype=jnp.int32),
        "buf_rewards": jnp.zeros((cfg["buf_cap"],), dtype=jnp.float32),
        "buf_next_states": jnp.zeros((cfg["buf_cap"],) + obs_shape, dtype=jnp.float32),
        "buf_dones": jnp.zeros((cfg["buf_cap"],), dtype=jnp.bool_),
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
        "loss_idx": jnp.int32(0),
        "ep_rets": jnp.zeros((total_steps,), dtype=jnp.float32),
        "ep_count": jnp.int32(0),
        "step_offset": jnp.int32(0),
    }


def run_config(cfg, total_steps, init_params):
    """Run one full DQN training loop.

    Returns (ep_rets, losses, best_params) where best_params are the weights
    at the peak rolling-50-episode mean return.
    """
    env_name = cfg.get("env_name", ENV_NAME)
    hidden_dim = int(cfg.get("hidden_dim", HIDDEN_DIM))
    num_layers = int(cfg.get("num_layers", NUM_LAYERS))

    p, tp = init_params
    opt_state = _get_tx().init(p)
    env, env_params, obs_shape, _, _ = _get_env_spec(env_name)
    runner = _make_runner(
        cfg["buf_cap"],
        cfg["batch_size"],
        total_steps,
        env_name,
        hidden_dim,
        num_layers,
    )

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
            "buf_states": jnp.zeros((cfg["buf_cap"],) + obs_shape, dtype=jnp.float32),
            "buf_actions": jnp.zeros((cfg["buf_cap"],), dtype=jnp.int32),
            "buf_rewards": jnp.zeros((cfg["buf_cap"],), dtype=jnp.float32),
            "buf_next_states": jnp.zeros((cfg["buf_cap"],) + obs_shape, dtype=jnp.float32),
            "buf_dones": jnp.zeros((cfg["buf_cap"],), dtype=jnp.bool_),
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
            "loss_idx": jnp.int32(0),
            "ep_rets": jnp.zeros((total_steps,), dtype=jnp.float32),
            "ep_count": jnp.int32(0),
            "step_offset": jnp.int32(0),
        }
    )

    ep_count = int(fc["ep_count"])
    ep_rets = np.asarray(fc["ep_rets"])[:ep_count].tolist()
    loss_count = int(fc["loss_idx"])
    losses = np.asarray(fc["loss_arr"])[:loss_count].tolist()

    return ep_rets, losses, fc["best_params"]


def __getattr__(name):
    # Backward compatibility for notebooks importing dqn.env / dqn.env_params.
    if name == "env":
        return _get_env_bundle(ENV_NAME)[0]
    if name == "env_params":
        return _get_env_bundle(ENV_NAME)[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
