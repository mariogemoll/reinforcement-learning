# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import ACTIVATION, LOG_STD_INIT, ORTHO_INIT, POLICY_HIDDEN_DIMS, VALUE_HIDDEN_DIMS, PPOConfig


def _activation_fn(name: str):
    return nn.relu if name.lower() == "relu" else nn.tanh


def _kernel_init(use_ortho: bool):
    if use_ortho:
        return nn.initializers.orthogonal()
    return nn.initializers.lecun_normal()


class MLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int
    activation: str
    ortho_init: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation_fn = _activation_fn(self.activation)
        kernel_init = _kernel_init(self.ortho_init)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=kernel_init)(x)
            x = activation_fn(x)
        return nn.Dense(self.output_dim, kernel_init=kernel_init)(x)


class ActorCritic(nn.Module):
    act_dim: int
    policy_hidden_dims: tuple[int, ...] = POLICY_HIDDEN_DIMS
    value_hidden_dims: tuple[int, ...] = VALUE_HIDDEN_DIMS
    activation: str = ACTIVATION
    log_std_init: float = LOG_STD_INIT
    ortho_init: bool = ORTHO_INIT

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        critic = MLP(self.value_hidden_dims, 1, self.activation, self.ortho_init)
        actor_mean = MLP(self.policy_hidden_dims, self.act_dim, self.activation, self.ortho_init)

        value = critic(obs).squeeze(-1)
        mean = actor_mean(obs)
        log_std = self.param(
            "actor_log_std",
            lambda key, shape, dtype=jnp.float32: jnp.full(shape, self.log_std_init, dtype=dtype),
            (self.act_dim,),
        )
        return mean, jnp.broadcast_to(log_std, mean.shape), value


def create_train_state(
    config: PPOConfig,
    obs_dim: int,
    act_dim: int,
    rng: jax.Array,
) -> tuple[TrainState, jax.Array]:
    model = ActorCritic(
        act_dim=act_dim,
        policy_hidden_dims=config.policy_hidden_dims,
        value_hidden_dims=config.value_hidden_dims,
        activation=config.activation,
        log_std_init=config.log_std_init,
        ortho_init=config.ortho_init,
    )
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros((config.num_envs, obs_dim), dtype=jnp.float32))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return train_state, rng
