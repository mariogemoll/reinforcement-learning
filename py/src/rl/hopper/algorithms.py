# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .types import PPOMetrics


def gaussian_sample(mean: jax.Array, log_std: jax.Array, rng: jax.Array) -> jax.Array:
    return mean + jnp.exp(log_std) * jax.random.normal(rng, shape=mean.shape)


def gaussian_log_prob(mean: jax.Array, log_std: jax.Array, action: jax.Array) -> jax.Array:
    var = jnp.exp(2.0 * log_std)
    return (
        -0.5 * (((action - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    ).sum(axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    return (0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + log_std).sum(axis=-1)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)

    def gae_step(last_gae, inputs):
        reward, value, done, next_val = inputs
        next_non_terminal = 1.0 - done
        delta = reward + gamma * next_val * next_non_terminal - value
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        return last_gae, last_gae

    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros_like(next_value),
        (rewards[::-1], values[::-1], dones[::-1], next_values[::-1]),
    )
    advantages = advantages[::-1]
    return advantages, advantages + values


def compute_explained_variance(y_pred: jax.Array, y_true: jax.Array) -> float:
    var_y = jnp.var(y_true)
    explained_var = jnp.where(var_y > 1e-8, 1.0 - jnp.var(y_true - y_pred) / var_y, jnp.nan)
    return float(explained_var)


@jax.jit
def sample_action_and_value(
    train_state: TrainState,
    obs: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean, log_std, value = train_state.apply_fn(train_state.params, obs)
    action = gaussian_sample(mean, log_std, rng)
    log_prob = gaussian_log_prob(mean, log_std, action)
    return action, log_prob, value


@jax.jit
def get_value(train_state: TrainState, obs: jax.Array) -> jax.Array:
    return train_state.apply_fn(train_state.params, obs)[2]


@jax.jit
def train_minibatch(
    train_state: TrainState,
    obs: jax.Array,
    action: jax.Array,
    old_logp: jax.Array,
    advantage: jax.Array,
    target_return: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[TrainState, PPOMetrics]:
    def loss_fn(params):
        mean, log_std, values = train_state.apply_fn(params, obs)
        new_logp = gaussian_log_prob(mean, log_std, action)
        entropy = gaussian_entropy(log_std).mean()

        log_ratio = new_logp - old_logp
        ratio = jnp.exp(log_ratio)

        pg_loss_1 = -advantage * ratio
        pg_loss_2 = -advantage * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg_loss_1, pg_loss_2).mean()
        v_loss = 0.5 * jnp.square(values - target_return).mean()
        loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        return loss, (pg_loss, v_loss, entropy, clip_fraction, approx_kl)

    grads, metrics = jax.grad(loss_fn, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=grads), metrics


@jax.jit
def train_epoch(
    train_state: TrainState,
    obs: jax.Array,
    action: jax.Array,
    old_logp: jax.Array,
    advantage: jax.Array,
    target_return: jax.Array,
    minibatches: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[TrainState, PPOMetrics]:
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    def run_minibatch(carry, mb):
        return train_minibatch(
            carry,
            obs[mb],
            action[mb],
            old_logp[mb],
            advantage[mb],
            target_return[mb],
            clip_eps,
            vf_coef,
            ent_coef,
        )

    train_state, metrics = jax.lax.scan(run_minibatch, train_state, minibatches)
    return train_state, tuple(metric.mean() for metric in metrics)
