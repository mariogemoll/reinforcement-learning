# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import jax
import jax.numpy as jnp

from rl.hopper.algorithms import compute_gae, train_epoch
from rl.hopper.config import PPOConfig
from rl.hopper.networks import create_train_state


def test_hopper_gae_shapes_and_finiteness():
    rewards = jnp.array([[1.0, 0.5], [0.2, -0.1], [0.0, 0.3]], dtype=jnp.float32)
    values = jnp.array([[0.1, 0.2], [0.0, 0.4], [0.3, -0.2]], dtype=jnp.float32)
    dones = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    next_value = jnp.array([0.5, -0.1], dtype=jnp.float32)

    advantages, returns = compute_gae(rewards, values, dones, next_value, 0.99, 0.95)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert jnp.all(jnp.isfinite(advantages))
    assert jnp.all(jnp.isfinite(returns))


def test_hopper_train_epoch_smoke():
    config = PPOConfig(num_envs=2, batch_size=4)
    train_state, _ = create_train_state(
        config,
        obs_dim=3,
        act_dim=2,
        rng=jax.random.PRNGKey(0),
    )

    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    action = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    old_logp = jax.random.normal(jax.random.PRNGKey(3), (8,))
    advantage = jax.random.normal(jax.random.PRNGKey(4), (8,))
    target_return = jax.random.normal(jax.random.PRNGKey(5), (8,))
    minibatches = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)

    new_state, metrics = train_epoch(
        train_state,
        obs,
        action,
        old_logp,
        advantage,
        target_return,
        minibatches,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
    )
    _ = jax.tree.map(lambda x: x.block_until_ready(), new_state.params)

    for metric in metrics:
        assert metric.shape == ()
        assert jnp.isfinite(metric)
