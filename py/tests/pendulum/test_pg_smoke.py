# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import jax
import jax.numpy as jnp

from rl.pendulum.policy_gradient import make_value_training_step


def test_pendulum_value_training_step_smoke():
    make_carry, training_step = make_value_training_step(
        n_rollouts=2,
        max_steps=8,
        lr_policy=3e-4,
        value_lr_mult=2.0,
        gamma=0.99,
    )

    carry = make_carry(0)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    carry, (losses, returns) = jax.lax.scan(training_step, carry, keys)
    _ = jax.tree.map(lambda x: x.block_until_ready(), carry)

    assert losses.shape == (2,)
    assert returns.shape == (2,)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.all(jnp.isfinite(returns))
