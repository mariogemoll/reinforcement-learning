# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import jax
import jax.numpy as jnp

from rl.core.policy_gradient import calculate_returns, forward_mlp, init_mlp_params

__all__ = [
    "init_mlp_params",
    "forward_mlp",
    "calculate_returns",
    "sample_action",
    "rollout_once",
    "generate_rollouts",
    "log_prob_trajectory",
    "log_prob_trajectories",
]


def sample_action(key, params, obs):
    logits = forward_mlp(params, obs, jax.nn.relu)
    return jax.random.categorical(key, logits)

def rollout_once(key, env, env_params, params, max_steps):
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env_params)

    obs_buf = jnp.zeros((max_steps + 1, obs.shape[0]))
    act_buf = jnp.zeros((max_steps,), dtype=jnp.int32)
    rew_buf = jnp.zeros((max_steps,))
    done_buf = jnp.zeros((max_steps,), dtype=jnp.bool_)

    obs_buf = obs_buf.at[0].set(obs)

    def step_fn(carry, t):
        key, state, obs, done, obs_buf, act_buf, rew_buf, done_buf = carry
        key, key_a, key_step = jax.random.split(key, 3)

        action = sample_action(key_a, params, obs)
        next_obs, next_state, reward, next_done, _ = env.step(key_step, state, action, env_params)

        obs_next = jnp.where(done, obs, next_obs)
        done_next = jnp.logical_or(done, next_done)
        reward_t = jnp.where(done, 0.0, reward)
        action_t = jnp.where(done, 0, action)

        obs_buf = obs_buf.at[t + 1].set(obs_next)
        act_buf = act_buf.at[t].set(action_t)
        rew_buf = rew_buf.at[t].set(reward_t)
        done_buf = done_buf.at[t].set(done_next)

        return (key, next_state, obs_next, done_next, obs_buf, act_buf, rew_buf, done_buf), None

    carry0 = (key, state, obs, False, obs_buf, act_buf, rew_buf, done_buf)
    (key, state, obs, done, obs_buf, act_buf, rew_buf, done_buf), _ = jax.lax.scan(
        step_fn, carry0, jnp.arange(max_steps)
    )

    done_int = done_buf.astype(jnp.int32)
    first_done = jnp.argmax(done_int)
    any_done = jnp.any(done_buf)
    length = jnp.where(any_done, first_done + 1, max_steps)

    return {
        "observations": obs_buf,
        "actions": act_buf,
        "rewards": rew_buf,
        "done": done_buf,
        "length": length,
    }

def generate_rollouts(key, env, env_params, params, num_rollouts, max_steps):
    keys = jax.random.split(key, num_rollouts)
    return jax.vmap(lambda k: rollout_once(k, env, env_params, params, max_steps))(keys)

def log_prob_trajectory(params, observations, actions, length):
    max_steps = actions.shape[0]

    def step(carry, t):
        obs = observations[t]
        action = actions[t]
        logits = forward_mlp(params, obs, jax.nn.relu)
        log_prob = jax.nn.log_softmax(logits)[action]
        log_prob = jnp.where(t < length, log_prob, 0.0)
        return carry + log_prob, None

    total_log_prob, _ = jax.lax.scan(step, 0.0, jnp.arange(max_steps))
    return total_log_prob

def log_prob_trajectories(params, rollouts):
    return jax.vmap(log_prob_trajectory, in_axes=(None, 0, 0, 0))(
        params, rollouts['observations'], rollouts['actions'], rollouts['length']
    )
