# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
import numpy as np


def _step_gymnasium(env, theta: float, theta_dot: float, elapsed_steps: int, torque: float):
    env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)
    env._elapsed_steps = elapsed_steps
    action = np.array([torque], dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
    return obs, reward, terminated, truncated, env.unwrapped.state.copy()


def _step_gymnax(env, env_params, state_template, theta: float, theta_dot: float, elapsed_steps: int, torque: float):
    state = state_template.replace(
        theta=jnp.array(theta, dtype=jnp.float32),
        theta_dot=jnp.array(theta_dot, dtype=jnp.float32),
        time=jnp.array(elapsed_steps, dtype=jnp.int32),
        last_u=jnp.array(0.0, dtype=jnp.float32),
    )
    action = jnp.array([torque], dtype=jnp.float32)
    obs, next_state, reward, done, _ = env.step(jax.random.key(0), state, action, env_params)
    return obs, next_state, reward, done


def test_gymnasium_matches_gymnax_nonterminal_pendulum_steps():
    gn_env, gn_params = gymnax.make("Pendulum-v1")
    _, template = gn_env.reset(jax.random.key(0), gn_params)

    gym_env = gym.make("Pendulum-v1")
    gym_env.reset(seed=0)

    rng = np.random.default_rng(42)
    for _ in range(1000):
        theta = float(rng.uniform(-4 * np.pi, 4 * np.pi))
        theta_dot = float(rng.uniform(-12.0, 12.0))
        elapsed_steps = int(rng.integers(0, 199))
        torque = float(rng.uniform(-4.0, 4.0))

        gym_obs, gym_reward, gym_terminated, gym_truncated, gym_next_state = _step_gymnasium(
            gym_env, theta, theta_dot, elapsed_steps, torque
        )
        gn_obs, gn_next_state, gn_reward, gn_done = _step_gymnax(
            gn_env, gn_params, template, theta, theta_dot, elapsed_steps, torque
        )

        assert not gym_terminated
        assert not gym_truncated
        assert not bool(gn_done)

        np.testing.assert_allclose(float(gn_next_state.theta), float(gym_next_state[0]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(gn_next_state.theta_dot), float(gym_next_state[1]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(gn_obs, dtype=np.float32), gym_obs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(gn_reward), float(gym_reward), rtol=1e-5, atol=1e-5)

    gym_env.close()


def test_gymnasium_matches_gymnax_terminal_step_reward_and_done():
    gn_env, gn_params = gymnax.make("Pendulum-v1")
    _, template = gn_env.reset(jax.random.key(0), gn_params)

    gym_env = gym.make("Pendulum-v1")
    gym_env.reset(seed=0)

    theta = 0.5
    theta_dot = -0.25
    elapsed_steps = 199
    torque = 1.2

    _gym_obs, gym_reward, gym_terminated, gym_truncated, _gym_next_state = _step_gymnasium(
        gym_env, theta, theta_dot, elapsed_steps, torque
    )
    _gn_obs, _gn_next_state, gn_reward, gn_done = _step_gymnax(
        gn_env, gn_params, template, theta, theta_dot, elapsed_steps, torque
    )

    assert not gym_terminated
    assert gym_truncated
    assert bool(gn_done)
    np.testing.assert_allclose(float(gn_reward), float(gym_reward), rtol=1e-5, atol=1e-5)

    gym_env.close()
