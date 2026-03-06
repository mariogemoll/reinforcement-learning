# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate golden Pendulum-v1 step vectors from Gymnax.

Writes one binary file consumed by the TypeScript vitest suite:
  ts/src/pendulum/environment.step-vectors.bin — N x 11 float32

Record layout (11 float32):
  input:   theta, theta_dot, elapsed_steps, torque                   (4)
  output:  theta, theta_dot, cos(theta), sin(theta), theta_dot       (5)
  reward:  reward                                                     (1)
  trunc:   0.0 or 1.0                                                (1)

Run with:
    .venv/bin/python generate_pendulum_vectors.py
"""

from pathlib import Path

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

STEP_STRIDE = 11
MAX_STEPS = 200

rows: list[list[float]] = []


def add_step(
    env,
    env_params,
    state_template,
    theta: float,
    theta_dot: float,
    elapsed_steps: int,
    torque: float,
) -> None:
    state = state_template.replace(
        theta=jnp.array(theta, dtype=jnp.float32),
        theta_dot=jnp.array(theta_dot, dtype=jnp.float32),
        time=jnp.array(elapsed_steps, dtype=jnp.int32),
        last_u=jnp.array(0.0, dtype=jnp.float32),
    )
    action = jnp.array([torque], dtype=jnp.float32)
    obs, next_state, reward, done, _ = env.step(jax.random.key(0), state, action, env_params)
    assert not bool(done)

    rows.append(
        [
            float(theta), float(theta_dot), float(elapsed_steps), float(torque),
            float(next_state.theta), float(next_state.theta_dot),
            float(obs[0]), float(obs[1]), float(obs[2]),
            float(reward),
            0.0,
        ]
    )


def main() -> None:
    env, env_params = gymnax.make("Pendulum-v1")
    _, state_template = env.reset(jax.random.key(0), env_params)

    # Systematic edge-ish coverage
    for theta in [-3.5, -np.pi, -1.0, 0.0, 1.0, np.pi, 3.5]:
        for theta_dot in [-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0]:
            for torque in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0]:
                add_step(env, env_params, state_template, theta, theta_dot, 0, torque)

    # Time-limit boundary without terminal auto-reset.
    for elapsed_steps in [197, 198]:
        for torque in [-2.0, 0.0, 2.0]:
            add_step(env, env_params, state_template, 0.5, -0.25, elapsed_steps, torque)

    # Random broad coverage
    rng = np.random.default_rng(42)
    for _ in range(500):
        theta = float(rng.uniform(-4 * np.pi, 4 * np.pi))
        theta_dot = float(rng.uniform(-12.0, 12.0))
        # Keep strictly pre-terminal for gymnax non-reset transition parity.
        elapsed_steps = int(rng.integers(0, MAX_STEPS - 1))
        torque = float(rng.uniform(-4.0, 4.0))
        add_step(env, env_params, state_template, theta, theta_dot, elapsed_steps, torque)

    out_dir = Path(__file__).parent.parent / "ts" / "src" / "pendulum"
    out_path = out_dir / "environment.step-vectors.bin"
    arr = np.array(rows, dtype=np.float32)
    assert arr.shape[1] == STEP_STRIDE
    arr.tofile(out_path)
    print(f"Wrote {len(rows)} step vectors to {out_path}")


if __name__ == "__main__":
    main()
