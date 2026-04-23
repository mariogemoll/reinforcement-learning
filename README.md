<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Reinforcement Learning

Jupyter notebooks and visualizations of RL algorithms.

Also used on
[mariogemoll.com/reinforcement-learning](https://mariogemoll.com/reinforcement-learning).

## Layout

- `ts/` — TypeScript visualizations (rendered on the website and embedded in notebooks).
- `py/` — JAX/Flax training code and Jupyter notebooks
- `scripts/` — Repo tooling.

## Visualizations

Located in `ts/src/visualizations/`:

- `gridworld` — Gridworld environment.
- `policy-iteration-v`, `policy-iteration-q` — P iteration over V and Q.
- `value-iteration-v`, `value-iteration-q` — Value iteration over V and Q.
- `monte-carlo` — Monte Carlo control.
- `cartpole` — CartPole environment and trained policy.
- `pendulum` — Pendulum environment and trained policy.
- `hopper` — Hopper (MuJoCo) environment and trained PPO policy.
- `pong`, `pong-policy` — Pong environment and trained policy.
- `pixel-pong`, `pixel-pong-policy` — Pixel-observation Pong and trained policy.
- `minatar-breakout` — MinAtar Breakout with a trained DQN.

## Notebooks

Located in `py/`:

- `ql.ipynb` — Tabular Q-learning on gridworld.
- `dqn.ipynb` — DQN.
- `dqn_minatar_breakout.ipynb`, `dqn_minatar_breakout_cnn.ipynb` — DQN on MinAtar Breakout (MLP and
  CNN).
- `cartpole_pg.ipynb`, `cartpole_pg_multiseed.ipynb` — REINFORCE on CartPole.
- `pendulum_pg.ipynb`, `pendulum_pg_multiseed.ipynb` — Policy gradient on Pendulum.
- `hopper_ppo.ipynb` — PPO on Hopper.
- `pong.ipynb` — Pong training.
- `pixel_pong_ql.ipynb` — Q-learning on pixel-observation Pong.
