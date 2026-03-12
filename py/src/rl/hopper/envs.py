# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import gymnasium as gym
from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward

from .config import PPOConfig


def make_envs(config: PPOConfig):
    vectorization_mode = "sync" if config.num_envs == 1 else "async"
    envs = gym.make_vec(config.env_id, num_envs=config.num_envs, vectorization_mode=vectorization_mode)
    if config.normalize_obs:
        envs = NormalizeObservation(envs)
    if config.normalize_reward:
        envs = NormalizeReward(envs, gamma=config.gamma)
    return envs
