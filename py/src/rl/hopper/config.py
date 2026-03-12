# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import re
from dataclasses import dataclass
from pathlib import Path


ENV_ID = "Hopper-v5"
NUM_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
STEPS_PER_ROLLOUT = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NORMALIZE_OBS = False
NORMALIZE_REWARD = False
POLICY_HIDDEN_DIMS = (64, 64)
VALUE_HIDDEN_DIMS = (64, 64)
ACTIVATION = "tanh"
LOG_STD_INIT = 0.0
ORTHO_INIT = False
TARGET_KL = 0.015
SEED = 42


@dataclass(frozen=True)
class PPOConfig:
    env_id: str = ENV_ID
    num_envs: int = NUM_ENVS
    total_timesteps: int = TOTAL_TIMESTEPS
    steps_per_rollout: int = STEPS_PER_ROLLOUT
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_eps: float = CLIP_EPS
    lr: float = LR
    ent_coef: float = ENT_COEF
    vf_coef: float = VF_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    normalize_obs: bool = NORMALIZE_OBS
    normalize_reward: bool = NORMALIZE_REWARD
    policy_hidden_dims: tuple[int, ...] = POLICY_HIDDEN_DIMS
    value_hidden_dims: tuple[int, ...] = VALUE_HIDDEN_DIMS
    activation: str = ACTIVATION
    log_std_init: float = LOG_STD_INIT
    ortho_init: bool = ORTHO_INIT
    target_kl: float | None = TARGET_KL
    seed: int = SEED
    output_dir: Path = Path(".")
    csv_name: str = "hopper_ppo_history.csv"
    checkpoint_name: str | None = None


def default_checkpoint_name(env_id: str) -> str:
    env_slug = re.sub(r"[^a-z0-9]+", "_", env_id.lower()).strip("_")
    return f"{env_slug}_ppo_jax.pkl"
