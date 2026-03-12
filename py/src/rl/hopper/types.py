# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from .config import PPOConfig


PPOMetrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class PPOTrainingResult:
    config: PPOConfig
    history: list[dict[str, float]]
    checkpoint_path: Path
    best_checkpoint_path: Path | None
    episode_returns: list[float]
    params: Any
