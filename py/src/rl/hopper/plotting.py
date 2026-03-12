# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import matplotlib.pyplot as plt
import numpy as np

from .types import PPOTrainingResult


def plot_training_history(
    result: PPOTrainingResult,
    title: str = "Hopper PPO training",
) -> None:
    history = result.history
    if not history:
        raise ValueError("No history to plot.")

    x = np.arange(1, len(history) + 1)
    mean_returns = np.array([row["mean_ep_return"] for row in history], dtype=float)
    pg_losses = np.array([row["pg_loss"] for row in history], dtype=float)
    vf_losses = np.array([row["vf_loss"] for row in history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, mean_returns)
    axes[0].set_title("Mean episode return")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Return")

    axes[1].plot(x, pg_losses, label="Policy loss")
    axes[1].plot(x, vf_losses, label="Value loss")
    axes[1].set_title("Losses")
    axes[1].set_xlabel("Update")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
