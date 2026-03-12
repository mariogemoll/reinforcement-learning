# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import csv
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import trange

from .algorithms import (
    compute_explained_variance,
    compute_gae,
    get_value,
    sample_action_and_value,
    train_epoch,
)
from .config import PPOConfig, default_checkpoint_name
from .envs import make_envs
from .networks import create_train_state
from .types import PPOTrainingResult


def log_completed_episodes(
    running_returns: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_returns: list[float],
) -> None:
    running_returns += rewards
    for env_idx in np.where(dones)[0]:
        episode_returns.append(float(running_returns[env_idx]))
        running_returns[env_idx] = 0.0


def train(config: PPOConfig) -> PPOTrainingResult:
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = make_envs(config)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    train_state, rng = create_train_state(config, obs_dim, act_dim, rng)

    obs_buf = np.zeros((config.steps_per_rollout, config.num_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((config.steps_per_rollout, config.num_envs, act_dim), dtype=np.float32)
    rew_buf = np.zeros((config.steps_per_rollout, config.num_envs), dtype=np.float32)
    done_buf = np.zeros((config.steps_per_rollout, config.num_envs), dtype=np.float32)
    logp_buf = np.zeros((config.steps_per_rollout, config.num_envs), dtype=np.float32)
    val_buf = np.zeros((config.steps_per_rollout, config.num_envs), dtype=np.float32)

    obs_np, _ = envs.reset(seed=config.seed)
    steps_per_update = config.steps_per_rollout * config.num_envs
    num_updates = config.total_timesteps // steps_per_update
    num_minibatches = steps_per_update // config.batch_size

    if steps_per_update % config.batch_size != 0:
        raise ValueError("steps_per_rollout * num_envs must be divisible by batch_size")

    history: list[dict[str, float]] = []
    episode_returns: list[float] = []
    running_returns = np.zeros(config.num_envs, dtype=np.float64)
    best_mean_return = -float("inf")
    best_checkpoint_path = output_dir / (
        "best_" + (config.checkpoint_name or default_checkpoint_name(config.env_id))
    )
    final_checkpoint_path = output_dir / (
        config.checkpoint_name or default_checkpoint_name(config.env_id)
    )

    csv_path = output_dir / config.csv_name
    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "update",
                "timestep",
                "mean_ep_return",
                "pg_loss",
                "vf_loss",
                "entropy",
                "clip_fraction",
                "approx_kl",
                "explained_variance",
            ],
        )
        csv_writer.writeheader()

        try:
            progress = trange(num_updates, desc="Hopper PPO", unit="update")
            for update_idx in progress:
                update = update_idx + 1
                for step in range(config.steps_per_rollout):
                    obs_buf[step] = obs_np
                    obs_t = jnp.asarray(obs_np, dtype=jnp.float32)

                    rng, action_rng = jax.random.split(rng)
                    action, log_prob, value = sample_action_and_value(
                        train_state, obs_t, action_rng
                    )

                    val_buf[step] = np.asarray(value)
                    logp_buf[step] = np.asarray(log_prob)
                    action_np = np.asarray(action)
                    act_buf[step] = action_np

                    next_obs, reward, terminated, truncated, _ = envs.step(action_np)
                    done_np = np.logical_or(terminated, truncated)

                    rew_buf[step] = reward
                    done_buf[step] = done_np.astype(np.float32)
                    log_completed_episodes(running_returns, reward, done_np, episode_returns)
                    obs_np = next_obs

                obs_t = jnp.asarray(obs_np, dtype=jnp.float32)
                obs_buf_t = jnp.asarray(obs_buf, dtype=jnp.float32)
                act_buf_t = jnp.asarray(act_buf, dtype=jnp.float32)
                rew_buf_t = jnp.asarray(rew_buf, dtype=jnp.float32)
                done_buf_t = jnp.asarray(done_buf, dtype=jnp.float32)
                logp_buf_t = jnp.asarray(logp_buf, dtype=jnp.float32)
                val_buf_t = jnp.asarray(val_buf, dtype=jnp.float32)

                next_value = get_value(train_state, obs_t)
                advantages, returns = compute_gae(
                    rew_buf_t,
                    val_buf_t,
                    done_buf_t,
                    next_value,
                    config.gamma,
                    config.gae_lambda,
                )

                b_obs = obs_buf_t.reshape(-1, obs_dim)
                b_act = act_buf_t.reshape(-1, act_dim)
                b_logp = logp_buf_t.reshape(-1)
                b_adv = advantages.reshape(-1)
                b_val = val_buf_t.reshape(-1)
                b_ret = returns.reshape(-1)
                explained_var = compute_explained_variance(b_val, b_ret)

                indices = np.arange(steps_per_update)
                pg_losses = []
                vf_losses = []
                entropies = []
                clip_fractions = []
                approx_kls = []
                for _ in range(config.n_epochs):
                    np.random.shuffle(indices)
                    minibatches = jnp.asarray(indices.reshape(num_minibatches, config.batch_size))
                    train_state, metrics = train_epoch(
                        train_state,
                        b_obs,
                        b_act,
                        b_logp,
                        b_adv,
                        b_ret,
                        minibatches,
                        config.clip_eps,
                        config.vf_coef,
                        config.ent_coef,
                    )
                    pg_loss, vf_loss, entropy, clip_fraction, approx_kl = metrics
                    pg_losses.append(float(pg_loss))
                    vf_losses.append(float(vf_loss))
                    entropies.append(float(entropy))
                    clip_fractions.append(float(clip_fraction))
                    approx_kls.append(float(approx_kl))
                    if config.target_kl is not None and float(approx_kl) > config.target_kl:
                        break

                mean_ep_return = float(np.mean(episode_returns[-10:])) if episode_returns else np.nan
                row = {
                    "update": float(update),
                    "timestep": float(update * steps_per_update),
                    "mean_ep_return": mean_ep_return,
                    "pg_loss": float(np.mean(pg_losses)),
                    "vf_loss": float(np.mean(vf_losses)),
                    "entropy": float(np.mean(entropies)),
                    "clip_fraction": float(np.mean(clip_fractions)),
                    "approx_kl": float(np.mean(approx_kls)),
                    "explained_variance": explained_var,
                }
                history.append(row)
                csv_writer.writerow(row)
                csv_file.flush()

                progress.set_postfix(
                    {
                        "ret": "nan" if np.isnan(mean_ep_return) else f"{mean_ep_return:.1f}",
                        "pg": f"{row['pg_loss']:.3f}",
                        "vf": f"{row['vf_loss']:.3f}",
                        "kl": f"{row['approx_kl']:.4f}",
                    }
                )

                if episode_returns and mean_ep_return > best_mean_return:
                    best_mean_return = mean_ep_return
                    with best_checkpoint_path.open("wb") as file:
                        pickle.dump(jax.device_get(train_state.params), file)
        finally:
            envs.close()

    params = jax.device_get(train_state.params)
    with final_checkpoint_path.open("wb") as file:
        pickle.dump(params, file)

    return PPOTrainingResult(
        config=config,
        history=history,
        checkpoint_path=final_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path if best_mean_return > -float("inf") else None,
        episode_returns=episode_returns,
        params=params,
    )


def result_to_dict(result: PPOTrainingResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["checkpoint_path"] = str(result.checkpoint_path)
    payload["best_checkpoint_path"] = (
        str(result.best_checkpoint_path) if result.best_checkpoint_path is not None else None
    )
    payload["config"]["output_dir"] = str(result.config.output_dir)
    return payload
