# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from collections.abc import Sequence
from typing import Any, cast


def _to_list(values: object) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = cast(Any, values).tolist()
    if isinstance(values, Sequence):
        return [float(v) for v in values]
    return []


def plot_dqn_metrics(
    losses: object,
    episode_returns: object,
) -> None:
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    np = importlib.import_module("numpy")
    plotted = False

    def rolling_mean(data: list[float], window: int) -> tuple[object, object]:
        if len(data) < window:
            return np.array([]), np.array([])
        cumsum = np.cumsum(data)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        means = cumsum[window - 1 :] / window
        return np.arange(window - 1, len(data)), means

    losses_list = _to_list(losses)
    returns_list = _to_list(episode_returns)

    if not losses_list and not returns_list:
        print("No loss or return values recorded. Run the training cell first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if losses_list:
        ax1.scatter(
            range(len(losses_list)),
            losses_list,
            s=1,
            alpha=0.2,
            color="#ffe5a9",
        )
        xs, ys = rolling_mean(losses_list, window=500)
        if len(xs):
            ax1.plot(xs, ys, color="#2f4e97", linewidth=1.5)
        ax1.set_xlabel("Update step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(alpha=0.3)
        plotted = True

    if returns_list:
        ax2.scatter(
            range(len(returns_list)),
            returns_list,
            s=8,
            alpha=0.3,
            color="#ffe5a9",
        )
        xs, ys = rolling_mean(returns_list, window=50)
        if len(xs):
            ax2.plot(xs, ys, color="#2f4e97", linewidth=1.5)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Return")
        ax2.set_title("Episode Returns")
        ax2.grid(alpha=0.3)
        plotted = True

    if plotted:
        plt.tight_layout()
        plt.show()


def eval_dqn_max_score(
    params: object,
    num_eval_episodes: int = 100,
    batch_size: int = 20,
    seed: int = 123,
    show_progress: bool = True,
) -> tuple[int, int, float, float]:
    """Evaluate a DQN policy and report how often it reaches max score."""
    import importlib

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    np = importlib.import_module("numpy")
    tqdm = importlib.import_module("tqdm.auto").tqdm
    dqn = importlib.import_module("dqn")

    env = dqn.env
    env_params = dqn.env_params
    forward = dqn.forward

    @jax.jit
    def rollout_episode(p, key):
        obs, env_state = env.reset(key, env_params)
        done = jnp.bool_(False)
        score = jnp.float32(0.0)

        def step_fn(carry, _):
            obs, env_state, done, score, key = carry
            key, step_key = jax.random.split(key)

            q_values = forward(p, obs)
            action = jnp.argmax(q_values).astype(jnp.int32)
            next_obs, next_env_state, reward, step_done, _ = env.step(
                step_key, env_state, action, env_params
            )

            obs = jax.tree.map(lambda n, o: jnp.where(done, o, n), next_obs, obs)
            env_state = jax.tree.map(lambda n, o: jnp.where(done, o, n), next_env_state, env_state)
            score = score + jnp.where(done, 0.0, reward)
            done = jnp.logical_or(done, step_done)
            return (obs, env_state, done, score, key), None

        (_, _, _, score, _), _ = jax.lax.scan(
            step_fn,
            (obs, env_state, done, score, key),
            xs=None,
            length=env_params.max_steps_in_episode,
        )
        return score

    @jax.jit
    def eval_batch(p, keys):
        return jax.vmap(lambda k: rollout_episode(p, k))(keys)

    key = jax.random.key(seed)
    num_batches = (num_eval_episodes + batch_size - 1) // batch_size
    scores: list[float] = []
    batch_iter = range(num_batches)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc="Evaluating")

    for batch_idx in batch_iter:
        n = min(batch_size, num_eval_episodes - batch_idx * batch_size)
        key, seed_key = jax.random.split(key)
        keys = jax.random.split(seed_key, n)
        batch_scores = np.asarray(eval_batch(params, keys))
        scores.extend(batch_scores.tolist())

    scores_arr = np.asarray(scores, dtype=np.float32)
    max_score = float(env_params.max_steps_in_episode)
    max_count = int(np.sum(scores_arr >= max_score))
    max_pct = 100.0 * max_count / num_eval_episodes
    return max_count, num_eval_episodes, max_score, max_pct
