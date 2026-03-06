# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import jax
import gymnax
import optax
import jax.numpy as jnp

from policygradient import init_mlp_params, forward_mlp, discounted_returns_to_go


OBS_DIM = 3      # [cos(θ), sin(θ), dθ/dt]
ACTION_DIM = 1
ACTION_LIMIT = 2.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


def _gaussian_log_prob(u, mean, log_std):
    std = jnp.exp(log_std)
    return -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)


def _policy_forward(params, obs):
    """Tanh-squashed MLP output scaled to [-ACTION_LIMIT, ACTION_LIMIT]."""
    return jax.nn.tanh(forward_mlp(params['layers'], obs, jax.nn.tanh)) * ACTION_LIMIT


def _value_forward(value_layers, obs):
    """Scalar value estimate V(s)."""
    return forward_mlp(value_layers, obs, jax.nn.tanh).squeeze()


def init_policy_params(key, sizes):
    """MLP layers + a learnable log_std for the Gaussian policy."""
    return {'layers': init_mlp_params(key, sizes), 'log_std': jnp.full((ACTION_DIM,), -0.5)}


def get_action(key, params, obs):
    """Sample a clipped Gaussian action from the policy."""
    mean = _policy_forward(params, obs)
    log_std = jnp.clip(params['log_std'], LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=(ACTION_DIM,))
    return jnp.clip(mean + std * noise, -ACTION_LIMIT, ACTION_LIMIT)


def sample_rollout(key, env, env_params, params, max_steps):
    """Sample a single rollout using the policy."""
    key_reset, key_episode = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)

    def policy_step(carry, _):
        key, state, obs, policy_params = carry
        key, key_step, key_net = jax.random.split(key, 3)
        action = get_action(key_net, policy_params, obs)
        next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)
        new_carry = [key, next_state, next_obs, policy_params]
        return new_carry, [obs, action, reward, done]

    carry, scan_out = jax.lax.scan(
        policy_step, [key_episode, env_state, obs, params], (), max_steps
    )
    obs_seq, action, reward, done = scan_out
    final_obs = carry[2]

    first_done = jnp.argmax(done)
    has_done = jnp.any(done)
    actual_length = jnp.where(has_done, first_done + 1, max_steps)
    obs_padded = jnp.concatenate([obs_seq, final_obs[None]], axis=0)

    return {
        'observations': obs_padded,
        'rewards': reward,
        'actions': action,
        'done': done,
        'length': actual_length
    }


def generate_n_rollouts(key, env, env_params, params, n_rollouts, max_steps):
    """Generate n rollouts using vmap for parallelization."""
    keys = jax.random.split(key, n_rollouts)
    return jax.vmap(
        sample_rollout, in_axes=(0, None, None, None, None)
    )(keys, env, env_params, params, max_steps)


def log_probs_per_step(params, observations, actions):
    """Compute log pi(a_t | s_t) for each step t. Returns (max_steps,)."""
    obs = observations[:-1]
    log_std = jnp.clip(params['log_std'], LOG_STD_MIN, LOG_STD_MAX)

    def per_step(o, a):
        mean = _policy_forward(params, o)
        return jnp.sum(_gaussian_log_prob(a, mean, log_std))

    return jax.vmap(per_step)(obs, actions)


def _returns_to_go(rollouts, gamma):
    """Discounted rewards-to-go G_t for all trajectories. Returns (m, max_steps)."""
    max_steps = rollouts['rewards'].shape[1]
    mask = jnp.arange(max_steps)[None, :] < rollouts['length'][:, None]
    masked_rewards = rollouts['rewards'] * mask
    return jax.vmap(discounted_returns_to_go, in_axes=(0, None))(masked_rewards, gamma)


def reinforce_loss(policy_params, value_params, rollouts, gamma):
    """REINFORCE with MC value baseline: advantage = G_t - V_φ(s_t)."""
    max_steps = rollouts['rewards'].shape[1]
    mask = jnp.arange(max_steps)[None, :] < rollouts['length'][:, None]

    returns_to_go = _returns_to_go(rollouts, gamma)

    obs = rollouts['observations'][:, :-1, :]
    values = jax.lax.stop_gradient(
        jax.vmap(jax.vmap(_value_forward, in_axes=(None, 0)), in_axes=(None, 0))(value_params, obs)
    )

    advantages = returns_to_go - values
    n_valid = jnp.sum(mask)
    mean_A = jnp.sum(advantages * mask) / n_valid
    var_A = jnp.sum((advantages - mean_A) ** 2 * mask) / n_valid
    advantages = (advantages - mean_A) / (jnp.sqrt(var_A) + 1e-8)

    log_probs = jax.vmap(log_probs_per_step, in_axes=(None, 0, 0))(
        policy_params, rollouts['observations'], rollouts['actions']
    )

    return -jnp.mean(jnp.sum(advantages * log_probs * mask, axis=1))


def value_loss(value_params, rollouts, gamma):
    """MSE: 1/(m·T) Σ_i Σ_t (V_φ(s_t^i) - G_t^i)² over valid steps."""
    max_steps = rollouts['rewards'].shape[1]
    mask = jnp.arange(max_steps)[None, :] < rollouts['length'][:, None]

    returns_to_go = _returns_to_go(rollouts, gamma)

    obs = rollouts['observations'][:, :-1, :]
    values = jax.vmap(jax.vmap(_value_forward, in_axes=(None, 0)), in_axes=(None, 0))(
        value_params, obs
    )

    return jnp.sum((values - returns_to_go) ** 2 * mask) / jnp.sum(mask)


def make_value_training_step(n_rollouts, max_steps, lr_policy, value_lr_mult, gamma):

    env, env_params = gymnax.make("Pendulum-v1")
    opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-1.0)
    )
    vp_lr = lr_policy * value_lr_mult

    def make_carry(seed):
        pp = init_policy_params(
            jax.random.PRNGKey(seed * 3), [OBS_DIM, 32, 32, ACTION_DIM]
        )
        vp = init_mlp_params(
            jax.random.PRNGKey(seed * 3 + 1), [OBS_DIM, 32, 32, 1]
        )
        return (pp, vp, opt.init(pp), opt.init(vp), jnp.float32(1.0))

    def training_step(carry, key):
        pp, vp, pos, vos, rew_std = carry
        rollouts = generate_n_rollouts(
            key, env, env_params, pp, n_rollouts, max_steps
        )
        new_rew_std = (
            0.99 * rew_std + 0.01 * (jnp.std(rollouts["rewards"]) + 1e-8)
        )
        scaled = {**rollouts, "rewards": rollouts["rewards"] / new_rew_std}
        loss, pg = jax.value_and_grad(reinforce_loss, argnums=0)(
            pp, vp, scaled, gamma
        )
        pu, new_pos = opt.update(pg, pos)
        new_pp = optax.apply_updates(
            pp, jax.tree.map(lambda u: lr_policy * u, pu)
        )
        _, vg = jax.value_and_grad(value_loss)(vp, scaled, gamma)
        vu, new_vos = opt.update(vg, vos)
        new_vp = optax.apply_updates(
            vp, jax.tree.map(lambda u: vp_lr * u, vu)
        )
        mask = jnp.arange(max_steps)[None, :] < rollouts["length"][:, None]
        mean_ret = jnp.mean(jnp.sum(rollouts["rewards"] * mask, axis=1))
        return (new_pp, new_vp, new_pos, new_vos, new_rew_std), (loss, mean_ret)

    return make_carry, training_step


def plot_training(losses, returns, title="Pendulum training"):
    import matplotlib.pyplot as plt

    x = range(len(losses))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, losses)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[1].plot(x, returns)
    axes[1].set_title("Mean return")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Return")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
