import jax
import jax.numpy as jnp


def init_mlp_params(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (din, dout) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params.append({'w': w, 'b': b})
    return params


def forward_mlp(layers, x, activation):
    for layer in layers[:-1]:
        x = activation(jnp.dot(x, layer['w']) + layer['b'])
    final = layers[-1]
    return jnp.dot(x, final['w']) + final['b']


def calculate_returns(rollouts):
    max_steps = rollouts['rewards'].shape[1]
    mask = jnp.arange(max_steps)[None, :] < rollouts['length'][:, None]
    return jnp.sum(rollouts['rewards'] * mask, axis=1)


def discounted_returns_to_go(rewards, gamma):
    """G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... via reverse scan."""
    def scan_fn(future_G, r):
        G = r + gamma * future_G
        return G, G
    _, returns = jax.lax.scan(scan_fn, jnp.float32(0.0), rewards, reverse=True)
    return returns
