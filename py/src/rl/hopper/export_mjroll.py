# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export trained Hopper PPO rollouts to the MuJoCo web viewer format."""

from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .config import PPOConfig, default_checkpoint_name
from .networks import ActorCritic

MAGIC = b"MJRV"
VERSION = 2
DEFAULT_FPS = 60
LANE_COLORS = [
    np.array([0.4, 0.6, 1.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.4, 0.3, 1.0], dtype=np.float32),
    np.array([0.3, 0.9, 0.4, 1.0], dtype=np.float32),
    np.array([1.0, 0.8, 0.2, 1.0], dtype=np.float32),
    np.array([0.8, 0.4, 1.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.6, 0.2, 1.0], dtype=np.float32),
]

GEOM_TYPE_IDS: dict[int, int] = {
    mujoco.mjtGeom.mjGEOM_PLANE: 0,
    mujoco.mjtGeom.mjGEOM_SPHERE: 1,
    mujoco.mjtGeom.mjGEOM_CAPSULE: 2,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: 3,
    mujoco.mjtGeom.mjGEOM_CYLINDER: 4,
    mujoco.mjtGeom.mjGEOM_BOX: 5,
}


def parse_hidden_dims(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def mat2quat(mat_flat: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a flat 3x3 rotation matrix to a quaternion in wxyz order."""
    m = mat_flat.reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (m[2, 1] - m[1, 2]) * scale
        y = (m[0, 2] - m[2, 0]) * scale
        z = (m[1, 0] - m[0, 1]) * scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale

    return (float(w), float(x), float(y), float(z))


def mat2angle_y(mat_flat: np.ndarray) -> float:
    """Extract the rotation angle around the Y axis from a flat 3x3 matrix."""
    m = mat_flat.reshape(3, 3)
    return float(np.arctan2(-m[2, 0], m[0, 0]))


def _collect_geom_info(
    mj_model: mujoco.MjModel,
) -> tuple[list[int], list[int], list[np.ndarray], list[np.ndarray], list[int]]:
    geom_indices: list[int] = []
    geom_type_ids: list[int] = []
    geom_sizes: list[np.ndarray] = []
    geom_rgbas: list[np.ndarray] = []
    plane_indices: list[int] = []

    for index in range(mj_model.ngeom):
        type_id = GEOM_TYPE_IDS.get(int(mj_model.geom_type[index]))
        if type_id is None:
            continue
        if mj_model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE:
            plane_indices.append(len(geom_indices))
        geom_indices.append(index)
        geom_type_ids.append(type_id)
        geom_sizes.append(mj_model.geom_size[index].copy())
        rgba = mj_model.geom_rgba[index]
        geom_rgbas.append(rgba.astype(np.float32))

    return geom_indices, geom_type_ids, geom_sizes, geom_rgbas, plane_indices


def _record_frame(
    mj_data: mujoco.MjData,
    geom_indices: list[int],
    ngeoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_frame = np.empty((ngeoms, 3), dtype=np.float32)
    quat_frame = np.empty((ngeoms, 4), dtype=np.float32)
    for geom_index, model_index in enumerate(geom_indices):
        pos_frame[geom_index] = mj_data.geom_xpos[model_index]
        quat_frame[geom_index] = mat2quat(mj_data.geom_xmat[model_index])
    return pos_frame, quat_frame


def _make_actor_critic(
    act_dim: int,
    policy_hidden_dims: tuple[int, ...],
    value_hidden_dims: tuple[int, ...],
    activation: str,
    log_std_init: float,
    ortho_init: bool,
) -> ActorCritic:
    return ActorCritic(
        act_dim=act_dim,
        policy_hidden_dims=policy_hidden_dims,
        value_hidden_dims=value_hidden_dims,
        activation=activation,
        log_std_init=log_std_init,
        ortho_init=ortho_init,
    )


def _make_action_fns(model: ActorCritic):
    @jax.jit
    def get_action_det(params: dict, obs: jax.Array) -> jax.Array:
        mean, _, _ = model.apply(params, obs)
        return mean

    @jax.jit
    def get_action_stochastic(params: dict, obs: jax.Array, rng: jax.Array) -> jax.Array:
        mean, log_std, _ = model.apply(params, obs)
        noise = jax.random.normal(rng, shape=mean.shape)
        return mean + jnp.exp(log_std) * noise

    return get_action_det, get_action_stochastic


def export_rollout(
    env_id: str,
    checkpoint: Path,
    out_path: Path,
    episodes: int,
    deterministic: bool,
    policy_hidden_dims: tuple[int, ...],
    value_hidden_dims: tuple[int, ...],
    activation: str,
    log_std_init: float,
    ortho_init: bool,
    seed: int = 0,
) -> None:
    with checkpoint.open("rb") as file:
        params = pickle.load(file)

    env = gym.make(env_id)
    act_dim = int(env.action_space.shape[0])
    model = _make_actor_critic(
        act_dim,
        policy_hidden_dims,
        value_hidden_dims,
        activation,
        log_std_init,
        ortho_init,
    )
    get_action_det, get_action_stochastic = _make_action_fns(model)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    geom_indices, geom_type_ids, geom_sizes, geom_rgbas, _ = _collect_geom_info(mj_model)
    ngeoms = len(geom_indices)
    steps_per_frame = max(1, round(1.0 / (DEFAULT_FPS * env.unwrapped.dt)))

    all_pos: list[np.ndarray] = []
    all_quat: list[np.ndarray] = []
    rng = jax.random.PRNGKey(seed)

    for episode_index in range(episodes):
        obs, _ = env.reset(seed=seed + episode_index)
        step = 0

        while True:
            if step % steps_per_frame == 0:
                pos_frame, quat_frame = _record_frame(mj_data, geom_indices, ngeoms)
                all_pos.append(pos_frame)
                all_quat.append(quat_frame)

            obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
            if deterministic:
                action = get_action_det(params, obs_jnp)
            else:
                rng, action_rng = jax.random.split(rng)
                action = get_action_stochastic(params, obs_jnp, action_rng)

            obs, _, terminated, truncated, _ = env.step(np.asarray(action))
            step += 1
            if terminated or truncated:
                break

    env.close()

    frame_count = len(all_pos)
    pos_buf = np.stack(all_pos)
    quat_buf = np.stack(all_quat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<IIII", 1, DEFAULT_FPS, ngeoms, frame_count))

        for index in range(ngeoms):
            file.write(struct.pack("<B3x", geom_type_ids[index]))
            file.write(struct.pack("<3f", *geom_sizes[index]))
            file.write(struct.pack("<4f", *geom_rgbas[index]))

        for frame_index in range(frame_count):
            file.write(pos_buf[frame_index].tobytes())
            file.write(quat_buf[frame_index].tobytes())


def export_race_rollout(
    env_id: str,
    checkpoints: list[Path],
    out_path: Path,
    deterministic: bool,
    policy_hidden_dims: tuple[int, ...],
    value_hidden_dims: tuple[int, ...],
    activation: str,
    log_std_init: float,
    ortho_init: bool,
    max_steps: int = 1000,
    seed: int = 0,
) -> None:
    all_params: list[dict] = []
    for checkpoint in checkpoints:
        with checkpoint.open("rb") as file:
            all_params.append(pickle.load(file))

    envs = [gym.make(env_id) for _ in checkpoints]
    act_dim = int(envs[0].action_space.shape[0])
    model = _make_actor_critic(
        act_dim,
        policy_hidden_dims,
        value_hidden_dims,
        activation,
        log_std_init,
        ortho_init,
    )
    get_action_det, get_action_stochastic = _make_action_fns(model)

    mj_model = envs[0].unwrapped.model
    geom_indices, geom_type_ids, geom_sizes, geom_rgbas, plane_indices = _collect_geom_info(mj_model)
    body_mask = [index not in plane_indices for index in range(len(geom_indices))]
    n_body_geoms = sum(body_mask)
    n_planes = len(plane_indices)
    total_geoms = n_planes + n_body_geoms * len(checkpoints)

    combined_type_ids: list[int] = []
    combined_sizes: list[np.ndarray] = []
    combined_rgbas: list[np.ndarray] = []

    for plane_index in plane_indices:
        combined_type_ids.append(geom_type_ids[plane_index])
        combined_sizes.append(geom_sizes[plane_index])
        combined_rgbas.append(geom_rgbas[plane_index])

    for agent_index in range(len(checkpoints)):
        color = LANE_COLORS[agent_index % len(LANE_COLORS)]
        for geom_index, is_body in enumerate(body_mask):
            if not is_body:
                continue
            combined_type_ids.append(geom_type_ids[geom_index])
            combined_sizes.append(geom_sizes[geom_index])
            combined_rgbas.append(color.copy())

    steps_per_frame = max(1, round(1.0 / (DEFAULT_FPS * envs[0].unwrapped.dt)))
    observations = [env.reset(seed=seed)[0] for env in envs]
    alive = [True] * len(checkpoints)
    last_body_pos2d: list[np.ndarray | None] = [None] * len(checkpoints)
    last_body_angle: list[np.ndarray | None] = [None] * len(checkpoints)
    all_pos2d_frames: list[np.ndarray] = []
    all_angle_frames: list[np.ndarray] = []
    rng = jax.random.PRNGKey(seed)

    def record_planar_frame() -> None:
        pos2d = np.zeros((total_geoms, 2), dtype=np.float32)
        angle = np.zeros(total_geoms, dtype=np.float32)

        for output_index, plane_index in enumerate(plane_indices):
            model_index = geom_indices[plane_index]
            xpos = envs[0].unwrapped.data.geom_xpos[model_index]
            pos2d[output_index] = [xpos[0], xpos[2]]
            angle[output_index] = mat2angle_y(envs[0].unwrapped.data.geom_xmat[model_index])

        for agent_index in range(len(checkpoints)):
            offset = n_planes + agent_index * n_body_geoms
            if alive[agent_index]:
                geom_positions = np.empty((n_body_geoms, 2), dtype=np.float32)
                geom_angles = np.empty(n_body_geoms, dtype=np.float32)
                body_index = 0
                for geom_index, model_index in enumerate(geom_indices):
                    if not body_mask[geom_index]:
                        continue
                    xpos = envs[agent_index].unwrapped.data.geom_xpos[model_index]
                    geom_positions[body_index] = [xpos[0], xpos[2]]
                    geom_angles[body_index] = mat2angle_y(
                        envs[agent_index].unwrapped.data.geom_xmat[model_index]
                    )
                    body_index += 1
                last_body_pos2d[agent_index] = geom_positions
                last_body_angle[agent_index] = geom_angles

            if last_body_pos2d[agent_index] is None or last_body_angle[agent_index] is None:
                continue

            pos2d[offset : offset + n_body_geoms] = last_body_pos2d[agent_index]
            angle[offset : offset + n_body_geoms] = last_body_angle[agent_index]

        all_pos2d_frames.append(pos2d)
        all_angle_frames.append(angle)

    for step in range(max_steps):
        if step % steps_per_frame == 0:
            record_planar_frame()

        any_alive = False
        for agent_index, params in enumerate(all_params):
            if not alive[agent_index]:
                continue

            obs_jnp = jnp.asarray(observations[agent_index], dtype=jnp.float32)
            if deterministic:
                action = get_action_det(params, obs_jnp)
            else:
                rng, action_rng = jax.random.split(rng)
                action = get_action_stochastic(params, obs_jnp, action_rng)

            obs, _, terminated, truncated, _ = envs[agent_index].step(np.asarray(action))
            observations[agent_index] = obs
            if terminated or truncated:
                alive[agent_index] = False
            else:
                any_alive = True

        if not any_alive:
            if step % steps_per_frame != 0:
                record_planar_frame()
            break

    for env in envs:
        env.close()

    frame_count = len(all_pos2d_frames)
    pos2d_buf = np.stack(all_pos2d_frames)
    angle_buf = np.stack(all_angle_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        file.write(MAGIC)
        file.write(
            struct.pack(
                "<IIIIIII",
                VERSION,
                DEFAULT_FPS,
                total_geoms,
                frame_count,
                len(checkpoints),
                0,
                1,
            )
        )

        for index in range(total_geoms):
            file.write(struct.pack("<B3x", combined_type_ids[index]))
            file.write(struct.pack("<3f", *combined_sizes[index]))
            file.write(struct.pack("<4f", *combined_rgbas[index]))

        for frame_index in range(frame_count):
            file.write(pos2d_buf[frame_index].tobytes())
            file.write(angle_buf[frame_index].tobytes())


def build_arg_parser() -> argparse.ArgumentParser:
    default_config = PPOConfig()
    parser = argparse.ArgumentParser(description="Export Hopper PPO rollouts to .mjroll")
    parser.add_argument("--env-id", default=default_config.env_id)
    parser.add_argument("--checkpoint", type=Path, default=Path(default_checkpoint_name(default_config.env_id)))
    parser.add_argument("--out", type=Path, default=Path("hopper_rollout.mjroll"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--activation", choices=("tanh", "relu"), default=default_config.activation)
    parser.add_argument("--log-std-init", type=float, default=default_config.log_std_init)
    parser.add_argument(
        "--policy-hidden-dims",
        type=parse_hidden_dims,
        default=default_config.policy_hidden_dims,
    )
    parser.add_argument(
        "--value-hidden-dims",
        type=parse_hidden_dims,
        default=default_config.value_hidden_dims,
    )
    parser.add_argument("--ortho-init", action="store_true", default=default_config.ortho_init)
    parser.add_argument("--race", nargs="+", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.race:
        export_race_rollout(
            env_id=args.env_id,
            checkpoints=args.race,
            out_path=args.out,
            deterministic=not args.stochastic,
            policy_hidden_dims=args.policy_hidden_dims,
            value_hidden_dims=args.value_hidden_dims,
            activation=args.activation,
            log_std_init=args.log_std_init,
            ortho_init=args.ortho_init,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return

    export_rollout(
        env_id=args.env_id,
        checkpoint=args.checkpoint,
        out_path=args.out,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        policy_hidden_dims=args.policy_hidden_dims,
        value_hidden_dims=args.value_hidden_dims,
        activation=args.activation,
        log_std_init=args.log_std_init,
        ortho_init=args.ortho_init,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
