# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import numpy as np

from rl.pong.pixel_q_learning import build_init_carry, fresh_params, get_chunk_runner, run_config

TOTAL_STEPS = 500_000
N_CHUNKS = 10
CHUNK_STEPS = TOTAL_STEPS // N_CHUNKS
MAX_WORKERS = 10

GRID = {
    "lr": [1e-4, 5e-4, 1e-3, 2e-3],
    "decay_dur": [50_000, 100_000, 200_000, 300_000],
    "hidden_dim": [128, 256, 512],
    "num_layers": [1, 2],
}

_TOP_N = 10
_BOARD_HEIGHT = 5 + _TOP_N  # status + timing + progress + header + separator + entries

grid_results = []
_start_time: float = 0.0
_status: str = ""


def _fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def peak_score(ep_rets, w=50):
    if len(ep_rets) < w:
        return sum(ep_rets) / max(len(ep_rets), 1)
    cs = np.cumsum(ep_rets)
    cs[w:] = cs[w:] - cs[:-w]
    return float(np.max(cs[w - 1:]) / w)


def _board_lines(results, total):
    top = sorted(results, key=lambda x: x[2], reverse=True)[:_TOP_N]
    n = len(results)
    elapsed = time.monotonic() - _start_time
    if n > 0:
        avg = elapsed / n
        eta = avg * (total - n)
        timing = (
            f"elapsed {_fmt_duration(elapsed)}"
            f"  avg {avg:.1f}s/cfg"
            f"  eta {_fmt_duration(eta)}"
        )
    else:
        timing = f"elapsed {_fmt_duration(elapsed)}"
    hdr = (
        f"{'#':>3}  {'peak':>6}  {'lr':>6}  {'decay':>6}"
        f"  {'hdim':>4}  {'nlyr':>4}"
    )
    lines = [_status, timing, f"Progress: {n}/{total}", hdr, "-" * len(hdr)]
    for i, (cfg, _, score) in enumerate(top):
        lines.append(
            f"  {i + 1:>2}  {score:>6.1f}"
            f"  {cfg['lr']:.0e}  {cfg['decay_dur'] // 1000:>4}k"
            f"  {cfg['hidden_dim']:>4}  {cfg['num_layers']:>4}"
        )
    while len(lines) < _BOARD_HEIGHT:
        lines.append("")
    return lines


_first_board = True


def _update_board(results, total):
    global _first_board
    lines = _board_lines(results, total)
    if sys.stdout.isatty():
        if not _first_board:
            sys.stdout.write(f"\033[{_BOARD_HEIGHT}A")
        for line in lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()
    else:
        if _status:
            print(_status, flush=True)
        if results:
            cfg, _, score = results[-1]
            best = max(r[2] for r in results)
            print(
                f"[{len(results):>3}/{total}]  score={score:>6.1f}  best={best:>6.1f}"
                f"  lr={cfg['lr']:.0e}  decay={cfg['decay_dur'] // 1000:>3}k"
                f"  hdim={cfg['hidden_dim']:>3}  nlyr={cfg['num_layers']}",
                flush=True,
            )
    _first_board = False


def _run_vmapped_group(group_configs, batch_size, total_configs):
    """Run same-architecture configs in parallel via jax.vmap, in chunks.

    Runs each vmap batch in N_CHUNKS slices so the progress bar can update
    between slices without blocking the entire training run.
    """
    global _status
    hd = group_configs[0]["hidden_dim"]
    nl = group_configs[0]["num_layers"]
    chunk_runner = get_chunk_runner(CHUNK_STEPS, hd, nl)
    vmapped_chunk = jax.vmap(chunk_runner)
    n_batches = (len(group_configs) + batch_size - 1) // batch_size

    results = []
    for b_idx, batch_start in enumerate(range(0, len(group_configs), batch_size)):
        batch = group_configs[batch_start : batch_start + batch_size]
        keys = jax.random.split(jax.random.key(batch_start), len(batch))
        carries = [
            build_init_carry(cfg, TOTAL_STEPS, fresh_params(hd, nl), keys[j])
            for j, cfg in enumerate(batch)
        ]
        batched_carry = jax.tree.map(lambda *xs: jnp.stack(xs), *carries)

        for chunk_idx in range(N_CHUNKS):
            steps_done = chunk_idx * CHUNK_STEPS
            bar_n = chunk_idx * 20 // N_CHUNKS
            bar = "█" * bar_n + "░" * (20 - bar_n)
            _status = (
                f"hd={hd} nl={nl}  "
                f"batch {b_idx + 1}/{n_batches}  "
                f"[{bar}] {steps_done // 1000:>3}k / {TOTAL_STEPS // 1000}k steps"
            )
            _update_board(grid_results, total_configs)
            batched_carry = vmapped_chunk(batched_carry)
            jax.block_until_ready(batched_carry)
            batched_carry = {
                **batched_carry,
                "step_offset": batched_carry["step_offset"] + jnp.int32(CHUNK_STEPS),
            }

        _status = ""
        for j, cfg in enumerate(batch):
            ep_count = int(batched_carry["ep_count"][j])
            ep_rets = np.asarray(batched_carry["ep_rets"][j])[:ep_count].tolist()
            results.append((cfg, ep_rets))
    return results


def _worker(cfg):
    """Runs in a separate process — each has its own JAX/XLA runtime."""
    ep_rets, _, __ = run_config(
        cfg, TOTAL_STEPS, fresh_params(cfg["hidden_dim"], cfg["num_layers"])
    )
    return cfg, ep_rets


def main():
    global grid_results, _start_time
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", action="store_true",
        help="Run in-process (no multiprocessing) for GPU use",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Number of same-architecture configs to vmap together (--gpu only)",
    )
    args = parser.parse_args()

    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"Grid search: {len(configs)} configs, {TOTAL_STEPS:,} steps each\n", flush=True)
    _start_time = time.monotonic()

    if args.gpu and args.batch > 1:
        # Group by architecture, then vmap each group in chunks of --batch.
        arch_groups: dict = {}
        for cfg in configs:
            arch_groups.setdefault((cfg["hidden_dim"], cfg["num_layers"]), []).append(cfg)
        for group_cfgs in arch_groups.values():
            for cfg, ep_rets in _run_vmapped_group(group_cfgs, args.batch, len(configs)):
                grid_results.append((cfg, ep_rets, peak_score(ep_rets)))
                _update_board(grid_results, len(configs))
    elif args.gpu:
        for cfg in configs:
            cfg_out, ep_rets = _worker(cfg)
            grid_results.append((cfg_out, ep_rets, peak_score(ep_rets)))
            _update_board(grid_results, len(configs))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_worker, cfg): cfg for cfg in configs}
            for future in as_completed(futures):
                cfg, ep_rets = future.result()
                grid_results.append((cfg, ep_rets, peak_score(ep_rets)))
                _update_board(grid_results, len(configs))

    grid_results.sort(key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    main()
