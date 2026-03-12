# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rl.cartpole.q_learning import fresh_params, run_config

TOTAL_STEPS = 200_000
MAX_WORKERS = 10

GRID = {
    'lr':         [5e-4, 1e-3, 2e-3],
    'decay_dur':  [15_000, 25_000, 40_000, 60_000],
    'hidden_dim': [8, 16, 32, 64],
    'num_layers': [1, 2],
}

_TOP_N = 10
_BOARD_HEIGHT = 4 + _TOP_N  # timing + progress + header + separator + entries

grid_results = []
_start_time: float = 0.0


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
    avg = elapsed / n
    eta = avg * (total - n)
    timing = (
        f"elapsed {_fmt_duration(elapsed)}"
        f"  avg {avg:.1f}s/cfg"
        f"  eta {_fmt_duration(eta)}"
    )
    hdr = (
        f"{'#':>3}  {'peak':>6}  {'lr':>6}  {'decay':>6}"
        f"  {'hdim':>4}  {'nlyr':>4}"
    )
    lines = [timing, f"Progress: {n}/{total}", hdr, "-" * len(hdr)]
    for i, (cfg, _, score) in enumerate(top):
        lines.append(
            f"  {i+1:>2}  {score:>6.1f}"
            f"  {cfg['lr']:.0e}  {cfg['decay_dur']//1000:>4}k"
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
        cfg, _, score = results[-1]
        best = max(r[2] for r in results)
        print(
            f"[{len(results):>3}/{total}]  score={score:>6.1f}  best={best:>6.1f}"
            f"  lr={cfg['lr']:.0e}  decay={cfg['decay_dur']//1000:>2}k"
            f"  hdim={cfg['hidden_dim']:>2}  nlyr={cfg['num_layers']}",
            flush=True,
        )
    _first_board = False


def _worker(cfg):
    """Runs in a separate process — each has its own JAX/XLA runtime."""
    ep_rets, _, __ = run_config(
        cfg, TOTAL_STEPS, fresh_params(cfg["hidden_dim"], cfg["num_layers"])
    )
    return cfg, ep_rets


def main():
    global grid_results, _start_time
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"Grid search: {len(configs)} configs, {TOTAL_STEPS:,} steps each\n", flush=True)
    _start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg, ep_rets = future.result()
            grid_results.append((cfg, ep_rets, peak_score(ep_rets)))
            _update_board(grid_results, len(configs))

    grid_results.sort(key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    main()
