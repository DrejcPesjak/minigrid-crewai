#!/usr/bin/env python3
# plot_sr_ensemble.py
# Aggregate multiple trials and plot mean SR vs episodes with bands.
# Adds tail handling to avoid cutting at the shortest trial.

# python3 plot_sr_ensemble.py --storage storage --match PickupLoc --tail-mode fill --tail-fill 0.99 --band std --thin 2 --show
# python3 plot_sr_ensemble.py --tail-mode cut --out storage/PickupLoc-ensemble-cut.png

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

VAL_RE = re.compile(r'^\[VAL[^\]]*\].*?SR=([0-9.]+).*?total_eps=(\d+)', re.ASCII)

def parse_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    eps, srs = [], []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                sr = float(m.group(1))
                te = int(m.group(2))
                eps.append(te)
                srs.append(sr)
    if not eps:
        raise RuntimeError(f"No validation lines found in {path}.")
    # sort + dedup (keep last SR per eps)
    pairs = {}
    for e, s in zip(eps, srs):
        pairs[e] = s
    eps_sorted = np.array(sorted(pairs.keys()), dtype=int)
    srs_sorted = np.array([pairs[e] for e in eps_sorted], dtype=float)
    return eps_sorted, srs_sorted

def find_trials(storage_dir: Path, name_substring: str) -> Dict[str, Path]:
    trials = {}
    for p in storage_dir.iterdir():
        if p.is_dir() and name_substring+'-' in p.name:
            logp = p / "log.txt"
            if logp.exists():
                trials[p.name] = logp
    if not trials:
        raise RuntimeError(f"No trials found in {storage_dir} containing '{name_substring}'.")
    return trials

def build_grid(trials_data: List[Tuple[np.ndarray, np.ndarray]], mode: str) -> np.ndarray:
    """mode: 'cut' -> [.. min(max_eps)], 'hold'/'fill' -> [.. max(max_eps)]"""
    max_eps_each = [eps[-1] for eps, _ in trials_data]
    if mode == "cut":
        x_max = int(min(max_eps_each))
    else:
        x_max = int(max(max_eps_each))
    union = np.unique(np.concatenate([eps[eps <= x_max] for eps, _ in trials_data]))
    # Make sure grid reaches x_max (even if no trial has a point exactly at x_max)
    if union[-1] < x_max:
        union = np.concatenate([union, [x_max]])
    return union

def interpolate_to_grid(eps: np.ndarray, srs: np.ndarray, grid: np.ndarray,
                        tail_mode: str, tail_fill: float) -> np.ndarray:
    # Need at least two points for interpolation; if only one, broadcast appropriately
    if len(eps) == 1:
        y = np.full_like(grid, srs[0], dtype=float)
        if tail_mode == "fill":
            # still respect the fill to the right of the single point
            mask_right = grid > eps[0]
            y[mask_right] = tail_fill
        return y

    # Base interpolation (linear) in the observed range
    # For left extrapolation, hold first; right depends on tail_mode
    right_val = srs[-1] if tail_mode == "hold" else (tail_fill if tail_mode == "fill" else srs[-1])
    y = np.interp(grid, xp=eps, fp=srs, left=srs[0], right=right_val)

    if tail_mode == "cut":
        # For 'cut', we won't even build grid beyond min max; nothing to do.
        pass
    elif tail_mode == "fill":
        # Force exact fill beyond last observed episode for this trial
        last_ep = eps[-1]
        y[grid > last_ep] = tail_fill
    else:
        # 'hold' already handled by right=last
        pass

    return y

def main():
    ap = argparse.ArgumentParser(description="Average SR vs episodes across trials and plot with bands.")
    ap.add_argument("--storage", type=Path, default=Path("storage"), help="Root dir containing trial folders.")
    ap.add_argument("--match", type=str, default="PickupLoc", help="Substring to match trial folder names.")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save PNG.")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively.")
    ap.add_argument("--band", choices=["ci", "std"], default="ci",
                    help="Band type: 95% CI over trials (ci) or ±1 SD (std).")
    ap.add_argument("--thin", type=int, default=1, help="Keep every k-th grid point for plotting.")
    ap.add_argument("--tail-mode", choices=["cut", "hold", "fill"], default="fill",
                    help="How to handle trials that finish earlier than others.")
    ap.add_argument("--tail-fill", type=float, default=0.99,
                    help="SR value to append after a trial finishes (used when --tail-mode=fill).")
    args = ap.parse_args()

    trials = find_trials(args.storage, args.match)

    trials_data = []
    bad = []
    for name, logp in trials.items():
        try:
            eps, srs = parse_log(logp)
            trials_data.append((eps, srs))
        except Exception as e:
            bad.append((name, str(e)))

    if not trials_data:
        raise RuntimeError("No valid trials to plot.\n" + "\n".join(f"{n}: {err}" for n, err in bad))

    grid = build_grid(trials_data, mode=args.tail_mode)
    if len(grid) == 0:
        raise RuntimeError("Grid is empty. Check logs.")

    # Interpolate each trial to the grid with the chosen tail behavior
    Ys = []
    for eps, srs in trials_data:
        Ys.append(interpolate_to_grid(eps, srs, grid, args.tail_mode, args.tail_fill))
    Y = np.vstack(Ys)  # (n_trials, n_points)

    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    n = Y.shape[0]
    if args.band == "ci" and n > 1:
        sem = std / np.sqrt(n)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
        band_label = "95% CI"
    else:
        lo = mean - std
        hi = mean + std
        band_label = "±1 SD"

    grid_plot = grid[::args.thin]
    mean_plot = mean[::args.thin]
    lo_plot = lo[::args.thin]
    hi_plot = hi[::args.thin]

    # plt.figure()
    # plt.plot(grid_plot, mean_plot, linewidth=2, label=f"Mean SR (n={n})")
    # plt.fill_between(grid_plot, lo_plot, hi_plot, alpha=0.25, label=band_label)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.figsize": (14, 8),
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "lines.linewidth": 2.5,
    })
    plt.figure(figsize=(14, 8))
    plt.plot(grid_plot, mean_plot, linewidth=2.5, label=f"Mean SR (n={n})")
    plt.fill_between(grid_plot, lo_plot, hi_plot, alpha=0.2, label=band_label)
    plt.xlabel("Episodes")
    plt.ylabel("Validation Success Rate (SR)")
    plt.title(f"Mean Validation SR vs Episodes — {args.match}")
    # plt.grid(True, linestyle='--', alpha=0.4)
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='lower right', fontsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.out}")

    if args.show or not args.out:
        plt.show()

    print(f"Trials used: {n}")
    print(f"Episode range: {int(grid[0])} .. {int(grid[-1])}")
    print(f"Mean SR at end: {mean[-1]:.4f}")
    if bad:
        print("Skipped trials due to errors:")
        for nme, err in bad:
            print(f"  - {nme}: {err}")

if __name__ == "__main__":
    main()
