#!/usr/bin/env python3
# plot_sr_vs_eps.py
# Parse [VAL*] lines from a BabyAI training log and plot SR vs total_eps,
# with optional smoothing (moving average or EWMA).

#python3 plot_sr_vs_eps.py storage/PickupLoc-v0/log.txt --ma 101 --show

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

VAL_RE = re.compile(r'^\[VAL[^\]]*\].*?SR=([0-9.]+).*?total_eps=(\d+)', re.ASCII)

def parse_log(path: Path):
    eps, srs = [], []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                sr = float(m.group(1))
                te = int(m.group(2))
                eps.append(te)
                srs.append(sr)
    zipped = sorted(zip(eps, srs), key=lambda x: x[0])
    if not zipped:
        raise RuntimeError("No validation lines found. Expected lines like: "
                           "[VAL512] SR=0.1582 ... total_eps=710")
    eps, srs = zip(*zipped)
    return list(eps), list(srs)

def moving_average(y, window):
    if window <= 1 or window > len(y):
        return np.array(y)
    # pad with edge values so length stays the same
    pad = window // 2
    ypad = np.pad(y, (pad, pad - (1 - window % 2)), mode='edge')
    kernel = np.ones(window) / window
    ysm = np.convolve(ypad, kernel, mode='valid')
    return ysm

def ewma(y, alpha):
    if alpha is None or not (0.0 < alpha < 1.0):
        return np.array(y)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def main():
    ap = argparse.ArgumentParser(description="Plot validation SR vs total_eps from log.")
    ap.add_argument("log_path", type=Path, help="Path to log.txt")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save PNG")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively")
    ap.add_argument("--ma", type=int, default=21, help="Moving average window (points). Set to 1 to disable.")
    ap.add_argument("--ewm", type=float, default=None, help="EWMA alpha in (0,1). If set, overrides --ma.")
    args = ap.parse_args()

    eps, srs = parse_log(args.log_path)
    # eps = eps[::5]
    # srs = srs[::5]
    srs_np = np.asarray(srs, dtype=float)

    # Choose smoothing
    if args.ewm is not None:
        srs_smooth = ewma(srs_np, args.ewm)
        smooth_label = f"EWMA (α={args.ewm})"
    else:
        srs_smooth = moving_average(srs_np, args.ma)
        smooth_label = f"MA (window={args.ma})" if args.ma > 1 else None

    plt.figure()
    # raw line
    plt.plot(eps, srs, marker='.', linewidth=1, label="SR (raw)")
    # smoothed line
    if smooth_label:
        plt.plot(eps, srs_smooth, linewidth=2.2, label=smooth_label)

    plt.xlabel("Episodes")
    plt.ylabel("Validation Success Rate (SR)")
    try:
        lvl_name = str(args.log_path).split('/')[1].split('-')[0]
    except Exception:
        lvl_name = str(args.log_path)
    plt.title(f"Success Rate vs Episodes — {lvl_name}")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.out}")

    if args.show or not args.out:
        plt.show()

    print(f"Parsed {len(eps)} validation points. "
          f"Last SR={srs[-1]:.4f} at total_eps={eps[-1]}.")

if __name__ == "__main__":
    main()
