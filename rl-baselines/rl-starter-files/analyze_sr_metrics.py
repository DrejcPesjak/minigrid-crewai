#!/usr/bin/env python3
# analyze_sr_metrics.py
# Calculate SR metrics across multiple trials per level.
# Metrics:
# 1. Success rate at 500 episodes (first total_eps with 5xx)
# 2. Number of episodes when 70% SR is first reached
# 3. Total episodes at final line if SR >= 99%

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np

VAL512_RE = re.compile(r'^\[VAL512\].*?SR=([0-9.]+).*?total_eps=(\d+)', re.ASCII)


def parse_val512_log(path: Path) -> Tuple[List[int], List[float]]:
    """Parse VAL512 rows from log file and return (episodes, success_rates)."""
    eps, srs = [], []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = VAL512_RE.search(line)
            if m:
                sr = float(m.group(1))
                te = int(m.group(2))
                eps.append(te)
                srs.append(sr)
    return eps, srs


def metric_sr_at_500(eps: List[int], srs: List[float]) -> Optional[float]:
    """Return SR at first episode count in 500-599 range."""
    for e, sr in zip(eps, srs):
        if 500 <= e < 600:
            return sr
    return None


def metric_eps_at_70_sr(eps: List[int], srs: List[float]) -> Optional[int]:
    """Return first episode count when SR >= 0.70."""
    for e, sr in zip(eps, srs):
        if sr >= 0.70:
            return e
    return None


# def metric_eps_at_5consecutive_99(eps: List[int], srs: List[float]) -> Optional[int]:
#     """Return episode count when 5 consecutive SR >= 0.99 are observed."""
#     if len(srs) < 5:
#         return None
#     
#     for i in range(len(srs) - 4):
#         # Check if current and next 4 are all >= 0.99
#         if all(srs[j] >= 0.99 for j in range(i, i + 5)):
#             return eps[i + 4]  # Return the episode count at the 5th consecutive occurrence
#     return None


def metric_eps_at_final_99(eps: List[int], srs: List[float]) -> Optional[int]:
    """Return episode count at last line if SR >= 0.99."""
    if not srs:
        return None
    
    if srs[-1] >= 0.99:
        return eps[-1]
    return None


def find_trials(storage_dir: Path, base_name: str) -> Dict[str, Path]:
    """Find all trial folders matching base_name pattern."""
    trials = {}
    # Match patterns like: PickupLoc-v0, PickupLoc-v1, PickupLoc-fastA, etc.
    for p in storage_dir.iterdir():
        if p.is_dir() and p.name.startswith(base_name + '-'):
            logp = p / "log.txt"
            if logp.exists():
                trials[p.name] = logp
    return trials


def calculate_metrics_for_trial(logpath: Path) -> Dict[str, Optional[float]]:
    """Calculate all three metrics for a single trial."""
    eps, srs = parse_val512_log(logpath)
    
    if not eps:
        return {
            'sr_at_500': None,
            'eps_at_70sr': None,
            'eps_at_final_99': None
        }
    
    return {
        'sr_at_500': metric_sr_at_500(eps, srs),
        'eps_at_70sr': metric_eps_at_70_sr(eps, srs),
        'eps_at_final_99': metric_eps_at_final_99(eps, srs)
    }


def aggregate_metrics(metrics_list: List[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across trials: compute min, max, mean."""
    result = {}
    
    for metric_name in ['sr_at_500', 'eps_at_70sr', 'eps_at_final_99']:
        values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]
        
        if values:
            result[metric_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'count': len(values),
                'total_trials': len(metrics_list)
            }
        else:
            result[metric_name] = {
                'min': None,
                'max': None,
                'mean': None,
                'count': 0,
                'total_trials': len(metrics_list)
            }
    
    return result


def format_metric_value(value: Optional[float], metric_name: str) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    
    if metric_name == 'sr_at_500':
        return f"{value:.4f}"
    else:
        return f"{int(value)}"


def print_results(level_results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print formatted results for all levels."""
    print("\n" + "=" * 80)
    print("SUCCESS RATE METRICS ANALYSIS")
    print("=" * 80)
    
    metric_titles = {
        'sr_at_500': 'Success Rate at 500 Episodes',
        'eps_at_70sr': 'Episodes to Reach 70% SR',
        'eps_at_final_99': 'Episodes at Final Line (if SR >= 99%)'
    }
    
    for level_name in sorted(level_results.keys()):
        results = level_results[level_name]
        
        print(f"\n{'─' * 80}")
        print(f"LEVEL: {level_name}")
        print(f"{'─' * 80}")
        
        for metric_name, metric_title in metric_titles.items():
            metric_data = results[metric_name]
            count = metric_data['count']
            total = metric_data['total_trials']
            
            print(f"\n  {metric_title}:")
            print(f"    Trials: {count}/{total} completed this metric")
            
            if count > 0:
                min_val = format_metric_value(metric_data['min'], metric_name)
                max_val = format_metric_value(metric_data['max'], metric_name)
                mean_val = format_metric_value(metric_data['mean'], metric_name)
                
                print(f"    Min:  {min_val}")
                print(f"    Max:  {max_val}")
                print(f"    Mean: {mean_val}")
            else:
                print(f"    No trials reached this metric")
    
    print(f"\n{'═' * 80}\n")


def export_csv(level_results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path):
    """Export results to CSV format."""
    lines = ["Level,Metric,Min,Max,Mean,Count,TotalTrials"]
    
    for level_name in sorted(level_results.keys()):
        results = level_results[level_name]
        
        for metric_name in ['sr_at_500', 'eps_at_70sr', 'eps_at_final_99']:
            metric_data = results[metric_name]
            min_val = metric_data['min'] if metric_data['min'] is not None else ""
            max_val = metric_data['max'] if metric_data['max'] is not None else ""
            mean_val = metric_data['mean'] if metric_data['mean'] is not None else ""
            count = metric_data['count']
            total = metric_data['total_trials']
            
            lines.append(f"{level_name},{metric_name},{min_val},{max_val},{mean_val},{count},{total}")
    
    output_path.write_text('\n'.join(lines) + '\n')
    print(f"Results exported to: {output_path}")


def main():
    # python3 analyze_sr_metrics.py --storage storage --levels PickupLoc GoToRedBall GoToRedBallGrey GoToLocal --csv sr_metrics_results_updated.csv
    ap = argparse.ArgumentParser(
        description="Analyze SR metrics across multiple trials per level.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all levels in storage directory
  python3 analyze_sr_metrics.py --storage storage
  
  # Analyze specific levels
  python3 analyze_sr_metrics.py --storage storage --levels PickupLoc GoToRedBall
  
  # Export to CSV
  python3 analyze_sr_metrics.py --storage storage --csv results.csv
        """
    )
    ap.add_argument("--storage", type=Path, default=Path("storage"),
                    help="Root directory containing trial folders.")
    ap.add_argument("--levels", nargs='+', type=str, default=None,
                    help="Specific level base names to analyze (e.g., PickupLoc GoToRedBall). "
                         "If not provided, auto-detect all levels.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to export results as CSV.")
    args = ap.parse_args()
    
    # Auto-detect levels if not specified
    if args.levels is None:
        # Find all unique base names
        level_set = set()
        for p in args.storage.iterdir():
            if p.is_dir() and '-' in p.name:
                # Extract base name (everything before the last dash+version)
                parts = p.name.rsplit('-', 1)
                if len(parts) == 2:
                    base_name = parts[0]
                    level_set.add(base_name)
        args.levels = sorted(level_set)
        print(f"Auto-detected levels: {', '.join(args.levels)}")
    
    # Process each level
    level_results = {}
    
    for level_name in args.levels:
        trials = find_trials(args.storage, level_name)
        
        if not trials:
            print(f"Warning: No trials found for level '{level_name}'")
            continue
        
        print(f"Processing {level_name}: {len(trials)} trials found...")
        
        metrics_list = []
        for trial_name, logpath in sorted(trials.items()):
            metrics = calculate_metrics_for_trial(logpath)
            metrics_list.append(metrics)
        
        if metrics_list:
            level_results[level_name] = aggregate_metrics(metrics_list)
    
    # Display results
    if level_results:
        print_results(level_results)
        
        if args.csv:
            export_csv(level_results, args.csv)
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()

