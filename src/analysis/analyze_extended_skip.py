#!/usr/bin/env python3
"""Analyze extended skip test results: adjacent vs spread patterns.

Reads the JSON output from extended_skip_test.py and produces:
1. Summary table: config → avg match%, first divergence position
2. Adjacent vs spread comparison at each skip count (1, 2, 3)
3. Per-problem breakdown showing which configs preserve reasoning
"""

import json
import sys
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def load_results(path: str) -> List[Dict]:
    """Load extended skip test results from JSON."""
    with open(path) as f:
        return json.loads(f.read())


def analyze(results: List[Dict]) -> None:
    """Print analysis of extended skip test results."""
    # Group by config
    by_config: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    n_problems = len(set(r["problem_idx"] for r in results))
    print(f"Extended Skip Test Analysis")
    print(f"Problems: {n_problems}, Configs: {len(by_config)}")
    print(f"Max tokens: {results[0]['max_new_tokens']}")
    print(f"Model: {results[0]['model']}")
    print()

    # Summary table
    print(f"{'Config':<30s} {'Match%':>7s} {'1st Div':>8s} {'Tokens':>7s}")
    print("-" * 55)
    for config, rows in sorted(by_config.items()):
        avg_pct = np.mean([r["pct"] for r in rows])
        avg_div = np.mean([r["first_divergence"] for r in rows])
        avg_tokens = np.mean([r["num_skip_tokens"] for r in rows])
        print(f"{config:<30s} {avg_pct:>6.1f}% {avg_div:>7.1f} {avg_tokens:>7.0f}")

    print()

    # Adjacent vs spread comparison
    print("=" * 55)
    print("Adjacent vs Spread comparison")
    print("=" * 55)

    skip_counts = defaultdict(dict)
    for config, rows in by_config.items():
        n_skipped = len(rows[0]["skip_layers"])
        is_adj = "adj" in config
        is_spread = "spread" in config
        if is_adj:
            skip_counts[n_skipped]["adjacent"] = {
                "config": config,
                "avg_pct": np.mean([r["pct"] for r in rows]),
                "avg_div": np.mean([r["first_divergence"] for r in rows]),
            }
        elif is_spread:
            key = f"spread_{config}"
            if "spread" not in skip_counts[n_skipped]:
                skip_counts[n_skipped]["spread"] = []
            skip_counts[n_skipped]["spread"].append({
                "config": config,
                "avg_pct": np.mean([r["pct"] for r in rows]),
                "avg_div": np.mean([r["first_divergence"] for r in rows]),
            })

    for n_skip in sorted(skip_counts.keys()):
        data = skip_counts[n_skip]
        print(f"\n{n_skip}-layer skip:")
        if "adjacent" in data:
            adj = data["adjacent"]
            print(f"  Adjacent {adj['config']}: {adj['avg_pct']:.1f}% match, "
                  f"diverges at pos {adj['avg_div']:.0f}")
        if "spread" in data:
            for sp in data["spread"]:
                print(f"  Spread   {sp['config']}: {sp['avg_pct']:.1f}% match, "
                      f"diverges at pos {sp['avg_div']:.0f}")

    print()

    # Per-problem breakdown
    print("=" * 55)
    print("Per-problem breakdown (match % for each config)")
    print("=" * 55)
    configs = sorted(by_config.keys())
    header = f"{'Prob':<5s}" + "".join(f"{c[:15]:>16s}" for c in configs)
    print(header)
    print("-" * len(header))

    problems = sorted(set(r["problem_idx"] for r in results))
    for p in problems:
        row = f"{p:<5d}"
        for c in configs:
            matches = [r for r in by_config[c] if r["problem_idx"] == p]
            if matches:
                row += f"{matches[0]['pct']:>15.0f}%"
            else:
                row += f"{'N/A':>16s}"
        print(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_extended_skip.py <results.json>")
        sys.exit(1)
    results = load_results(sys.argv[1])
    analyze(results)
