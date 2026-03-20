#!/usr/bin/env python3
"""Generate Figure 1: Accuracy vs FLOP reduction decomposition.

THE key figure of the paper. Three curves showing how accuracy degrades
under FFN-only skip (gentle), full-layer skip (collapse), and
attention-only skip (collapse) at iso-FLOP matched levels.

Reads JSONL files from run_full_experiment.py output directory.

Usage:
    python src/analysis/plot_decomposition.py \
        --input_dir results/ \
        --output_dir figures/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors matching paper aesthetics
COLORS = {
    "ffn_only": "#2196F3",       # Blue
    "full_layer": "#F44336",     # Red
    "attention_only": "#FF9800", # Orange
    "baseline": "#4CAF50",       # Green
}

LABELS = {
    "ffn_only": "FFN-only skip",
    "full_layer": "Full-layer skip",
    "attention_only": "Attention-only skip",
}

MARKERS = {
    "ffn_only": "o",
    "full_layer": "s",
    "attention_only": "^",
}


@dataclass
class ConfigResult:
    """Aggregated results for one experiment config."""

    config_name: str
    skip_type: Optional[str]
    skip_pct: float
    flop_reduction_pct: float
    accuracy: float
    accuracy_se: float
    n_problems: int
    mean_tokens: float
    mean_wall_clock: float


def load_all_results(input_dir: str) -> List[Dict]:
    """Load all JSONL files from input directory."""
    records = []
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(input_dir, fname)
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def aggregate_by_config(records: List[Dict]) -> List[ConfigResult]:
    """Group records by config and compute accuracy stats."""
    by_config = defaultdict(list)
    for rec in records:
        config_name = rec.get("config", "unknown")
        by_config[config_name].append(rec)

    results = []
    for config_name, recs in sorted(by_config.items()):
        accuracies = [r["accuracy"] for r in recs]
        mean_acc = np.mean(accuracies)
        se_acc = np.std(accuracies) / np.sqrt(len(accuracies))
        mean_tokens = np.mean([r["actual_tokens_generated"] for r in recs])
        mean_clock = np.mean([r["wall_clock_seconds"] for r in recs])

        results.append(ConfigResult(
            config_name=config_name,
            skip_type=recs[0].get("skip_type"),
            skip_pct=recs[0].get("skip_pct", 0),
            flop_reduction_pct=recs[0].get("flop_reduction_pct", 0),
            accuracy=mean_acc,
            accuracy_se=se_acc,
            n_problems=len(recs),
            mean_tokens=mean_tokens,
            mean_wall_clock=mean_clock,
        ))

    return results


def plot_decomposition_figure(
    results: List[ConfigResult],
    output_path: str,
    benchmark_name: str = "MATH-500",
) -> None:
    """Plot Figure 1: Accuracy vs FLOP reduction.

    Three curves: FFN-only (blue), full-layer (red), attention-only (orange).
    Baseline shown as horizontal dashed line.

    Args:
        results: List of ConfigResult from aggregate_by_config.
        output_path: Path to save figure.
        benchmark_name: Name for title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Find baseline
    baseline = [r for r in results if r.skip_type is None]
    baseline_acc = baseline[0].accuracy if baseline else None

    if baseline_acc is not None:
        ax.axhline(
            y=baseline_acc * 100, color=COLORS["baseline"],
            linestyle='--', alpha=0.7, linewidth=1.5,
            label=f"Baseline ({baseline_acc*100:.1f}%)",
        )

    # Plot each skip type
    for skip_type in ["ffn_only", "full_layer", "attention_only"]:
        configs = sorted(
            [r for r in results if r.skip_type == skip_type],
            key=lambda r: r.flop_reduction_pct,
        )
        if not configs:
            continue

        x = [0] + [c.flop_reduction_pct for c in configs]
        y = [baseline_acc * 100 if baseline_acc else 0] + [c.accuracy * 100 for c in configs]
        yerr = [0] + [c.accuracy_se * 100 for c in configs]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=MARKERS[skip_type],
            color=COLORS[skip_type],
            label=LABELS[skip_type],
            linewidth=2, markersize=8, capsize=4,
        )

    ax.set_xlabel("FLOP Reduction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Decomposition: Accuracy vs Compute Reduction ({benchmark_name})")
    ax.legend(loc="lower left")
    ax.set_xlim(-1, max(r.flop_reduction_pct for r in results) + 3)
    if baseline_acc:
        ax.set_ylim(0, baseline_acc * 100 + 10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_accuracy_table(results: List[ConfigResult], output_path: str) -> None:
    """Save a summary table as text."""
    lines = [
        f"{'Config':<30} {'Skip%':>6} {'FLOP↓%':>7} {'Acc%':>6} "
        f"{'±SE':>6} {'N':>4} {'Tok':>6} {'Time':>6}",
        "-" * 80,
    ]
    for r in sorted(results, key=lambda x: (x.skip_type or "", x.flop_reduction_pct)):
        lines.append(
            f"{r.config_name:<30} {r.skip_pct:>6.1f} "
            f"{r.flop_reduction_pct:>7.1f} {r.accuracy*100:>6.1f} "
            f"{r.accuracy_se*100:>6.1f} {r.n_problems:>4} "
            f"{r.mean_tokens:>6.0f} {r.mean_wall_clock:>6.1f}"
        )
    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nSaved: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate decomposition figure (Figure 1)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with JSONL result files")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory for output figures")
    parser.add_argument("--benchmark", type=str, default="MATH-500")
    return parser.parse_args()


def main():
    """Load results and generate decomposition figure."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.input_dir}")
    records = load_all_results(args.input_dir)
    print(f"Loaded {len(records)} total records")

    if not records:
        print("No records found!")
        return

    results = aggregate_by_config(records)
    print(f"Found {len(results)} configs")

    plot_decomposition_figure(
        results,
        os.path.join(args.output_dir, "figure1_decomposition.png"),
        benchmark_name=args.benchmark,
    )

    plot_accuracy_table(
        results,
        os.path.join(args.output_dir, "results_table.txt"),
    )


if __name__ == "__main__":
    main()
