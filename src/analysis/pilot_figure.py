#!/usr/bin/env python3
"""Generate preliminary Figure 1 from pilot data.

Creates the decomposition bar chart: baseline vs FFN-only vs full-layer
accuracy on MATH-500. This is the go/no-go visual.

Usage:
    python src/analysis/pilot_figure.py --results_dir results/ --output figures/pilot_decomposition.pdf
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results(results_dir: str) -> list:
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.jsonl'):
            with open(os.path.join(results_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return records


def bootstrap_ci(values, n_boot=2000):
    arr = np.array(values)
    if len(arr) < 2:
        return arr.mean(), 0
    boot = [np.random.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    return arr.mean(), 1.96 * np.std(boot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="figures/pilot_decomposition.pdf")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print("No records found.")
        return

    # Group by condition
    by_cond = defaultdict(list)
    for r in records:
        key = f"{r.get('skip_type', 'none')}_{int(r.get('ffn_skip_pct', 0))}pct"
        by_cond[key].append(r['accuracy'])

    # Prepare data
    conditions = ['none_0pct', 'ffn_only_30pct', 'full_layer_30pct']
    labels = ['Baseline\n(no skip)', 'FFN-only\n(30% skip)', 'Full-layer\n(30% skip)']
    colors = ['#757575', '#2196F3', '#F44336']

    accs = []
    errs = []
    ns = []
    for c in conditions:
        if c in by_cond:
            mean, ci = bootstrap_ci(by_cond[c])
            accs.append(mean * 100)
            errs.append(ci * 100)
            ns.append(len(by_cond[c]))
        else:
            accs.append(0)
            errs.append(0)
            ns.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, accs, yerr=errs, capsize=5, color=colors, edgecolor='black',
                  linewidth=0.8, width=0.6, alpha=0.85)

    # Add value labels
    for i, (bar, acc, n) in enumerate(zip(bars, accs, ns)):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errs[i] + 1,
                    f'{acc:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Phase 1.1 Decomposition Pilot — MATH-500\n'
                 'DeepSeek-R1-Distill-Qwen-7B, ~20% FLOP reduction')
    ax.set_ylim(0, 100)
    ax.axhline(y=accs[0] if accs[0] > 0 else 83, color='gray', linestyle='--',
               alpha=0.5, label='Baseline')
    ax.grid(axis='y', alpha=0.3)

    # Gap annotation
    if all(a > 0 for a in accs):
        gap = accs[1] - accs[2]
        mid_y = (accs[1] + accs[2]) / 2
        ax.annotate(f'Gap: {gap:+.1f}%',
                    xy=(1.5, mid_y), fontsize=12, fontweight='bold',
                    ha='center', color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved: {args.output}")

    # Also save PNG
    png_path = args.output.replace('.pdf', '.png')
    plt.savefig(png_path)
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
