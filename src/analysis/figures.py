#!/usr/bin/env python3
"""Figure generation for the paper.

Figure 1 — Decomposition: Accuracy vs FLOP reduction (3 curves)
Figure 2 — Generalization: Decomposition across model families
Figure 3 — Surface: Heatmaps over (FFN-skip%, token-budget)
Figure 4 — Compute-optimal allocation
Figure 5 — Lossless speedup (self-speculative decoding)

All figures use a consistent style suitable for EMNLP publication.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

# Paper-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'ffn_only': '#2196F3',       # Blue
    'full_layer': '#F44336',      # Red
    'attention_only': '#4CAF50',  # Green
    'baseline': '#757575',        # Gray
}

LABELS = {
    'ffn_only': 'FFN-only skip',
    'full_layer': 'Full-layer skip',
    'attention_only': 'Attention-only skip',
}

MARKERS = {
    'ffn_only': 'o',
    'full_layer': 's',
    'attention_only': '^',
}


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all JSONL results into a DataFrame."""
    records = []
    for f in os.listdir(results_dir):
        if f.endswith('.jsonl'):
            with open(os.path.join(results_dir, f)) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return pd.DataFrame(records)


def compute_accuracy_with_ci(
    df: pd.DataFrame,
    group_cols: List[str],
    ci: float = 0.95,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute accuracy with bootstrap confidence intervals."""
    results = []
    for name, group in df.groupby(group_cols):
        acc = group['accuracy'].mean()
        n = len(group)
        se = np.sqrt(acc * (1 - acc) / n) if n > 0 else 0

        # Bootstrap CI
        if n >= 10:
            boot_accs = []
            for _ in range(n_bootstrap):
                boot_sample = group['accuracy'].sample(n=n, replace=True)
                boot_accs.append(boot_sample.mean())
            ci_low = np.percentile(boot_accs, (1 - ci) / 2 * 100)
            ci_high = np.percentile(boot_accs, (1 + ci) / 2 * 100)
        else:
            ci_low = acc - 1.96 * se
            ci_high = acc + 1.96 * se

        row = dict(zip(group_cols, name if isinstance(name, tuple) else [name]))
        row.update({
            'accuracy': acc,
            'accuracy_pct': acc * 100,
            'se': se,
            'ci_low': ci_low * 100,
            'ci_high': ci_high * 100,
            'n': n,
        })
        results.append(row)

    return pd.DataFrame(results)


def figure1_decomposition(
    df: pd.DataFrame,
    output_path: str,
    benchmarks: List[str] = None,
):
    """Figure 1: THE figure. Accuracy vs FLOP reduction for 3 skip types.

    Shows FFN-only degrades gently, full-layer collapses.
    """
    if benchmarks is None:
        benchmarks = ['math500', 'gpqa']

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4))
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, benchmark in zip(axes, benchmarks):
        bench_df = df[df['benchmark'] == benchmark]

        for skip_type in ['ffn_only', 'full_layer', 'attention_only']:
            type_df = bench_df[bench_df['skip_type'] == skip_type]
            if type_df.empty:
                continue

            grouped = compute_accuracy_with_ci(type_df, ['flop_reduction_pct'])
            grouped = grouped.sort_values('flop_reduction_pct')

            ax.plot(
                grouped['flop_reduction_pct'], grouped['accuracy_pct'],
                color=COLORS[skip_type], marker=MARKERS[skip_type],
                label=LABELS[skip_type], linewidth=2, markersize=6,
            )
            ax.fill_between(
                grouped['flop_reduction_pct'],
                grouped['ci_low'], grouped['ci_high'],
                color=COLORS[skip_type], alpha=0.15,
            )

        # Baseline (no skip)
        baseline = bench_df[bench_df['skip_type'] == 'none']
        if not baseline.empty:
            baseline_acc = baseline['accuracy'].mean() * 100
            ax.axhline(y=baseline_acc, color=COLORS['baseline'],
                        linestyle='--', alpha=0.7, label='Full model')

        ax.set_xlabel('FLOP Reduction (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(benchmark.upper().replace('500', '-500'))
        ax.legend(loc='lower left')
        ax.set_xlim(-2, 55)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def figure2_generalization(
    df: pd.DataFrame,
    output_path: str,
    benchmark: str = 'math500',
):
    """Figure 2: Decomposition curves across model families."""
    models = df['model'].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[(df['model'] == model) & (df['benchmark'] == benchmark)]
        model_short = model.split('/')[-1]

        for skip_type in ['ffn_only', 'full_layer', 'attention_only']:
            type_df = model_df[model_df['skip_type'] == skip_type]
            if type_df.empty:
                continue

            grouped = compute_accuracy_with_ci(type_df, ['flop_reduction_pct'])
            grouped = grouped.sort_values('flop_reduction_pct')

            ax.plot(
                grouped['flop_reduction_pct'], grouped['accuracy_pct'],
                color=COLORS[skip_type], marker=MARKERS[skip_type],
                label=LABELS[skip_type], linewidth=2, markersize=6,
            )

        ax.set_xlabel('FLOP Reduction (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(model_short, fontsize=10)
        if ax == axes[0]:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle(f'Decomposition Generalization — {benchmark.upper()}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 2 to {output_path}")


def figure3_surface(
    df: pd.DataFrame,
    output_path: str,
    benchmarks: List[str] = None,
):
    """Figure 3: Heatmaps of accuracy over (FFN-skip%, token-budget)."""
    if benchmarks is None:
        benchmarks = ['math500', 'gpqa', 'mmlu_pro']

    fig, axes = plt.subplots(1, len(benchmarks),
                              figsize=(4.5 * len(benchmarks), 4))
    if len(benchmarks) == 1:
        axes = [axes]

    # Only FFN-only skip for surface
    surface_df = df[(df['skip_type'].isin(['ffn_only', 'none']))]

    for ax, benchmark in zip(axes, benchmarks):
        bench_df = surface_df[surface_df['benchmark'] == benchmark]

        # Create pivot table
        grouped = compute_accuracy_with_ci(
            bench_df, ['ffn_skip_pct', 'token_budget']
        )

        # Handle unlimited budget
        grouped['token_budget_label'] = grouped['token_budget'].apply(
            lambda x: '∞' if x is None or pd.isna(x) else str(int(x))
        )

        # Pivot
        skip_pcts = sorted(grouped['ffn_skip_pct'].unique())
        budget_labels = ['512', '1024', '2048', '4096', '∞']
        budget_labels = [b for b in budget_labels
                         if b in grouped['token_budget_label'].values]

        heatmap = np.full((len(budget_labels), len(skip_pcts)), np.nan)
        for _, row in grouped.iterrows():
            if row['token_budget_label'] in budget_labels:
                j = budget_labels.index(row['token_budget_label'])
                i = skip_pcts.index(row['ffn_skip_pct'])
                heatmap[j, i] = row['accuracy_pct']

        im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=100)
        ax.set_xticks(range(len(skip_pcts)))
        ax.set_xticklabels([f'{int(s)}%' for s in skip_pcts])
        ax.set_yticks(range(len(budget_labels)))
        ax.set_yticklabels(budget_labels)
        ax.set_xlabel('FFN Skip %')
        ax.set_ylabel('Token Budget')
        ax.set_title(benchmark.upper().replace('500', '-500'))

        # Annotate cells
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if not np.isnan(heatmap[i, j]):
                    ax.text(j, i, f'{heatmap[i,j]:.0f}',
                            ha='center', va='center', fontsize=8,
                            color='white' if heatmap[i,j] < 50 else 'black')

    fig.colorbar(im, ax=axes, label='Accuracy (%)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 3 to {output_path}")


def figure4_compute_optimal(
    df: pd.DataFrame,
    output_path: str,
    benchmark: str = 'math500',
):
    """Figure 4: Compute-optimal allocation.

    Given a total FLOP budget, what's the best (FFN-skip%, length) split?
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    bench_df = df[(df['benchmark'] == benchmark) &
                  (df['skip_type'].isin(['ffn_only', 'none']))]

    # Group by approximate total FLOP budget
    # Total FLOPs ∝ (1 - flop_reduction_pct/100) × actual_tokens_generated
    bench_df = bench_df.copy()
    bench_df['relative_flops'] = (
        (1 - bench_df['flop_reduction_pct'] / 100) *
        bench_df['actual_tokens_generated']
    )

    # Bin into FLOP budget ranges
    grouped = compute_accuracy_with_ci(bench_df, ['ffn_skip_pct', 'token_budget'])

    # Plot strategies
    strategies = {
        'All length (no skip)': grouped[grouped['ffn_skip_pct'] == 0],
        'Balanced (20% FFN skip)': grouped[grouped['ffn_skip_pct'] == 20],
        'Heavy skip (40% FFN)': grouped[grouped['ffn_skip_pct'] == 40],
    }

    for name, strat_df in strategies.items():
        if strat_df.empty:
            continue
        strat_df = strat_df.sort_values('token_budget')
        ax.plot(strat_df['token_budget'].fillna(8192),
                strat_df['accuracy_pct'],
                marker='o', label=name, linewidth=2)

    ax.set_xlabel('Token Budget')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Compute-Optimal Allocation — {benchmark.upper()}')
    ax.legend()
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 4 to {output_path}")


def generate_all_figures(results_dir: str, figures_dir: str):
    """Generate all paper figures from results."""
    os.makedirs(figures_dir, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} results from {results_dir}")
    print(f"Models: {df['model'].unique()}")
    print(f"Benchmarks: {df['benchmark'].unique()}")
    print(f"Skip types: {df['skip_type'].unique()}")

    # Figure 1
    figure1_decomposition(df, os.path.join(figures_dir, "fig1_decomposition.pdf"))

    # Figure 2
    figure2_generalization(df, os.path.join(figures_dir, "fig2_generalization.pdf"))

    # Figure 3
    figure3_surface(df, os.path.join(figures_dir, "fig3_surface.pdf"))

    # Figure 4
    figure4_compute_optimal(df, os.path.join(figures_dir, "fig4_compute_optimal.pdf"))

    print("\nAll figures generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.figures_dir)
