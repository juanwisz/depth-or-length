#!/usr/bin/env python3
"""Statistical analysis for the paper.

- Bootstrap confidence intervals on all accuracy numbers
- Significance tests for decomposition (FFN-only vs full-layer)
- Effect sizes
- HORL oracle analysis
"""

import json
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        np.mean(np.random.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def mcnemar_test(correct_a, correct_b):
    """McNemar's test for comparing two classifiers on the same data.

    Tests whether the disagreement pattern is symmetric.
    More appropriate than chi-squared for paired accuracy comparisons.

    Args:
        correct_a: Binary array of correctness for method A.
        correct_b: Binary array of correctness for method B.

    Returns:
        (statistic, p_value)
    """
    a = np.array(correct_a)
    b = np.array(correct_b)

    # Contingency table
    both_correct = np.sum(a & b)
    a_only = np.sum(a & ~b)
    b_only = np.sum(~a & b)
    both_wrong = np.sum(~a & ~b)

    # McNemar's test (with continuity correction)
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0

    statistic = (abs(a_only - b_only) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


def cohens_h(p1, p2):
    """Cohen's h effect size for comparing two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def decomposition_significance(
    df: pd.DataFrame,
    benchmark: str,
    flop_level: float = None,
) -> Dict:
    """Test significance of FFN-only vs full-layer decomposition.

    Args:
        df: Results DataFrame.
        benchmark: Benchmark to analyze.
        flop_level: Specific FLOP reduction level to compare at.
                    If None, compare across all levels.

    Returns:
        Dict with test results.
    """
    bench_df = df[df['benchmark'] == benchmark]

    ffn_df = bench_df[bench_df['skip_type'] == 'ffn_only']
    full_df = bench_df[bench_df['skip_type'] == 'full_layer']

    if flop_level is not None:
        # Filter to specific FLOP level (with tolerance)
        ffn_df = ffn_df[abs(ffn_df['flop_reduction_pct'] - flop_level) < 5]
        full_df = full_df[abs(full_df['flop_reduction_pct'] - flop_level) < 5]

    if ffn_df.empty or full_df.empty:
        return {"error": "Insufficient data"}

    # Match on problem_id for paired test
    common_ids = set(ffn_df['problem_id']) & set(full_df['problem_id'])
    if len(common_ids) < 10:
        return {"error": f"Only {len(common_ids)} common problems"}

    ffn_matched = ffn_df[ffn_df['problem_id'].isin(common_ids)].set_index('problem_id')
    full_matched = full_df[full_df['problem_id'].isin(common_ids)].set_index('problem_id')

    common_ids = sorted(common_ids)
    ffn_correct = ffn_matched.loc[common_ids, 'accuracy'].values
    full_correct = full_matched.loc[common_ids, 'accuracy'].values

    # McNemar's test
    stat, p_value = mcnemar_test(ffn_correct.astype(bool), full_correct.astype(bool))

    # Accuracies
    ffn_acc = ffn_correct.mean()
    full_acc = full_correct.mean()
    gap = (ffn_acc - full_acc) * 100

    # Effect size
    h = cohens_h(ffn_acc, full_acc)

    # Bootstrap CIs
    ffn_ci = bootstrap_ci(ffn_correct)
    full_ci = bootstrap_ci(full_correct)

    return {
        "benchmark": benchmark,
        "flop_level": flop_level,
        "n_problems": len(common_ids),
        "ffn_only_accuracy": round(ffn_acc * 100, 2),
        "ffn_only_ci": [round(x * 100, 2) for x in ffn_ci],
        "full_layer_accuracy": round(full_acc * 100, 2),
        "full_layer_ci": [round(x * 100, 2) for x in full_ci],
        "accuracy_gap_pct": round(gap, 2),
        "mcnemar_statistic": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant_001": p_value < 0.01,
        "cohens_h": round(h, 4),
        "effect_size_interpretation": (
            "small" if abs(h) < 0.2 else
            "medium" if abs(h) < 0.5 else
            "large"
        ),
    }


def horl_analysis(df: pd.DataFrame, benchmark: str) -> Dict:
    """Analyze Hindsight-Optimal Reasoning Length.

    For full-model, full-length traces, compute how much of the
    generation was necessary to reach the correct answer.
    """
    # Filter to full-model, full-length, correct answers
    full_df = df[
        (df['benchmark'] == benchmark) &
        (df['skip_type'] == 'none') &
        (df['token_budget'].isna() | (df['token_budget'] == 0)) &
        (df['accuracy'] == 1) &
        (df['horl_position'].notna())
    ].copy()

    if full_df.empty:
        return {"error": "No data for HORL analysis"}

    # HORL ratio: position of correct answer / total generation length
    full_df['horl_ratio'] = full_df['horl_position'] / full_df['actual_tokens_generated']

    return {
        "benchmark": benchmark,
        "n_correct": len(full_df),
        "mean_total_tokens": round(full_df['actual_tokens_generated'].mean(), 0),
        "mean_horl_position": round(full_df['horl_position'].mean(), 0),
        "mean_horl_ratio": round(full_df['horl_ratio'].mean(), 4),
        "median_horl_ratio": round(full_df['horl_ratio'].median(), 4),
        "theoretical_max_savings_pct": round(
            (1 - full_df['horl_ratio'].mean()) * 100, 1
        ),
        "p10_horl_ratio": round(full_df['horl_ratio'].quantile(0.1), 4),
        "p90_horl_ratio": round(full_df['horl_ratio'].quantile(0.9), 4),
    }


def full_statistical_report(results_dir: str) -> str:
    """Generate a full statistical report from results."""
    from src.analysis.figures import load_results

    df = load_results(results_dir)
    if df.empty:
        return "No results found."

    report = []
    report.append("# Statistical Report — Depth or Length?\n")

    # Decomposition significance
    report.append("## Decomposition Significance Tests\n")
    for benchmark in ['math500', 'gpqa', 'mmlu_pro']:
        bench_df = df[df['benchmark'] == benchmark]
        if bench_df.empty:
            continue

        flop_levels = sorted(bench_df['flop_reduction_pct'].unique())
        for flop in flop_levels:
            if flop <= 0:
                continue
            result = decomposition_significance(df, benchmark, flop)
            if 'error' in result:
                continue

            report.append(f"### {benchmark} @ {flop:.0f}% FLOP reduction")
            report.append(f"  FFN-only: {result['ffn_only_accuracy']}% "
                          f"({result['ffn_only_ci'][0]:.1f}-{result['ffn_only_ci'][1]:.1f})")
            report.append(f"  Full-layer: {result['full_layer_accuracy']}% "
                          f"({result['full_layer_ci'][0]:.1f}-{result['full_layer_ci'][1]:.1f})")
            report.append(f"  Gap: {result['accuracy_gap_pct']}pp")
            report.append(f"  McNemar p={result['p_value']:.6f} "
                          f"{'***' if result['significant_001'] else 'ns'}")
            report.append(f"  Cohen's h={result['cohens_h']:.3f} "
                          f"({result['effect_size_interpretation']})\n")

    # HORL analysis
    report.append("\n## HORL Oracle Analysis\n")
    for benchmark in ['math500', 'gpqa', 'mmlu_pro']:
        result = horl_analysis(df, benchmark)
        if 'error' in result:
            continue
        report.append(f"### {benchmark}")
        report.append(f"  Correct problems: {result['n_correct']}")
        report.append(f"  Mean tokens generated: {result['mean_total_tokens']:.0f}")
        report.append(f"  Mean HORL position: {result['mean_horl_position']:.0f}")
        report.append(f"  Theoretical max savings: {result['theoretical_max_savings_pct']:.1f}%\n")

    return "\n".join(report)
