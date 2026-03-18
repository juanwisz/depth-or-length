#!/usr/bin/env python3
"""Quick analysis of Phase 1.1 decomposition pilot results.

Run this locally on downloaded JSONL files to check progress and
make the go/no-go decision on the decomposition finding.

Usage:
    python src/analysis/pilot_analysis.py --results_dir results/

Outputs:
    - Per-condition accuracy with CI
    - Decomposition gap (FFN-only vs full-layer)
    - Per-problem timing stats
    - Answer extraction success rate
    - Go/no-go recommendation
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all JSONL files from a directory."""
    all_records = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.jsonl'):
            path = os.path.join(results_dir, fname)
            records = load_jsonl(path)
            all_records.extend(records)
            print(f"  {fname}: {len(records)} records")
    return all_records


def bootstrap_ci(values: List[float], n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    arr = np.array(values)
    if len(arr) < 2:
        return arr.mean(), arr.mean(), arr.mean()
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    low = np.percentile(boot_means, (1 - ci) / 2 * 100)
    high = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return arr.mean(), low, high


def analyze_pilot(records: List[Dict]) -> None:
    """Analyze pilot results and print summary."""
    if not records:
        print("No records found.")
        return

    # Group by condition
    by_condition = defaultdict(list)
    for r in records:
        skip_type = r.get('skip_type', 'none')
        skip_pct = r.get('ffn_skip_pct', 0)
        key = f"{skip_type}_{int(skip_pct)}pct"
        by_condition[key].append(r)

    print(f"\n{'='*70}")
    print("PHASE 1.1 DECOMPOSITION PILOT — RESULTS")
    print(f"{'='*70}")
    print(f"Total records: {len(records)}")
    print(f"Conditions: {list(by_condition.keys())}")

    # Per-condition stats
    print(f"\n{'─'*70}")
    print(f"{'Condition':<25} {'N':>5} {'Acc%':>7} {'95% CI':>15} {'Avg Tok':>8} {'Avg Time':>9}")
    print(f"{'─'*70}")

    condition_accs = {}
    for key in sorted(by_condition.keys()):
        recs = by_condition[key]
        accs = [r['accuracy'] for r in recs]
        tokens = [r.get('actual_tokens_generated', 0) for r in recs]
        times = [r.get('wall_clock_seconds', 0) for r in recs]

        mean_acc, ci_low, ci_high = bootstrap_ci(accs)
        condition_accs[key] = (mean_acc, ci_low, ci_high, len(accs))

        print(f"{key:<25} {len(accs):>5} {mean_acc*100:>6.1f}% "
              f"[{ci_low*100:>5.1f}, {ci_high*100:>5.1f}] "
              f"{np.mean(tokens):>7.0f} {np.mean(times):>8.1f}s")

    # Decomposition gap analysis
    print(f"\n{'─'*70}")
    print("DECOMPOSITION GAP ANALYSIS")
    print(f"{'─'*70}")

    baseline_key = 'none_0pct'
    ffn_key = 'ffn_only_30pct'
    full_key = 'full_layer_30pct'

    if baseline_key in condition_accs:
        base_acc = condition_accs[baseline_key][0]
        print(f"Baseline accuracy: {base_acc*100:.1f}%")

    if ffn_key in condition_accs and full_key in condition_accs:
        ffn_acc = condition_accs[ffn_key][0]
        full_acc = condition_accs[full_key][0]
        gap = (ffn_acc - full_acc) * 100

        print(f"FFN-only accuracy:    {ffn_acc*100:.1f}%")
        print(f"Full-layer accuracy:  {full_acc*100:.1f}%")
        print(f"Decomposition gap:    {gap:+.1f}% (FFN-only - full-layer)")

        if baseline_key in condition_accs:
            ffn_drop = (base_acc - ffn_acc) * 100
            full_drop = (base_acc - full_acc) * 100
            print(f"FFN-only drop from baseline:    {ffn_drop:+.1f}%")
            print(f"Full-layer drop from baseline:  {full_drop:+.1f}%")

        # Go/no-go
        print(f"\n{'─'*70}")
        print("GO / NO-GO DECISION")
        print(f"{'─'*70}")
        if gap >= 10:
            print(f"GO — Gap is {gap:.1f}% >= 10%. Decomposition finding is real.")
            print("Proceed to Phase 2 full experiments.")
        elif gap >= 5:
            print(f"MARGINAL — Gap is {gap:.1f}%. Between 5-10%.")
            print("Consider running more problems or adjusting skip percentage.")
        else:
            print(f"NO-GO — Gap is {gap:.1f}% < 5%. FFN-only is NOT special.")
            print("Abandon decomposition angle. Consider pivot to surface-only paper.")
    elif ffn_key in condition_accs or full_key in condition_accs:
        print("Only partial skip conditions available. Waiting for more data.")
    else:
        print("No skip conditions completed yet. Only baseline available.")
        if baseline_key in condition_accs:
            base_acc = condition_accs[baseline_key][0]
            n = condition_accs[baseline_key][3]
            print(f"Baseline: {base_acc*100:.1f}% on {n} problems")
            # Expected range for DeepSeek-R1-Distill-Qwen-7B on MATH-500: ~83%
            if base_acc > 0.75:
                print("Baseline looks reasonable (expected ~83% for this model).")
            elif base_acc > 0.60:
                print("Baseline lower than expected. Check answer extraction.")
            else:
                print("WARNING: Baseline very low. Possible extraction bug.")

    # Answer extraction analysis
    print(f"\n{'─'*70}")
    print("ANSWER EXTRACTION")
    print(f"{'─'*70}")
    extraction_fails = sum(1 for r in records if r.get('extracted_answer') is None)
    print(f"Extraction failures: {extraction_fails}/{len(records)} "
          f"({extraction_fails/len(records)*100:.1f}%)")
    if extraction_fails / len(records) > 0.10:
        print("WARNING: >10% extraction failures. Review extraction logic.")
    else:
        print("OK: Extraction success rate >= 90%.")

    # Token stats
    print(f"\n{'─'*70}")
    print("TOKEN GENERATION STATS")
    print(f"{'─'*70}")
    for key in sorted(by_condition.keys()):
        recs = by_condition[key]
        tokens = [r.get('actual_tokens_generated', 0) for r in recs]
        hit_budget = sum(1 for r in recs if r.get('hit_budget', False))
        print(f"{key:<25} tokens: mean={np.mean(tokens):.0f} "
              f"median={np.median(tokens):.0f} "
              f"max={max(tokens)} "
              f"hit_budget={hit_budget}/{len(recs)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with JSONL result files")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    records = load_all_results(args.results_dir)
    analyze_pilot(records)


if __name__ == "__main__":
    main()
