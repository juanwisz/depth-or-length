#!/usr/bin/env python3
"""Rescore experiment JSONL results using hendrycks_math is_equiv.

Re-evaluates accuracy using the standardized is_equiv checker to ensure
our numbers match published baselines. Also reports answer extraction
success rate.

Usage:
    python src/analysis/rescore_results.py results/experiment.jsonl
    python src/analysis/rescore_results.py results/  # all JSONL in dir
"""

import json
import os
import sys
from collections import defaultdict

from lm_eval.tasks.hendrycks_math.utils import is_equiv


def rescore_file(path: str) -> dict:
    """Rescore a single JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {"path": path, "n": 0}

    correct_old = 0
    correct_new = 0
    extraction_success = 0
    mismatches = []

    for r in records:
        extracted = r.get("extracted_answer")
        gt = r.get("ground_truth", "")
        old_acc = r.get("accuracy", 0)
        bench = r.get("benchmark", "math")

        correct_old += old_acc

        if extracted is not None and extracted != "":
            extraction_success += 1

        if bench in ("math", "aime", "math500"):
            if extracted is not None:
                clean_ext = str(extracted).strip().strip('$').strip()
                clean_gt = str(gt).strip().strip('$').strip()
                new_acc = 1 if is_equiv(clean_ext, clean_gt) else 0
            else:
                new_acc = 0
        else:
            new_acc = old_acc

        correct_new += new_acc

        if old_acc != new_acc:
            mismatches.append({
                "problem_id": r.get("problem_id"),
                "extracted": extracted,
                "ground_truth": gt,
                "old": old_acc,
                "new": new_acc,
            })

    n = len(records)
    return {
        "path": os.path.basename(path),
        "n": n,
        "old_accuracy": correct_old / n * 100 if n else 0,
        "new_accuracy": correct_new / n * 100 if n else 0,
        "extraction_rate": extraction_success / n * 100 if n else 0,
        "mismatches": mismatches,
        "skip_type": records[0].get("skip_type", "unknown"),
        "benchmark": records[0].get("benchmark", "unknown"),
        "model": records[0].get("model", "unknown"),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python rescore_results.py <file.jsonl or directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in os.listdir(target) if f.endswith('.jsonl')]
    else:
        files = [target]

    if not files:
        print("No JSONL files found")
        sys.exit(1)

    print(f"Rescoring {len(files)} file(s) with is_equiv\n")

    for path in sorted(files):
        result = rescore_file(path)
        if result["n"] == 0:
            print(f"{result['path']}: empty")
            continue

        delta = result["new_accuracy"] - result["old_accuracy"]
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

        print(f"{result['path']}")
        print(f"  {result['skip_type']} | {result['benchmark']} | n={result['n']}")
        print(f"  Old: {result['old_accuracy']:.1f}% -> New: {result['new_accuracy']:.1f}% ({delta_str}%)")
        print(f"  Extraction rate: {result['extraction_rate']:.1f}%")

        if result["mismatches"]:
            print(f"  Mismatches ({len(result['mismatches'])}):")
            for m in result["mismatches"][:5]:
                print(f"    {m['problem_id']}: '{m['extracted']}' vs '{m['ground_truth']}' "
                      f"({m['old']}→{m['new']})")
            if len(result["mismatches"]) > 5:
                print(f"    ... and {len(result['mismatches']) - 5} more")
        print()


if __name__ == "__main__":
    main()
