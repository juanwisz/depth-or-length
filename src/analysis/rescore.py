#!/usr/bin/env python3
"""Re-score existing JSONL results with improved answer normalizer.

Reads JSONL files, re-applies check_answer_correct with the latest
normalization logic, and reports original vs rescored accuracy.
Does NOT modify the JSONL files.

Usage:
    python src/analysis/rescore.py results/
    python src/analysis/rescore.py results/specific_file.jsonl
"""

import json
import os
import sys
from pathlib import Path

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from infrastructure.generation import check_answer_correct, normalize_math_answer


def rescore_file(filepath: str) -> dict:
    """Re-score a single JSONL file."""
    lines = [json.loads(l) for l in open(filepath)]
    if not lines:
        return None

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa", "mmlu_pro": "mmlu_pro",
        "aime": "aime",
    }

    original_correct = 0
    rescored_correct = 0
    flipped = []

    for r in lines:
        benchmark = r.get("benchmark", "math500")
        bench_type = bench_type_map.get(benchmark, "math")

        original_correct += r.get("accuracy", 0)

        extracted = r.get("extracted_answer")
        ground_truth = r.get("ground_truth", "")
        new_correct = check_answer_correct(extracted, ground_truth, bench_type)
        if new_correct:
            rescored_correct += 1

        if bool(new_correct) != bool(r.get("accuracy", 0)):
            flipped.append({
                "problem_id": r["problem_id"],
                "extracted": extracted,
                "ground_truth": ground_truth,
                "original": r.get("accuracy", 0),
                "rescored": 1 if new_correct else 0,
            })

    total = len(lines)
    avg_tokens = sum(r.get("actual_tokens_generated", 0) for r in lines) / total

    return {
        "file": os.path.basename(filepath),
        "total": total,
        "original_correct": original_correct,
        "original_pct": original_correct / total * 100,
        "rescored_correct": rescored_correct,
        "rescored_pct": rescored_correct / total * 100,
        "flipped": len(flipped),
        "flipped_details": flipped,
        "avg_tokens": avg_tokens,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python rescore.py <path_to_results_dir_or_file>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = sorted(Path(target).glob("*.jsonl"))
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    for f in files:
        result = rescore_file(str(f))
        if result is None:
            continue

        name = result["file"].replace("deepseek_r1_distill_qwen_7b__math500__", "").replace("__seed42.jsonl", "")
        print(f"\n{name}:")
        print(f"  Original:  {result['original_correct']}/{result['total']} = {result['original_pct']:.1f}%")
        print(f"  Rescored:  {result['rescored_correct']}/{result['total']} = {result['rescored_pct']:.1f}%")
        print(f"  Flipped:   {result['flipped']} answers")
        print(f"  Avg tokens: {result['avg_tokens']:.0f}")

        if result["flipped_details"]:
            for f_detail in result["flipped_details"][:5]:
                direction = "0→1" if f_detail["rescored"] else "1→0"
                print(f"    {direction} {f_detail['problem_id']}: "
                      f"ext='{(f_detail['extracted'] or 'None')[:30]}' "
                      f"gt='{f_detail['ground_truth'][:30]}'")


if __name__ == "__main__":
    main()
