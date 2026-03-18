#!/usr/bin/env python3
"""Run multiple skip conditions in a single process.

Loads the model ONCE, then iterates over all (skip_type, skip_pct) configs.
Saves results to separate JSONL files per condition. Crash-safe via resume.

This is the efficient version for Phase 2+ experiments where model loading
time (~2min) would otherwise dominate with many conditions.

Usage:
    python src/experiments/run_multi_condition.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 \
        --configs none:0 ffn_only:30 full_layer:30 \
        --output_dir /content/drive/MyDrive/depth_or_length \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from infrastructure.model_loader import load_model_and_tokenizer, resolve_model_name
from infrastructure.generation import (
    generate_with_budget, extract_answer, check_answer_correct, find_horl,
)
from infrastructure.checkpoint import (
    get_experiment_id, load_completed, append_result,
    save_run_metadata, save_crash_log,
)
from depth_control.skip_manager import get_skip_layers, apply_skip
from depth_control.flop_counter import compute_total_flops, get_architecture
from benchmarks.loader import load_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-condition experiment runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "mmlu_pro", "aime",
                                 "humaneval", "livecodebench"])
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--configs", nargs="+", required=True,
                        help="skip_type:skip_pct pairs, e.g. none:0 ffn_only:30 full_layer:30")
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--quantize_4bit", action="store_true")
    parser.add_argument("--cold_start", type=int, default=4)
    parser.add_argument("--cold_end", type=int, default=4)
    return parser.parse_args()


def parse_config(config_str: str) -> tuple:
    """Parse 'skip_type:skip_pct' into (str, float)."""
    parts = config_str.split(':')
    return parts[0], float(parts[1])


def run_condition(
    model, tokenizer, problems, skip_type, skip_pct,
    benchmark, token_budget, seed, output_dir,
    cold_start, cold_end, hf_name, resume,
):
    """Run one condition on all problems."""
    experiment_id = get_experiment_id(
        hf_name, benchmark, skip_type, skip_pct, token_budget, seed,
    )
    results_dir = os.path.join(output_dir, "results")
    results_path = os.path.join(results_dir, f"{experiment_id}.jsonl")
    os.makedirs(results_dir, exist_ok=True)

    completed = set()
    if resume:
        completed = load_completed(results_path)
        if completed:
            logger.info(f"  Resuming: {len(completed)} already done")

    remaining = [p for p in problems if p['problem_id'] not in completed]
    if not remaining:
        logger.info(f"  All problems completed, skipping")
        return

    # Compute skip layers
    try:
        arch = get_architecture(hf_name)
        num_layers = arch.num_layers
    except ValueError:
        from infrastructure.model_loader import get_layer_modules
        num_layers = len(get_layer_modules(model))

    skip_layers = []
    if skip_type != "none" and skip_pct > 0:
        skip_layers = get_skip_layers(
            num_layers, skip_pct,
            cold_start=cold_start, cold_end=cold_end,
        )
        logger.info(f"  Skip layers: {skip_layers}")

    # FLOP info
    flop_info = {}
    try:
        flop_info = compute_total_flops(hf_name, skip_type, skip_layers)
        logger.info(f"  FLOP reduction: {flop_info.get('flop_reduction_pct', 0):.1f}%")
    except ValueError:
        pass

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa", "mmlu_pro": "mmlu_pro",
        "aime": "aime", "humaneval": "humaneval", "livecodebench": "livecodebench",
    }
    bench_type = bench_type_map.get(benchmark, "math")

    correct = 0
    total = 0

    for i, problem in enumerate(remaining):
        pid = problem["problem_id"]
        logger.info(f"  [{i+1}/{len(remaining)}] {pid}")

        if skip_layers and skip_type != "none":
            with apply_skip(model, skip_type, skip_layers):
                gen_result = generate_with_budget(
                    model, tokenizer,
                    prompt=problem["prompt"],
                    token_budget=token_budget,
                    benchmark_type=bench_type,
                    seed=seed,
                )
        else:
            gen_result = generate_with_budget(
                model, tokenizer,
                prompt=problem["prompt"],
                token_budget=token_budget,
                benchmark_type=bench_type,
                seed=seed,
            )

        extracted = extract_answer(gen_result["generation_text"], bench_type)
        is_correct = check_answer_correct(extracted, problem["ground_truth"], bench_type)
        accuracy = 1 if is_correct else 0

        horl_pos = None
        if skip_type == "none" and token_budget is None:
            horl_pos = find_horl(gen_result["generation_text"], problem["ground_truth"], bench_type)

        record = {
            "problem_id": pid,
            "model": hf_name,
            "benchmark": benchmark,
            "skip_type": skip_type,
            "ffn_skip_pct": skip_pct,
            "skip_layers": skip_layers,
            "token_budget": token_budget,
            "seed": seed,
            "accuracy": accuracy,
            "extracted_answer": extracted,
            "ground_truth": problem["ground_truth"],
            "actual_tokens_generated": gen_result["actual_tokens_generated"],
            "hit_budget": gen_result["hit_budget"],
            "generation_text": gen_result["generation_text"],
            "wall_clock_seconds": gen_result["wall_clock_seconds"],
            "peak_memory_mb": gen_result["peak_memory_mb"],
            "flop_reduction_pct": flop_info.get("flop_reduction_pct", 0),
            "horl_position": horl_pos,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            import subprocess as sp
            record["git_sha"] = sp.check_output(
                ["git", "rev-parse", "HEAD"], stderr=sp.DEVNULL
            ).decode().strip()
        except Exception:
            record["git_sha"] = "unknown"

        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)

        append_result(results_path, record)
        total += 1
        correct += accuracy
        running_acc = correct / total * 100

        logger.info(
            f"    {'Y' if accuracy else 'N'} | "
            f"ans={extracted} gt={problem['ground_truth']} | "
            f"tok={gen_result['actual_tokens_generated']} "
            f"t={gen_result['wall_clock_seconds']:.1f}s | "
            f"acc={running_acc:.1f}%"
        )

    if total > 0:
        logger.info(f"  Condition done: {correct}/{total} = {correct/total*100:.1f}%")


def main():
    args = parse_args()
    hf_name = resolve_model_name(args.model)

    configs = [parse_config(c) for c in args.configs]
    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Configs: {configs}")
    logger.info(f"Subsample: {args.subsample or 'all'}")

    # Load model ONCE
    logger.info("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        hf_name, dtype=args.dtype, quantize_4bit=args.quantize_4bit,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Load benchmark ONCE
    problems = load_benchmark(args.benchmark, subsample=args.subsample, seed=args.seed)
    logger.info(f"Loaded {len(problems)} problems")

    # Run each condition
    for skip_type, skip_pct in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {skip_type} @ {skip_pct}%")
        logger.info(f"{'='*60}")

        try:
            run_condition(
                model, tokenizer, problems,
                skip_type, skip_pct,
                args.benchmark, args.token_budget, args.seed,
                args.output_dir, args.cold_start, args.cold_end,
                hf_name, args.resume,
            )
        except Exception as e:
            logger.error(f"Condition {skip_type}@{skip_pct}% FAILED: {e}")
            debug_dir = os.path.join(args.output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            exp_id = f"{skip_type}_{int(skip_pct)}pct"
            save_crash_log(debug_dir, exp_id, e, None)
            continue

    logger.info("\nAll conditions complete.")


if __name__ == "__main__":
    main()
