#!/usr/bin/env python3
"""Run the compute-accuracy surface experiment (SUPPORTING FINDING).

Maps accuracy over (FFN-skip%, token-budget) grid for each benchmark.
Loads model ONCE per benchmark, runs all (skip%, budget) cells in-process.

Usage:
    python src/experiments/run_surface.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmarks math500 gpqa \
        --output_dir /content/drive/MyDrive/depth_or_length \
        --resume
"""

import argparse
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

FFN_SKIP_PCTS = [0, 10, 20, 30, 40, 50]
TOKEN_BUDGETS = [512, 1024, 2048, 4096, None]  # None = unlimited


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "gpqa"])
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--cold_start", type=int, default=4)
    parser.add_argument("--cold_end", type=int, default=4)
    return parser.parse_args()


def run_cell(
    model, tokenizer, problems, skip_pct, token_budget,
    benchmark, seed, output_dir, cold_start, cold_end, hf_name, resume,
):
    """Run one (skip_pct, token_budget) cell on all problems."""
    skip_type = "ffn_only" if skip_pct > 0 else "none"
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
            logger.info(f"    Resuming: {len(completed)} done")

    remaining = [p for p in problems if p['problem_id'] not in completed]
    if not remaining:
        logger.info(f"    All done, skipping")
        return

    try:
        arch = get_architecture(hf_name)
        num_layers = arch.num_layers
    except ValueError:
        from infrastructure.model_loader import get_layer_modules
        num_layers = len(get_layer_modules(model))

    skip_layers = []
    if skip_pct > 0:
        skip_layers = get_skip_layers(
            num_layers, skip_pct, cold_start=cold_start, cold_end=cold_end,
        )

    flop_info = {}
    try:
        flop_info = compute_total_flops(hf_name, skip_type, skip_layers)
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

        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)

        append_result(results_path, record)
        total += 1
        correct += accuracy

        if (i + 1) % 10 == 0 or i == len(remaining) - 1:
            logger.info(f"    [{i+1}/{len(remaining)}] acc={correct/total*100:.1f}%")

    if total > 0:
        logger.info(f"    Cell done: {correct}/{total} = {correct/total*100:.1f}%")


def main():
    args = parse_args()
    hf_name = resolve_model_name(args.model)

    total_cells = len(FFN_SKIP_PCTS) * len(TOKEN_BUDGETS) * len(args.benchmarks)
    logger.info(f"Surface experiment: {total_cells} cells across {len(args.benchmarks)} benchmarks")

    # Load model ONCE
    logger.info("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(hf_name, dtype=args.dtype)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    cell_num = 0
    for benchmark in args.benchmarks:
        problems = load_benchmark(benchmark, subsample=args.subsample, seed=args.seed)
        logger.info(f"\nBenchmark: {benchmark} ({len(problems)} problems)")

        for skip_pct in FFN_SKIP_PCTS:
            for budget in TOKEN_BUDGETS:
                cell_num += 1
                budget_str = str(budget) if budget else "unlimited"
                logger.info(f"\n  [{cell_num}/{total_cells}] "
                            f"FFN-skip={skip_pct}% budget={budget_str}")

                try:
                    run_cell(
                        model, tokenizer, problems,
                        skip_pct, budget,
                        benchmark, args.seed, args.output_dir,
                        args.cold_start, args.cold_end, hf_name, args.resume,
                    )
                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    continue

    logger.info(f"\nSurface experiment complete. {cell_num} cells processed.")


if __name__ == "__main__":
    main()
