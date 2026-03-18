#!/usr/bin/env python3
"""Main experiment runner: self-contained entry point for Colab.

This script does EVERYTHING:
1. Loads model and tokenizer
2. Loads benchmark problems
3. Applies skip configuration
4. Generates with token budget
5. Extracts and evaluates answers
6. Saves per-problem results to JSONL (append-only, crash-safe)
7. Supports --resume to skip completed problems

Usage from Colab launcher:
    python src/experiments/run_experiment.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 \
        --skip_type ffn_only \
        --skip_pct 30 \
        --token_budget 2048 \
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

# Add project root to path (works from both repo root and src/experiments/)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from infrastructure.model_loader import load_model_and_tokenizer, resolve_model_name, get_model_info
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
    parser = argparse.ArgumentParser(description="Depth or Length? Experiment Runner")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (HF path or short name)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--quantize_4bit", action="store_true",
                        help="Use 4-bit quantization (for 14B scale test)")

    # Benchmark
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "mmlu_pro", "aime",
                                 "humaneval", "livecodebench"])
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N problems (for quick tests)")

    # Skip configuration
    parser.add_argument("--skip_type", type=str, default="none",
                        choices=["none", "ffn_only", "full_layer", "attention_only"])
    parser.add_argument("--skip_pct", type=float, default=0,
                        help="Percentage of eligible layers to skip (0-100)")
    parser.add_argument("--skip_strategy", type=str, default="middle",
                        choices=["middle", "uniform", "random"])
    parser.add_argument("--cold_start", type=int, default=4,
                        help="Number of initial layers to protect")
    parser.add_argument("--cold_end", type=int, default=4,
                        help="Number of final layers to protect")

    # Length control
    parser.add_argument("--token_budget", type=int, default=None,
                        help="Absolute token budget. None = unlimited.")

    # Generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy decoding")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory (e.g., /content/drive/MyDrive/depth_or_length)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing JSONL, skip completed problems")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model name
    hf_name = resolve_model_name(args.model)
    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Skip: {args.skip_type} @ {args.skip_pct}%")
    logger.info(f"Token budget: {args.token_budget or 'unlimited'}")

    # Generate experiment ID
    experiment_id = get_experiment_id(
        hf_name, args.benchmark, args.skip_type,
        args.skip_pct, args.token_budget, args.seed,
    )
    logger.info(f"Experiment ID: {experiment_id}")

    # Set up paths
    results_dir = os.path.join(args.output_dir, "results")
    logs_dir = os.path.join(args.output_dir, "logs")
    metadata_dir = os.path.join(args.output_dir, "metadata")
    debug_dir = os.path.join(args.output_dir, "debug")

    results_path = os.path.join(results_dir, f"{experiment_id}.jsonl")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # Set up file logging
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, f"{experiment_id}.log")
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # Load completed problems (for resume)
    completed = set()
    if args.resume:
        completed = load_completed(results_path)
        logger.info(f"Resuming: {len(completed)} problems already completed")

    # Determine skip layers
    try:
        arch = get_architecture(hf_name)
        num_layers = arch.num_layers
    except ValueError:
        # Unknown model, will determine from loaded model
        num_layers = None

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        hf_name,
        dtype=args.dtype,
        quantize_4bit=args.quantize_4bit,
    )
    logger.info("Model loaded successfully")

    # Get num_layers if not already known
    if num_layers is None:
        from infrastructure.model_loader import get_layer_modules
        num_layers = len(get_layer_modules(model))

    # Compute skip layers
    skip_layers = []
    if args.skip_type != "none" and args.skip_pct > 0:
        skip_layers = get_skip_layers(
            num_layers, args.skip_pct,
            cold_start=args.cold_start, cold_end=args.cold_end,
            strategy=args.skip_strategy,
        )
        logger.info(f"Skipping layers: {skip_layers} ({len(skip_layers)}/{num_layers})")

    # Compute FLOPs
    flop_info = {}
    try:
        flop_info = compute_total_flops(hf_name, args.skip_type, skip_layers)
        logger.info(f"FLOP reduction: {flop_info.get('flop_reduction_pct', 0):.1f}%")
    except ValueError:
        logger.warning("Could not compute FLOPs (model not in registry)")

    # Save run metadata
    config = vars(args).copy()
    config["skip_layers"] = skip_layers
    config["flop_info"] = flop_info
    save_run_metadata(
        os.path.join(metadata_dir, f"{experiment_id}.json"),
        experiment_id, config,
    )

    # Load benchmark
    logger.info(f"Loading benchmark: {args.benchmark}")
    problems = load_benchmark(args.benchmark, subsample=args.subsample, seed=args.seed)
    logger.info(f"Loaded {len(problems)} problems")

    # Determine benchmark type for answer extraction
    benchmark_type_map = {
        "math500": "math",
        "gpqa": "gpqa",
        "mmlu_pro": "mmlu_pro",
        "aime": "aime",
        "humaneval": "humaneval",
        "livecodebench": "livecodebench",
    }
    bench_type = benchmark_type_map.get(args.benchmark, "math")

    # Run experiment
    correct = 0
    total = 0
    last_problem_id = None

    try:
        for i, problem in enumerate(problems):
            pid = problem["problem_id"]

            # Skip completed
            if pid in completed:
                continue

            logger.info(f"[{i+1}/{len(problems)}] {pid}")

            # Generate with skip applied
            if skip_layers and args.skip_type != "none":
                with apply_skip(model, args.skip_type, skip_layers):
                    gen_result = generate_with_budget(
                        model, tokenizer,
                        prompt=problem["prompt"],
                        token_budget=args.token_budget,
                        benchmark_type=bench_type,
                        seed=args.seed,
                    )
            else:
                gen_result = generate_with_budget(
                    model, tokenizer,
                    prompt=problem["prompt"],
                    token_budget=args.token_budget,
                    benchmark_type=bench_type,
                    seed=args.seed,
                )

            # Extract and check answer
            extracted = extract_answer(gen_result["generation_text"], bench_type)
            is_correct = check_answer_correct(
                extracted, problem["ground_truth"], bench_type
            )
            accuracy = 1 if is_correct else 0

            # HORL analysis (only for full-model, full-length runs)
            horl_pos = None
            if args.skip_type == "none" and args.token_budget is None:
                horl_pos = find_horl(
                    gen_result["generation_text"],
                    problem["ground_truth"],
                    bench_type,
                )

            # Build result record
            record = {
                "problem_id": pid,
                "model": hf_name,
                "benchmark": args.benchmark,
                "skip_type": args.skip_type,
                "ffn_skip_pct": args.skip_pct,
                "skip_layers": skip_layers,
                "token_budget": args.token_budget,
                "seed": args.seed,
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

            # Try to add git SHA
            try:
                import subprocess
                record["git_sha"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                record["git_sha"] = "unknown"

            # GPU name
            if torch.cuda.is_available():
                record["gpu_name"] = torch.cuda.get_device_name(0)

            # Save immediately (crash-safe)
            append_result(results_path, record)
            last_problem_id = pid

            # Update running stats
            total += 1
            correct += accuracy
            running_acc = correct / total * 100

            logger.info(
                f"  {'✓' if accuracy else '✗'} | "
                f"Extracted: {extracted} | GT: {problem['ground_truth']} | "
                f"Tokens: {gen_result['actual_tokens_generated']} | "
                f"Hit budget: {gen_result['hit_budget']} | "
                f"Time: {gen_result['wall_clock_seconds']:.1f}s | "
                f"Running acc: {running_acc:.1f}%"
            )

    except Exception as e:
        logger.error(f"CRASH: {str(e)}")
        save_crash_log(debug_dir, experiment_id, e, last_problem_id)
        raise

    # Final summary
    if total > 0:
        final_acc = correct / total * 100
        logger.info(f"\n{'='*60}")
        logger.info(f"DONE: {experiment_id}")
        logger.info(f"Accuracy: {correct}/{total} = {final_acc:.1f}%")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'='*60}")
    else:
        logger.info("No new problems to run (all completed)")


if __name__ == "__main__":
    main()
