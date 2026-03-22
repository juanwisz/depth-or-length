#!/usr/bin/env python3
"""Scaling laws pilot: measure token counts and accuracy across model sizes.

No skipping — just normal inference. The goal is to measure how many tokens
each model size generates per problem and whether smaller models inflate.

Usage from Colab:
    python src/scaling_laws/experiments/run_scaling_pilot.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --benchmark math500 \
        --subsample 200 \
        --num_samples 1 \
        --output_dir /content/drive/MyDrive/scaling_laws \
        --resume

    # Then run k=4 with temperature sampling:
    python src/scaling_laws/experiments/run_scaling_pilot.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --benchmark math500 \
        --subsample 200 \
        --num_samples 4 \
        --temperature 0.6 \
        --output_dir /content/drive/MyDrive/scaling_laws \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from infrastructure.model_loader import load_model_and_tokenizer, resolve_model_name
from infrastructure.generation import (
    generate_with_budget, extract_answer, check_answer_correct,
)
from infrastructure.checkpoint import load_completed, append_result
from benchmarks.loader import load_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingConfig:
    """Configuration for a scaling law experiment run."""
    model_id: str
    benchmark: str
    subsample: Optional[int]
    num_samples: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int

    @property
    def experiment_id(self) -> str:
        model_short = self.model_id.split('/')[-1]
        sub_str = f"_sub{self.subsample}" if self.subsample else ""
        return f"scaling_{model_short}_{self.benchmark}{sub_str}_k{self.num_samples}_t{self.temperature}"


def parse_args() -> ScalingConfig:
    """Parse command line arguments into a ScalingConfig."""
    parser = argparse.ArgumentParser(description="Scaling Laws Pilot Experiment")

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "aime", "mmlu_pro"])
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N problems")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem (k for majority voting)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0.0=greedy, 0.6=DeepSeek default")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="Max tokens to generate per sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # For k>1, force temperature > 0
    if args.num_samples > 1 and args.temperature == 0.0:
        args.temperature = 0.6
        logger.info("Forcing temperature=0.6 for k>1 sampling")

    return ScalingConfig(
        model_id=resolve_model_name(args.model),
        benchmark=args.benchmark,
        subsample=args.subsample,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    ), args.output_dir, args.resume


def get_param_count(model: torch.nn.Module) -> int:
    """Get total parameter count of a model."""
    return sum(p.numel() for p in model.parameters())


def compute_sample_flops(param_count: int, num_tokens: int) -> float:
    """Compute FLOPs for a single forward pass.

    Approximation: 2 * N_params * num_tokens (standard transformer FLOP estimate).
    """
    return 2.0 * param_count * num_tokens


def main() -> None:
    config, output_dir, resume = parse_args()

    logger.info(f"Model: {config.model_id}")
    logger.info(f"Benchmark: {config.benchmark} (subsample={config.subsample})")
    logger.info(f"Samples per problem: {config.num_samples}")
    logger.info(f"Temperature: {config.temperature}")

    # Set up paths
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config.experiment_id}.jsonl")

    # File logging
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, f"{config.experiment_id}.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    # Resume
    completed = set()
    if resume:
        completed = _load_completed_samples(results_path)
        logger.info(f"Resuming: {len(completed)} (problem, sample_idx) pairs already done")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config.model_id, dtype="auto")
    param_count = get_param_count(model)
    logger.info(f"Model loaded. Parameters: {param_count:,} ({param_count/1e9:.1f}B)")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load benchmark
    problems = load_benchmark(config.benchmark, subsample=config.subsample, seed=config.seed)
    logger.info(f"Loaded {len(problems)} problems")

    benchmark_type_map = {
        "math500": "math", "gpqa": "gpqa", "aime": "aime", "mmlu_pro": "mmlu_pro",
    }
    bench_type = benchmark_type_map.get(config.benchmark, "math")

    # Save run metadata
    meta_path = os.path.join(output_dir, "metadata", f"{config.experiment_id}.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "param_count": param_count,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "num_problems": len(problems),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    # Run
    total_correct = 0
    total_samples = 0

    try:
        for i, problem in enumerate(problems):
            pid = problem["problem_id"]

            for sample_idx in range(config.num_samples):
                sample_key = f"{pid}_s{sample_idx}"
                if sample_key in completed:
                    continue

                # Vary seed per sample for diverse generations
                sample_seed = config.seed + sample_idx * 1000

                gen_result = generate_with_budget(
                    model, tokenizer,
                    prompt=problem["prompt"],
                    token_budget=config.max_new_tokens,
                    benchmark_type=bench_type,
                    seed=sample_seed,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

                extracted = extract_answer(gen_result["generation_text"], bench_type)
                is_correct = check_answer_correct(
                    extracted, problem["ground_truth"], bench_type
                )

                tokens_generated = gen_result["actual_tokens_generated"]
                sample_flops = compute_sample_flops(param_count, tokens_generated)

                record = {
                    "problem_id": pid,
                    "sample_idx": sample_idx,
                    "model": config.model_id,
                    "param_count": param_count,
                    "benchmark": config.benchmark,
                    "num_samples_config": config.num_samples,
                    "temperature": config.temperature,
                    "seed": sample_seed,
                    "accuracy": 1 if is_correct else 0,
                    "extracted_answer": extracted,
                    "ground_truth": problem["ground_truth"],
                    "tokens_generated": tokens_generated,
                    "sample_flops": sample_flops,
                    "hit_budget": gen_result["hit_budget"],
                    "wall_clock_seconds": gen_result["wall_clock_seconds"],
                    "generation_text": gen_result["generation_text"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                if torch.cuda.is_available():
                    record["gpu_name"] = torch.cuda.get_device_name(0)

                append_result(results_path, record)

                total_samples += 1
                total_correct += record["accuracy"]
                running_acc = total_correct / total_samples * 100

                logger.info(
                    f"[{i+1}/{len(problems)}] s{sample_idx} | "
                    f"{'Y' if is_correct else 'N'} | "
                    f"tokens={tokens_generated:,} | "
                    f"flops={sample_flops:.2e} | "
                    f"time={gen_result['wall_clock_seconds']:.1f}s | "
                    f"acc={running_acc:.1f}%"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"CRASH: {e}")
        raise

    if total_samples > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"DONE: {config.experiment_id}")
        logger.info(f"Samples: {total_samples}, Accuracy: {total_correct/total_samples*100:.1f}%")
        logger.info(f"Results: {results_path}")
        logger.info(f"{'='*60}")


def _load_completed_samples(results_path: str) -> set[str]:
    """Load completed (problem_id, sample_idx) pairs from JSONL."""
    completed = set()
    if not os.path.exists(results_path):
        return completed
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                completed.add(f"{r['problem_id']}_s{r.get('sample_idx', 0)}")
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


if __name__ == "__main__":
    main()
