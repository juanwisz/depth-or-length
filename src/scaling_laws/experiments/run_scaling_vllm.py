#!/usr/bin/env python3
"""Scaling laws pilot with vLLM for fast inference.

Measures token counts and accuracy across model sizes. No skipping.
Uses vLLM offline batched inference for maximum throughput.

Usage from Colab:
    python src/scaling_laws/experiments/run_scaling_vllm.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --benchmark math500 \
        --subsample 200 \
        --num_samples 1 \
        --output_dir /content/drive/MyDrive/scaling_laws \
        --resume

    # k=4 with temperature sampling:
    python src/scaling_laws/experiments/run_scaling_vllm.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
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
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))


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


def parse_args() -> tuple:
    """Parse CLI args. Returns (ScalingConfig, output_dir, resume)."""
    parser = argparse.ArgumentParser(description="Scaling Laws vLLM Experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "aime", "mmlu_pro"])
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Samples per problem (k for majority voting)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0=greedy, 0.6=sampling")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_samples > 1 and args.temperature == 0.0:
        args.temperature = 0.6
        logger.info("Forcing temperature=0.6 for k>1")

    config = ScalingConfig(
        model_id=args.model,
        benchmark=args.benchmark,
        subsample=args.subsample,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    return config, args.output_dir, args.resume


def load_completed_samples(results_path: str) -> set[str]:
    """Load completed (problem_id, sample_idx) keys from JSONL."""
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


def extract_math_answer(text: str) -> str:
    """Extract boxed answer from math reasoning trace."""
    boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Fallback: look for "answer is X" pattern
    match = re.search(r'(?:answer|result)\s+(?:is|=)\s+[\\$]*([^\s,.$]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_mcq_answer(text: str) -> str:
    """Extract letter answer from MCQ reasoning trace."""
    match = re.search(r'(?:answer is|answer:)\s*\(?([A-D])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: last standalone letter
    letters = re.findall(r'\b([A-D])\b', text[-200:])
    if letters:
        return letters[-1].upper()
    return ""


def check_math_correct(extracted: str, ground_truth: str) -> bool:
    """Check if extracted math answer matches ground truth."""
    if not extracted:
        return False
    # Normalize
    e = extracted.strip().rstrip('.').replace(' ', '')
    g = ground_truth.strip().rstrip('.').replace(' ', '')
    if e == g:
        return True
    # Try numeric comparison
    try:
        return abs(float(e) - float(g)) < 1e-6
    except (ValueError, OverflowError):
        return False


def check_mcq_correct(extracted: str, ground_truth: str) -> bool:
    """Check MCQ answer."""
    return extracted.strip().upper() == ground_truth.strip().upper()


def main() -> None:
    config, output_dir, resume = parse_args()

    logger.info(f"Config: {asdict(config)}")

    # Set up paths
    results_dir = os.path.join(output_dir, "results")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f"{config.experiment_id}.jsonl")

    fh = logging.FileHandler(os.path.join(logs_dir, f"{config.experiment_id}.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    # Resume
    completed = set()
    if resume:
        completed = load_completed_samples(results_path)
        logger.info(f"Resuming: {len(completed)} samples already done")

    # Load benchmark
    from benchmarks.loader import load_benchmark
    problems = load_benchmark(config.benchmark, subsample=config.subsample, seed=config.seed)
    logger.info(f"Loaded {len(problems)} problems")

    is_mcq = config.benchmark in ("gpqa", "mmlu_pro")
    extract_fn = extract_mcq_answer if is_mcq else extract_math_answer
    check_fn = check_mcq_correct if is_mcq else check_math_correct

    # Build request list: (problem_idx, sample_idx, prompt, seed)
    requests = []
    for pi, problem in enumerate(problems):
        for si in range(config.num_samples):
            key = f"{problem['problem_id']}_s{si}"
            if key not in completed:
                sample_seed = config.seed + si * 1000
                requests.append((pi, si, problem["prompt"], sample_seed))

    logger.info(f"Total requests to run: {len(requests)} (skipped {len(completed)} completed)")

    if not requests:
        logger.info("Nothing to do — all samples completed")
        return

    # Load vLLM
    logger.info(f"Loading vLLM model: {config.model_id}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=config.model_id,
        dtype="auto",
        max_model_len=config.max_new_tokens + 2048,  # input + output
        seed=config.seed,
        trust_remote_code=True,
        enforce_eager=False,
    )

    # Get param count from model config (avoids vLLM internal API changes)
    PARAM_COUNTS = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 1_500_000_000,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 7_000_000_000,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 14_000_000_000,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 32_000_000_000,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 8_000_000_000,
        "Qwen/Qwen3-1.7B": 1_700_000_000,
        "Qwen/Qwen3-8B": 8_000_000_000,
        "Qwen/Qwen3-14B": 14_000_000_000,
        "Qwen/Qwen3-32B": 32_000_000_000,
        "Qwen/Qwen2.5-1.5B-Instruct": 1_500_000_000,
        "Qwen/Qwen2.5-7B-Instruct": 7_000_000_000,
        "Qwen/Qwen2.5-14B-Instruct": 14_000_000_000,
        "Qwen/Qwen2.5-32B-Instruct": 32_000_000_000,
    }
    param_count = PARAM_COUNTS.get(config.model_id, 0)
    if param_count == 0:
        logger.warning(f"Unknown model {config.model_id}, using 0 for param_count")
    logger.info(f"Parameters: {param_count:,} ({param_count/1e9:.1f}B)")

    # Save metadata
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{config.experiment_id}.json"), "w") as f:
        json.dump({
            "config": asdict(config),
            "param_count": param_count,
            "num_problems": len(problems),
            "total_requests": len(requests),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    # Process in batches to save incrementally
    BATCH_SIZE = 50
    total_correct = 0
    total_done = 0
    t_start = time.time()

    for batch_start in range(0, len(requests), BATCH_SIZE):
        batch = requests[batch_start:batch_start + BATCH_SIZE]

        # Build prompts with chat template
        tokenizer = llm.get_tokenizer()
        formatted_prompts = []
        for _, _, prompt, _ in batch:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)

        # Generate
        sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 0.0,
            top_p=config.top_p if config.temperature > 0 else 1.0,
            max_tokens=config.max_new_tokens,
            seed=batch[0][3] if config.temperature == 0.0 else None,
        )

        t_batch = time.time()
        outputs = llm.generate(formatted_prompts, sampling_params)
        batch_time = time.time() - t_batch

        # Process results
        for req_idx, (pi, si, prompt, sample_seed) in enumerate(batch):
            output = outputs[req_idx]
            gen_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            sample_flops = 2.0 * param_count * tokens_generated

            problem = problems[pi]
            extracted = extract_fn(gen_text)
            is_correct = check_fn(extracted, problem["ground_truth"])

            record = {
                "problem_id": problem["problem_id"],
                "sample_idx": si,
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
                "hit_budget": tokens_generated >= config.max_new_tokens - 10,
                "wall_clock_seconds": batch_time / len(batch),  # avg per sample
                "generation_text": gen_text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            # Append to JSONL immediately
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            total_done += 1
            total_correct += record["accuracy"]

        running_acc = total_correct / total_done * 100
        elapsed = time.time() - t_start
        rate = total_done / elapsed
        eta = (len(requests) - total_done) / rate if rate > 0 else 0

        logger.info(
            f"Batch {batch_start//BATCH_SIZE + 1}: "
            f"{total_done}/{len(requests)} done | "
            f"acc={running_acc:.1f}% | "
            f"rate={rate:.1f} samples/s | "
            f"ETA={eta/60:.0f}min"
        )

    elapsed_total = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {config.experiment_id}")
    logger.info(f"Total samples: {total_done}")
    logger.info(f"Accuracy: {total_correct}/{total_done} = {total_correct/total_done*100:.1f}%")
    logger.info(f"Total time: {elapsed_total/60:.1f} min")
    logger.info(f"Results: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
