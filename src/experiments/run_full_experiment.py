#!/usr/bin/env python3
"""Run all experiment configs in ONE model-loading session.

Single GPU session does:
1. Baseline inference with activation caching (norms per layer per token)
2. FFN-only skip at multiple FLOP levels
3. Full-layer skip at iso-FLOP matched levels
4. Attention-only skip at iso-FLOP matched levels

Each config saves to its own JSONL file with --resume support.
Model loads ONCE. No wasted credits.

Usage:
    python src/experiments/run_full_experiment.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 --subsample 100 \
        --output_dir /content/experiment_output \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any

import torch
import torch.nn as nn

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from experiments.configs import (
    CONFIGS, ExperimentConfig, append_result, load_completed_pairs,
)
from infrastructure.model_loader import load_model_and_tokenizer, resolve_model_name
from infrastructure.generation import extract_answer, check_answer_correct
from benchmarks.loader import load_benchmark
from depth_control.skip_manager import apply_skip, get_skip_layers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)




class ActivationCache:
    """Hooks into every layer to cache FFN/attention output norms."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: list = []
        self.layers = self._get_layers()
        self.num_layers = len(self.layers)
        self.ffn_output_norms: Dict[int, List[float]] = {}
        self.attn_output_norms: Dict[int, List[float]] = {}
        self.residual_pre_norms: Dict[int, List[float]] = {}
        self.residual_post_norms: Dict[int, List[float]] = {}

    def _get_layers(self) -> list:
        """Get transformer layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        raise ValueError(f"Cannot find layers in {type(self.model)}")

    def reset(self) -> None:
        """Clear cached data for next problem."""
        for idx in range(self.num_layers):
            self.ffn_output_norms[idx] = []
            self.attn_output_norms[idx] = []
            self.residual_pre_norms[idx] = []
            self.residual_post_norms[idx] = []

    def install_hooks(self) -> None:
        """Register forward hooks on all layers."""
        for idx, layer in enumerate(self.layers):
            self.ffn_output_norms[idx] = []
            self.attn_output_norms[idx] = []
            self.residual_pre_norms[idx] = []
            self.residual_post_norms[idx] = []

            mlp = self._find_sub(layer, ['mlp', 'feed_forward', 'ffn'])
            self.hooks.append(
                mlp.register_forward_hook(self._make_norm_hook(self.ffn_output_norms, idx))
            )

            attn = self._find_sub(layer, ['self_attn', 'attention', 'attn'])
            self.hooks.append(
                attn.register_forward_hook(self._make_norm_hook(self.attn_output_norms, idx))
            )

            self.hooks.append(
                layer.register_forward_pre_hook(self._make_pre_hook(idx))
            )
            self.hooks.append(
                layer.register_forward_hook(self._make_post_hook(idx))
            )

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @staticmethod
    def _find_sub(layer: nn.Module, attrs: list) -> nn.Module:
        """Find submodule by attribute name."""
        for attr in attrs:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(f"Cannot find {attrs} in {type(layer)}")

    def _make_norm_hook(self, storage: dict, layer_idx: int):
        """Hook that records output norm of a sublayer."""
        def hook(module, args, output):
            with torch.no_grad():
                out = output[0] if isinstance(output, tuple) else output
                while isinstance(out, tuple):
                    out = out[0]
                norm = out[0, -1, :].float().norm().item()
                storage[layer_idx].append(norm)
            return output
        return hook

    def _make_pre_hook(self, layer_idx: int):
        """Record residual norm before layer."""
        def hook(module, args):
            with torch.no_grad():
                hidden = args[0] if isinstance(args, tuple) else args
                while isinstance(hidden, tuple):
                    hidden = hidden[0]
                norm = hidden[0, -1, :].float().norm().item()
                self.residual_pre_norms[layer_idx].append(norm)
        return hook

    def _make_post_hook(self, layer_idx: int):
        """Record residual norm after layer."""
        def hook(module, args, output):
            with torch.no_grad():
                hidden = output[0] if isinstance(output, tuple) else output
                while isinstance(hidden, tuple):
                    hidden = hidden[0]
                norm = hidden[0, -1, :].float().norm().item()
                self.residual_post_norms[layer_idx].append(norm)
            return output
        return hook

    def get_cached_data(self) -> Dict[str, Any]:
        """Return cached norms as serializable dict."""
        return {
            "num_layers": self.num_layers,
            "ffn_output_norms": {str(k): v for k, v in self.ffn_output_norms.items()},
            "attn_output_norms": {str(k): v for k, v in self.attn_output_norms.items()},
            "residual_pre_norms": {str(k): v for k, v in self.residual_pre_norms.items()},
            "residual_post_norms": {str(k): v for k, v in self.residual_post_norms.items()},
        }


def _tokenize_prompts(
    tokenizer, prompts: List[str], device: torch.device,
) -> dict:
    """Tokenize a batch of prompts with left-padding for batched generation."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_input_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        all_input_ids.append(ids.squeeze(0))

    # Left-pad to same length
    max_len = max(ids.shape[0] for ids in all_input_ids)
    padded_ids = []
    attention_masks = []
    input_lengths = []
    for ids in all_input_ids:
        pad_len = max_len - ids.shape[0]
        input_lengths.append(ids.shape[0])
        padded = torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype),
            ids,
        ])
        mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(ids.shape[0], dtype=torch.long),
        ])
        padded_ids.append(padded)
        attention_masks.append(mask)

    return {
        "input_ids": torch.stack(padded_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "input_lengths": input_lengths,
    }


def run_batch(
    model, tokenizer, prompts: List[str],
    config: ExperimentConfig,
    num_layers: int,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
) -> List[Dict[str, Any]]:
    """Run batched inference on multiple prompts with the same config.

    Returns:
        List of result dicts, one per prompt.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = next(model.parameters()).device
    tokenized = _tokenize_prompts(tokenizer, prompts, device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    input_lengths = tokenized["input_lengths"]

    gen_kwargs = {}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    start_time = time.time()

    if config.is_baseline:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **gen_kwargs, pad_token_id=tokenizer.pad_token_id,
            )
    else:
        skip_layers = get_skip_layers(
            num_layers, config.skip_pct,
            cold_start=4, cold_end=4, strategy="middle",
        )
        with apply_skip(model, config.skip_type, skip_layers):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs, pad_token_id=tokenizer.pad_token_id,
                )

    wall_clock = time.time() - start_time
    total_padded_input = input_ids.shape[1]

    results = []
    for i, inp_len in enumerate(input_lengths):
        # Account for left-padding: actual content starts at (total - inp_len)
        gen_start = total_padded_input
        gen_ids = outputs[i][gen_start:]
        # Strip padding tokens from generated output
        gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
        actual_tokens = len(gen_ids)
        generation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        result = {
            "generation_text": generation_text,
            "actual_tokens_generated": actual_tokens,
            "wall_clock_seconds": round(wall_clock / len(prompts), 3),
            "batch_wall_clock_seconds": round(wall_clock, 3),
            "batch_size": len(prompts),
        }
        if not config.is_baseline:
            result["skip_layers"] = get_skip_layers(
                num_layers, config.skip_pct,
                cold_start=4, cold_end=4, strategy="middle",
            )
        results.append(result)

    return results




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full decomposition experiment in one session"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "mmlu_pro", "aime"])
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Run only these configs (by name). Default: all."
    )
    return parser.parse_args()


def main():
    """Run all configs interleaved: for each problem, run all configs.

    This ensures that even if the VM dies after N problems, we have
    N data points for EVERY config — enough to see signal early.
    """
    args = parse_args()
    hf_name = resolve_model_name(args.model)
    model_short = hf_name.split("/")[-1].lower().replace("-", "_")

    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark} (n={args.subsample})")
    logger.info(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa",
        "mmlu_pro": "mmlu_pro", "aime": "aime",
    }
    bench_type = bench_type_map.get(args.benchmark, "math")

    # Load model ONCE
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(hf_name, attn_implementation="sdpa")
    logger.info("Model loaded")

    # Get layer count
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        raise ValueError("Cannot determine layer count")
    logger.info(f"Layers: {num_layers}")

    # Activation cache created lazily — hooks installed only for baseline
    act_cache = ActivationCache(model)
    logger.info(f"Activation cache ready ({num_layers} layers)")

    # Load benchmark
    problems = load_benchmark(
        args.benchmark, subsample=args.subsample, seed=args.seed
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Filter configs if specified
    configs_to_run = list(CONFIGS)
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c.name in args.configs]
    logger.info(f"Configs ({len(configs_to_run)}): {[c.name for c in configs_to_run]}")

    # Load completed (problem_id, config) pairs for resume
    completed = set()
    if args.resume:
        completed = load_completed_pairs(args.output_dir)
        logger.info(f"Resuming: {len(completed)} (problem, config) pairs already done")

    batch_size = args.batch_size

    # Track per-config accuracy
    config_correct: Dict[str, int] = {c.name: 0 for c in configs_to_run}
    config_total: Dict[str, int] = {c.name: 0 for c in configs_to_run}

    # BATCHED: outer loop = configs, inner loop = batches of problems
    total_start = time.time()
    for c_idx, config in enumerate(configs_to_run):
        # Collect problems not yet done for this config
        todo = [
            p for p in problems
            if (p["problem_id"], config.name) not in completed
        ]
        if not todo:
            logger.info(f"\n[{config.name}] all done, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(
            f"CONFIG [{c_idx+1}/{len(configs_to_run)}]: {config.name} "
            f"({len(todo)} problems remaining, batch_size={batch_size})"
        )
        logger.info(f"{'='*60}")

        # Set up skip layers once per config
        skip_layers = None
        if not config.is_baseline:
            skip_layers = get_skip_layers(
                num_layers, config.skip_pct,
                cold_start=4, cold_end=4, strategy="middle",
            )

        # Process in batches
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start:batch_start + batch_size]
            batch_prompts = [p["prompt"] for p in batch]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(todo) + batch_size - 1) // batch_size

            logger.info(
                f"  Batch {batch_num}/{total_batches} "
                f"({len(batch)} problems)..."
            )

            try:
                results = run_batch(
                    model, tokenizer, batch_prompts, config,
                    num_layers, seed=args.seed,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                logger.error(f"    BATCH CRASH on {config.name}: {e}")
                continue

            # Process each result in the batch
            for i, (problem, result) in enumerate(zip(batch, results)):
                pid = problem["problem_id"]
                extracted = extract_answer(
                    result["generation_text"], bench_type
                )
                is_correct = check_answer_correct(
                    extracted, problem["ground_truth"], bench_type
                )
                accuracy = 1 if is_correct else 0
                config_total[config.name] += 1
                config_correct[config.name] += accuracy

                record = {
                    "problem_id": pid,
                    "config": config.name,
                    "model": hf_name,
                    "benchmark": args.benchmark,
                    "skip_type": config.skip_type,
                    "skip_pct": config.skip_pct,
                    "flop_reduction_pct": config.flop_reduction_pct,
                    "accuracy": accuracy,
                    "extracted_answer": extracted,
                    "ground_truth": problem["ground_truth"],
                    "actual_tokens_generated":
                        result["actual_tokens_generated"],
                    "wall_clock_seconds": result["wall_clock_seconds"],
                    "batch_wall_clock_seconds":
                        result.get("batch_wall_clock_seconds"),
                    "batch_size": result.get("batch_size"),
                    "generation_text": result["generation_text"],
                    "seed": args.seed,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                if "skip_layers" in result:
                    record["skip_layers"] = result["skip_layers"]

                append_result(
                    args.output_dir, model_short, args.benchmark,
                    config, record,
                )

                running_acc = (
                    config_correct[config.name]
                    / config_total[config.name] * 100
                )
                logger.info(
                    f"    {'✓' if accuracy else '✗'} {pid} | "
                    f"Ans: {extracted} | "
                    f"GT: {problem['ground_truth']} | "
                    f"Tok: {result['actual_tokens_generated']} | "
                    f"Time: {result['wall_clock_seconds']:.1f}s | "
                    f"Acc: {running_acc:.1f}%"
                )

        # Summary after each config
        elapsed = time.time() - total_start
        t = config_total[config.name]
        if t > 0:
            acc = config_correct[config.name] / t * 100
            logger.info(
                f"\n  {config.name}: {acc:.1f}% "
                f"({t} done, {elapsed/60:.1f} min elapsed)"
            )

    # Cleanup
    act_cache.remove_hooks()
    total_elapsed = time.time() - total_start
    logger.info(f"\nALL DONE in {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
