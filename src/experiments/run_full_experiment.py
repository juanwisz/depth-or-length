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
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

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


@dataclass(frozen=True)
class ExperimentConfig:
    """One experimental condition to run."""

    name: str
    skip_type: Optional[str]  # None for baseline
    skip_pct: float  # 0 for baseline
    flop_reduction_pct: float  # estimated FLOP savings

    @property
    def is_baseline(self) -> bool:
        return self.skip_type is None


# FFN ≈ 2/3 of layer compute, attention ≈ 1/3.
# For 28 layers (DeepSeek-R1-Distill-Qwen-7B):
# - skip_pct=25 means skip 5 of 20 eligible layers (cold_start=4, cold_end=4)
# - FFN-only skip 5 layers: saves 5/28 * 2/3 ≈ 11.9% FLOPs
# - Full-layer skip 5 layers: saves 5/28 ≈ 17.9% FLOPs
# - To match 11.9% FLOPs with full-layer: skip ~3.3 layers → skip_pct≈16.7
#
# Iso-FLOP matching table (for 28-layer model, 20 eligible):
# FFN-only skip_pct → full-layer skip_pct at same FLOPs
# FFN 25% (5 layers, 11.9% FLOP) ↔ Full 16.7% (3.3 layers, 11.9%)
# FFN 50% (10 layers, 23.8% FLOP) ↔ Full 33.3% (6.7 layers, 23.8%)
# FFN 75% (15 layers, 35.7% FLOP) ↔ Full 50% (10 layers, 35.7%)
#
# Attention-only at same FLOPs: attn is 1/3 of layer
# FFN 25% (11.9% FLOP) ↔ Attn 50% (10 layers, 11.9%)
# FFN 50% (23.8% FLOP) ↔ Attn 100% (20 layers, 23.8%)

CONFIGS = [
    # Baseline with activation caching
    ExperimentConfig("baseline", None, 0, 0),
    # FFN-only skip at 3 levels
    ExperimentConfig("ffn_skip_25", "ffn_only", 25, 11.9),
    ExperimentConfig("ffn_skip_50", "ffn_only", 50, 23.8),
    ExperimentConfig("ffn_skip_75", "ffn_only", 75, 35.7),
    # Full-layer skip at ISO-FLOP matched levels
    ExperimentConfig("full_skip_isoflop_12", "full_layer", 16.7, 11.9),
    ExperimentConfig("full_skip_isoflop_24", "full_layer", 33.3, 23.8),
    ExperimentConfig("full_skip_isoflop_36", "full_layer", 50, 35.7),
    # Attention-only skip at ISO-FLOP matched levels
    ExperimentConfig("attn_skip_isoflop_12", "attention_only", 50, 11.9),
    ExperimentConfig("attn_skip_isoflop_24", "attention_only", 100, 23.8),
]


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
                norm = out[0, -1, :].float().norm().item()
                storage[layer_idx].append(norm)
            return output
        return hook

    def _make_pre_hook(self, layer_idx: int):
        """Record residual norm before layer."""
        def hook(module, args):
            with torch.no_grad():
                hidden = args[0] if isinstance(args, tuple) else args
                norm = hidden[0, -1, :].float().norm().item()
                self.residual_pre_norms[layer_idx].append(norm)
        return hook

    def _make_post_hook(self, layer_idx: int):
        """Record residual norm after layer."""
        def hook(module, args, output):
            with torch.no_grad():
                hidden = output[0] if isinstance(output, tuple) else output
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


def run_single_problem(
    model, tokenizer, prompt: str,
    config: ExperimentConfig,
    num_layers: int,
    cache: Optional[ActivationCache] = None,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
) -> Dict[str, Any]:
    """Run inference on one problem with optional skip and caching.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        prompt: Problem prompt.
        config: Experiment configuration.
        num_layers: Total transformer layers.
        cache: ActivationCache for baseline runs (None for skip runs).
        seed: Random seed.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        max_new_tokens: Max generation length.

    Returns:
        Dict with generation_text, tokens, wall_clock, optional activations.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = next(model.parameters()).device

    if cache is not None:
        cache.reset()

    # Tokenize
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        else:
            raise AttributeError("no chat template")
    except Exception:
        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}

    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    start_time = time.time()

    # Apply skip if needed
    if config.is_baseline:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
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
                    **inputs, max_new_tokens=max_new_tokens,
                    **gen_kwargs, pad_token_id=tokenizer.pad_token_id,
                )

    gen_ids = outputs[0][input_len:]
    actual_tokens = len(gen_ids)
    generation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    wall_clock = time.time() - start_time

    result = {
        "generation_text": generation_text,
        "actual_tokens_generated": actual_tokens,
        "wall_clock_seconds": round(wall_clock, 3),
    }

    if cache is not None:
        result["activations"] = cache.get_cached_data()

    if not config.is_baseline:
        result["skip_layers"] = get_skip_layers(
            num_layers, config.skip_pct,
            cold_start=4, cold_end=4, strategy="middle",
        )

    return result


def load_completed_ids(path: str) -> set:
    """Load completed problem IDs from JSONL file."""
    completed = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def run_config(
    model, tokenizer, problems: list,
    config: ExperimentConfig,
    num_layers: int,
    output_dir: str,
    model_name: str,
    benchmark: str,
    cache: Optional[ActivationCache] = None,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
    resume: bool = True,
) -> None:
    """Run one config across all problems.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        problems: List of problem dicts.
        config: Experiment config to run.
        num_layers: Total transformer layers.
        output_dir: Base output directory.
        model_name: HuggingFace model name.
        benchmark: Benchmark name.
        cache: ActivationCache (only for baseline).
        seed: Random seed.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        max_new_tokens: Max generation tokens.
        resume: Whether to skip completed problems.
    """
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = os.path.join(
        output_dir, f"{model_short}__{benchmark}__{config.name}.jsonl"
    )

    completed = load_completed_ids(out_path) if resume else set()
    remaining = [p for p in problems if p["problem_id"] not in completed]

    if not remaining:
        logger.info(f"[{config.name}] All {len(problems)} problems done, skipping")
        return

    logger.info(
        f"[{config.name}] Running {len(remaining)}/{len(problems)} problems "
        f"(skip_type={config.skip_type}, skip_pct={config.skip_pct}, "
        f"flop_reduction≈{config.flop_reduction_pct}%)"
    )

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa",
        "mmlu_pro": "mmlu_pro", "aime": "aime",
    }
    bench_type = bench_type_map.get(benchmark, "math")

    correct = 0
    total = 0

    for i, problem in enumerate(remaining):
        pid = problem["problem_id"]
        logger.info(f"  [{i+1}/{len(remaining)}] {pid}")

        result = run_single_problem(
            model, tokenizer, problem["prompt"], config, num_layers,
            cache=cache if config.is_baseline else None,
            seed=seed, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        extracted = extract_answer(result["generation_text"], bench_type)
        is_correct = check_answer_correct(
            extracted, problem["ground_truth"], bench_type
        )
        accuracy = 1 if is_correct else 0
        total += 1
        correct += accuracy

        record = {
            "problem_id": pid,
            "config": config.name,
            "model": model_name,
            "benchmark": benchmark,
            "skip_type": config.skip_type,
            "skip_pct": config.skip_pct,
            "flop_reduction_pct": config.flop_reduction_pct,
            "accuracy": accuracy,
            "extracted_answer": extracted,
            "ground_truth": problem["ground_truth"],
            "actual_tokens_generated": result["actual_tokens_generated"],
            "wall_clock_seconds": result["wall_clock_seconds"],
            "generation_text": result["generation_text"],
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if "activations" in result:
            record["activations"] = result["activations"]
        if "skip_layers" in result:
            record["skip_layers"] = result["skip_layers"]

        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        running_acc = correct / total * 100
        logger.info(
            f"    {'✓' if accuracy else '✗'} | "
            f"Ans: {extracted} | GT: {problem['ground_truth']} | "
            f"Tok: {result['actual_tokens_generated']} | "
            f"Time: {result['wall_clock_seconds']:.1f}s | "
            f"Acc: {running_acc:.1f}%"
        )

    logger.info(
        f"[{config.name}] DONE: {correct}/{total} = "
        f"{correct/total*100:.1f}%"
    )


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Run only these configs (by name). Default: all."
    )
    return parser.parse_args()


def main():
    """Run all configs sequentially with one model load."""
    args = parse_args()
    hf_name = resolve_model_name(args.model)

    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark} (n={args.subsample})")
    logger.info(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

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

    # Install activation cache (used only for baseline)
    act_cache = ActivationCache(model)
    act_cache.install_hooks()
    logger.info(f"Activation cache hooks installed on {num_layers} layers")

    # Load benchmark
    problems = load_benchmark(
        args.benchmark, subsample=args.subsample, seed=args.seed
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Filter configs if specified
    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c.name in args.configs]
        logger.info(f"Running {len(configs_to_run)} configs: {[c.name for c in configs_to_run]}")

    # Run each config
    total_start = time.time()
    for config in configs_to_run:
        config_start = time.time()
        run_config(
            model, tokenizer, problems, config, num_layers,
            args.output_dir, hf_name, args.benchmark,
            cache=act_cache,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )
        elapsed = time.time() - config_start
        logger.info(f"[{config.name}] took {elapsed/60:.1f} min")

    # Cleanup
    act_cache.remove_hooks()
    total_elapsed = time.time() - total_start
    logger.info(f"\nALL CONFIGS DONE in {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
