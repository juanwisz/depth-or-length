#!/usr/bin/env python3
"""Cache per-layer activations during normal inference for post-hoc analysis.

Luciano's approach: run inference ONCE per problem with NO skipping, but hook
every layer to record FFN/attention contributions. Then simulate any skip
pattern offline without GPU.

Caches per generated token, per layer:
- FFN output L2 norm (scalar)
- Attention output L2 norm (scalar)
- Residual stream norm before/after each sublayer
- Attention entropy per head (scalar per head)

This is compact (~28 layers × N_tokens × ~35 floats ≈ ~100KB per problem)
and sufficient for:
- Identifying "dead" FFN layers (low contribution)
- Comparing FFN vs attention importance across layers
- Predicting which layers are safe to skip
- Understanding why full-layer skip destroys reasoning but FFN-only doesn't

Usage:
    python src/experiments/run_cache_activations.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 \
        --subsample 10 \
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

from infrastructure.model_loader import load_model_and_tokenizer, resolve_model_name
from infrastructure.generation import extract_answer, check_answer_correct
from infrastructure.checkpoint import load_completed, get_experiment_id
from benchmarks.loader import load_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class ActivationCache:
    """Hooks into every layer to cache FFN/attention output norms and entropy."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: list = []
        self.layers = self._get_layers()
        self.num_layers = len(self.layers)

        # Storage: populated during generation, reset per problem
        self.ffn_output_norms: Dict[int, List[float]] = {}
        self.attn_output_norms: Dict[int, List[float]] = {}
        self.residual_pre_attn_norms: Dict[int, List[float]] = {}
        self.residual_post_attn_norms: Dict[int, List[float]] = {}
        self.residual_post_ffn_norms: Dict[int, List[float]] = {}
        # Attention entropy per head: Dict[layer_idx, List[List[float]]]
        # outer list = tokens, inner list = heads
        self.attn_entropy: Dict[int, List[List[float]]] = {}

    def _get_layers(self) -> list:
        """Get transformer layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        raise ValueError(f"Cannot find layers in {type(self.model)}")

    def _find_mlp(self, layer: nn.Module) -> nn.Module:
        """Find MLP module."""
        for attr in ['mlp', 'feed_forward', 'ffn']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(f"Cannot find MLP in {type(layer)}")

    def _find_attn(self, layer: nn.Module) -> nn.Module:
        """Find attention module."""
        for attr in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(f"Cannot find attention in {type(layer)}")

    def reset(self) -> None:
        """Clear cached data for next problem."""
        for d in [self.ffn_output_norms, self.attn_output_norms,
                  self.residual_pre_attn_norms, self.residual_post_attn_norms,
                  self.residual_post_ffn_norms, self.attn_entropy]:
            d.clear()

    def install_hooks(self) -> None:
        """Register forward hooks on all layers."""
        for idx, layer in enumerate(self.layers):
            # Initialize storage
            self.ffn_output_norms[idx] = []
            self.attn_output_norms[idx] = []
            self.residual_pre_attn_norms[idx] = []
            self.residual_post_attn_norms[idx] = []
            self.residual_post_ffn_norms[idx] = []
            self.attn_entropy[idx] = []

            # Hook on MLP
            mlp = self._find_mlp(layer)
            self.hooks.append(
                mlp.register_forward_hook(self._make_mlp_hook(idx))
            )

            # Hook on attention — need output_attentions=True for entropy
            attn = self._find_attn(layer)
            self.hooks.append(
                attn.register_forward_hook(self._make_attn_hook(idx))
            )

            # Hook on the full layer to get residual norms
            self.hooks.append(
                layer.register_forward_pre_hook(self._make_layer_pre_hook(idx))
            )
            self.hooks.append(
                layer.register_forward_hook(self._make_layer_post_hook(idx))
            )

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_mlp_hook(self, layer_idx: int):
        """Hook that records FFN output norm."""
        def hook(module, args, output):
            # output shape: (batch, seq_len, hidden_size)
            # For autoregressive generation, seq_len=1 for each new token
            with torch.no_grad():
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                # L2 norm of the last token's FFN output
                norm = out[0, -1, :].float().norm().item()
                self.ffn_output_norms[layer_idx].append(norm)
            return output
        return hook

    def _make_attn_hook(self, layer_idx: int):
        """Hook that records attention output norm and entropy."""
        def hook(module, args, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    attn_out = output[0]
                    # attention weights: (batch, num_heads, seq_len, seq_len)
                    attn_weights = output[1] if len(output) > 1 else None
                else:
                    attn_out = output
                    attn_weights = None

                # Attention output norm
                norm = attn_out[0, -1, :].float().norm().item()
                self.attn_output_norms[layer_idx].append(norm)

                # Attention entropy per head
                if attn_weights is not None and attn_weights.numel() > 0:
                    # attn_weights: (1, num_heads, q_len, kv_len)
                    # Take last query token
                    w = attn_weights[0, :, -1, :].float()  # (num_heads, kv_len)
                    # Entropy: -sum(p * log(p)), clamp for numerical stability
                    w_clamped = w.clamp(min=1e-10)
                    entropy = -(w_clamped * w_clamped.log()).sum(dim=-1)  # (num_heads,)
                    self.attn_entropy[layer_idx].append(entropy.tolist())
                else:
                    self.attn_entropy[layer_idx].append([])
            return output
        return hook

    def _make_layer_pre_hook(self, layer_idx: int):
        """Record residual norm before layer."""
        def hook(module, args):
            with torch.no_grad():
                if isinstance(args, tuple):
                    hidden = args[0]
                else:
                    hidden = args
                norm = hidden[0, -1, :].float().norm().item()
                self.residual_pre_attn_norms[layer_idx].append(norm)
        return hook

    def _make_layer_post_hook(self, layer_idx: int):
        """Record residual norm after full layer."""
        def hook(module, args, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                norm = hidden[0, -1, :].float().norm().item()
                self.residual_post_ffn_norms[layer_idx].append(norm)
            return output
        return hook

    def get_cached_data(self) -> Dict[str, Any]:
        """Return all cached data as a serializable dict.

        Returns per-layer, per-token norms and entropy values.
        """
        return {
            "num_layers": self.num_layers,
            "ffn_output_norms": {str(k): v for k, v in self.ffn_output_norms.items()},
            "attn_output_norms": {str(k): v for k, v in self.attn_output_norms.items()},
            "residual_pre_attn_norms": {str(k): v for k, v in self.residual_pre_attn_norms.items()},
            "residual_post_ffn_norms": {str(k): v for k, v in self.residual_post_ffn_norms.items()},
            "attn_entropy": {str(k): v for k, v in self.attn_entropy.items()},
        }


def generate_with_cache(
    model, tokenizer, prompt: str, cache: ActivationCache,
    seed: int = 42, temperature: float = 0.6, top_p: float = 0.95,
    max_new_tokens: int = 32768,
) -> Dict[str, Any]:
    """Generate and cache activations simultaneously.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        cache: ActivationCache instance (hooks already installed).
        seed: Random seed.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        max_new_tokens: Max generation length.

    Returns:
        Dict with generation_text, tokens, wall_clock, cached activations.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = next(model.parameters()).device
    cache.reset()

    # Tokenize with chat template
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

    with torch.no_grad():
        # output_attentions=True on the MODEL config so each attention layer
        # returns weights in its forward() — hooks capture them.
        # We don't need generate() to accumulate them.
        model.config.output_attentions = True
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.config.output_attentions = False

    gen_ids = outputs[0][input_len:]
    actual_tokens = len(gen_ids)
    generation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    wall_clock = time.time() - start_time

    return {
        "generation_text": generation_text,
        "actual_tokens_generated": actual_tokens,
        "wall_clock_seconds": round(wall_clock, 3),
        "activations": cache.get_cached_data(),
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cache activations during inference for post-hoc analysis"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gpqa", "mmlu_pro", "aime"])
    parser.add_argument("--subsample", type=int, default=10,
                        help="Number of problems to run (default: 10)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    """Run caching inference on benchmark problems."""
    args = parse_args()
    hf_name = resolve_model_name(args.model)

    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark} (subsample={args.subsample})")

    # Paths
    cache_dir = os.path.join(args.output_dir, "cached_activations")
    os.makedirs(cache_dir, exist_ok=True)

    # Make experiment ID for the cache file
    model_short = hf_name.split("/")[-1].lower().replace("-", "_")
    cache_path = os.path.join(
        cache_dir,
        f"{model_short}__{args.benchmark}__n{args.subsample}__seed{args.seed}.jsonl"
    )

    # Resume support
    completed = set()
    if args.resume and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                completed.add(rec["problem_id"])
        logger.info(f"Resuming: {len(completed)} problems already cached")

    # Load model with eager attention (needed to get attention weights for entropy)
    # SDPA fuses attention and doesn't expose weights. Eager is ~20% slower
    # but gives us attention probabilities for entropy computation.
    logger.info("Loading model (eager attention for weight capture)...")
    model, tokenizer = load_model_and_tokenizer(
        hf_name, attn_implementation="eager"
    )
    logger.info("Model loaded")

    # Install activation cache hooks
    cache = ActivationCache(model)
    cache.install_hooks()
    logger.info(f"Installed hooks on {cache.num_layers} layers")

    # Load benchmark
    problems = load_benchmark(args.benchmark, subsample=args.subsample, seed=args.seed)
    logger.info(f"Loaded {len(problems)} problems")

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa",
        "mmlu_pro": "mmlu_pro", "aime": "aime",
    }
    bench_type = bench_type_map.get(args.benchmark, "math")

    correct = 0
    total = 0

    try:
        for i, problem in enumerate(problems):
            pid = problem["problem_id"]
            if pid in completed:
                continue

            logger.info(f"[{i+1}/{len(problems)}] {pid}")
            t0 = time.time()

            result = generate_with_cache(
                model, tokenizer, problem["prompt"], cache,
                seed=args.seed, temperature=args.temperature,
                top_p=args.top_p, max_new_tokens=args.max_new_tokens,
            )

            # Extract and check answer
            extracted = extract_answer(result["generation_text"], bench_type)
            is_correct = check_answer_correct(
                extracted, problem["ground_truth"], bench_type
            )
            accuracy = 1 if is_correct else 0
            total += 1
            correct += accuracy

            # Build record (activations included)
            record = {
                "problem_id": pid,
                "model": hf_name,
                "benchmark": args.benchmark,
                "accuracy": accuracy,
                "extracted_answer": extracted,
                "ground_truth": problem["ground_truth"],
                "actual_tokens_generated": result["actual_tokens_generated"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                "generation_text": result["generation_text"],
                "activations": result["activations"],
                "seed": args.seed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            # Append to JSONL (crash-safe)
            with open(cache_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            elapsed = time.time() - t0
            running_acc = correct / total * 100
            logger.info(
                f"  {'✓' if accuracy else '✗'} | "
                f"Extracted: {extracted} | GT: {problem['ground_truth']} | "
                f"Tokens: {result['actual_tokens_generated']} | "
                f"Time: {elapsed:.1f}s | "
                f"Running acc: {running_acc:.1f}%"
            )

    except Exception as e:
        logger.error(f"CRASH at problem {i}: {e}")
        raise
    finally:
        cache.remove_hooks()

    if total > 0:
        logger.info(f"\nDONE: {correct}/{total} = {correct/total*100:.1f}%")
        logger.info(f"Cached activations saved to: {cache_path}")
    else:
        logger.info("No new problems to run")


if __name__ == "__main__":
    main()
