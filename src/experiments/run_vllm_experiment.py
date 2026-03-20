#!/usr/bin/env python3
"""Run decomposition experiment using vLLM for maximum throughput.

Strategy: one vLLM instance per config. Patch model layers BEFORE first
inference so CUDA graphs capture the patched (skipped) computation.
Destroy and recreate between configs.

This gives ~20-30x speedup over HuggingFace generate() via:
- Continuous batching (all problems in one call)
- CUDA graphs (baked-in skip logic, no Python overhead per token)
- PagedAttention (efficient KV cache)

Usage:
    python src/experiments/run_vllm_experiment.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 --subsample 100 \
        --output_dir results/vllm_pilot \
        --resume
"""

import argparse
import gc
import logging
import os
import sys
import time
from typing import Dict, List

import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from experiments.configs import (
    CONFIGS, ExperimentConfig, append_result, load_completed_pairs,
)
from infrastructure.generation import extract_answer, check_answer_correct
from infrastructure.model_loader import resolve_model_name, MODEL_REGISTRY
from benchmarks.loader import load_benchmark
from depth_control.skip_manager import get_skip_layers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)



def _get_num_layers(model_name: str) -> int:
    """Get layer count from registry or model config."""
    for info in MODEL_REGISTRY.values():
        if info["hf_name"] == model_name:
            return info["num_layers"]
    # Fallback: load config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return config.num_hidden_layers


def _find_vllm_model(llm):
    """Navigate vLLM internals to get the actual nn.Module.

    vLLM's internal structure varies by version. Try multiple paths.
    """
    # vLLM >= 0.4.x
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    # vLLM >= 0.5.x
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model_runner.model
    except AttributeError:
        pass
    # vLLM with Ray / tensor parallel
    try:
        workers = llm.llm_engine.model_executor.workers
        return workers[0].model_runner.model
    except (AttributeError, IndexError):
        pass
    raise RuntimeError(
        "Cannot find model inside vLLM LLM object. "
        "Check vLLM version compatibility."
    )


def _find_layers(model) -> list:
    """Find transformer layers in a vLLM model."""
    # vLLM Qwen2/Llama models: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    # Some vLLM versions wrap differently
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    raise RuntimeError(f"Cannot find layers in vLLM model: {type(model)}")


def _patch_model_for_config(
    llm, config: ExperimentConfig, num_layers: int,
) -> None:
    """Patch vLLM's internal model BEFORE first inference.

    CUDA graphs haven't been captured yet, so patched forwards
    will be baked into the graphs for maximum speed.
    """
    if config.is_baseline:
        return

    model = _find_vllm_model(llm)
    layers = _find_layers(model)
    skip_layer_indices = get_skip_layers(
        num_layers, config.skip_pct,
        cold_start=4, cold_end=4, strategy="middle",
    )

    logger.info(
        f"  Patching {len(skip_layer_indices)} layers for {config.skip_type}: "
        f"{skip_layer_indices}"
    )

    for idx in skip_layer_indices:
        if idx >= len(layers):
            continue
        layer = layers[idx]

        if config.skip_type == "ffn_only":
            mlp = _find_submodule(layer, ['mlp', 'feed_forward', 'ffn'])
            original = mlp.forward

            def make_skip(orig):
                def skipped(hidden_states):
                    return torch.zeros_like(hidden_states)
                return skipped

            mlp.forward = make_skip(original)

        elif config.skip_type == "attention_only":
            # vLLM attention: forward(positions, hidden_states) -> Tensor
            attn = _find_submodule(layer, ['self_attn', 'attention', 'attn'])
            original = attn.forward

            def make_attn_skip(orig):
                def skipped(positions, hidden_states, *args, **kwargs):
                    return torch.zeros_like(hidden_states)
                return skipped

            attn.forward = make_attn_skip(original)

        elif config.skip_type == "full_layer":
            # vLLM layer: forward(positions, hidden_states, residual)
            #   -> tuple[Tensor, Tensor]
            original = layer.forward

            def make_layer_skip(orig):
                def skipped(positions, hidden_states, residual, *args, **kwargs):
                    # In vLLM's residual pattern, the caller does:
                    #   hidden, residual = layer(pos, hidden, residual)
                    # For first layer residual=None; layer inits it.
                    # To skip: just pass through unchanged.
                    if residual is None:
                        # First layer: init residual = hidden, hidden = 0
                        # (matches what a real layer does before adding)
                        return hidden_states, hidden_states
                    return hidden_states, residual
                return skipped

            layer.forward = make_layer_skip(original)


def _find_submodule(layer, names: list):
    """Find a submodule by trying multiple attribute names."""
    for name in names:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Cannot find {names} in {type(layer)}")



def run_config(
    model_name: str,
    config: ExperimentConfig,
    problems: List[Dict],
    completed: set,
    num_layers: int,
    output_dir: str,
    model_short: str,
    benchmark: str,
    bench_type: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gpu_mem: float,
) -> Dict[str, int]:
    """Run one config: create vLLM instance, patch, infer, destroy.

    Returns:
        Dict with 'correct' and 'total' counts.
    """
    from vllm import LLM, SamplingParams

    # Filter to problems not yet done for this config
    todo = [p for p in problems if (p["problem_id"], config.name) not in completed]
    if not todo:
        logger.info(f"  [{config.name}] all done, skipping")
        return {"correct": 0, "total": 0}

    logger.info(f"  [{config.name}] {len(todo)} problems to run")

    # Create fresh vLLM instance
    logger.info(f"  Loading vLLM instance for {config.name}...")
    load_start = time.time()
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_new_tokens + 2048,  # input + output headroom
        trust_remote_code=True,
    )
    load_time = time.time() - load_start
    logger.info(f"  Model loaded in {load_time:.1f}s")

    # Patch BEFORE first inference (before CUDA graph capture)
    _patch_model_for_config(llm, config, num_layers)

    # Build chat messages for all problems
    conversations = []
    for p in todo:
        conversations.append([{"role": "user", "content": p["prompt"]}])

    # Sampling params
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    # Run all problems in one batched call
    logger.info(f"  Generating {len(todo)} problems...")
    gen_start = time.time()
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling,
        use_tqdm=True,
    )
    gen_time = time.time() - gen_start
    avg_time = gen_time / len(todo)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / gen_time if gen_time > 0 else 0
    logger.info(
        f"  Done: {len(todo)} problems in {gen_time:.1f}s "
        f"({avg_time:.1f}s/problem, {throughput:.0f} tok/s aggregate)"
    )

    # Process and save results
    correct = 0
    total = 0
    skip_layers = None
    if not config.is_baseline:
        skip_layers = get_skip_layers(
            num_layers, config.skip_pct,
            cold_start=4, cold_end=4, strategy="middle",
        )

    for problem, output in zip(todo, outputs):
        gen_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)

        extracted = extract_answer(gen_text, bench_type)
        is_correct = check_answer_correct(
            extracted, problem["ground_truth"], bench_type,
        )
        accuracy = 1 if is_correct else 0
        correct += accuracy
        total += 1

        record = {
            "problem_id": problem["problem_id"],
            "config": config.name,
            "model": model_name,
            "benchmark": benchmark,
            "skip_type": config.skip_type,
            "skip_pct": config.skip_pct,
            "flop_reduction_pct": config.flop_reduction_pct,
            "accuracy": accuracy,
            "extracted_answer": extracted,
            "ground_truth": problem["ground_truth"],
            "actual_tokens_generated": num_tokens,
            "wall_clock_seconds": round(avg_time, 3),
            "total_generation_seconds": round(gen_time, 3),
            "throughput_tok_s": round(throughput, 1),
            "generation_text": gen_text,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine": "vllm",
        }
        if skip_layers is not None:
            record["skip_layers"] = skip_layers

        append_result(output_dir, model_short, benchmark, config, record)

        logger.info(
            f"    {'✓' if accuracy else '✗'} {problem['problem_id']} | "
            f"Ans: {extracted} | GT: {problem['ground_truth']} | "
            f"Tok: {num_tokens}"
        )

    acc_pct = correct / total * 100 if total > 0 else 0
    logger.info(f"  [{config.name}] Accuracy: {acc_pct:.1f}% ({correct}/{total})")

    # Destroy vLLM instance to free GPU memory for next config
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {"correct": correct, "total": total}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run decomposition experiment with vLLM (fast batched inference)"
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
    parser.add_argument("--gpu_mem", type=float, default=0.90,
                        help="vLLM gpu_memory_utilization (0-1)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Run only these configs (by name). Default: all."
    )
    return parser.parse_args()


def main():
    """Run all configs sequentially, each with a fresh vLLM instance."""
    args = parse_args()
    hf_name = resolve_model_name(args.model)
    model_short = hf_name.split("/")[-1].lower().replace("-", "_")

    logger.info(f"Model: {hf_name}")
    logger.info(f"Benchmark: {args.benchmark} (n={args.subsample})")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Engine: vLLM (batched, CUDA graphs)")

    os.makedirs(args.output_dir, exist_ok=True)

    bench_type_map = {
        "math500": "math", "gpqa": "gpqa",
        "mmlu_pro": "mmlu_pro", "aime": "aime",
    }
    bench_type = bench_type_map.get(args.benchmark, "math")

    num_layers = _get_num_layers(hf_name)
    logger.info(f"Layers: {num_layers}")

    # Load benchmark
    problems = load_benchmark(
        args.benchmark, subsample=args.subsample, seed=args.seed,
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Filter configs
    configs_to_run = list(CONFIGS)
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c.name in args.configs]
    logger.info(f"Configs ({len(configs_to_run)}): {[c.name for c in configs_to_run]}")

    # Load completed pairs for resume
    completed = set()
    if args.resume:
        completed = load_completed_pairs(args.output_dir)
        logger.info(f"Resuming: {len(completed)} pairs already done")

    # Run each config with a fresh vLLM instance
    total_start = time.time()
    summary = {}

    for c_idx, config in enumerate(configs_to_run):
        logger.info(f"\n{'='*60}")
        logger.info(f"CONFIG [{c_idx+1}/{len(configs_to_run)}]: {config.name}")
        logger.info(f"{'='*60}")

        result = run_config(
            model_name=hf_name,
            config=config,
            problems=problems,
            completed=completed,
            num_layers=num_layers,
            output_dir=args.output_dir,
            model_short=model_short,
            benchmark=args.benchmark,
            bench_type=bench_type,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            gpu_mem=args.gpu_mem,
        )
        summary[config.name] = result

    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {total_time/60:.1f} min")
    logger.info(f"{'='*60}")
    for name, res in summary.items():
        if res["total"] > 0:
            acc = res["correct"] / res["total"] * 100
            logger.info(f"  {name}: {acc:.1f}% ({res['correct']}/{res['total']})")


if __name__ == "__main__":
    main()
