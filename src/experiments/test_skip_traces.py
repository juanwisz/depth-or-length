#!/usr/bin/env python3
"""Quick test: run 3 MATH problems with baseline vs ffn_skip_25.

Prints full reasoning traces side-by-side so we can verify the
decode-only skip fix produces coherent output.
"""

import json
import logging
import sys
import time

import torch
from vllm import LLM, SamplingParams

# Add project root to path
sys.path.insert(0, ".")
from src.benchmarks.loader import load_benchmark
from src.depth_control.skip_manager import get_skip_layers
from src.experiments.run_vllm_experiment import (
    _find_layers,
    _find_submodule,
    _find_vllm_model,
)
from src.infrastructure.generation import extract_answer, check_answer_correct

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEMS = 3
MAX_TOKENS = 8192
SKIP_PCT = 25  # 25% FFN skip = 5 middle layers


def patch_ffn_decode_only(llm, num_layers: int, skip_pct: float):
    """Patch FFN to skip during decode only."""
    model = _find_vllm_model(llm)
    layers = _find_layers(model)
    skip_indices = get_skip_layers(
        num_layers, skip_pct,
        cold_start=4, cold_end=4, strategy="middle",
    )
    logger.info(f"Skipping FFN in layers: {skip_indices}")

    for idx in skip_indices:
        layer = layers[idx]
        mlp = _find_submodule(layer, ['mlp', 'feed_forward', 'ffn'])
        original = mlp.forward

        def make_skip(orig):
            def skipped(hidden_states):
                if hidden_states.shape[0] > 1:
                    return orig(hidden_states)
                return torch.zeros_like(hidden_states)
            return skipped

        mlp.forward = make_skip(original)


def run_problems(llm, problems, label: str):
    """Run problems and return results with traces."""
    params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
    )
    results = []
    for p in problems:
        prompt = p["prompt"]
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        out = llm.chat([messages], params, use_tqdm=False)
        elapsed = time.time() - t0
        text = out[0].outputs[0].text
        n_tokens = len(out[0].outputs[0].token_ids)
        extracted = extract_answer(text, "math")
        correct = check_answer_correct(extracted, p["ground_truth"], "math")

        results.append({
            "label": label,
            "problem_id": p.get("problem_id", "?"),
            "ground_truth": p["ground_truth"],
            "extracted": extracted,
            "correct": correct,
            "n_tokens": n_tokens,
            "wall_s": elapsed,
            "trace": text,
        })
    return results


def main():
    problems = load_benchmark("math500", subsample=N_PROBLEMS, seed=42)
    logger.info(f"Loaded {len(problems)} problems")

    # --- Baseline ---
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (no skip)")
    logger.info("=" * 60)
    llm = LLM(
        model=MODEL, dtype="bfloat16", seed=42,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_TOKENS + 2048,
        trust_remote_code=True,
    )
    num_layers = len(_find_layers(_find_vllm_model(llm)))
    baseline = run_problems(llm, problems, "baseline")
    del llm
    torch.cuda.empty_cache()

    # --- FFN skip 25% (decode-only) ---
    logger.info("\n" + "=" * 60)
    logger.info(f"FFN_SKIP_{SKIP_PCT}% (decode-only)")
    logger.info("=" * 60)
    llm = LLM(
        model=MODEL, dtype="bfloat16", seed=42,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_TOKENS + 2048,
        trust_remote_code=True,
        enforce_eager=True,
    )
    patch_ffn_decode_only(llm, num_layers, SKIP_PCT)
    skip = run_problems(llm, problems, f"ffn_skip_{SKIP_PCT}")
    del llm
    torch.cuda.empty_cache()

    # --- Print results ---
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 80)

    for b, s in zip(baseline, skip):
        pid = b["problem_id"]
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Problem: {pid}  |  Ground truth: {b['ground_truth']}")
        logger.info(f"{'─' * 80}")
        logger.info(
            f"  BASELINE:  tokens={b['n_tokens']:5d}  "
            f"answer={b['extracted']}  "
            f"correct={'✓' if b['correct'] else '✗'}  "
            f"time={b['wall_s']:.1f}s"
        )
        logger.info(
            f"  SKIP_{SKIP_PCT}%:  tokens={s['n_tokens']:5d}  "
            f"answer={s['extracted']}  "
            f"correct={'✓' if s['correct'] else '✗'}  "
            f"time={s['wall_s']:.1f}s"
        )

    # Print full traces
    for b, s in zip(baseline, skip):
        pid = b["problem_id"]
        logger.info(f"\n{'═' * 80}")
        logger.info(f"TRACE: {pid} — BASELINE ({b['n_tokens']} tokens)")
        logger.info(f"{'═' * 80}")
        # Print first 2000 chars + last 500
        trace = b["trace"]
        if len(trace) > 3000:
            logger.info(trace[:2000])
            logger.info(f"\n... [{len(trace) - 2500} chars omitted] ...\n")
            logger.info(trace[-500:])
        else:
            logger.info(trace)

        logger.info(f"\n{'═' * 80}")
        logger.info(
            f"TRACE: {pid} — FFN_SKIP_{SKIP_PCT}% ({s['n_tokens']} tokens)"
        )
        logger.info(f"{'═' * 80}")
        trace = s["trace"]
        if len(trace) > 3000:
            logger.info(trace[:2000])
            logger.info(f"\n... [{len(trace) - 2500} chars omitted] ...\n")
            logger.info(trace[-500:])
        else:
            logger.info(trace)


if __name__ == "__main__":
    main()
