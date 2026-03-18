#!/usr/bin/env python3
"""Run Phase 1 pilot experiments.

1.1 — Coarse decomposition pilot: FFN-only vs full-layer vs attention-only at ~20% FLOP reduction
1.2 — Coarse surface pilot: 3x3 grid (FFN-skip: 0/25/50%, budget: 1024/2048/unlimited)

Both on DeepSeek-R1-Distill-Qwen-7B, MATH-500 only.

Usage:
    python src/experiments/run_pilot.py \
        --output_dir /content/drive/MyDrive/depth_or_length \
        --resume
"""

import argparse
import logging
import os
import sys
import subprocess

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))
from depth_control.flop_counter import find_iso_flop_configs

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BENCHMARK = "math500"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_experiment(model, benchmark, skip_type, skip_pct, token_budget, output_dir, resume, seed):
    """Run a single experiment config."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_experiment.py"),
        "--model", model,
        "--benchmark", benchmark,
        "--skip_type", skip_type,
        "--skip_pct", str(skip_pct),
        "--output_dir", output_dir,
        "--seed", str(seed),
    ]
    if token_budget is not None:
        cmd.extend(["--token_budget", str(token_budget)])
    if resume:
        cmd.append("--resume")

    logger.info(f"CMD: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=False)


def main():
    args = parse_args()

    # ===== 1.0: Baseline (full model, full length) =====
    logger.info("\n" + "="*60)
    logger.info("BASELINE: Full model, full length")
    logger.info("="*60)
    run_experiment(MODEL, BENCHMARK, "none", 0, None, args.output_dir, args.resume, args.seed)

    # ===== 1.1: Coarse decomposition pilot =====
    logger.info("\n" + "="*60)
    logger.info("PILOT 1.1: Coarse decomposition at ~20% FLOP reduction")
    logger.info("="*60)

    iso_configs = find_iso_flop_configs(MODEL, target_flop_reduction_pct=20.0)
    for skip_type, config in iso_configs.items():
        logger.info(f"\n{skip_type}: {config['num_layers_skipped']} layers skipped, "
                     f"FLOP reduction: {config['actual_flop_reduction_pct']:.1f}%")
        run_experiment(
            MODEL, BENCHMARK, skip_type,
            config['skip_pct_of_eligible'],
            None,  # Full length
            args.output_dir, args.resume, args.seed,
        )

    # ===== 1.2: Coarse surface pilot =====
    logger.info("\n" + "="*60)
    logger.info("PILOT 1.2: Coarse 3x3 surface")
    logger.info("="*60)

    for skip_pct in [0, 25, 50]:
        for budget in [1024, 2048, None]:
            budget_str = str(budget) if budget else "unlimited"
            skip_type = "ffn_only" if skip_pct > 0 else "none"
            logger.info(f"\nFFN-skip={skip_pct}%, budget={budget_str}")
            run_experiment(
                MODEL, BENCHMARK, skip_type, skip_pct,
                budget, args.output_dir, args.resume, args.seed,
            )

    logger.info("\n" + "="*60)
    logger.info("PILOT COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
