#!/usr/bin/env python3
"""Run the full decomposition experiment (LEAD FINDING).

Runs FFN-only, full-layer, and attention-only skip at iso-FLOP levels
across multiple benchmarks. This is the core experiment of the paper.

Usage:
    python src/experiments/run_decomposition.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmarks math500 gpqa mmlu_pro \
        --flop_reductions 10 20 30 40 50 \
        --output_dir /content/drive/MyDrive/depth_or_length \
        --resume
"""

import argparse
import logging
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.depth_control.flop_counter import find_iso_flop_configs, get_architecture

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "gpqa", "mmlu_pro"])
    parser.add_argument("--flop_reductions", nargs="+", type=float, default=[10, 20, 30, 40, 50])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # For each target FLOP reduction, find iso-FLOP configs for all skip types
    for target_flop in args.flop_reductions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Target FLOP reduction: {target_flop}%")

        configs = find_iso_flop_configs(args.model, target_flop, tolerance_pct=3.0)

        for skip_type, config in configs.items():
            logger.info(f"  {skip_type}: {config['num_layers_skipped']} layers, "
                        f"actual FLOP reduction: {config['actual_flop_reduction_pct']:.1f}%")

            for benchmark in args.benchmarks:
                logger.info(f"    Running {skip_type} on {benchmark}...")

                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "run_experiment.py"),
                    "--model", args.model,
                    "--benchmark", benchmark,
                    "--skip_type", skip_type,
                    "--skip_pct", str(config['skip_pct_of_eligible']),
                    "--output_dir", args.output_dir,
                    "--seed", str(args.seed),
                ]
                if args.resume:
                    cmd.append("--resume")

                logger.info(f"    CMD: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=False)

                if result.returncode != 0:
                    logger.error(f"    FAILED with return code {result.returncode}")
                else:
                    logger.info(f"    DONE")


if __name__ == "__main__":
    main()
