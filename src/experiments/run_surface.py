#!/usr/bin/env python3
"""Run the compute-accuracy surface experiment (SUPPORTING FINDING).

Maps accuracy over (FFN-skip%, token-budget) grid for each benchmark.

Usage:
    python src/experiments/run_surface.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmarks math500 gpqa mmlu_pro \
        --output_dir /content/drive/MyDrive/depth_or_length \
        --resume
"""

import argparse
import logging
import os
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Surface grid
FFN_SKIP_PCTS = [0, 10, 20, 30, 40, 50]
TOKEN_BUDGETS = [512, 1024, 2048, 4096, None]  # None = unlimited


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "gpqa", "mmlu_pro"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    total_configs = len(FFN_SKIP_PCTS) * len(TOKEN_BUDGETS) * len(args.benchmarks)
    config_num = 0

    for benchmark in args.benchmarks:
        for skip_pct in FFN_SKIP_PCTS:
            for budget in TOKEN_BUDGETS:
                config_num += 1
                budget_str = str(budget) if budget else "unlimited"
                logger.info(f"\n[{config_num}/{total_configs}] "
                            f"{benchmark} | FFN-skip={skip_pct}% | budget={budget_str}")

                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "run_experiment.py"),
                    "--model", args.model,
                    "--benchmark", benchmark,
                    "--skip_type", "ffn_only" if skip_pct > 0 else "none",
                    "--skip_pct", str(skip_pct),
                    "--output_dir", args.output_dir,
                    "--seed", str(args.seed),
                ]
                if budget is not None:
                    cmd.extend(["--token_budget", str(budget)])
                if args.resume:
                    cmd.append("--resume")

                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    logger.error(f"  FAILED with return code {result.returncode}")


if __name__ == "__main__":
    main()
