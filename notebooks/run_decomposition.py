#!/usr/bin/env python3
"""Colab launcher for decomposition experiment.

Copy-paste this into a Colab cell or run as a script.
Clones the repo, installs deps, and runs the full decomposition.

Expected runtime: ~5h on A100 for pilot (50 problems × 3 skip types × 3 FLOP levels).
"""

# --- Cell 1: Setup ---
# !pip install -q transformers accelerate torch datasets sympy
# !git clone https://github.com/juanwisznia/elias_lab_grants.git /content/repo
# %cd /content/repo

# --- Cell 2: Mount Drive ---
# from google.colab import drive
# drive.mount('/content/drive')

# --- Cell 3: Run Decomposition ---
import subprocess
import sys
import os

REPO = "/content/repo"
DRIVE = "/content/drive/MyDrive/depth_or_length_decomposition"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

os.makedirs(DRIVE, exist_ok=True)

# Run baseline first, then each skip type at multiple FLOP levels
CONFIGS = [
    # (skip_type, skip_pct, description)
    ("none", 0, "baseline"),
    ("ffn_only", 15, "FFN-only ~10% FLOP reduction (5 layers)"),
    ("ffn_only", 30, "FFN-only ~19% FLOP reduction (6 layers)"),
    ("ffn_only", 50, "FFN-only ~31% FLOP reduction (10 layers)"),
    ("ffn_only", 65, "FFN-only ~41% FLOP reduction (13 layers)"),
    ("full_layer", 15, "Full-layer ~11% FLOP reduction (3 layers)"),
    ("full_layer", 30, "Full-layer ~21% FLOP reduction (6 layers)"),
    ("full_layer", 50, "Full-layer ~36% FLOP reduction (10 layers)"),
    ("full_layer", 65, "Full-layer ~46% FLOP reduction (13 layers)"),
    ("attention_only", 50, "Attn-only ~5% FLOP reduction (10 layers)"),
    ("attention_only", 100, "Attn-only ~9% FLOP reduction (20 layers)"),
]

for skip_type, skip_pct, desc in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        os.path.join(REPO, "src/experiments/run_experiment.py"),
        "--model", MODEL,
        "--benchmark", "math500",
        "--skip_type", skip_type,
        "--skip_pct", str(skip_pct),
        "--output_dir", DRIVE,
        "--resume",
    ]
    subprocess.run(cmd, check=False)

print("\n\nAll decomposition configs complete!")
print(f"Results in: {DRIVE}/results/")
