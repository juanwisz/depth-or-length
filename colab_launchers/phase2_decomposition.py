#!/usr/bin/env python3
"""Colab launcher for Phase 2 — Full Decomposition (LEAD FINDING).

Copy this into a Colab cell after Phase 1 pilot shows GO.

Runs the full decomposition grid:
- 3 skip types × multiple FLOP levels
- On MATH-500 and GPQA
- Primary model: DeepSeek-R1-Distill-Qwen-7B

Uses run_multi_condition.py to load model once per benchmark.
"""

# Cell 1: Setup (same as pilot)
SETUP = """
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/repo
!git clone https://github.com/juanwisz/depth-or-length.git /content/repo
!pip install -q -r /content/repo/requirements.txt
import os
for d in ['results', 'logs', 'metadata', 'debug']:
    os.makedirs(f'/content/drive/MyDrive/depth_or_length/{d}', exist_ok=True)
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""

# Cell 2: Phase 2.1 — Full decomposition on MATH-500
# Skip levels chosen to span 10-50% FLOP reduction
# FFN-only: 10%,20%,30%,40%,50% of eligible layers
# Full-layer: same percentages (different FLOP reductions due to FFN vs full)
MATH500_DECOMPOSITION = """
import subprocess

SCRIPT = '/content/repo/src/experiments/run_multi_condition.py'
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
OUTPUT = '/content/drive/MyDrive/depth_or_length'

# Run all FFN-only levels + baseline + all full-layer levels in one process
configs = ' '.join([
    'none:0',
    'ffn_only:10', 'ffn_only:20', 'ffn_only:30', 'ffn_only:40', 'ffn_only:50',
    'full_layer:10', 'full_layer:20', 'full_layer:30', 'full_layer:40', 'full_layer:50',
])

cmd = f'''python {SCRIPT} \\
    --model {MODEL} \\
    --benchmark math500 \\
    --configs {configs} \\
    --output_dir {OUTPUT} \\
    --resume'''

print(f"Running: {cmd}")
!{cmd}
print("MATH-500 decomposition complete!")
"""

# Cell 3: Phase 2.1 continued — GPQA Diamond
GPQA_DECOMPOSITION = """
import subprocess

SCRIPT = '/content/repo/src/experiments/run_multi_condition.py'
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
OUTPUT = '/content/drive/MyDrive/depth_or_length'

configs = ' '.join([
    'none:0',
    'ffn_only:10', 'ffn_only:20', 'ffn_only:30', 'ffn_only:40', 'ffn_only:50',
    'full_layer:10', 'full_layer:20', 'full_layer:30', 'full_layer:40', 'full_layer:50',
])

cmd = f'''python {SCRIPT} \\
    --model {MODEL} \\
    --benchmark gpqa \\
    --configs {configs} \\
    --output_dir {OUTPUT} \\
    --resume'''

print(f"Running: {cmd}")
!{cmd}
print("GPQA decomposition complete!")
"""

# Cell 4: Phase 2.2 — Second model (Llama-8B) on MATH-500 + GPQA
SECOND_MODEL = """
SCRIPT = '/content/repo/src/experiments/run_multi_condition.py'
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
OUTPUT = '/content/drive/MyDrive/depth_or_length'

configs = 'none:0 ffn_only:20 ffn_only:30 ffn_only:40 full_layer:20 full_layer:30 full_layer:40'

for bench in ['math500', 'gpqa']:
    cmd = f'python {SCRIPT} --model {MODEL} --benchmark {bench} --configs {configs} --output_dir {OUTPUT} --resume'
    print(f"Running: {cmd}")
    !{cmd}

print("Second model complete!")
"""

if __name__ == '__main__':
    print("This file contains Colab cell contents.")
    print("Copy the relevant cell content into Colab notebook cells.")
    print()
    print("Phase 2 cells:")
    print("  Cell 1: SETUP")
    print("  Cell 2: MATH500_DECOMPOSITION")
    print("  Cell 3: GPQA_DECOMPOSITION")
    print("  Cell 4: SECOND_MODEL")
