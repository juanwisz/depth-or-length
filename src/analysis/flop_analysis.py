#!/usr/bin/env python3
"""FLOP analysis and iso-FLOP comparison figures.

Generates:
1. Bar chart showing FFN vs attention compute fraction per layer
2. Table of iso-FLOP configs: how many layers each skip type needs
3. Theoretical max FLOP reduction per skip type
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from depth_control.flop_counter import (
    get_architecture, flops_per_token_per_layer,
    find_iso_flop_configs, compute_total_flops, ARCHITECTURES,
)

# Paper-quality settings
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def figure_compute_breakdown(output_dir: str) -> None:
    """Bar chart: FFN vs attention fraction for each model."""
    models = list(ARCHITECTURES.keys())
    ffn_fracs = []
    attn_fracs = []
    labels = []

    for name in models:
        arch = get_architecture(name)
        flops = flops_per_token_per_layer(arch)
        ffn_fracs.append(flops['ffn_fraction'] * 100)
        attn_fracs.append((1 - flops['ffn_fraction']) * 100)
        labels.append(name.split('/')[-1])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(labels))
    w = 0.6

    ax.barh(x, ffn_fracs, w, label='FFN (MLP)', color='#2196F3', alpha=0.85)
    ax.barh(x, attn_fracs, w, left=ffn_fracs, label='Attention', color='#F44336', alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Compute Fraction (%)')
    ax.set_title('Per-Layer Compute Breakdown (GQA models)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)

    for i, (f, a) in enumerate(zip(ffn_fracs, attn_fracs)):
        ax.text(f / 2, i, f'{f:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(f + a / 2, i, f'{a:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')

    path = os.path.join(output_dir, 'compute_breakdown.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f'Saved {path}')


def figure_iso_flop_layers(output_dir: str, model: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B') -> None:
    """Plot: number of layers skipped at each target FLOP reduction for each skip type."""
    targets = [5, 10, 15, 20, 25, 30, 40, 50]
    data = {'ffn_only': [], 'full_layer': [], 'attention_only': []}
    valid_targets = {'ffn_only': [], 'full_layer': [], 'attention_only': []}

    for t in targets:
        configs = find_iso_flop_configs(model, t, tolerance_pct=5.0)
        for st in data:
            if st in configs:
                data[st].append(configs[st]['num_layers_skipped'])
                valid_targets[st].append(t)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'ffn_only': '#2196F3', 'full_layer': '#F44336', 'attention_only': '#4CAF50'}
    labels = {'ffn_only': 'FFN-only', 'full_layer': 'Full-layer', 'attention_only': 'Attention-only'}
    markers = {'ffn_only': 'o', 'full_layer': 's', 'attention_only': '^'}

    for st in data:
        if data[st]:
            ax.plot(valid_targets[st], data[st], color=colors[st],
                    marker=markers[st], label=labels[st], linewidth=2, markersize=7)

    arch = get_architecture(model)
    ax.axhline(y=arch.num_layers - 8, color='gray', linestyle=':', alpha=0.5,
               label=f'Max eligible ({arch.num_layers - 8} layers)')

    ax.set_xlabel('Target FLOP Reduction (%)')
    ax.set_ylabel('Layers Skipped')
    ax.set_title(f'Iso-FLOP Layer Requirements\n{model.split("/")[-1]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'iso_flop_layers.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f'Saved {path}')


def print_iso_flop_table(model: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B') -> None:
    """Print iso-FLOP comparison table."""
    targets = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f'\nIso-FLOP table for {model.split("/")[-1]}')
    print(f'{"Target":>8s} | {"FFN-only":>22s} | {"Full-layer":>22s} | {"Attn-only":>22s}')
    print('-' * 82)
    for t in targets:
        configs = find_iso_flop_configs(model, t, tolerance_pct=5.0)
        parts = [f'{t:>6d}%']
        for st in ['ffn_only', 'full_layer', 'attention_only']:
            if st in configs:
                c = configs[st]
                parts.append(f'{c["num_layers_skipped"]:>2d}L = {c["actual_flop_reduction_pct"]:>5.1f}%')
            else:
                parts.append(f'{"N/A":>22s}')
        print(' | '.join(parts))


if __name__ == '__main__':
    output_dir = os.path.join(_project_root, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    figure_compute_breakdown(output_dir)
    figure_iso_flop_layers(output_dir)

    for model in ARCHITECTURES:
        print_iso_flop_table(model)
