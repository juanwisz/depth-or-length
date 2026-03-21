#!/usr/bin/env python3
"""Measure FFN redundancy per layer via cosine similarity.

Implements the core insight from FFN-SkipLLM (EMNLP 2024):
FFN blocks in middle layers have high cosine similarity between
input and output, indicating redundant computation.

We measure this on a calibration set, then rank layers by redundancy
to determine which are safest to skip.

Usage:
    # As a library
    from depth_control.ffn_redundancy import measure_ffn_redundancy
    scores = measure_ffn_redundancy(model, tokenizer, calibration_texts)
    # scores[i] = mean cosine similarity for layer i's FFN

    # Or get skip layers ranked by redundancy (most redundant first)
    from depth_control.ffn_redundancy import get_redundancy_skip_layers
    layers = get_redundancy_skip_layers(model, tokenizer, texts, skip_pct=25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _get_layers(model: nn.Module) -> list:
    """Get transformer layers from model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    raise ValueError(f"Cannot find layers in {type(model)}")


def _find_mlp(layer: nn.Module) -> nn.Module:
    """Find MLP submodule in a transformer layer."""
    for attr in ['mlp', 'feed_forward', 'ffn']:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise ValueError(f"Cannot find MLP in {type(layer)}")


def measure_ffn_redundancy(
    model: nn.Module,
    tokenizer,
    calibration_texts: List[str],
    max_tokens: int = 128,
) -> List[float]:
    """Measure cosine similarity between FFN input and output per layer.

    High cosine similarity = FFN is doing little = safe to skip.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        calibration_texts: Short texts for calibration.
        max_tokens: Max tokens per calibration text.

    Returns:
        List of mean cosine similarities, one per layer.
        Higher values = more redundant = safer to skip.
    """
    layers = _get_layers(model)
    num_layers = len(layers)
    similarities: list[list[float]] = [[] for _ in range(num_layers)]
    hooks = []

    def _make_hook(layer_idx: int):
        """Create a forward hook that records cosine similarity."""
        def hook_fn(module, args, output):
            # args[0] is the input hidden_states (post-layernorm)
            inp = args[0] if isinstance(args, tuple) else args
            if isinstance(inp, torch.Tensor) and isinstance(output, torch.Tensor):
                # Cosine similarity across embedding dimension
                cos_sim = F.cosine_similarity(
                    inp.flatten(0, -2), output.flatten(0, -2), dim=-1
                ).mean().item()
                similarities[layer_idx].append(cos_sim)
        return hook_fn

    try:
        # Register hooks on all MLP modules
        for i, layer in enumerate(layers):
            mlp = _find_mlp(layer)
            h = mlp.register_forward_hook(_make_hook(i))
            hooks.append(h)

        # Run calibration texts through model
        model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                inputs = tokenizer(
                    text, return_tensors="pt",
                    max_length=max_tokens, truncation=True,
                ).to(model.device)
                model(**inputs)

    finally:
        for h in hooks:
            h.remove()

    # Average similarities per layer
    mean_sims = []
    for i in range(num_layers):
        if similarities[i]:
            mean_sims.append(sum(similarities[i]) / len(similarities[i]))
        else:
            mean_sims.append(0.0)

    return mean_sims


def get_redundancy_skip_layers(
    model: nn.Module,
    tokenizer,
    calibration_texts: List[str],
    skip_pct: float,
    cold_start: int = 4,
    cold_end: int = 4,
    max_tokens: int = 128,
) -> List[int]:
    """Get layers to skip ranked by redundancy (most redundant first).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        calibration_texts: Texts for calibration.
        skip_pct: Percentage of eligible layers to skip (0-100).
        cold_start: Protect first N layers.
        cold_end: Protect last N layers.
        max_tokens: Max tokens per calibration input.

    Returns:
        Sorted list of layer indices to skip (most redundant layers).
    """
    sims = measure_ffn_redundancy(model, tokenizer, calibration_texts, max_tokens)
    num_layers = len(sims)

    # Eligible layers (exclude cold start/end)
    eligible = list(range(cold_start, num_layers - cold_end))
    if not eligible:
        return []

    num_to_skip = max(1, round(len(eligible) * skip_pct / 100))
    num_to_skip = min(num_to_skip, len(eligible))

    # Rank eligible layers by cosine similarity (highest = most redundant)
    ranked = sorted(eligible, key=lambda i: sims[i], reverse=True)
    skip_layers = sorted(ranked[:num_to_skip])

    return skip_layers


def print_redundancy_profile(sims: List[float]) -> None:
    """Print a visual redundancy profile across layers."""
    print(f"{'Layer':>6} {'CosSim':>8} {'Bar'}")
    print("-" * 50)
    for i, s in enumerate(sims):
        bar_len = int(s * 40) if s > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ cold" if i < 4 or i >= len(sims) - 4 else ""
        print(f"{i:>6} {s:>8.4f} {bar}{marker}")
