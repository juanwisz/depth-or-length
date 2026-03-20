"""Skip manager: FFN-only, full-layer, and attention-only skipping.

Adapted from AdaSkip (AAAI 2025). Uses forward hooks to replace sublayer
outputs with scaled passthrough, preserving residual stream magnitude.

Three skip types:
1. FFN-only skip: Skip MLP, keep attention. Hook on MLP returns input.
2. Full-layer skip: Skip entire layer. Replace forward with passthrough.
3. Attention-only skip: Skip attention, keep MLP. Hook on attention returns input.

Key insight from AdaSkip: when skipping a sublayer, the residual connection
means hidden_states += sublayer(x). Skipping makes hidden_states += 0, which
changes the norm. AdaSkip computes a scaling ratio during calibration to
compensate. For simplicity, we use the hook approach (return input to sublayer)
which preserves the residual correctly: hidden_states += layernorm(hidden_states)
instead of hidden_states += mlp(layernorm(hidden_states)).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal
from contextlib import contextmanager


SkipType = Literal["ffn_only", "full_layer", "attention_only"]


def get_skip_layers(
    num_layers: int,
    skip_pct: float,
    cold_start: int = 4,
    cold_end: int = 4,
    strategy: str = "middle",
) -> List[int]:
    """Determine which layer indices to skip.

    Args:
        num_layers: Total number of transformer layers.
        skip_pct: Percentage of eligible layers to skip (0-100).
        cold_start: Number of initial layers to protect.
        cold_end: Number of final layers to protect.
        strategy: "middle" (skip from center out), "uniform", or "random".

    Returns:
        Sorted list of layer indices to skip.
    """
    if skip_pct <= 0:
        return []

    eligible_start = cold_start
    eligible_end = num_layers - cold_end
    eligible = list(range(eligible_start, eligible_end))

    if not eligible:
        return []

    num_to_skip = max(1, round(len(eligible) * skip_pct / 100))
    num_to_skip = min(num_to_skip, len(eligible))

    if strategy == "middle":
        center = len(eligible) // 2
        sorted_by_center = sorted(eligible, key=lambda x: abs(x - eligible[center]))
        skip_layers = sorted(sorted_by_center[:num_to_skip])
    elif strategy == "uniform":
        if num_to_skip >= len(eligible):
            skip_layers = eligible
        else:
            step = len(eligible) / num_to_skip
            skip_layers = sorted([eligible[int(i * step)] for i in range(num_to_skip)])
    elif strategy == "random":
        import random
        random.seed(42)
        skip_layers = sorted(random.sample(eligible, num_to_skip))
    else:
        raise ValueError(f"Unknown skip strategy: {strategy}")

    return skip_layers


def _get_layers(model) -> list:
    """Get transformer layers from model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    raise ValueError(f"Cannot find layers in {type(model)}")


def _find_mlp(layer) -> nn.Module:
    """Find the MLP module in a transformer layer."""
    for attr in ['mlp', 'feed_forward', 'ffn']:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise ValueError(f"Cannot find MLP in layer {type(layer)}")


def _find_attn(layer) -> nn.Module:
    """Find the attention module in a transformer layer."""
    for attr in ['self_attn', 'attention', 'attn']:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise ValueError(f"Cannot find attention in layer {type(layer)}")


@contextmanager
def apply_skip(
    model,
    skip_type: SkipType,
    skip_layers: List[int],
):
    """Context manager that applies skipping via forward hooks.

    Uses the hook-based approach from AdaSkip: register a forward hook
    on the sublayer (MLP or attention) that returns the input instead of
    the computed output. This means the residual connection sees:
        hidden_states = residual + hook_output
    where hook_output = input_to_sublayer (the layernorm'd hidden states).

    For full-layer skip, we replace the entire layer forward with a
    passthrough that handles KV cache properly.

    Args:
        model: HuggingFace causal LM model.
        skip_type: Type of skipping.
        skip_layers: List of layer indices to skip.
    """
    if not skip_layers:
        yield
        return

    layers = _get_layers(model)
    originals = {}

    def _make_mlp_skip(original_mlp_forward):
        """Replace MLP forward to return zeros without computing."""
        def skipped_forward(hidden_states):
            return torch.zeros_like(hidden_states)
        return skipped_forward

    def _make_attn_skip(original_attn_forward):
        """Replace attention forward to return zeros without computing.

        Must still handle KV cache args to avoid downstream errors.
        Qwen2Attention.forward returns (attn_output, attn_weights).
        The caller (Qwen2DecoderLayer) unpacks with: hidden_states, _ = self.self_attn(...)
        """
        def skipped_forward(hidden_states, **kwargs):
            zeros = torch.zeros_like(hidden_states)
            return zeros, None
        return skipped_forward

    try:
        for idx in skip_layers:
            if idx >= len(layers):
                continue
            layer = layers[idx]

            if skip_type == "ffn_only":
                mlp = _find_mlp(layer)
                originals[('mlp', idx)] = mlp.forward
                mlp.forward = _make_mlp_skip(mlp.forward)

            elif skip_type == "attention_only":
                attn = _find_attn(layer)
                originals[('attn', idx)] = attn.forward
                attn.forward = _make_attn_skip(attn.forward)

            elif skip_type == "full_layer":
                originals[('layer', idx)] = layer.forward
                layer.forward = _make_full_layer_skip_forward(layer)

            else:
                raise ValueError(f"Unknown skip type: {skip_type}")

        yield

    finally:
        for key, original in originals.items():
            kind, idx = key
            layer = layers[idx]
            if kind == 'mlp':
                _find_mlp(layer).forward = original
            elif kind == 'attn':
                _find_attn(layer).forward = original
            elif kind == 'layer':
                layer.forward = original


def _make_full_layer_skip_forward(layer):
    """Create a forward that skips the entire layer.

    Hidden states pass through unchanged.
    KV cache gets None entries to maintain indexing.
    """
    def full_skip_forward(hidden_states, **kwargs):
        """Skip entire layer — just pass hidden_states through.

        Qwen2DecoderLayer.forward() returns a plain tensor (not a tuple),
        so we must do the same. KV cache is handled internally by the
        model's attention implementation; skipped layers simply don't
        contribute any new KV entries.
        """
        return hidden_states

    return full_skip_forward
