"""Skip manager: FFN-only, full-layer, and attention-only skipping.

This is the core contribution of the paper. We implement three skip types:
1. FFN-only skip: Replace selected layers' MLP forward with identity
2. Full-layer skip: Skip entire transformer layers
3. Attention-only skip: Replace selected layers' attention forward with identity

All implementations use monkey-patching on the model's forward methods.
This avoids modifying model code and works with any HuggingFace model.

Key design decisions:
- Skip middle layers, protect first N and last N ("cold regions" per FFN-SkipLLM)
- Default cold region = 4 layers on each side
- Skip scheduler returns layer indices to skip given a percentage
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Literal
from contextlib import contextmanager
from functools import partial


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
        strategy: "middle" (skip from center out), "uniform" (evenly spaced),
                  "random" (random selection — deterministic with seed).

    Returns:
        Sorted list of layer indices to skip.
    """
    if skip_pct <= 0:
        return []

    # Eligible layers (excluding cold regions)
    eligible_start = cold_start
    eligible_end = num_layers - cold_end
    eligible = list(range(eligible_start, eligible_end))

    if not eligible:
        return []

    num_to_skip = max(1, round(len(eligible) * skip_pct / 100))
    num_to_skip = min(num_to_skip, len(eligible))

    if strategy == "middle":
        # Skip from the center outward
        center = len(eligible) // 2
        # Sort by distance from center (closest first)
        sorted_by_center = sorted(eligible, key=lambda x: abs(x - eligible[center]))
        skip_layers = sorted(sorted_by_center[:num_to_skip])

    elif strategy == "uniform":
        # Evenly spaced
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


class _IdentityModule(nn.Module):
    """Identity module that returns input unchanged.

    Used to replace MLP or attention modules during skipping.
    Handles the residual connection pattern: output = input + module(input)
    by returning zero (so input + 0 = input).
    """
    def forward(self, *args, **kwargs):
        # For MLP modules: return zeros matching the hidden states shape
        # The residual connection in the transformer layer will add this to input
        if args:
            hidden_states = args[0]
        elif 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        else:
            # Fallback: try first positional arg
            raise ValueError("Cannot determine hidden_states from args")
        return torch.zeros_like(hidden_states)


class _IdentityAttention(nn.Module):
    """Identity attention that returns zeros + empty cache.

    Attention forward returns (output, attention_weights, past_key_value).
    We return zeros for output, None for weights, None for cache.
    """
    def forward(self, *args, **kwargs):
        if args:
            hidden_states = args[0]
        elif 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        else:
            raise ValueError("Cannot determine hidden_states from args")
        zeros = torch.zeros_like(hidden_states)
        return zeros, None, None


@contextmanager
def apply_skip(
    model,
    skip_type: SkipType,
    skip_layers: List[int],
):
    """Context manager that applies skipping to a model.

    Usage:
        with apply_skip(model, "ffn_only", [4, 5, 6, 7]):
            outputs = model.generate(...)

    Args:
        model: HuggingFace causal LM model.
        skip_type: Type of skipping ("ffn_only", "full_layer", "attention_only").
        skip_layers: List of layer indices to skip.
    """
    if not skip_layers:
        yield
        return

    # Get layer modules
    layers = _get_layers(model)
    originals = {}

    try:
        if skip_type == "ffn_only":
            for idx in skip_layers:
                if idx < len(layers):
                    layer = layers[idx]
                    mlp_attr = _find_mlp_attr(layer)
                    originals[(idx, 'mlp')] = getattr(layer, mlp_attr)
                    setattr(layer, mlp_attr, _IdentityModule())

        elif skip_type == "full_layer":
            # For full-layer skip, we replace the entire layer's forward
            for idx in skip_layers:
                if idx < len(layers):
                    layer = layers[idx]
                    originals[(idx, 'forward')] = layer.forward
                    # Replace with identity that just passes hidden states through
                    layer.forward = _make_identity_layer_forward(layer)

        elif skip_type == "attention_only":
            for idx in skip_layers:
                if idx < len(layers):
                    layer = layers[idx]
                    attn_attr = _find_attn_attr(layer)
                    originals[(idx, 'attn')] = getattr(layer, attn_attr)
                    setattr(layer, attn_attr, _IdentityAttention())

        else:
            raise ValueError(f"Unknown skip type: {skip_type}")

        yield

    finally:
        # Restore original modules
        for (idx, component), original in originals.items():
            layer = layers[idx]
            if component == 'mlp':
                mlp_attr = _find_mlp_attr(layer)
                setattr(layer, mlp_attr, original)
            elif component == 'forward':
                layer.forward = original
            elif component == 'attn':
                attn_attr = _find_attn_attr(layer)
                setattr(layer, attn_attr, original)


def _get_layers(model) -> list:
    """Get transformer layers from model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    raise ValueError(f"Cannot find layers in {type(model)}")


def _find_mlp_attr(layer) -> str:
    """Find the MLP attribute name in a transformer layer."""
    for attr in ['mlp', 'feed_forward', 'ffn']:
        if hasattr(layer, attr):
            return attr
    raise ValueError(f"Cannot find MLP in layer {type(layer)}")


def _find_attn_attr(layer) -> str:
    """Find the attention attribute name in a transformer layer."""
    for attr in ['self_attn', 'attention', 'attn']:
        if hasattr(layer, attr):
            return attr
    raise ValueError(f"Cannot find attention in layer {type(layer)}")


def _make_identity_layer_forward(layer):
    """Create an identity forward function for full-layer skip.

    The layer forward typically does:
        hidden = hidden + attn(norm1(hidden))
        hidden = hidden + mlp(norm2(hidden))

    We replace it with: just return hidden_states unchanged.
    """
    def identity_forward(hidden_states, **kwargs):
        # Return in the format expected by the model
        # Most models return a tuple: (hidden_states, ...) or a BaseModelOutputWithPast
        return (hidden_states,) + (None,) * 2  # (hidden_states, attn_weights, past_kv)

    return identity_forward
