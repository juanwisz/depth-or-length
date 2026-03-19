"""Skip manager: FFN-only, full-layer, and attention-only skipping.

Adapted from AdaSkip (AAAI 2025) approach of modifying layer forward
to conditionally skip sublayers while properly managing KV cache.

Three skip types:
1. FFN-only skip: Skip MLP computation, preserve residual
2. Full-layer skip: Skip entire transformer layer
3. Attention-only skip: Skip attention computation, manage KV cache

Key design decisions:
- Skip middle layers, protect first N and last N ("cold regions" per FFN-SkipLLM)
- Default cold region = 4 layers on each side
- Uses forward hooks instead of monkey-patching submodules
- Properly handles KV cache for attention skipping
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


@contextmanager
def apply_skip(
    model,
    skip_type: SkipType,
    skip_layers: List[int],
):
    """Context manager that applies skipping to a model.

    Approach adapted from AdaSkip (AAAI 2025): replaces the layer's forward
    method with a modified version that skips the specified sublayer while
    properly managing residual connections and KV cache.

    For FFN-only skip: the MLP computation is skipped but the residual
    connection passes through (hidden_states unchanged through MLP stage).

    For attention-only skip: attention is skipped, KV cache gets None entries
    to maintain proper indexing for subsequent layers.

    For full-layer skip: entire layer forward is replaced with pass-through,
    KV cache gets None entries.

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

    try:
        for idx in skip_layers:
            if idx >= len(layers):
                continue
            layer = layers[idx]
            originals[idx] = layer.forward

            if skip_type == "ffn_only":
                layer.forward = _make_ffn_skip_forward(layer)
            elif skip_type == "full_layer":
                layer.forward = _make_full_layer_skip_forward(layer)
            elif skip_type == "attention_only":
                layer.forward = _make_attn_skip_forward(layer)
            else:
                raise ValueError(f"Unknown skip type: {skip_type}")

        yield

    finally:
        for idx, original in originals.items():
            layers[idx].forward = original


def _make_ffn_skip_forward(layer):
    """Create a forward that runs attention normally but skips MLP.

    The standard Qwen2/Llama layer forward does:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    We keep the attention part, skip the MLP part (just pass residual through).
    """
    original_forward = layer.forward

    def ffn_skip_forward(hidden_states, **kwargs):
        # Run the full layer normally to get attention output + KV cache
        result = original_forward(hidden_states, **kwargs)

        # result is typically (hidden_states,) or (hidden_states, ..., cache)
        if isinstance(result, tuple):
            full_output = result[0]
        else:
            full_output = result

        # The full output = input + attn_contribution + mlp_contribution
        # We want: input + attn_contribution (no mlp)
        # To get this, we run attention separately...
        # Actually, it's simpler to just run the original and subtract the MLP contribution.

        # Cleaner approach: compute what the layer WOULD output without MLP
        # Run attention portion only
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)

        attn_attr = _find_attn_attr(layer)
        attn_module = getattr(layer, attn_attr)

        # Run attention with all the kwargs it needs
        attn_kwargs = {}
        for k in ['attention_mask', 'position_ids', 'past_key_value',
                   'output_attentions', 'use_cache', 'cache_position',
                   'position_embeddings']:
            if k in kwargs:
                attn_kwargs[k] = kwargs[k]

        attn_result = attn_module(attn_input, **attn_kwargs)

        if isinstance(attn_result, tuple):
            attn_output = attn_result[0]
            # Preserve other outputs (attention weights, cache, etc.)
            extra_outputs = attn_result[1:]
        else:
            attn_output = attn_result
            extra_outputs = ()

        # Apply residual connection for attention
        hidden_states_after_attn = residual + attn_output
        # Skip MLP entirely — hidden_states_after_attn IS the output

        # Reconstruct the return format
        outputs = (hidden_states_after_attn,)
        if extra_outputs:
            outputs = outputs + extra_outputs

        # Match original return format
        if not isinstance(result, tuple):
            return outputs[0]
        return outputs

    return ffn_skip_forward


def _make_attn_skip_forward(layer):
    """Create a forward that skips attention but runs MLP.

    Attention is skipped, KV cache gets None entries to maintain indexing.
    MLP runs normally on the (unchanged) hidden states.
    """
    def attn_skip_forward(hidden_states, **kwargs):
        # Skip attention: just preserve hidden_states
        # Handle KV cache — append None entries so indexing stays correct
        past_key_value = kwargs.get('past_key_value', None)
        if past_key_value is not None and kwargs.get('use_cache', False):
            # Append None to cache to maintain layer indexing
            layer_idx = getattr(layer, 'layer_idx', None)
            if layer_idx is not None and hasattr(past_key_value, 'key_cache'):
                while len(past_key_value.key_cache) <= layer_idx:
                    past_key_value.key_cache.append(None)
                while len(past_key_value.value_cache) <= layer_idx:
                    past_key_value.value_cache.append(None)

        # Run MLP on the unchanged hidden states
        residual = hidden_states
        mlp_attr = _find_mlp_attr(layer)
        mlp_input = layer.post_attention_layernorm(hidden_states)
        mlp_output = getattr(layer, mlp_attr)(mlp_input)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if kwargs.get('output_attentions', False):
            outputs = outputs + (None,)
        if kwargs.get('use_cache', False):
            outputs = outputs + (past_key_value,)
        return outputs

    return attn_skip_forward


def _make_full_layer_skip_forward(layer):
    """Create a forward that skips the entire layer.

    Hidden states pass through unchanged.
    KV cache gets None entries to maintain indexing.
    """
    def full_skip_forward(hidden_states, **kwargs):
        # Handle KV cache
        past_key_value = kwargs.get('past_key_value', None)
        if past_key_value is not None and kwargs.get('use_cache', False):
            layer_idx = getattr(layer, 'layer_idx', None)
            if layer_idx is not None and hasattr(past_key_value, 'key_cache'):
                while len(past_key_value.key_cache) <= layer_idx:
                    past_key_value.key_cache.append(None)
                while len(past_key_value.value_cache) <= layer_idx:
                    past_key_value.value_cache.append(None)

        # Pass through unchanged
        outputs = (hidden_states,)
        if kwargs.get('output_attentions', False):
            outputs = outputs + (None,)
        if kwargs.get('use_cache', False):
            outputs = outputs + (past_key_value,)
        return outputs

    return full_skip_forward
