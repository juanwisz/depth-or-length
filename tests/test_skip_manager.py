"""Tests for skip_manager: FFN-only, full-layer, and attention-only skipping.

Tests verify:
1. Skip layer selection (middle-out, cold regions protected)
2. Identity modules return correct shapes
3. apply_skip context manager properly patches and restores model
4. Output changes when skipping (skip is not no-op)
5. Output is restored after context manager exits
"""

import torch
import torch.nn as nn
import pytest
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_control.skip_manager import (
    get_skip_layers,
    apply_skip,
    _IdentityModule,
    _IdentityAttention,
)


# ── Skip layer selection tests ──


class TestGetSkipLayers:
    def test_basic_middle_out(self):
        layers = get_skip_layers(28, skip_pct=30, cold_start=4, cold_end=4)
        assert len(layers) == 6  # 30% of 20 eligible = 6
        assert all(4 <= l < 24 for l in layers)  # within eligible range
        # Middle layers should be selected
        center = 14  # middle of 4..23
        assert center in layers or center - 1 in layers

    def test_cold_regions_protected(self):
        layers = get_skip_layers(28, skip_pct=50, cold_start=4, cold_end=4)
        for l in layers:
            assert l >= 4, f"Layer {l} in cold start region"
            assert l < 24, f"Layer {l} in cold end region"

    def test_zero_skip(self):
        layers = get_skip_layers(28, skip_pct=0)
        assert layers == []

    def test_100_pct_skip(self):
        layers = get_skip_layers(28, skip_pct=100, cold_start=4, cold_end=4)
        assert len(layers) == 20  # all eligible layers

    def test_small_model(self):
        layers = get_skip_layers(12, skip_pct=30, cold_start=2, cold_end=2)
        assert len(layers) > 0
        assert all(2 <= l < 10 for l in layers)

    def test_uniform_strategy(self):
        layers = get_skip_layers(28, skip_pct=30, strategy="uniform")
        assert len(layers) == 6
        # Check roughly evenly spaced
        diffs = [layers[i+1] - layers[i] for i in range(len(layers)-1)]
        assert max(diffs) - min(diffs) <= 2  # roughly uniform

    def test_sorted_output(self):
        for strategy in ["middle", "uniform", "random"]:
            layers = get_skip_layers(28, skip_pct=30, strategy=strategy)
            assert layers == sorted(layers)


# ── Identity module tests ──


class TestIdentityModules:
    def test_identity_module_returns_zeros(self):
        mod = _IdentityModule()
        x = torch.randn(2, 10, 64)
        out = mod(x)
        assert out.shape == x.shape
        assert torch.all(out == 0)

    def test_identity_module_kwargs(self):
        mod = _IdentityModule()
        x = torch.randn(2, 10, 64)
        out = mod(hidden_states=x)
        assert out.shape == x.shape
        assert torch.all(out == 0)

    def test_identity_attention_returns_tuple(self):
        attn = _IdentityAttention()
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == x.shape
        assert torch.all(out[0] == 0)
        assert out[1] is None


# ── Mock model for apply_skip tests ──


class MockMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states), None


class MockLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = MockAttention(dim)
        self.mlp = MockMLP(dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(h)
        h = residual + attn_out

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h


class MockModel(nn.Module):
    """Mimics HuggingFace model structure: model.model.layers."""
    def __init__(self, num_layers: int = 8, dim: int = 32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLayer(dim) for _ in range(num_layers)]
        )


class TestApplySkip:
    def setup_method(self):
        self.model = MockModel(num_layers=8, dim=32)
        self.x = torch.randn(1, 5, 32)

    def test_ffn_skip_patches_mlp(self):
        layer = self.model.model.layers[4]
        original_mlp = layer.mlp

        with apply_skip(self.model, 'ffn_only', [4]):
            # MLP should be replaced
            assert not isinstance(layer.mlp, MockMLP)
            assert isinstance(layer.mlp, _IdentityModule)

        # Restored after context
        assert layer.mlp is original_mlp

    def test_attention_skip_patches_attn(self):
        layer = self.model.model.layers[4]
        original_attn = layer.self_attn

        with apply_skip(self.model, 'attention_only', [4]):
            assert not isinstance(layer.self_attn, MockAttention)
            assert isinstance(layer.self_attn, _IdentityAttention)

        assert layer.self_attn is original_attn

    def test_full_layer_skip_changes_output(self):
        """Full-layer skip should pass hidden states through unchanged."""
        layer = self.model.model.layers[4]
        x = torch.randn(1, 5, 32)

        # Normal output (single tensor for modern transformers)
        normal_out = layer(x)

        with apply_skip(self.model, 'full_layer', [4]):
            skip_out = layer(x)
            # Full-layer skip returns input unchanged
            assert torch.allclose(skip_out, x, atol=1e-6)

        # After restore, normal behavior returns
        restored_out = layer(x)
        assert torch.allclose(restored_out, normal_out, atol=1e-6)

    def test_multiple_layers_skipped(self):
        skip_layers = [3, 4, 5]
        originals = {i: self.model.model.layers[i].mlp for i in skip_layers}

        with apply_skip(self.model, 'ffn_only', skip_layers):
            for i in skip_layers:
                assert isinstance(self.model.model.layers[i].mlp, _IdentityModule)
            # Non-skipped layers unchanged
            assert isinstance(self.model.model.layers[0].mlp, MockMLP)
            assert isinstance(self.model.model.layers[7].mlp, MockMLP)

        for i in skip_layers:
            assert self.model.model.layers[i].mlp is originals[i]

    def test_empty_skip_layers_is_noop(self):
        layer = self.model.model.layers[4]
        original_mlp = layer.mlp

        with apply_skip(self.model, 'ffn_only', []):
            assert layer.mlp is original_mlp

    def test_ffn_skip_changes_output(self):
        """FFN skip should produce different output than no skip."""
        # Run without skip
        h = self.x.clone()
        for layer in self.model.model.layers:
            h = layer(h)
        out_normal = h.clone()

        # Run with FFN skip on middle layers
        h = self.x.clone()
        with apply_skip(self.model, 'ffn_only', [3, 4, 5]):
            for layer in self.model.model.layers:
                h = layer(h)
        out_skip = h.clone()

        assert not torch.allclose(out_normal, out_skip, atol=1e-6), \
            "FFN skip should change output"

    def test_restore_on_exception(self):
        """Model should be restored even if exception occurs inside context."""
        layer = self.model.model.layers[4]
        original_mlp = layer.mlp

        with pytest.raises(ValueError):
            with apply_skip(self.model, 'ffn_only', [4]):
                assert isinstance(layer.mlp, _IdentityModule)
                raise ValueError("test error")

        assert layer.mlp is original_mlp


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
