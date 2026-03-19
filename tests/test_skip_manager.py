"""Tests for skip_manager: FFN-only, full-layer, and attention-only skipping.

Tests verify:
1. Skip layer selection (middle-out, cold regions protected)
2. apply_skip context manager properly patches and restores layer forward
3. FFN skip preserves attention output but removes MLP contribution
4. Full-layer skip passes hidden states through unchanged
5. Forward is restored after context manager exits
"""

import torch
import torch.nn as nn
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_control.skip_manager import (
    get_skip_layers,
    apply_skip,
)


# ── Skip layer selection tests ──


class TestGetSkipLayers:
    def test_basic_middle_out(self):
        layers = get_skip_layers(28, skip_pct=30, cold_start=4, cold_end=4)
        assert len(layers) == 6
        assert all(4 <= l < 24 for l in layers)

    def test_cold_regions_protected(self):
        layers = get_skip_layers(28, skip_pct=50, cold_start=4, cold_end=4)
        for l in layers:
            assert l >= 4
            assert l < 24

    def test_zero_skip(self):
        assert get_skip_layers(28, skip_pct=0) == []

    def test_100_pct_skip(self):
        layers = get_skip_layers(28, skip_pct=100, cold_start=4, cold_end=4)
        assert len(layers) == 20

    def test_small_model(self):
        layers = get_skip_layers(12, skip_pct=30, cold_start=2, cold_end=2)
        assert len(layers) > 0
        assert all(2 <= l < 10 for l in layers)

    def test_uniform_strategy(self):
        layers = get_skip_layers(28, skip_pct=30, strategy="uniform")
        assert len(layers) == 6
        diffs = [layers[i+1] - layers[i] for i in range(len(layers)-1)]
        assert max(diffs) - min(diffs) <= 2

    def test_sorted_output(self):
        for strategy in ["middle", "uniform", "random"]:
            layers = get_skip_layers(28, skip_pct=30, strategy=strategy)
            assert layers == sorted(layers)


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
    def __init__(self, dim: int, layer_idx: int = 0):
        super().__init__()
        self.self_attn = MockAttention(dim)
        self.mlp = MockMLP(dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.layer_idx = layer_idx

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
    def __init__(self, num_layers: int = 8, dim: int = 32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLayer(dim, layer_idx=i) for i in range(num_layers)]
        )


class TestApplySkip:
    def setup_method(self):
        self.model = MockModel(num_layers=8, dim=32)
        self.x = torch.randn(1, 5, 32)

    def test_ffn_skip_replaces_forward(self):
        layer = self.model.model.layers[4]
        x = torch.randn(1, 5, 32)
        normal_out = layer(x)

        with apply_skip(self.model, 'ffn_only', [4]):
            skip_out = layer(x)
            if isinstance(skip_out, tuple):
                skip_out = skip_out[0]
            # FFN skip should produce different output
            assert not torch.allclose(normal_out, skip_out, atol=1e-4)

    def test_full_layer_skip_pass_through(self):
        layer = self.model.model.layers[4]
        x = torch.randn(1, 5, 32)

        with apply_skip(self.model, 'full_layer', [4]):
            skip_out = layer(x)
            if isinstance(skip_out, tuple):
                skip_out = skip_out[0]
            assert torch.allclose(skip_out, x, atol=1e-6)

    def test_forward_restored_after_context(self):
        layer = self.model.model.layers[4]
        x = torch.randn(1, 5, 32)
        normal_out = layer(x)

        with apply_skip(self.model, 'full_layer', [4]):
            pass

        restored_out = layer(x)
        assert torch.allclose(restored_out, normal_out, atol=1e-6)

    def test_multiple_layers_skipped(self):
        skip_layers = [3, 4, 5]
        x = torch.randn(1, 5, 32)

        # Get normal output through all layers
        h_normal = x.clone()
        for layer in self.model.model.layers:
            h_normal = layer(h_normal)

        # Get output with FFN skip on middle layers
        h_skip = x.clone()
        with apply_skip(self.model, 'ffn_only', skip_layers):
            for layer in self.model.model.layers:
                out = layer(h_skip)
                h_skip = out[0] if isinstance(out, tuple) else out

        assert not torch.allclose(h_normal, h_skip, atol=1e-4)

    def test_empty_skip_layers_is_noop(self):
        x = torch.randn(1, 5, 32)
        layer = self.model.model.layers[4]
        normal_out = layer(x)

        with apply_skip(self.model, 'ffn_only', []):
            noop_out = layer(x)

        assert torch.allclose(normal_out, noop_out, atol=1e-6)

    def test_ffn_skip_changes_output(self):
        h = self.x.clone()
        for layer in self.model.model.layers:
            h = layer(h)
        out_normal = h.clone()

        h = self.x.clone()
        with apply_skip(self.model, 'ffn_only', [3, 4, 5]):
            for layer in self.model.model.layers:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
        out_skip = h

        assert not torch.allclose(out_normal, out_skip, atol=1e-4)

    def test_all_skip_types_work(self):
        for skip_type in ['ffn_only', 'full_layer', 'attention_only']:
            with apply_skip(self.model, skip_type, [4]):
                layer = self.model.model.layers[4]
                out = layer(self.x)
                if isinstance(out, tuple):
                    out = out[0]
                assert out.shape == self.x.shape
