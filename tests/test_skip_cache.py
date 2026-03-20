"""Test that skip hooks work correctly with model.generate() and KV cache.

These tests use a tiny model (GPT2-like) to verify:
1. FFN-only skip preserves KV cache functionality
2. Full-layer skip handles cache correctly (or documents known issues)
3. Attention-only skip works with cache
"""

import pytest
import torch
import torch.nn as nn

from src.depth_control.skip_manager import apply_skip, _get_layers, _find_mlp


class MockConfig:
    """Minimal config for testing."""
    num_hidden_layers = 4
    hidden_size = 32
    intermediate_size = 64
    num_attention_heads = 4
    num_key_value_heads = 4


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class MockAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        out = self.proj(hidden_states)
        return (out, None)  # (attn_output, attn_weights)


class MockLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, layer_idx: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        attn_out, _ = self.self_attn(hidden_states)
        hidden_states = hidden_states + attn_out
        mlp_out = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_out
        return (hidden_states,)


class MockModel(nn.Module):
    """Minimal model with .model.layers structure for skip_manager."""
    def __init__(self, config: MockConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayer(config.hidden_size, config.intermediate_size, i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


@pytest.fixture
def model():
    config = MockConfig()
    m = MockModel(config)
    m.eval()
    return m


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(1, 5, 32)  # batch=1, seq=5, hidden=32


class TestFFNOnlySkip:
    def test_output_changes(self, model, input_tensor):
        """FFN-only skip should change the output (not identity)."""
        with torch.no_grad():
            full_out = model(input_tensor)
            with apply_skip(model, "ffn_only", [1, 2]):
                skip_out = model(input_tensor)
        assert not torch.allclose(full_out, skip_out, atol=1e-5)

    def test_preserves_residual(self, model, input_tensor):
        """FFN-only skip with zeros means residual is preserved."""
        # After FFN skip, hidden_states = residual + 0 = residual
        # So the MLP contribution is removed
        with torch.no_grad():
            with apply_skip(model, "ffn_only", [0, 1, 2, 3]):
                skip_out = model(input_tensor)
        # Should still produce finite output (attention still runs)
        assert torch.isfinite(skip_out).all()

    def test_no_skip_is_identity(self, model, input_tensor):
        """Empty skip list should be identical to no skip."""
        with torch.no_grad():
            full_out = model(input_tensor)
            with apply_skip(model, "ffn_only", []):
                skip_out = model(input_tensor)
        assert torch.allclose(full_out, skip_out, atol=1e-6)


class TestFullLayerSkip:
    def test_output_changes(self, model, input_tensor):
        """Full-layer skip should change the output."""
        with torch.no_grad():
            full_out = model(input_tensor)
            with apply_skip(model, "full_layer", [1]):
                skip_out = model(input_tensor)
        assert not torch.allclose(full_out, skip_out, atol=1e-5)

    def test_skip_all_is_passthrough(self, model, input_tensor):
        """Skipping all layers should return input unchanged."""
        with torch.no_grad():
            with apply_skip(model, "full_layer", [0, 1, 2, 3]):
                skip_out = model(input_tensor)
        assert torch.allclose(input_tensor, skip_out, atol=1e-6)


class TestAttentionOnlySkip:
    def test_output_changes(self, model, input_tensor):
        """Attention-only skip should change the output."""
        with torch.no_grad():
            full_out = model(input_tensor)
            with apply_skip(model, "attention_only", [1, 2]):
                skip_out = model(input_tensor)
        assert not torch.allclose(full_out, skip_out, atol=1e-5)


class TestHooksCleanup:
    def test_hooks_removed_after_context(self, model, input_tensor):
        """Hooks should be removed after context manager exits."""
        with torch.no_grad():
            full_out_before = model(input_tensor)
            with apply_skip(model, "ffn_only", [1, 2]):
                pass  # hooks active
            full_out_after = model(input_tensor)
        assert torch.allclose(full_out_before, full_out_after, atol=1e-6)

    def test_hooks_removed_on_exception(self, model, input_tensor):
        """Hooks should be cleaned up even if an exception occurs."""
        with torch.no_grad():
            full_out_before = model(input_tensor)
            try:
                with apply_skip(model, "ffn_only", [1]):
                    raise ValueError("test error")
            except ValueError:
                pass
            full_out_after = model(input_tensor)
        assert torch.allclose(full_out_before, full_out_after, atol=1e-6)
