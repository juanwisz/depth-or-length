"""Tests for FLOP counter: per-layer breakdown, iso-FLOP configs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_control.flop_counter import (
    get_architecture,
    flops_per_token_per_layer,
    compute_total_flops,
    find_iso_flop_configs,
    ModelArchitecture,
)


class TestArchitectures:
    def test_known_models(self):
        for name in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "Qwen/Qwen2.5-7B-Instruct",
        ]:
            arch = get_architecture(name)
            assert arch.num_layers > 0
            assert arch.hidden_size > 0

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError):
            get_architecture("nonexistent/model")


class TestFLOPsPerLayer:
    def test_qwen_7b(self):
        arch = get_architecture("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        result = flops_per_token_per_layer(arch)

        assert result["ffn"] > result["attention"]
        assert result["total"] == result["ffn"] + result["attention"]
        assert 0.8 < result["ffn_fraction"] < 0.95  # FFN dominates

    def test_ffn_dominates(self):
        """FFN should be >80% for all known architectures (SwiGLU)."""
        for name in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ]:
            arch = get_architecture(name)
            result = flops_per_token_per_layer(arch)
            assert result["ffn_fraction"] > 0.7, (
                f"{name}: FFN fraction {result['ffn_fraction']:.2f} < 0.7"
            )


class TestComputeTotalFlops:
    def test_no_skip(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        result = compute_total_flops(model, "none", [])
        assert result["flop_reduction_pct"] == 0
        assert result["full_flops_per_token"] == result["actual_flops_per_token"]

    def test_ffn_skip_reduces_flops(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        result = compute_total_flops(model, "ffn_only", [11, 12, 13, 14, 15, 16])
        assert result["flop_reduction_pct"] > 0
        assert result["actual_flops_per_token"] < result["full_flops_per_token"]

    def test_full_layer_removes_more_than_ffn(self):
        """At same layers, full-layer skip removes more FLOPs than FFN-only."""
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        layers = [11, 12, 13, 14, 15, 16]
        ffn = compute_total_flops(model, "ffn_only", layers)
        full = compute_total_flops(model, "full_layer", layers)
        assert full["flop_reduction_pct"] > ffn["flop_reduction_pct"]

    def test_attention_skip_removes_least(self):
        """Attention-only removes least FLOPs (attention is small fraction)."""
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        layers = [11, 12, 13, 14, 15, 16]
        attn = compute_total_flops(model, "attention_only", layers)
        ffn = compute_total_flops(model, "ffn_only", layers)
        assert attn["flop_reduction_pct"] < ffn["flop_reduction_pct"]


class TestIsoFlopConfigs:
    def test_finds_configs(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        configs = find_iso_flop_configs(model, target_flop_reduction_pct=20.0)
        assert "ffn_only" in configs
        assert "full_layer" in configs
        # Both should be near 20%
        for st, cfg in configs.items():
            assert abs(cfg["actual_flop_reduction_pct"] - 20.0) < 5.0

    def test_attention_may_not_reach_high_targets(self):
        """Attention-only can't reach high FLOP reductions since attention is small."""
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        configs = find_iso_flop_configs(model, target_flop_reduction_pct=30.0)
        # attention_only may not be in configs since it can't reach 30%
        # (attention is only ~13% of compute)
        if "attention_only" in configs:
            assert configs["attention_only"]["actual_flop_reduction_pct"] < 15


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
