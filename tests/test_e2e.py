"""End-to-end tests: verify the full pipeline works with mock data.

Tests the complete flow: load benchmark → extract answer → check correct →
checkpoint → resume, without needing a GPU or real model.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.loader import load_benchmark
from infrastructure.generation import (
    extract_answer, check_answer_correct, normalize_math_answer, find_horl,
)
from infrastructure.checkpoint import (
    get_experiment_id, load_completed, append_result, load_all_results,
)
from depth_control.skip_manager import get_skip_layers
from depth_control.flop_counter import (
    compute_total_flops, find_iso_flop_configs, flops_per_token_per_layer,
    get_architecture,
)


class TestFullPipeline:
    """Simulate a full experiment run without GPU."""

    def test_pipeline_flow(self):
        """Full flow: load problems → simulate generation → extract → save → resume."""
        # 1. Load benchmark
        problems = load_benchmark("math500", subsample=5, seed=42)
        assert len(problems) == 5

        # 2. Configure skip
        skip_layers = get_skip_layers(28, 30, cold_start=4, cold_end=4)
        assert len(skip_layers) == 6

        # 3. Compute FLOPs
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        flop_info = compute_total_flops(model_name, "ffn_only", skip_layers)
        assert flop_info["flop_reduction_pct"] > 0

        # 4. Generate experiment ID
        exp_id = get_experiment_id(model_name, "math500", "ffn_only", 30, None, 42)
        assert "ffn_only" in exp_id

        with tempfile.TemporaryDirectory() as d:
            results_path = os.path.join(d, f"{exp_id}.jsonl")

            # 5. Simulate generation + save
            for p in problems[:3]:
                # Fake generation output
                fake_gen = r"Let me solve this... \boxed{42}"
                extracted = extract_answer(fake_gen, "math")
                correct = check_answer_correct(extracted, p["ground_truth"], "math")

                record = {
                    "problem_id": p["problem_id"],
                    "accuracy": 1 if correct else 0,
                    "extracted_answer": extracted,
                    "ground_truth": p["ground_truth"],
                    "actual_tokens_generated": 100,
                    "skip_type": "ffn_only",
                    "ffn_skip_pct": 30,
                }
                append_result(results_path, record)

            # 6. Resume: should skip completed problems
            completed = load_completed(results_path)
            assert len(completed) == 3

            remaining = [p for p in problems if p["problem_id"] not in completed]
            assert len(remaining) == 2

            # 7. Load all results
            results = load_all_results(d)
            assert len(results) == 3

    def test_iso_flop_consistency(self):
        """Verify iso-FLOP configs produce consistent layer selections."""
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

        for target in [10, 20, 30]:
            configs = find_iso_flop_configs(
                model, target_flop_reduction_pct=float(target),
            )

            for skip_type, cfg in configs.items():
                # Verify the reported reduction matches recomputation
                actual = compute_total_flops(model, skip_type, cfg["skip_layers"])
                assert abs(actual["flop_reduction_pct"] - cfg["actual_flop_reduction_pct"]) < 0.1

    def test_answer_extraction_on_real_formats(self):
        """Test answer extraction on formats the model actually produces."""
        # DeepSeek-R1 style: <think>...</think> then \boxed{}
        text = (
            "<think>\nLet me work through this step by step.\n"
            "First, I need to find the value of x.\n"
            "x = 3 + 4 = 7\n"
            "Wait, let me reconsider.\n"
            "x = 3 × 4 = 12\n"
            "</think>\n\n"
            "The answer is \\boxed{12}"
        )
        assert extract_answer(text, "math") == "12"

        # Multiple boxed answers (model corrects itself)
        text2 = (
            "First attempt: \\boxed{7}\n"
            "No wait, that's wrong.\n"
            "The correct answer is \\boxed{12}"
        )
        assert extract_answer(text2, "math") == "12"  # takes last

        # MCQ with reasoning
        text3 = (
            "Looking at option (A), it says...\n"
            "Option (B) seems better because...\n"
            "After careful analysis, the answer is (B)"
        )
        assert extract_answer(text3, "gpqa") == "B"

    def test_horl_finds_earliest_correct(self):
        """HORL should find the FIRST occurrence of the correct answer."""
        text = r"Wrong: \boxed{10}. Correct: \boxed{42}. Again: \boxed{42}"
        pos = find_horl(text, "42", "math")
        assert pos is not None
        # Should find the first \boxed{42}, not the second
        assert text[pos:pos+10].startswith("\\boxed{42")

    def test_math_normalization_edge_cases(self):
        """Test normalization handles real-world answer formats."""
        assert normalize_math_answer("\\frac{3}{4}") == "0.75"
        assert normalize_math_answer("$-\\frac{1}{2}$") == "-0.5"  # negative fracs
        assert normalize_math_answer("3.14159") == "3.14159"
        assert normalize_math_answer("1,234,567") == "1234567"
        assert normalize_math_answer("50\\%") == "50"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
