"""Tests for benchmark loaders."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.loader import load_benchmark


class TestMATH500:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load once for all tests in this class."""
        self.problems = load_benchmark("math500", subsample=10, seed=42)

    def test_loads_correct_count(self):
        assert len(self.problems) == 10

    def test_problem_structure(self):
        p = self.problems[0]
        assert "problem_id" in p
        assert "prompt" in p
        assert "ground_truth" in p
        assert "metadata" in p

    def test_problem_ids_unique(self):
        ids = [p["problem_id"] for p in self.problems]
        assert len(ids) == len(set(ids))

    def test_prompt_not_empty(self):
        for p in self.problems:
            assert len(p["prompt"]) > 20

    def test_ground_truth_not_empty(self):
        for p in self.problems:
            assert len(p["ground_truth"]) > 0

    def test_deterministic_subsample(self):
        """Same seed should give same subsample."""
        p1 = load_benchmark("math500", subsample=10, seed=42)
        p2 = load_benchmark("math500", subsample=10, seed=42)
        ids1 = [p["problem_id"] for p in p1]
        ids2 = [p["problem_id"] for p in p2]
        assert ids1 == ids2

    def test_different_seeds_different_samples(self):
        p1 = load_benchmark("math500", subsample=10, seed=42)
        p2 = load_benchmark("math500", subsample=10, seed=123)
        ids1 = set(p["problem_id"] for p in p1)
        ids2 = set(p["problem_id"] for p in p2)
        # Very unlikely to be identical with different seeds
        assert ids1 != ids2


class TestMMLUPro:
    def test_loads(self):
        problems = load_benchmark("mmlu_pro", subsample=5, seed=42)
        assert len(problems) == 5
        assert all("prompt" in p for p in problems)

    def test_has_choices(self):
        problems = load_benchmark("mmlu_pro", subsample=3, seed=42)
        for p in problems:
            # MCQ prompts should contain (A), (B), etc.
            assert "(A)" in p["prompt"]


class TestUnknownBenchmark:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("nonexistent_benchmark")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
