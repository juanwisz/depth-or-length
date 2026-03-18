"""Tests for checkpoint module: append-only JSONL and crash recovery."""

import json
import os
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.checkpoint import (
    get_experiment_id,
    load_completed,
    append_result,
    save_run_metadata,
    load_all_results,
)


class TestExperimentId:
    def test_basic_id(self):
        eid = get_experiment_id(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "math500", "ffn_only", 30, None, 42,
        )
        assert "deepseek_r1_distill_qwen_7b" in eid
        assert "math500" in eid
        assert "ffn_only" in eid
        assert "30" in eid
        assert "unlimited" in eid
        assert "seed42" in eid

    def test_with_budget(self):
        eid = get_experiment_id(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "math500", "none", 0, 2048, 42,
        )
        assert "budget2048" in eid

    def test_different_configs_different_ids(self):
        id1 = get_experiment_id("model", "math500", "ffn_only", 30, None, 42)
        id2 = get_experiment_id("model", "math500", "full_layer", 30, None, 42)
        assert id1 != id2


class TestAppendAndLoad:
    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")

            append_result(path, {"problem_id": "p1", "accuracy": 1})
            append_result(path, {"problem_id": "p2", "accuracy": 0})
            append_result(path, {"problem_id": "p3", "accuracy": 1})

            completed = load_completed(path)
            assert "p1" in completed
            assert "p2" in completed
            assert "p3" in completed
            assert "p4" not in completed

    def test_load_nonexistent_file(self):
        completed = load_completed("/nonexistent/path.jsonl")
        assert completed == set()

    def test_append_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "new.jsonl")
            assert not os.path.exists(path)

            append_result(path, {"problem_id": "p1"})
            assert os.path.exists(path)

    def test_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")

            # Simulate first run: 3 problems done
            for i in range(3):
                append_result(path, {"problem_id": f"p{i}", "accuracy": 1})

            # Simulate resume
            completed = load_completed(path)
            assert len(completed) == 3

            problems = [{"problem_id": f"p{i}"} for i in range(5)]
            remaining = [p for p in problems if p["problem_id"] not in completed]
            assert len(remaining) == 2
            assert remaining[0]["problem_id"] == "p3"
            assert remaining[1]["problem_id"] == "p4"

    def test_load_all_results(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            append_result(path, {"problem_id": "p1", "accuracy": 1, "tokens": 100})
            append_result(path, {"problem_id": "p2", "accuracy": 0, "tokens": 200})

            results = load_all_results(d)  # takes directory, not file
            assert len(results) == 2
            assert results[0]["problem_id"] == "p1"
            assert results[1]["tokens"] == 200

    def test_corrupt_line_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            with open(path, 'w') as f:
                f.write('{"problem_id": "p1"}\n')
                f.write('corrupt line\n')
                f.write('{"problem_id": "p2"}\n')

            completed = load_completed(path)
            assert "p1" in completed
            assert "p2" in completed
            assert len(completed) == 2


class TestSaveMetadata:
    def test_save_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "meta.json")
            save_run_metadata(path, "test_exp", {"model": "test", "seed": 42})

            with open(path) as f:
                meta = json.load(f)
            assert meta["experiment_id"] == "test_exp"
            assert meta["config"]["model"] == "test"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
