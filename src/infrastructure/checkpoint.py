"""Append-only JSONL checkpointing for crash recovery.

Each experiment writes one JSONL line per problem to a results file.
On resume, completed (problem_id, config) pairs are loaded and skipped.

This is the foundation for surviving Colab session crashes.
Google Drive is the storage backend — files persist across sessions.
"""

import json
import os
import time
import subprocess
from typing import Set, Dict, Any, Optional, List
from pathlib import Path


def get_experiment_id(
    model_name: str,
    benchmark: str,
    skip_type: str,
    ffn_skip_pct: float,
    token_budget: Optional[int],
    seed: int = 42,
) -> str:
    """Generate a unique experiment ID from config."""
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    budget_str = f"budget{token_budget}" if token_budget else "unlimited"
    skip_str = f"{skip_type}_{int(ffn_skip_pct)}pct"
    return f"{model_short}__{benchmark}__{skip_str}__{budget_str}__seed{seed}"


def load_completed(results_path: str) -> Set[str]:
    """Load completed problem IDs from existing JSONL file.

    Args:
        results_path: Path to the JSONL results file.

    Returns:
        Set of completed problem_id strings.
    """
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def append_result(results_path: str, record: Dict[str, Any]):
    """Append a single result record to JSONL file.

    Writes immediately and flushes to ensure data survives crashes.

    Args:
        results_path: Path to JSONL file.
        record: Result dict to append.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def save_run_metadata(
    metadata_path: str,
    experiment_id: str,
    config: Dict[str, Any],
):
    """Save per-run metadata (config, environment info, etc.)."""
    import platform
    import sys

    metadata = {
        "experiment_id": experiment_id,
        "config": config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Try to get git SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["git_sha"] = sha
    except Exception:
        metadata["git_sha"] = "unknown"

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            metadata["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            metadata["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    # Try to get pip freeze
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL
        ).decode()
        metadata["pip_freeze"] = freeze.split('\n')
    except Exception:
        pass

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_crash_log(debug_dir: str, experiment_id: str, error: Exception, last_problem_id: str = None):
    """Save crash information for debugging."""
    import traceback

    os.makedirs(debug_dir, exist_ok=True)
    crash_path = os.path.join(debug_dir, f"{experiment_id}_crash.log")

    with open(crash_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Crash at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        if last_problem_id:
            f.write(f"Last successful problem: {last_problem_id}\n")
        f.write(f"Error: {str(error)}\n")
        f.write(f"Traceback:\n{traceback.format_exc()}\n")


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all results from a results directory.

    Reads all JSONL files and returns a flat list of records.
    Used for analysis scripts.
    """
    all_records = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    for jsonl_file in sorted(results_path.glob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = jsonl_file.name
                    all_records.append(record)
                except json.JSONDecodeError:
                    continue

    return all_records
