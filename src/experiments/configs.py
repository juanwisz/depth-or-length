"""Shared experiment configuration and I/O utilities.

Used by both HuggingFace and vLLM experiment runners.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """One experimental condition to run."""

    name: str
    skip_type: Optional[str]  # None for baseline
    skip_pct: float  # 0 for baseline
    flop_reduction_pct: float  # estimated FLOP savings

    @property
    def is_baseline(self) -> bool:
        return self.skip_type is None


# FFN ≈ 2/3 of layer compute, attention ≈ 1/3.
# For 28 layers (DeepSeek-R1-Distill-Qwen-7B):
# Iso-FLOP matching table (for 28-layer model, 20 eligible):
# FFN 25% (5 layers, 11.9% FLOP) ↔ Full 16.7% (3.3 layers, 11.9%)
# FFN 50% (10 layers, 23.8% FLOP) ↔ Full 33.3% (6.7 layers, 23.8%)
# FFN 75% (15 layers, 35.7% FLOP) ↔ Full 50% (10 layers, 35.7%)
# Attn iso-FLOP: FFN 25% ↔ Attn 50%; FFN 50% ↔ Attn 100%

CONFIGS = [
    ExperimentConfig("baseline", None, 0, 0),
    ExperimentConfig("ffn_skip_25", "ffn_only", 25, 11.9),
    ExperimentConfig("ffn_skip_50", "ffn_only", 50, 23.8),
    ExperimentConfig("ffn_skip_75", "ffn_only", 75, 35.7),
    ExperimentConfig("full_skip_isoflop_12", "full_layer", 16.7, 11.9),
    ExperimentConfig("full_skip_isoflop_24", "full_layer", 33.3, 23.8),
    ExperimentConfig("full_skip_isoflop_36", "full_layer", 50, 35.7),
    ExperimentConfig("attn_skip_isoflop_12", "attention_only", 50, 11.9),
    ExperimentConfig("attn_skip_isoflop_24", "attention_only", 100, 23.8),
]


def load_completed_pairs(output_dir: str) -> set:
    """Load all completed (problem_id, config_name) pairs from JSONL files.

    Returns:
        Set of (problem_id, config_name) tuples already completed.
    """
    completed = set()
    if not os.path.exists(output_dir):
        return completed
    for fname in os.listdir(output_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(output_dir, fname)
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add((rec["problem_id"], rec["config"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def append_result(
    output_dir: str, model_short: str, benchmark: str,
    config: ExperimentConfig, record: Dict,
) -> None:
    """Append one result record to the appropriate JSONL file."""
    path = os.path.join(
        output_dir, f"{model_short}__{benchmark}__{config.name}.jsonl"
    )
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
