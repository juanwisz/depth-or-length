"""Benchmark loaders for all target benchmarks.

Loads problems from HuggingFace datasets or local files.
Each benchmark returns a list of dicts with: problem_id, prompt, ground_truth, metadata.

Benchmarks:
- MATH-500: 500 competition math problems (primary quantitative workhorse)
- GPQA Diamond: 198 graduate-level science questions
- MMLU-Pro: 1000-problem subsample of MMLU-Pro
- AIME 2024: 30 competition math problems (qualitative only)
- HumanEval: 164 code generation problems
- LiveCodeBench: Latest code reasoning problems
"""

import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset


def load_benchmark(
    benchmark: str,
    subsample: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """Load benchmark problems.

    Args:
        benchmark: Benchmark name ("math500", "gpqa", "mmlu_pro", "aime", "humaneval", "livecodebench").
        subsample: If set, randomly sample this many problems.
        seed: Random seed for subsampling.

    Returns:
        List of problem dicts with keys: problem_id, prompt, ground_truth, metadata.
    """
    loaders = {
        "math500": _load_math500,
        "gpqa": _load_gpqa_diamond,
        "mmlu_pro": _load_mmlu_pro,
        "aime": _load_aime,
        "humaneval": _load_humaneval,
        "livecodebench": _load_livecodebench,
    }

    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(loaders.keys())}")

    problems = loaders[benchmark]()

    if subsample and subsample < len(problems):
        random.seed(seed)
        problems = random.sample(problems, subsample)

    return problems


def _load_math500() -> List[Dict]:
    """Load MATH-500 benchmark.

    Uses the HuggingFace datasets version of MATH (Hendrycks et al.).
    We use a fixed 500-problem subset for consistency.
    """
    try:
        ds = load_dataset("HuggingFaceTB/MATH-500", split="test")
    except Exception:
        # Fallback: try the full MATH dataset and take 500
        ds = load_dataset("hendrycks/competition_math", split="test")
        # Take first 500 for determinism
        ds = ds.select(range(min(500, len(ds))))

    problems = []
    for i, item in enumerate(ds):
        problem_text = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("answer", ""))

        # Extract answer from solution (look for \boxed{})
        import re
        boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
        ground_truth = boxed[-1] if boxed else solution

        prompt = _format_math_prompt(problem_text)

        problems.append({
            "problem_id": f"math500_{i:03d}",
            "prompt": prompt,
            "ground_truth": ground_truth,
            "metadata": {
                "benchmark": "math500",
                "subject": item.get("type", item.get("subject", "unknown")),
                "level": item.get("level", "unknown"),
                "original_problem": problem_text,
            }
        })

    return problems


def _load_gpqa_diamond() -> List[Dict]:
    """Load GPQA Diamond (198 graduate-level science questions)."""
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception:
        # Alternative name
        ds = load_dataset("gpqa/gpqa", "diamond", split="train")

    problems = []
    for i, item in enumerate(ds):
        question = item.get("Question", item.get("question", ""))
        correct = item.get("Correct Answer", item.get("correct_answer", ""))

        # Build MCQ prompt
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if key in item and item[key]:
                choices.append(item[key])

        # Shuffle choices deterministically
        random.seed(42 + i)
        random.shuffle(choices)
        correct_idx = choices.index(correct)
        correct_letter = chr(65 + correct_idx)  # A, B, C, D

        prompt = _format_mcq_prompt(question, choices)

        problems.append({
            "problem_id": f"gpqa_{i:03d}",
            "prompt": prompt,
            "ground_truth": correct_letter,
            "metadata": {
                "benchmark": "gpqa",
                "domain": item.get("Subdomain", "unknown"),
                "original_question": question,
                "choices": choices,
                "correct_index": correct_idx,
            }
        })

    return problems


def _load_mmlu_pro() -> List[Dict]:
    """Load MMLU-Pro (1000-problem subsample)."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Deterministic 1000-problem subsample
    random.seed(42)
    indices = random.sample(range(len(ds)), min(1000, len(ds)))
    indices.sort()

    problems = []
    for j, i in enumerate(indices):
        item = ds[i]
        question = item.get("question", "")
        choices = item.get("options", [])
        answer = item.get("answer", "")

        prompt = _format_mcq_prompt(question, choices)

        problems.append({
            "problem_id": f"mmlu_pro_{j:03d}",
            "prompt": prompt,
            "ground_truth": answer,
            "metadata": {
                "benchmark": "mmlu_pro",
                "category": item.get("category", "unknown"),
                "original_question": question,
            }
        })

    return problems


def _load_aime() -> List[Dict]:
    """Load AIME 2024 (30 problems). Qualitative only due to small size."""
    try:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    except Exception:
        # Manual AIME 2024 problems would go here
        # For now, return empty and handle gracefully
        return []

    problems = []
    for i, item in enumerate(ds):
        problem_text = item.get("problem", item.get("question", ""))
        answer = str(item.get("answer", ""))

        prompt = _format_math_prompt(problem_text)

        problems.append({
            "problem_id": f"aime_{i:03d}",
            "prompt": prompt,
            "ground_truth": answer,
            "metadata": {
                "benchmark": "aime",
                "year": item.get("year", 2024),
                "original_problem": problem_text,
            }
        })

    return problems[:30]  # Cap at 30


def _load_humaneval() -> List[Dict]:
    """Load HumanEval (164 code generation problems)."""
    ds = load_dataset("openai/openai_humaneval", split="test")

    problems = []
    for i, item in enumerate(ds):
        prompt = item.get("prompt", "")
        canonical = item.get("canonical_solution", "")
        test = item.get("test", "")
        entry_point = item.get("entry_point", "")

        problems.append({
            "problem_id": f"humaneval_{i:03d}",
            "prompt": prompt,
            "ground_truth": canonical,
            "metadata": {
                "benchmark": "humaneval",
                "task_id": item.get("task_id", f"HumanEval/{i}"),
                "entry_point": entry_point,
                "test": test,
            }
        })

    return problems


def _load_livecodebench() -> List[Dict]:
    """Load LiveCodeBench (latest split)."""
    try:
        ds = load_dataset("livecodebench/livecodebench", split="test")
        problems = []
        for i, item in enumerate(ds):
            problems.append({
                "problem_id": f"lcb_{i:03d}",
                "prompt": item.get("question", item.get("prompt", "")),
                "ground_truth": item.get("solution", item.get("answer", "")),
                "metadata": {
                    "benchmark": "livecodebench",
                    "difficulty": item.get("difficulty", "unknown"),
                }
            })
        return problems
    except Exception:
        return []


# ============ Prompt formatting ============

def _format_math_prompt(problem: str) -> str:
    """Format a math problem for reasoning models.

    Uses a simple prompt that works well with DeepSeek-R1 reasoning models.
    """
    return (
        f"Please solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:"
    )


def _format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format a multiple-choice question."""
    choice_str = "\n".join(
        f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"Please answer the following multiple-choice question. "
        f"Think step by step, then give your final answer as a single letter (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        f"Answer:"
    )
