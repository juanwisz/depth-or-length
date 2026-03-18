"""Generation utilities with token budget enforcement and answer extraction.

Handles:
- Standard generation with configurable max_new_tokens
- Forced-answer suffix injection at budget limit
- Recording of actual tokens generated, hit_budget flag
- Answer extraction from reasoning traces
"""

import re
import time
import torch
from typing import Optional, Dict, Any, Tuple


# Forced-answer suffixes by benchmark type
FORCED_ANSWER_SUFFIXES = {
    "math": "\n\nTherefore, the final answer is: \\boxed{",
    "gpqa": "\n\nThe answer is (",
    "mmlu_pro": "\n\nThe answer is (",
    "humaneval": "\n",  # Just let it finish the function
    "livecodebench": "\n",
    "aime": "\n\nTherefore, the final answer is: \\boxed{",
    "default": "\n\nThe final answer is: ",
}


def generate_with_budget(
    model,
    tokenizer,
    prompt: str,
    token_budget: Optional[int] = None,
    benchmark_type: str = "math",
    seed: int = 42,
    temperature: float = 0.0,
    top_p: float = 1.0,
    forced_answer_max_tokens: int = 50,
) -> Dict[str, Any]:
    """Generate with optional token budget enforcement.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input prompt string.
        token_budget: Max new tokens to generate. None = unlimited (model decides).
        benchmark_type: Type of benchmark for forced-answer suffix.
        seed: Random seed.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        forced_answer_max_tokens: Extra tokens after forced suffix.

    Returns:
        Dict with: generation_text, actual_tokens_generated, hit_budget,
                    wall_clock_seconds, extracted_answer, peak_memory_mb
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()

    device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generation kwargs
    gen_kwargs = {
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if temperature > 0 else None,
    }
    # Remove None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # Set max tokens
    if token_budget is not None:
        max_new = token_budget
    else:
        max_new = 8192  # Default max for unlimited

    start_time = time.time()

    # First generation pass
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][input_len:]
    actual_tokens = len(gen_ids)
    generation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Check if we hit the budget
    hit_budget = (token_budget is not None) and (actual_tokens >= token_budget)

    # If hit budget, apply forced-answer suffix and generate a bit more
    if hit_budget:
        suffix = FORCED_ANSWER_SUFFIXES.get(benchmark_type,
                                             FORCED_ANSWER_SUFFIXES["default"])
        forced_prompt = prompt + generation_text + suffix
        forced_inputs = tokenizer(forced_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            forced_outputs = model.generate(
                **forced_inputs,
                max_new_tokens=forced_answer_max_tokens,
                **gen_kwargs,
                pad_token_id=tokenizer.pad_token_id,
            )

        forced_gen_ids = forced_outputs[0][forced_inputs["input_ids"].shape[1]:]
        forced_text = tokenizer.decode(forced_gen_ids, skip_special_tokens=True)
        generation_text = generation_text + suffix + forced_text

    wall_clock = time.time() - start_time

    # Peak memory
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "generation_text": generation_text,
        "actual_tokens_generated": actual_tokens,
        "hit_budget": hit_budget,
        "wall_clock_seconds": round(wall_clock, 3),
        "peak_memory_mb": round(peak_memory_mb, 1),
    }


def extract_answer(text: str, benchmark_type: str) -> Optional[str]:
    """Extract final answer from generation text.

    Uses benchmark-specific extraction patterns.
    Based on DeepSeek-Math evaluation scripts.

    Args:
        text: Full generation text.
        benchmark_type: Type of benchmark.

    Returns:
        Extracted answer string, or None if extraction fails.
    """
    if benchmark_type in ("math", "aime"):
        return extract_math_answer(text)
    elif benchmark_type == "gpqa":
        return extract_mcq_answer(text)
    elif benchmark_type == "mmlu_pro":
        return extract_mcq_answer(text)
    elif benchmark_type in ("humaneval", "livecodebench"):
        return extract_code_answer(text)
    else:
        return extract_math_answer(text)  # Default


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} or common math patterns.

    Based on DeepSeek-Math extraction logic.
    """
    # Try \boxed{} first (most reliable)
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Try "the answer is X" patterns
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.]+)',
        r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.]+)',
        r'=\s*([^\n,]+)\s*$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].strip()
            # Clean up common artifacts
            answer = answer.strip('.')
            answer = answer.strip('$')
            answer = answer.strip()
            if answer:
                return answer

    return None


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract multiple-choice answer (A, B, C, D, etc.)."""
    # Try "the answer is (X)" pattern
    patterns = [
        r'(?:the\s+)?answer\s+is\s*\(?([A-J])\)?',
        r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?answer\s+is\s*\(?([A-J])\)?',
        r'\b([A-J])\s*(?:is\s+)?(?:the\s+)?(?:correct|right|best)\s+answer',
        r'(?:choose|select|pick)\s+\(?([A-J])\)?',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Last resort: find any standalone letter near the end
    last_500 = text[-500:]
    matches = re.findall(r'\b([A-J])\b', last_500)
    if matches:
        return matches[-1].upper()

    return None


def extract_code_answer(text: str) -> Optional[str]:
    """Extract code from generation (for HumanEval/LiveCodeBench)."""
    # Try to find code blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Try to find function definitions
    func_matches = re.findall(r'(def\s+\w+.*?)(?=\ndef\s|\Z)', text, re.DOTALL)
    if func_matches:
        return func_matches[-1].strip()

    return text.strip()


def check_answer_correct(
    extracted: Optional[str],
    ground_truth: str,
    benchmark_type: str,
) -> bool:
    """Check if extracted answer matches ground truth.

    Args:
        extracted: Extracted answer string.
        ground_truth: Ground truth answer.
        benchmark_type: Benchmark type for comparison logic.

    Returns:
        True if correct.
    """
    if extracted is None:
        return False

    if benchmark_type in ("gpqa", "mmlu_pro"):
        # Multiple choice: exact letter match
        return extracted.strip().upper() == ground_truth.strip().upper()

    elif benchmark_type in ("math", "aime"):
        # Numeric/symbolic: normalize and compare
        return normalize_math_answer(extracted) == normalize_math_answer(ground_truth)

    elif benchmark_type in ("humaneval", "livecodebench"):
        # Code: would need execution. For now, return None.
        return None  # Requires separate evaluation

    return extracted.strip() == ground_truth.strip()


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison.

    Based on DeepSeek-Math normalization.
    """
    if answer is None:
        return ""

    # Remove whitespace, $, \text{}, etc.
    s = answer.strip()
    s = s.replace('$', '')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)  # \text{foo} -> foo
    s = s.replace('\\%', '')
    s = s.replace('%', '')
    s = s.replace(' ', '')
    s = s.replace(',', '')  # Remove thousands separator

    # Try fraction BEFORE stripping braces
    frac_match = re.match(r'^\\?frac\{([^}]+)\}\{([^}]+)\}$', s)
    if frac_match:
        try:
            num, den = float(frac_match.group(1)), float(frac_match.group(2))
            if den != 0:
                val = num / den
                if val == int(val):
                    return str(int(val))
                return str(val)
        except ValueError:
            pass

    # Try to parse as number
    try:
        val = float(s)
        # Normalize to remove trailing zeros
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    return s.lower()


def find_horl(generation_text: str, ground_truth: str, benchmark_type: str) -> Optional[int]:
    """Find Hindsight-Optimal Reasoning Length (HORL).

    Scans the generation text to find the earliest position where
    the correct answer first appears.

    Args:
        generation_text: Full generation text.
        ground_truth: Ground truth answer.
        benchmark_type: Benchmark type.

    Returns:
        Token position (approximate, character-based) where correct answer
        first appears, or None if it never appears.
    """
    if benchmark_type in ("gpqa", "mmlu_pro"):
        # Look for the correct letter
        gt = ground_truth.strip().upper()
        # Find first occurrence of a pattern that would be extracted as this answer
        patterns = [
            rf'answer\s+is\s*\(?{gt}\)?',
            rf'\b{gt}\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, generation_text, re.IGNORECASE)
            if match:
                return match.start()
    elif benchmark_type in ("math", "aime"):
        normalized_gt = normalize_math_answer(ground_truth)
        # Look for boxed answer
        for match in re.finditer(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', generation_text):
            extracted = match.group(1).strip()
            if normalize_math_answer(extracted) == normalized_gt:
                return match.start()

    return None
