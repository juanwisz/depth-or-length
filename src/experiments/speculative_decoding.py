#!/usr/bin/env python3
"""FFN-only self-speculative decoding.

The practical contribution: use an FFN-skipped draft model for
speculative decoding, verified by the full model. This gives
LOSSLESS acceleration (same output distribution as full model).

Key insight: Since FFN-only skipping preserves attention (our lead finding),
the draft model maintains better CoT coherence than full-layer-skipped drafts.
This means higher acceptance rates → faster decoding.

Comparison baseline: LayerSkip's full-layer-skip draft.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from depth_control.skip_manager import apply_skip, get_skip_layers


def speculative_decode(
    model,
    tokenizer,
    prompt: str,
    draft_skip_type: str = "ffn_only",
    draft_skip_pct: float = 30,
    draft_k: int = 5,
    max_new_tokens: int = 2048,
    seed: int = 42,
    cold_start: int = 4,
    cold_end: int = 4,
) -> Dict:
    """Self-speculative decoding with skip-based draft model.

    Args:
        model: Full model (used for both drafting and verification).
        tokenizer: Tokenizer.
        prompt: Input prompt.
        draft_skip_type: Skip type for draft ("ffn_only" or "full_layer").
        draft_skip_pct: Skip percentage for draft model.
        draft_k: Number of draft tokens per step.
        max_new_tokens: Maximum total tokens to generate.
        seed: Random seed.
        cold_start: Protected initial layers.
        cold_end: Protected final layers.

    Returns:
        Dict with: generation_text, total_tokens, accepted_tokens,
                    acceptance_rate, wall_clock_seconds, draft_time, verify_time.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    # Determine skip layers
    from infrastructure.model_loader import get_layer_modules
    num_layers = len(get_layer_modules(model))
    skip_layers = get_skip_layers(
        num_layers, draft_skip_pct,
        cold_start=cold_start, cold_end=cold_end,
    )

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_draft_time = 0
    total_verify_time = 0

    start_time = time.time()

    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        # === Draft phase: generate k tokens with skipped model ===
        draft_start = time.time()
        draft_tokens = []
        draft_logits = []
        current = generated.clone()

        with apply_skip(model, draft_skip_type, skip_layers):
            for _ in range(draft_k):
                with torch.no_grad():
                    outputs = model(current)
                    logits = outputs.logits[:, -1, :]
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    draft_tokens.append(next_token)
                    draft_logits.append(logits)
                    current = torch.cat([current, next_token], dim=-1)

        total_draft_time += time.time() - draft_start
        total_draft_tokens += draft_k

        # === Verify phase: full model forward pass on all draft tokens ===
        verify_start = time.time()
        verify_input = torch.cat([generated] + draft_tokens, dim=-1)

        with torch.no_grad():
            verify_outputs = model(verify_input)
            verify_logits = verify_outputs.logits

        total_verify_time += time.time() - verify_start

        # === Accept/reject draft tokens ===
        n_accepted = 0
        for j in range(draft_k):
            # Position of the j-th draft token in the verify sequence
            pos = generated.shape[1] + j - 1
            if pos >= verify_logits.shape[1]:
                break

            verify_next = verify_logits[:, pos, :].argmax(dim=-1)
            draft_next = draft_tokens[j].squeeze(-1)

            if verify_next.item() == draft_next.item():
                n_accepted += 1
            else:
                # Reject: use verify model's token instead
                generated = torch.cat([
                    generated,
                    *draft_tokens[:j],
                    verify_next.unsqueeze(-1),
                ], dim=-1)
                break
        else:
            # All accepted, also get the bonus token from verify
            bonus_pos = generated.shape[1] + draft_k - 1
            if bonus_pos < verify_logits.shape[1]:
                bonus_token = verify_logits[:, bonus_pos, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated] + draft_tokens + [bonus_token], dim=-1)
                n_accepted += 1
            else:
                generated = torch.cat([generated] + draft_tokens, dim=-1)

        total_accepted_tokens += n_accepted

        # Check for EOS
        if generated[0, -1].item() == tokenizer.eos_token_id:
            break

    wall_clock = time.time() - start_time

    # Decode
    gen_ids = generated[0, input_ids.shape[1]:]
    generation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0

    return {
        "generation_text": generation_text,
        "total_tokens": len(gen_ids),
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "acceptance_rate": round(acceptance_rate, 4),
        "wall_clock_seconds": round(wall_clock, 3),
        "draft_time_seconds": round(total_draft_time, 3),
        "verify_time_seconds": round(total_verify_time, 3),
        "draft_skip_type": draft_skip_type,
        "draft_skip_pct": draft_skip_pct,
        "draft_k": draft_k,
    }


def compare_speculative_methods(
    model,
    tokenizer,
    prompts: List[str],
    output_path: str = None,
) -> Dict:
    """Compare FFN-only vs full-layer self-speculative decoding.

    This generates Figure 5 data.
    """
    import json

    results = {
        "ffn_only": [],
        "full_layer": [],
        "baseline": [],
    }

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}]")

        # FFN-only draft
        ffn_result = speculative_decode(
            model, tokenizer, prompt,
            draft_skip_type="ffn_only", draft_skip_pct=30,
        )
        results["ffn_only"].append(ffn_result)

        # Full-layer draft
        full_result = speculative_decode(
            model, tokenizer, prompt,
            draft_skip_type="full_layer", draft_skip_pct=20,
        )
        results["full_layer"].append(full_result)

        # Baseline: no speculative decoding
        from infrastructure.generation import generate_with_budget
        baseline = generate_with_budget(model, tokenizer, prompt)
        results["baseline"].append(baseline)

    # Summary
    summary = {}
    for method in ["ffn_only", "full_layer"]:
        method_results = results[method]
        summary[method] = {
            "mean_acceptance_rate": round(
                sum(r["acceptance_rate"] for r in method_results) / len(method_results), 4
            ),
            "mean_wall_clock": round(
                sum(r["wall_clock_seconds"] for r in method_results) / len(method_results), 3
            ),
            "mean_tokens": round(
                sum(r["total_tokens"] for r in method_results) / len(method_results), 0
            ),
        }

    baseline_time = sum(
        r["wall_clock_seconds"] for r in results["baseline"]
    ) / len(results["baseline"])

    for method in ["ffn_only", "full_layer"]:
        summary[method]["speedup"] = round(
            baseline_time / summary[method]["mean_wall_clock"], 2
        )

    summary["baseline_mean_wall_clock"] = round(baseline_time, 3)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({"summary": summary, "details": results}, f, indent=2)
        print(f"Saved results to {output_path}")

    return summary
