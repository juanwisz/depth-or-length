"""FLOP counter for iso-FLOP comparisons.

Computes actual FLOPs per forward pass under different skip configurations.
This is CRITICAL for the decomposition finding — we must compare FFN-only
vs full-layer vs attention-only at equal FLOP reduction, not equal skip%.

We use analytical FLOP counting based on model architecture parameters,
which is exact for transformer models (no profiling noise).

FLOP formulas for a single transformer layer per token:
- Attention: 4 * h * d_head * n_heads (Q,K,V projections + output) + 2 * seq_len * d_head * n_heads (attention scores)
  Simplified (ignoring sequence length term for per-token): 4 * h * h_attn
  where h_attn = d_head * n_heads (usually = h for MHA)
  For GQA: Q proj = 2*h*d, KV proj = 2*h*d_kv, O proj = 2*h*d → total ≈ 2h*(2d + d_kv) per token
  Plus attention score computation: 2*seq_len*d per head per token
- FFN (SwiGLU): 3 * 2 * h * intermediate_size = 6 * h * intermediate_size
  (gate, up, down projections, each 2*h*i FLOPs)
- LayerNorm: ~4*h (negligible)

For simplicity, we count multiply-accumulate operations (MACs) × 2 = FLOPs.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ModelArchitecture:
    """Model architecture parameters for FLOP counting."""
    num_layers: int
    hidden_size: int  # h
    intermediate_size: int  # FFN intermediate dim
    num_attention_heads: int  # n_heads
    num_kv_heads: int  # n_kv_heads (for GQA)
    head_dim: Optional[int] = None  # d_head, defaults to h // n_heads

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


# Known model architectures
ARCHITECTURES = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelArchitecture(
        num_layers=28, hidden_size=3584, intermediate_size=18944,
        num_attention_heads=28, num_kv_heads=4,
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ModelArchitecture(
        num_layers=32, hidden_size=4096, intermediate_size=14336,
        num_attention_heads=32, num_kv_heads=8,
    ),
    "Qwen/Qwen3-8B": ModelArchitecture(
        num_layers=36, hidden_size=4096, intermediate_size=12288,
        num_attention_heads=32, num_kv_heads=8,
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelArchitecture(
        num_layers=28, hidden_size=3584, intermediate_size=18944,
        num_attention_heads=28, num_kv_heads=4,
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelArchitecture(
        num_layers=48, hidden_size=5120, intermediate_size=13824,
        num_attention_heads=40, num_kv_heads=8,
    ),
}


def get_architecture(model_name: str) -> ModelArchitecture:
    """Get architecture for a known model."""
    if model_name in ARCHITECTURES:
        return ARCHITECTURES[model_name]
    raise ValueError(f"Unknown model: {model_name}. Add to ARCHITECTURES dict.")


def flops_per_token_per_layer(arch: ModelArchitecture) -> Dict[str, float]:
    """Compute FLOPs per token for one transformer layer, broken down by component.

    Returns dict with 'ffn', 'attention', 'total' (in FLOPs, not MACs).
    """
    h = arch.hidden_size
    i = arch.intermediate_size
    n_heads = arch.num_attention_heads
    n_kv = arch.num_kv_heads
    d_head = arch.head_dim

    # Attention FLOPs per token (projection only, ignoring seq-length-dependent score computation)
    # Q projection: 2 * h * (n_heads * d_head) = 2 * h * h_q
    # K projection: 2 * h * (n_kv * d_head)
    # V projection: 2 * h * (n_kv * d_head)
    # Output projection: 2 * (n_heads * d_head) * h
    h_q = n_heads * d_head
    h_kv = n_kv * d_head
    attn_proj_flops = 2 * h * h_q + 2 * h * h_kv + 2 * h * h_kv + 2 * h_q * h
    # Note: we ignore the O(seq_len) attention score computation since it varies per token position
    # For generation (autoregressive), the attention score computation per token is:
    # 2 * n_heads * d_head * seq_len (for Q*K^T) + 2 * n_heads * d_head * seq_len (for attn*V)
    # This grows with seq_len but we report the fixed per-token cost for comparability
    attn_flops = attn_proj_flops

    # FFN FLOPs per token (SwiGLU: gate, up, down projections)
    # Gate projection: 2 * h * i
    # Up projection: 2 * h * i
    # Down projection: 2 * i * h
    # Total: 6 * h * i
    ffn_flops = 6 * h * i

    return {
        "attention": attn_flops,
        "ffn": ffn_flops,
        "total": attn_flops + ffn_flops,
        "ffn_fraction": ffn_flops / (attn_flops + ffn_flops),
    }


def compute_total_flops(
    model_name: str,
    skip_type: str,
    skip_layers: List[int],
    num_tokens: int = 1,
) -> Dict[str, float]:
    """Compute total FLOPs for a given skip configuration.

    Args:
        model_name: Model name (must be in ARCHITECTURES).
        skip_type: "ffn_only", "full_layer", "attention_only", or "none".
        skip_layers: Layer indices being skipped.
        num_tokens: Number of tokens generated.

    Returns:
        Dict with total_flops, flop_reduction_pct, per_layer_breakdown.
    """
    arch = get_architecture(model_name)
    layer_flops = flops_per_token_per_layer(arch)

    # Full model FLOPs (no skipping)
    full_flops_per_token = layer_flops["total"] * arch.num_layers
    full_total = full_flops_per_token * num_tokens

    # Compute FLOPs with skipping
    skipped_flops_per_token = 0
    for layer_idx in range(arch.num_layers):
        if layer_idx in skip_layers:
            if skip_type == "ffn_only":
                # Skip FFN, keep attention
                skipped_flops_per_token += layer_flops["attention"]
            elif skip_type == "full_layer":
                # Skip entire layer
                skipped_flops_per_token += 0
            elif skip_type == "attention_only":
                # Skip attention, keep FFN
                skipped_flops_per_token += layer_flops["ffn"]
            else:
                skipped_flops_per_token += layer_flops["total"]
        else:
            skipped_flops_per_token += layer_flops["total"]

    actual_total = skipped_flops_per_token * num_tokens
    reduction_pct = (1 - actual_total / full_total) * 100 if full_total > 0 else 0

    return {
        "full_flops_per_token": full_flops_per_token,
        "actual_flops_per_token": skipped_flops_per_token,
        "full_total_flops": full_total,
        "actual_total_flops": actual_total,
        "flop_reduction_pct": round(reduction_pct, 2),
        "num_layers_skipped": len(skip_layers),
        "skip_type": skip_type,
        "ffn_fraction_per_layer": layer_flops["ffn_fraction"],
    }


def find_iso_flop_configs(
    model_name: str,
    target_flop_reduction_pct: float,
    tolerance_pct: float = 2.0,
) -> Dict[str, dict]:
    """Find skip configurations that achieve approximately the same FLOP reduction
    across different skip types.

    This is essential for iso-FLOP comparisons in the decomposition experiment.

    Args:
        model_name: Model name.
        target_flop_reduction_pct: Target FLOP reduction (e.g., 20%).
        tolerance_pct: Acceptable tolerance in FLOP reduction.

    Returns:
        Dict mapping skip_type -> {skip_pct, skip_layers, actual_flop_reduction}.
    """
    arch = get_architecture(model_name)
    layer_flops = flops_per_token_per_layer(arch)

    results = {}

    for skip_type in ["ffn_only", "full_layer", "attention_only"]:
        best_config = None
        best_diff = float('inf')

        # Binary search over skip percentages
        for skip_pct in range(1, 100):
            skip_layers = get_skip_layers_for_flop_target(
                arch, skip_type, layer_flops, target_flop_reduction_pct
            )
            if skip_layers is None:
                continue

            result = compute_total_flops(model_name, skip_type, skip_layers)
            diff = abs(result["flop_reduction_pct"] - target_flop_reduction_pct)

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    "skip_pct": skip_pct,
                    "skip_layers": skip_layers,
                    "actual_flop_reduction_pct": result["flop_reduction_pct"],
                    "num_layers_skipped": len(skip_layers),
                }

        # Simpler approach: iterate over number of layers to skip
        eligible_start = 4
        eligible_end = arch.num_layers - 4
        eligible = list(range(eligible_start, eligible_end))

        best_config = None
        best_diff = float('inf')

        for n_skip in range(1, len(eligible) + 1):
            # Middle-out selection
            center = len(eligible) // 2
            sorted_by_center = sorted(eligible, key=lambda x: abs(x - eligible[center]))
            skip_layers = sorted(sorted_by_center[:n_skip])

            result = compute_total_flops(model_name, skip_type, skip_layers)
            diff = abs(result["flop_reduction_pct"] - target_flop_reduction_pct)

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    "skip_layers": skip_layers,
                    "actual_flop_reduction_pct": result["flop_reduction_pct"],
                    "num_layers_skipped": n_skip,
                    "skip_pct_of_eligible": round(n_skip / len(eligible) * 100, 1),
                }

        if best_config and best_diff <= tolerance_pct:
            results[skip_type] = best_config

    return results


def get_skip_layers_for_flop_target(arch, skip_type, layer_flops, target_pct):
    """Helper to find skip layers for a FLOP reduction target."""
    # This is handled by the outer function's iteration
    return None
