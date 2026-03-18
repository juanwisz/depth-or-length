# Literature Verification — "Depth or Length?"
**Date**: 2026-03-18
**Status**: ✅ NOT SCOOPED — Proceed with full confidence

## Core Claims Verified as Novel

1. **FFN-only skipping preserves reasoning while full-layer skipping destroys it** — NOVEL. No paper decomposes FFN vs full-layer vs attention-only at iso-FLOP on reasoning benchmarks.
2. **Compute-accuracy surface over (FFN-skip%, absolute token budget)** — NOVEL. No paper maps this 2D surface.
3. **Mechanistic explanation: attention maintains CoT chain integrity** — NOVEL. Closest is "Layer Pruning Harms Test-Time Scaling" (Oct 2025) which shows full-layer pruning breaks reasoning but doesn't explain why or decompose by component.
4. **Self-speculative decoding with FFN-only drafts** — NOVEL. LayerSkip uses full-layer early exit; nobody has tried FFN-only drafts.

## Key Related Work (must cite and differentiate)

| Paper | Year | What They Do | Gap vs. Us |
|-------|------|-------------|-----------|
| FFN-SkipLLM (EMNLP 2024) | 2024 | FFN redundancy on knowledge tasks | Not tested on reasoning; no decomposition vs full-layer |
| Layer Pruning Harms Test-Time Scaling | 2025 | Shows 1-2 layer removal breaks reasoning scaling | Doesn't decompose by component (FFN vs attn); we explain WHY |
| Token-Budget-Aware LLM Reasoning (ACL Findings 2025) | 2025 | Token budget sensitivity | Length-only; no depth interaction |
| Think Deep, Not Just Long (Feb 2026) | 2026 | Deep-thinking ratio metric | Observational; we provide mechanistic explanation |
| Art of Efficient Reasoning (Feb 2026) | 2026 | RL reward shaping for length control | Length-only; no layer decomposition; complementary |
| SWIFT (ICLR 2025) | 2025 | On-the-fly layer skipping | General; we specialize to FFN-only + explain why attn matters |
| LayerSkip (Meta, ACL 2024) | 2024 | Self-speculative decoding via layer early exit | Full-layer only; we show FFN-only is the safe axis |
| SelfBudgeter (May 2025) | 2025 | Adaptive token allocation | Length-only; orthogonal to depth |
| Survey of Efficient Reasoning (Dec 2025) | 2025 | Taxonomy of methods | Descriptive; doesn't synthesize our FFN-only finding |
| Theory of Inference Compute (June 2025) | 2025 | Theoretical framework for inference scaling | Different axis (algorithm-level, not architecture-level) |

## Searches Conducted

1. "FFN skipping" AND "reasoning" — No direct overlap found
2. "layer skipping" AND "chain-of-thought" — Layer Pruning Harms paper is closest but doesn't decompose
3. "depth" AND "length" AND "tradeoff" AND "reasoning" — Think Deep Not Just Long is observational only
4. "compute allocation" AND "reasoning" AND "inference" — Token budget work exists but all length-only
5. "self-speculative decoding" AND "reasoning" — Not yet published for reasoning models
6. "FFN redundancy" AND "reasoning model" — Nobody has tested FFN-SkipLLM on reasoning
7. SkipDecode citations 2025-2026 — Extensions to vision/multimodal, not reasoning
8. FFN-SkipLLM citations 2025-2026 — Applied to general LLMs, not reasoning models
9. "overthinking" AND "layer" — Reasoning Theater, TERMINATOR focus on length not depth
10. FlexiDepth (March 2026) — Dynamic depth for general efficiency, not reasoning-specific decomposition

## EMNLP 2026 Timeline
- **ARR submission deadline: May 25, 2026** (11:59 PM UTC-12)
- ~2 months from today — tight but feasible

## Conclusion
All three core claims are novel. The competitive landscape is active but orthogonal. Proceed immediately to Phase 0.
