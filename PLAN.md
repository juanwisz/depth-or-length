# Execution Plan: "Depth or Length?" — Revised After Critical Review

**Target**: Strong accept at EMNLP 2026 main (or equivalent tier-1: ACL, NeurIPS, ICML)
**Hardware**: Google Colab A100 (40GB) via computer use
**Execution**: Fully autonomous by Claude

**Key structural change**: The decomposition finding (FFN-only skipping preserves reasoning, full-layer skipping destroys it) is the **lead** contribution. The compute-accuracy surface is **supporting** evidence. This is more defensible because the decomposition has a clean mechanistic prediction, while the surface is subject to interaction confounds.

---

## Critical Fixes Incorporated

| # | Problem | Fix |
|---|---------|-----|
| 1 | % of natural length confounded with difficulty | Absolute token budgets: {512, 1024, 2048, 4096, 8192, unlimited}. Plus HORL oracle from TERMINATOR. |
| 2 | "Depth" overstates claim when FFN-only preserves attention | Rename axis to "per-token FFN compute fraction." Reserve "depth" for decomposition experiment. |
| 3 | FFN-skip changes generation behavior | Record actual tokens generated + hit_budget flag at every cell. 3-variable dataset. |
| 4 | 25% FFN-skip ≠ 25% full-layer-skip in FLOPs | Iso-FLOP comparisons. Measure actual FLOPs. Plot accuracy vs. actual FLOP reduction. |
| 5 | AIME has 30 problems → ~9% SE per cell | AIME qualitative case studies only. MATH-500 is the quantitative workhorse. |
| 6 | "depth=recall, length=computation" may not hold | Let data speak. Don't commit upfront. The measurement is the contribution. |
| 7 | Colab sessions die unpredictably | Per-problem append-only JSONL. `--resume` skips completed entries. Google Drive storage. |
| 8 | Forced-answer extraction may fail | Validate on 50-problem subsample. Use DeepSeek-Math extraction scripts. Must be ≥90% success. |
| 9 | Surface is subject to interaction confounds | Lead with decomposition (clean mechanistic prediction). Surface is supporting evidence. |
| 10 | No real baselines | Reproduce TokenSkip, LayerSkip self-speculative decoding, HORL oracle. |

---

## Colab Execution Model: Stateless Lambda

Colab is hostile infrastructure. Sessions restart without warning, local disk is wiped, GPU can be preempted. Treat every Colab session as a stateless lambda function: it clones, runs, saves, and dies. Nothing lives only in Colab.

### Two sources of truth
- **GitHub repo** = code (experiment scripts, analysis, all logic)
- **Google Drive** = data (results, logs, traces, metadata, debug info)

### Colab notebook is a ~20-line launcher
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<repo> /content/repo
!cd /content/repo && git checkout <sha>
!pip install -r /content/repo/requirements.txt
!python /content/repo/src/experiments/run_experiment.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --benchmark math500 --ffn_skip_pct 30 --token_budget 2048 \
    --output_dir /content/drive/MyDrive/depth_or_length --resume
```
**No experiment logic in the notebook.** All logic in the repo.

### Every experiment script saves per-problem (non-negotiable)
Append one JSONL line to `results/{experiment_id}.jsonl` on Drive:
```json
{
  "problem_id": "math500_042",
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "benchmark": "math500",
  "ffn_skip_pct": 30,
  "token_budget": 2048,
  "skip_type": "ffn_only",
  "skip_layers": [4,5,6,7,8,9,10,11,12,13],
  "seed": 42,
  "accuracy": 1,
  "extracted_answer": "42",
  "ground_truth": "42",
  "actual_tokens_generated": 1847,
  "hit_budget": false,
  "generation_text": "Let me think step by step...",
  "wall_clock_seconds": 12.3,
  "peak_memory_mb": 14200,
  "timestamp": "2026-03-19T14:32:01Z",
  "git_sha": "abc123",
  "gpu_name": "NVIDIA A100-SXM4-40GB"
}
```

Per-run (not per-problem):
- `logs/{experiment_id}.log` — full stdout + stderr
- `metadata/{experiment_id}.json` — full config, pip freeze, CUDA version
- On crash: `debug/{experiment_id}_crash.log` — traceback + last successful problem_id

### Crash recovery
`--resume` loads existing JSONL, extracts completed (problem_id, config) pairs, skips them. Session dies at problem 247/500 → next session resumes at 248. No work lost. Multiple sessions can run different configs in parallel (different experiment_ids).

### Drive folder structure
```
depth_or_length/
  results/        # JSONL per experiment (append-only)
  logs/           # stdout/stderr per run
  traces/         # Full generation text for every problem
  metadata/       # config + environment per run
  checkpoints/    # Model caches, intermediate state
  debug/          # Stack traces, OOM dumps, anomalies
```

### Local analysis
After experiments, sync Drive → local machine. All analysis runs locally on JSONL files. Colab is GPU-only.

---

## Phase -1: Literature Verification (1 day, BEFORE any code)

No changes from original. Search web/arxiv/Semantic Scholar for overlap. Read suspicious papers from corpus. Document in `literature_verification.md`. If scooped, pivot before coding.

### -1.1 — Targeted searches (web + arxiv + semantic scholar)
Run the following searches. For each, check the top 20 results:
- [ ] `"depth" AND "length" AND "tradeoff" AND "reasoning" AND "LLM"` (direct overlap)
- [ ] `"FFN skipping" AND "reasoning"` (has anyone applied FFN-skip to reasoning models?)
- [ ] `"layer skipping" AND "chain-of-thought"` (adaptive depth for CoT specifically)
- [ ] `"overthinking" AND "layer" AND "depth"` (does overthinking literature touch depth?)
- [ ] `"compute allocation" AND "reasoning" AND "inference"` (compute-optimal reasoning)
- [ ] `"token budget" AND "reasoning" AND "early exit"` (length control for reasoning)
- [ ] `"FFN redundancy" AND "reasoning model"` (FFN analysis on reasoning models)
- [ ] `"self-speculative decoding" AND "reasoning"` (has anyone done spec-dec on reasoning?)
- [ ] Check papers citing SkipDecode (2023) published in 2026 — any that extend to reasoning?
- [ ] Check papers citing FFN-SkipLLM (2024) published in 2025-2026 — any that do reasoning?

### -1.2 — Check specific concurrent work
- [ ] Search arxiv for March 2026 papers on: efficient reasoning, adaptive depth reasoning, layer skipping reasoning
- [ ] Check EMNLP 2026 / ACL 2026 submission deadlines — what's the timeline?
- [ ] Check if FlexiDepth (March 2026) or any other recent paper does depth-length tradeoff analysis

### -1.3 — Read key papers from corpus that might overlap
- [ ] `A_Theory_of_Inference_Compute__2025.pdf` — partially read, confirmed no overlap. Read fully to be sure.
- [ ] `The_Art_of_Efficient_Reasoning_2026.pdf` — title suggests possible overlap.
- [ ] `Efficient_Reasoning_Through_Su_2025.pdf` — check if it studies depth vs. length.
- [ ] `A_Survey_of_Efficient_Reasonin_2025.pdf` — check what's covered in the survey.
- [ ] `Learning_Query-Aware_Budget-Ti_2026.pdf` — "budget" in title, check for overlap.
- [ ] Any paper with "overthinking" or "compute budget" in the title.

### -1.4 — Outcome documentation
- [ ] Create `literature_verification.md` with search results, relevance assessments, confirmed novelty claims, and parts that are already done by others.
- [ ] Update PLAN.md if any findings change the experimental design.

**Exit criterion**: Written documentation that our core claims (FFN-only decomposition for reasoning, compute-accuracy surface) are NOT scooped. If scooped, pivot before writing any code.

---

## Phase 0: Infrastructure — Clone, Reproduce, Extend (3 days)

**Philosophy: NEVER build from scratch. Clone repos, reproduce results, extend minimally.**

### 0.1 — Clone and set up existing repos
- [ ] Clone `EleutherAI/lm-evaluation-harness` — benchmark evaluation backbone
- [ ] Clone `facebookresearch/LayerSkip` — baseline + self-speculative decoding reference
- [ ] Clone `hemingkx/TokenSkip` — length-reduction baseline
- [ ] Clone DeepSeek-Math evaluation scripts — answer extraction for MATH/AIME
- [ ] Clone GPQA official evaluation scripts
- [ ] Search for and clone FFN-SkipLLM code (check paper for GitHub link)
- [ ] Set up Google Drive folder structure: `gdrive/depth_or_length/{results, logs, traces, metadata, checkpoints, debug}`

### 0.2 — Reproduce published baselines (CRITICAL)
Before ANY modification, reproduce known numbers:
- [ ] DeepSeek-R1-Distill-Qwen-7B on MATH-500 via lm-eval-harness. Target: ~83%. Must be within 2%.
- [ ] Same on GPQA Diamond. Target: ~50-55%.
- [ ] Same on MMLU-Pro subsample.
- [ ] TokenSkip on MATH-500 (their checkpoint). Reproduce their published numbers.
- [ ] Document all in `results/baselines.jsonl`.

**These reproduced baselines ARE our full-model control numbers.** Everything else is measured as delta from these.

### 0.3 — Implement per-token compute control (~50 lines total)
- [ ] **FFN skip**: monkey-patch `layer.mlp.forward` → identity for selected layers. ~20 lines.
- [ ] **Full-layer skip**: skip entire layer in forward loop. ~10 lines.
- [ ] **Attention skip**: monkey-patch `layer.self_attn.forward` → identity. ~20 lines.
- [ ] **Skip scheduler**: given skip_pct and total_layers, return middle-layer indices (protect first 4, last 4 per FFN-SkipLLM cold regions). ~15 lines.
- [ ] **FLOP counter**: measure actual FLOPs per configuration (use torchprofile or manual counting based on architecture). Required for iso-FLOP comparisons.

### 0.4 — Implement length control
- [ ] `max_new_tokens` = absolute budget from {512, 1024, 2048, 4096, 8192}. No percentages.
- [ ] At budget limit: append forced-answer suffix, generate ≤50 more answer tokens.
- [ ] Record per-problem: (imposed_budget, actual_tokens_generated, hit_budget_flag).
- [ ] HORL analysis script: for full-model full-length traces, find first occurrence of correct answer token in the generation. Compute theoretical max length savings.

### 0.5 — Implement checkpointing
- [ ] Append-only JSONL per experiment.
- [ ] Each line: full schema from "Every experiment script saves" section above.
- [ ] Runner loads JSONL on start, skips completed (problem_id, config) pairs.
- [ ] Save to Google Drive after each problem. Not batched.

### 0.6 — Validation (BEFORE any real experiments)
- [ ] Reproduce full-model baselines (already done in 0.2).
- [ ] Run 50 MATH-500 problems at 30% FFN-skip, full length → verify coherent output, modest accuracy drop.
- [ ] Run 50 MATH-500 problems at full model, 1024-token budget → verify forced-answer extraction works.
- [ ] Manual check: do extracted answers match expected for the 50-problem subsample?
- [ ] Report extraction success rate. **Must be ≥90%.** If >10% failure, revise extraction methodology before proceeding.

**Exit criterion**: Baselines reproduced within 2%. Skip mechanisms produce coherent output. Answer extraction validated at ≥90% success rate.

---

## Phase 1: Pilot (2 days)

### 1.1 — Coarse decomposition pilot (the LEAD finding)
- Model: DeepSeek-R1-Distill-Qwen-7B
- Benchmark: MATH-500
- At ~20% total FLOP reduction (iso-FLOP):
  - FFN-only skip at ~30% of FFN blocks
  - Full-layer skip at ~20% of layers
  - Attention-only skip at ~60% of attention blocks
- Full length (no truncation)
- 500 problems per condition = 1,500 runs
- ~2 hours on A100

**Key question**: Does FFN-only dramatically outperform full-layer on MATH-500? If yes (≥10% accuracy gap), the decomposition finding is real. Proceed.

### 1.2 — Coarse surface pilot
- Same model, MATH-500
- FFN-skip: 0%, 25%, 50%
- Token budget: 1024, 2048, unlimited
- 9 cells × 500 problems = 4,500 runs, ~3 hours

**Key question**: Does the 3×3 surface show non-uniform structure?

### 1.3 — Sanity checks
- [ ] Full model accuracy matches reproduced baselines (within 2%)
- [ ] Generation at reduced FFN compute is coherent text, not garbage
- [ ] Actual tokens generated recorded correctly at every cell
- [ ] FLOP measurements are sane (FFN-skip FLOPs < full-model FLOPs)

**Exit criterion**: Decomposition pilot shows ≥10% gap between FFN-only and full-layer at iso-FLOP on MATH-500. Surface pilot shows some visible structure (not flat).

**Pivot if**: Decomposition gap <5% → FFN-only isn't special for reasoning. Abandon decomposition angle, focus purely on the surface. Surface is flat → depth axis is uninformative. Pivot to pure decomposition paper without surface.

---

## Phase 2: Core Experiments (7 days)

### 2.1 — Full decomposition (LEAD FINDING)

Model: DeepSeek-R1-Distill-Qwen-7B

For each skip type (FFN-only, full-layer, attention-only):
- Measure accuracy at FLOP reductions of {10%, 20%, 30%, 40%, 50%}
- On: MATH-500, GPQA Diamond, MMLU-Pro
- Full length (no truncation)
- Total: 3 skip_types × 5 levels × 3 benchmarks × ~500 problems = ~22,500 runs
- ~15 hours on A100 (across 2 sessions)

### 2.2 — Replicate decomposition on second model
DeepSeek-R1-Distill-Llama-8B on MATH-500 + GPQA (~8h)

### 2.3 — Replicate decomposition on third model
Qwen3-8B (thinking mode) on MATH-500 + GPQA (~8h)

### 2.4 — Compute-accuracy surface (SUPPORTING FINDING)

Model: DeepSeek-R1-Distill-Qwen-7B

FFN-skip rates: {0%, 10%, 20%, 30%, 40%, 50%}
Token budgets: {512, 1024, 2048, 4096, unlimited}

On: MATH-500, GPQA Diamond, MMLU-Pro
Total: 6 × 5 × 3 × ~500 = ~45,000 runs
~25 hours (across 3 sessions)

### 2.5 — HORL oracle analysis
For all full-model full-length traces (already generated), find first occurrence of correct answer. Compute theoretical max length savings per benchmark. Compare: actual truncation accuracy vs HORL oracle.

### 2.6 — Instruct baseline surface
Qwen2.5-7B-Instruct on MATH-500 + MMLU-Pro, same grid (~12h)
Purpose: show reasoning models have different surface shape than instruct models.

**Exit criterion**: Decomposition finding replicates across ≥2 model families. Surface shows some task-dependent or model-type-dependent structure.

---

## Phase 3: Baselines and Comparisons (3 days)

### 3.1 — TokenSkip baseline
Run their Qwen2.5-7B-Instruct checkpoint on MATH-500. Plot their accuracy-vs-compression curve. Compare our length-reduction results against theirs.

### 3.2 — LayerSkip self-speculative decoding
Run LayerSkip on DeepSeek-R1-Distill-Qwen-7B on MATH-500. Measure speedup and accuracy. Note: LayerSkip may require their training recipe. If so, use their released checkpoint (Llama-based) and compare on that model family instead.

### 3.3 — Our lossless variant: FFN-only self-speculative decoding
- Draft: skip 30-40% FFN blocks, generate k tokens
- Verify: full model forward pass
- Measure acceptance rate and wall-clock speedup
- Compare against LayerSkip's full-layer-skip draft

---

## Phase 4: Analysis and Figures (4 days)

### 4.1 — Core figures (in order of importance)

1. **Figure 1 — The decomposition (THE figure)**: Accuracy vs. FLOP reduction. Three curves (FFN-only, full-layer, attention-only) on MATH-500 and GPQA. Shows FFN-only degrades gently, full-layer collapses. Immediately communicates the finding.
2. **Figure 2 — Generalization**: Same decomposition curves for 3 model families, overlaid. Shows the finding is universal.
3. **Figure 3 — The surface**: Heatmaps of accuracy over (FFN-skip%, token-budget) for MATH-500 vs GPQA vs MMLU-Pro. Shows task-dependent structure.
4. **Figure 4 — Compute-optimal allocation**: Given a total FLOP budget, accuracy under different allocation strategies. Shows optimal is neither all-depth nor all-length.
5. **Figure 5 — Lossless speedup**: Acceptance rate and wall-clock speedup of FFN-only self-speculative decoding.

### 4.2 — Statistical analysis
- [ ] Bootstrap CIs on all accuracy numbers
- [ ] Significance tests for decomposition (FFN-only vs full-layer), p < 0.01
- [ ] Effect sizes for key comparisons

### 4.3 — Token-level analysis
- [ ] For problems where FFN-skip hurts: which tokens diverge? Planning tokens ("Let me think") or computation tokens ("3×7=21")? This tests Juan's intuition about which tokens carry heaviest computation.
- [ ] Per-problem analysis: are depth-sensitive problems systematically different from length-sensitive problems? (Difficulty, topic, solution length?)

### 4.4 — HORL comparison
- [ ] For each benchmark: HORL theoretical max savings vs. our actual truncation results
- [ ] How close do we get to the oracle?

---

## Phase 5: Robustness and Extensions (5 days)

### 5.1 — Scale test
DeepSeek-R1-Distill-Qwen-14B (4-bit quantized) on MATH-500 + GPQA. Does the decomposition finding hold at larger scale?

### 5.2 — Seed robustness
Re-run core experiment (DeepSeek-R1-Distill-Qwen-7B on MATH-500) with 3 different seeds. Confirm decomposition curves and heatmap structure are stable.

### 5.3 — Ablations
- Which layers to skip? (middle-only vs uniform vs random)
- Cold region size (how many first/last layers to protect?)
- Forced-answer mechanism: does answer quality depend on when we force it?

### 5.4 — LiveCodeBench + HumanEval extension
Confirm findings on code generation benchmarks.

---

## Phase 6: Paper (ONLY after Phase 4 complete)

Written around the data. Likely structure:

1. **Introduction** — two axes of reasoning compute, nobody has decomposed them
2. **Background** — SkipDecode, FFN-SkipLLM, LayerSkip, overthinking, LLMs-to-LRMs
3. **Experimental setup** — models, benchmarks, skip implementations, length control, FLOP measurement
4. **Attention Holds the Chain: FFN-only vs full-layer decomposition** (LEAD)
5. **The Compute-Accuracy Surface** (SUPPORTING)
6. **Compute-Optimal Reasoning: practical allocation**
7. **Related Work**
8. **Conclusion**

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Decomposition gap <5% (FFN-only ≈ full-layer) | Medium | High | Pivot to pure surface paper. Decomposition becomes a negative result. |
| Surface is flat (no task-dependent structure) | Medium | Medium | Lead exclusively with decomposition. Surface becomes a minor section. |
| Answer extraction fails at high truncation | Medium | High | Use DeepSeek-Math scripts. Validate at ≥90%. If failing, increase forced-answer budget or use regex extraction. |
| Actual FLOPs hard to measure accurately | Low | Medium | Use manual counting based on architecture (parameter count × tokens). torchprofile as backup. |
| AIME too noisy for any conclusions | High | Low | Already mitigated: AIME is qualitative only. MATH-500 is the workhorse. |
| Colab A100 unavailable | Low | High | Use T4/V100 with 4-bit quantization. Slower but works. |
| TokenSkip/LayerSkip hard to reproduce | Medium | Medium | Best-effort reproduction. If impossible, cite their numbers and compare directionally. |
| Concurrent work scoops decomposition finding | Medium | High | Move fast. Check arxiv weekly. Our specific framing (FFN-only as safe axis for reasoning) is novel as of March 2026. |

---

## Decision Points

**After Phase 1 (pilot)**:
- **GO** if decomposition shows ≥10% gap at iso-FLOP AND surface shows structure → proceed to Phase 2
- **PARTIAL GO** if only decomposition works → drop surface, proceed with decomposition-only paper
- **PARTIAL GO** if only surface works → drop decomposition lead, surface becomes the contribution
- **STOP** if neither shows signal → discuss with user, this direction is dead

**After Phase 2 (core)**:
- **GO** if decomposition replicates across ≥2 model families → proceed to Phase 3
- **NARROW** if only 1 model works → focus on that model family, investigate why others differ
- **PIVOT** if decomposition doesn't replicate → reframe as model-specific finding, emphasize surface

**After Phase 3 (baselines)**:
- **GO** if our FFN-only self-spec-dec outperforms LayerSkip → triple finding (decomposition + surface + practical)
- **DOWNGRADE** if self-spec-dec is comparable → practical section becomes minor

**After Phase 4 (analysis)**:
- Strong decomposition + strong surface → full paper as planned
- Strong decomposition only → decomposition paper with surface as appendix
- Strong surface only → surface paper with decomposition as appendix
- Neither strong enough for EMNLP main → target Findings of EMNLP or workshop

---

## Verification Checklist (after all experiments)

- [ ] All reproduced baselines within 2% of published numbers
- [ ] Decomposition: FFN-only outperforms full-layer by ≥10% accuracy at iso-FLOP on ≥2 reasoning benchmarks and ≥2 model families
- [ ] Surface: visible non-uniform structure in ≥1 heatmap
- [ ] All figures self-explanatory (show to someone unfamiliar — can they understand the finding?)
- [ ] Statistical significance on key comparisons (p < 0.01)
- [ ] Answer extraction validated at ≥90% success rate on held-out subsample
- [ ] Results saved as JSONL with full metadata for reproducibility
- [ ] HORL oracle computed for all benchmarks
- [ ] Actual FLOPs measured for all decomposition comparisons
- [ ] Actual tokens generated recorded at every (skip_rate, budget) cell
