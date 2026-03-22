# The Cost of Thinking: Inference Scaling Laws for Reasoning Models

## One-Pager — EMNLP 2026

### Core Question

Wu et al. (ICLR 2025) established that smaller models + more samples beat larger models at fixed FLOP budgets. Snell et al. (2024) showed test-time compute can substitute for 14x more parameters. But both studied **non-reasoning models** (Pythia, Llemma, PaLM-2). Do these scaling laws hold for natively reasoning models that produce long, variable-length chain-of-thought?

We hypothesize they **invert**: smaller reasoning models generate disproportionately more tokens (OckBench, NeurIPS 2025 Workshop; "When More is Less", ICML 2025 Spotlight), making each sample MORE expensive in total FLOPs despite cheaper per-token cost. Combined with quadratic attention cost over long sequences (Kinetics, Sadhukhan et al., 2025), larger reasoning models may be compute-optimal even at low budgets — the opposite of what holds for non-reasoning models.

### Why This Matters

Every production system routes between reasoning models of different sizes. The community assumes "small + many samples" is cheaper (Wu et al., Snell et al.). If this inverts for reasoning models, practitioners are making systematically wrong cost decisions. This paper provides the first FLOP-accurate scaling laws for reasoning model inference.

### Models

DeepSeek-R1-Distill-Qwen family — same architecture (Qwen2.5), same teacher (DeepSeek-R1), same distillation recipe. Clean scaling ladder:
- **1.5B** (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **7B** (deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- **14B** (deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- **32B** (deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)

### Experiments

For each model, we run k ∈ {1, 2, 4, 8} independent samples per problem (temperature=0.6), then evaluate via majority voting and best-of-k (with answer matching). We record: accuracy, output token count per sample, and compute total FLOPs = 2 × N_params × total_tokens.

#### Dataset 1: MATH-500 (500 problems)
**Why:** The standard benchmark used by Wu et al. (ICLR 2025) and Snell et al. (2024). Enables direct comparison of our reasoning-model scaling curves against their non-reasoning baselines. Covers difficulty levels 1-5, allowing difficulty-stratified analysis (as in Snell et al.'s compute-optimal strategy). Verifiable answers via symbolic matching.

#### Dataset 2: AIME 2024 + 2025 (60 problems)
**Why:** Competition-level math with near-zero contamination risk (problems released after model training cutoffs). Used as the flagship benchmark by DeepSeek-R1, s1 (Muennighoff et al., EMNLP 2025), and "Can 1B Surpass 405B?" (Liu et al., 2025). Small but high-signal — the difficulty is extreme enough that token inflation in smaller models should be maximally visible. Integer answers enable exact verification.

#### Dataset 3: GPQA Diamond (198 problems)
**Why:** Graduate-level science questions requiring domain knowledge, not just computation. Tests whether the scaling law inversion is task-dependent — "When More is Less" (Wu et al., ICML 2025) predicts optimal CoT length varies by task type. GPQA is knowledge-heavy vs MATH being computation-heavy. If the inversion profile differs across task types, that's a richer finding. Used by Conformal Thinking (2026) and Art of Scaling TTC (2025). Multiple-choice with verifiable answers.

#### Dataset 4: LiveCodeBench (latest split, ~500 problems)
**Why:** Code reasoning with recent problems (post-training contamination minimal). Tests a third modality — neither pure math nor knowledge QA. R-HORIZON (ICLR 2026) showed reasoning models fail at multi-step horizons; code problems naturally require sustained multi-step reasoning. Used by ShorterBetter (NeurIPS 2025) and DeepSeek-R1's own evaluation. Verifiable via execution-based testing.

#### Dataset 5: Omni-MATH (subsample of 500 from 4,428 problems)
**Why:** Olympiad-level math across 6 domains and 4 difficulty tiers. Used by "o3 thinks harder, not longer" (Ballon et al., 2025) as their primary benchmark. The difficulty tier structure enables fine-grained analysis: does the scaling law inversion appear only on the hardest problems (Tier 4), or across all difficulties? Stratified subsampling ensures coverage across tiers and domains. Omni-Judge provides automatic evaluation.

### Key Measurements

For each (model_size, dataset, k) configuration, we measure:
1. **pass@1 accuracy** (greedy, k=1)
2. **majority@k accuracy** (k=2,4,8)
3. **Mean output tokens per sample** — the token inflation measurement
4. **Total FLOPs per problem** = 2 × N_params × Σ(tokens across k samples) — the iso-FLOP comparison
5. **FLOPs per correct answer** = total FLOPs / accuracy — the efficiency metric

### Expected Findings

1. **Token inflation scales inversely with model size.** The 1.5B model generates 3-5x more tokens than the 32B on the same problems (consistent with OckBench and "When More is Less"). This alone may flip which model is cheaper in total FLOPs.

2. **The inference scaling law inverts.** At iso-FLOP budget, 14B×k=2 beats 7B×k=4 beats 1.5B×k=8 on reasoning benchmarks. The crossover point (where smaller models stop being compute-optimal) shifts dramatically left compared to Wu et al.'s non-reasoning curves.

3. **The inversion is task-dependent.** Computation-heavy tasks (MATH, AIME, Omni-MATH) show strong inversion; knowledge-heavy tasks (GPQA) may show a weaker effect because token inflation is less extreme when the model either knows the answer or doesn't. Code (LiveCodeBench) may show a distinct profile.

4. **Difficulty modulates the effect.** On easy problems (MATH level 1-2, Omni-MATH Tier 1), smaller models don't inflate much and may still be compute-optimal. On hard problems (level 5, Tier 4), inflation is extreme and larger models dominate. This mirrors Snell et al.'s difficulty-dependent compute-optimal strategy, but with inverted conclusions for reasoning models.

### Related Work Positioning

- **Extends** Wu et al. (ICLR 2025) and Snell et al. (2024) to reasoning models — the first iso-FLOP scaling laws for natively reasoning model families.
- **Provides FLOP-level evidence** for OckBench's token-count observation that smaller reasoning models are more expensive.
- **Complements** Kinetics (Sadhukhan et al., 2025) which shows attention cost dominates for long CoT — we show this has concrete scaling law implications.
- **Validates** "When More is Less" (ICML 2025) prediction that optimal CoT length decreases with model capability, in a scaling law context they did not study.
- **Explains** the practical observation (documented in "o3 thinks harder, not longer") that better models reason more efficiently per token — we show this has quantifiable cost implications across a model family.
- **Differs from** all length-control papers (ShorterBetter, BudgetThinker, TOPS, s1, Token-Budget-Aware) which optimize a single model's efficiency. We study the cross-model scaling question: given a FLOP budget, which model size should you use?

### Contribution

1. First inference scaling laws for natively reasoning (distilled) models with proper FLOP accounting including variable output length.
2. Demonstration that the "smaller + more samples" scaling law inverts for reasoning models.
3. Task-dependent and difficulty-dependent characterization of the inversion.
4. Practical lookup table: given a FLOP budget and task type, which model size is optimal?

### Cost Estimate

~105,000 API requests across 4 models × 5 datasets × 4 sample counts. Estimated ~$200-250 on AWS SageMaker Batch Transform or Vast.ai spot instances. Pilot (MATH-500 only, 2 models, k=1,4): ~$20-30.
