# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains materials for applying to compute grants for the **ELIAS Applied Lab** (Efficient Language Intelligence for Applied Autonomous Systems) at Universidad de San Andrés, Buenos Aires, Argentina.

This is **not a software project** — it is a grant-application workspace. The primary task is drafting, refining, and submitting proposals for cloud compute credits from multiple providers.

## Key People

- **PI:** Luciano Del Corro — Professor at UdeSA, former Senior Research Engineer at Microsoft Research (AI Frontiers), co-author of Orca 2, AgentInstruct, SkipDecode, ClausIE.
- **Co-investigator:** Juan Wisznia — Researcher/Graduate Student, first author on ACL 2025 paper (cost-aware LLM ranking), co-author on EMNLP 2024.

## Target Grants (priority order)

1. **Google TPU Research Cloud (TRC)** — free TPU access, short form, fast turnaround
2. **Google Cloud Research Credits** — up to $5,000 (faculty) / $1,000 (PhD)
3. **Hugging Face Community GPU Grants** — requires a working HF Space demo
4. **AWS Cloud Credit for Research** — up to $20,000 (faculty) / $5,000 (student)
5. **NVIDIA Academic Grant Program** — faculty-only, must emphasize NVIDIA stack (NeMo, TensorRT, Triton)
6. **Microsoft Azure Research Credits** — leverage Luciano's MSR coauthor network for referral

## Research Themes to Emphasize in Proposals

- **Model distillation & synthetic data:** Orca 2 (230 cites), AgentInstruct (70 cites)
- **Inference acceleration:** SkipDecode — 2-5x speedups via token-level early exit (105 cites)
- **Cost-aware algorithms:** 40-90% LLM inference cost reduction for ranking (ACL 2025)
- **Moral alignment:** Greatest Good Benchmark across 15 LLMs (EMNLP 2024)

## Active Research: EMNLP 2026 Paper (Autonomous Mode)

Claude is autonomously pursuing an EMNLP 2026 main conference paper. This section governs that work.

### Paper: "Depth or Length? The Computational Tradeoff in Chain-of-Thought Reasoning"

**Core idea**: Reasoning models spend inference compute along two axes — *per-token FFN compute* (how many FFN blocks process each token) and *length* (how many CoT tokens are generated). We decompose per-token compute into its FFN and attention components and discover that FFN-only skipping preserves reasoning while full-layer skipping destroys it. This mechanistic finding — attention maintains chain-of-thought integrity, FFN is often redundant — explains why prior adaptive-depth methods (SkipDecode, dynamic routing) break on reasoning models. We then map the compute-accuracy surface over (FFN-skip rate, absolute token budget) and find task-dependent structure across benchmarks.

**Contribution hierarchy**:
1. **Lead**: FFN-only skip preserves reasoning; full-layer skip destroys it. Mechanistic explanation: attention maintains CoT chain integrity.
2. **Supporting**: The compute-accuracy surface shows different degradation profiles across task types (MATH-500 vs GPQA vs MMLU-Pro).
3. **Practical**: Self-speculative decoding with FFN-only drafts yields lossless speedup.

**Key methodological choices** (from critical review):
- "Per-token FFN compute fraction" instead of "depth" — since FFN-only skipping preserves full attention depth.
- Absolute token budgets {512, 1024, 2048, 4096, 8192, unlimited} instead of percentages of natural length — avoids confounding with problem difficulty.
- Iso-FLOP comparisons for decomposition — FFN-only vs full-layer must be compared at equal FLOP reduction, not equal skip percentage.
- AIME (30 problems) is qualitative case studies only, not in quantitative heatmaps (~9% SE per cell is too noisy).
- HORL (hindsight-optimal reasoning length) from TERMINATOR as post-hoc oracle for length analysis.
- No upfront commitment to "depth=recall, length=computation" — let data speak.

### Autonomy Scope
- Claude may create directories, Python files, experiment scripts, notebooks, and figures **without asking** as long as they serve the paper.
- Claude may run experiments on Google Colab (A100) **without asking**.
- Claude may install Python packages into `.venv` or Colab environments **without asking**.
- Claude may read any PDF in `papers_early_exit/` **without asking**.
- Claude should still **ask before**: any git push, any external API call that costs money, any deletion of existing files, any commit to main.

### Hardware & Execution Model
- **Primary**: Google Colab with A100 (40GB), accessed via computer use.
- **Colab is stateless.** Treat every session as a lambda: clone repo, run script, save to Drive, die. Nothing lives only in Colab.
- **Code** lives in a GitHub repo (source of truth). Colab notebook is a ~20-line launcher that clones the repo and runs a script. No experiment logic in notebooks.
- **Data** lives on Google Drive (append-only JSONL per experiment). Every problem saves immediately. `--resume` flag skips completed work on restart.
- **Every run saves**: full generation text, extracted answers, accuracy, actual token counts, wall-clock time, GPU info, git SHA, full config. Plus logs, metadata, and crash dumps.
- **Analysis runs locally** on downloaded JSONL. Colab is GPU-only.

### Models (all HuggingFace, all fit in A100-40GB in fp16/bf16)
- **Reasoning (primary)**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Reasoning (secondary)**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Reasoning (extension)**: `Qwen/Qwen3-8B` (thinking mode)
- **Instruct baseline**: `Qwen/Qwen2.5-7B-Instruct`
- **Scale test**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (4-bit quantization)

### Benchmarks
- **Computation-heavy**: MATH-500, AIME 2024 (30 problems)
- **Knowledge-heavy**: GPQA Diamond (198 problems)
- **Code reasoning**: LiveCodeBench (latest available split)
- **Standard control**: MMLU-Pro (1000-problem subsample), HumanEval
- Avoid saturated benchmarks (GSM8K, MMLU, ARC-Easy).

### Method Constraints
- **Training is OK** for extensions: LoRA, probes, small MLPs, calibration on <10K samples.
- **No large-scale pretraining.** We're not training foundation models.
- **Publicly available models only.**
- **Reproducibility is sacred.** Every experiment fully scripted. Seeds fixed. No manual steps.
- **The bar is "strong accept."** If results look borderline, pivot or add experiments. Do not spin.

### Engineering Philosophy: Reproduce and Extend
**NEVER write experiment infrastructure from scratch.** The correct workflow is:
1. **Clone** existing repos that implement the baselines or similar methods (FFN-SkipLLM, LayerSkip, lm-evaluation-harness, etc.)
2. **Reproduce** their published results first. Verify the numbers match. This validates the setup.
3. **Extend minimally** from there — add our depth/length control on top of working, validated code.
4. **Claims are grounded** in reproduced baselines, not in numbers from scratch implementations that might have subtle bugs.

Key repos to build on:
- `EleutherAI/lm-evaluation-harness` — benchmark evaluation (MATH, GPQA, MMLU-Pro, HumanEval)
- `facebookresearch/LayerSkip` — self-speculative decoding, layer dropout training
- `hemingkx/TokenSkip` — trained length-reduction baseline on Qwen2.5
- DeepSeek-Math evaluation scripts — answer extraction for MATH/AIME
- FFN-SkipLLM code (if released) — FFN saturation measurement, skip logic
- HuggingFace `transformers` — model loading, generation
- GPQA official evaluation scripts

**Do NOT reinvent evaluation.** Use lm-evaluation-harness or the benchmark's official eval scripts. Our contribution is the depth/length manipulation, not a new eval framework.

### Research Direction — Grounded in Literature
The paper connects:
1. **SkipDecode** (Del Corro et al., 2023) — PI's work. Positional exit schedule. We show this is one point on the compute-accuracy surface, and explain why full-layer skipping fails for reasoning (attention disruption).
2. **Cost-aware LLM inference** (Wisznia et al., ACL 2025) — Juan's work. We extend cost-optimal allocation from between-models to within-model (FFN compute vs. length allocation).
3. **FFN-SkipLLM** (EMNLP 2024) — FFN blocks are 2/3 of compute and redundant in middle layers. We show this matters specifically for reasoning because attention must be preserved while FFN can be safely reduced.
4. **From LLMs to LRMs** (2026) — Dynamic depth pruning fails on reasoning. We explain WHY (attention disruption) and show FFN-only is the safe axis.
5. **LayerSkip** (Meta, ACL 2024) — Self-speculative decoding. We use their verification framework for our FFN-only lossless variant.
6. **TERMINATOR / Art of Efficient Reasoning** — HORL methodology for theoretical max length reduction. We adopt absolute token budgets and HORL oracle.
7. **TokenSkip** — Trained length-reduction baseline. We compare against their compression curves.

### Workflow: Experiments First, Paper Last
**Do NOT write LaTeX until experiments are done and the story is clear from data.**

Order:
1. **Literature verification**: Confirm novelty before writing code.
2. **Infrastructure**: Clone repos, reproduce baselines, implement FFN-skip/full-layer-skip/attention-skip and absolute token budgets. Validate answer extraction.
3. **Pilot**: Coarse decomposition pilot (FFN-only vs full-layer at iso-FLOP) + coarse surface pilot. Must show signal.
4. **Core — Decomposition (LEAD)**: Full decomposition across 3 skip types × 5 FLOP levels × 3 benchmarks × 3 model families.
5. **Core — Surface (SUPPORTING)**: Full grid over (FFN-skip%, token-budget) on all benchmarks + HORL oracle analysis.
6. **Baselines**: TokenSkip, LayerSkip self-speculative decoding, HORL oracle.
7. **Analysis**: Figures, iso-FLOP curves, compute-optimal allocation, token-level analysis.
8. **Robustness**: Scale test, seeds, ablations, LiveCodeBench + HumanEval.
9. **Paper**: Written last, around the data.

### Directory Structure
```
src/
  infrastructure/       # Model loading, generation, answer extraction
  depth_control/        # FFN-skip, layer-skip, attention-skip implementations
  length_control/       # Token budget enforcement, forced-answer extraction
  benchmarks/           # Benchmark loaders (MATH-500, GPQA, LiveCodeBench, etc.)
  experiments/          # Experiment runners (one script per experiment)
  analysis/             # Plotting, statistics, figure generation
results/                # Raw outputs (JSON). Organized by experiment/model/benchmark.
figures/                # Generated figures
paper/                  # LaTeX — CREATED LAST
```

### Quality Gates
1. **Decomposition pilot must show signal.** FFN-only must outperform full-layer by ≥10% accuracy at iso-FLOP on MATH-500 before scaling. If gap <5%, abandon decomposition angle.
2. **Surface must show structure.** If the compute-accuracy surface is flat, pivot to pure decomposition paper without surface.
3. **Task-type differentiation is a bonus, not a requirement.** The measurement itself is the contribution. Don't commit upfront to "depth=recall, length=computation."
4. **3+ model families.** Decomposition finding must replicate across ≥2 model families.
5. **Answer extraction must be validated at ≥90% success rate** on held-out subsample before running full experiments.
6. **Iso-FLOP comparisons required** for all decomposition claims. Never compare at equal skip percentage.
7. **Every figure must be self-explanatory.** A reader looking only at figures should understand the paper.
8. **Related work must cite and differentiate from**: SkipDecode, LayerSkip, FFN-SkipLLM, CLaSp, From LLMs to LRMs, Pruning as Cooperative Game, AdaPonderLM, MoDA, overthinking papers (Reasoning Theater, TERMINATOR), TokenSkip, Art of Efficient Reasoning, and any relevant 2026 concurrent work.

### What Success Looks Like
- **Figure 1 is THE figure**: Accuracy vs. FLOP reduction with three curves (FFN-only, full-layer, attention-only). FFN-only degrades gently, full-layer collapses. Immediately communicates the decomposition finding.
- FFN-only outperforms full-layer by ≥10% accuracy at iso-FLOP on reasoning benchmarks, across ≥2 model families
- The compute-accuracy surface heatmaps show visible task-dependent structure (different shapes for MATH-500 vs GPQA vs MMLU-Pro)
- A practical compute-optimal allocation rule (given budget C, allocate X% to FFN compute, Y% to length)
- FFN-only self-speculative decoding yields measurable lossless speedup
- All baselines reproduced within 2% of published numbers
- Answer extraction validated at ≥90% success rate
- Results are **broad and confident**, not cherry-picked

## Writing Guidelines

- All applications are in **English**.
- Both Luciano and Juan can apply separately to most grants (they are per-person, not per-institution).
- Always mention: open-source commitment, planned publications at ACL/EMNLP/NeurIPS, and existing publication track record.
- For NVIDIA specifically: explicitly reference NeMo, TensorRT-LLM, Triton Inference Server, and pretrained models from ai.nvidia.com.
- For Microsoft: mention coauthors Ahmed Awadallah, Subhabrata Mukherjee, Corby Rosset, Guoqing Zheng, Hamid Palangi.
