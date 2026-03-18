# Early Exit / Layer Skipping / Adaptive Depth — HuggingFace Trending Papers
## Mined: Jan–Mar 2026 | Sorted by HF upvotes (relevance proxy)

| # | Paper | HF Votes | Date | Topic |
|---|-------|----------|------|-------|
| 1 | **Attention Residuals** (Moonshot AI) | 1,070 | Mar 17 | Residual connections in attention for efficient computation |
| 2 | **DFlash: Block Diffusion for Flash Speculative Decoding** | 640 | Feb 7 | Speculative decoding acceleration |
| 3 | **PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning** | 335 | Jan 13 | Adaptive test-time compute scaling |
| 4 | **Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights** | 222 | Mar 14 | Task experts near pretrained weights (depth/param efficiency) |
| 5 | **MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head** | 132 | Jan 13 | Token-level computation in attention |
| 6 | **Scaling Embeddings Outperforms Scaling Experts in Language Models** | 102 | Jan 31 | Token-level compute allocation via scaling strategy |
| 7 | **Mixture-of-Depths Attention** (ByteDance Seed) | 70 | Mar 17 | Adaptive depth — different tokens processed at different depths |
| 8 | **ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation** | 42 | Jan 31 | Token-level adaptive compute allocation |
| 9 | **LoopViT: Scaling Visual ARC with Looped Transformers** | 37 | Feb 3 | Looped/recurrent transformers = adaptive computational depth |
| 10 | **One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers** | 34 | Mar 14 | Elastic compute budgets at inference |
| 11 | **dVoting: Fast Voting for dLLMs** | 31 | Feb 14 | Fast inference for distributed LLMs |
| 12 | **Shaping capabilities with token-level data filtering** | 27 | Jan 31 | Token-level efficiency |
| 13 | **Hybrid Linear Attention: Efficient Distillation and Effective Architectures** | 27 | Jan 31 | Efficient architecture for long contexts |
| 14 | **VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning** | 25 | Jan 31 | Compression for efficient inference |
| 15 | **Forest Before Trees: Latent Superposition for Efficient Visual Reasoning** | 18 | Jan 13 | Hierarchical efficient reasoning |
| 16 | **Elastic Attention: Test-time Adaptive Sparsity Ratios** | 17 | Jan 27 | Adaptive sparsity at test time |
| 17 | **Least-Loaded Expert Parallelism: Load Balancing MoE** | 16 | Jan 27 | MoE load balancing for efficiency |
| 18 | **Training LLMs for Divide-and-Conquer Reasoning** | 13 | Feb 3 | Hierarchical depth allocation |
| 19 | **Fast KVzip: Efficient LLM Inference with Gated KV Eviction** | 13 | Jan 27 | KV cache efficiency |
| 20 | **Scalable Power Sampling: Training-Free Reasoning for LLMs** | 13 | Jan 31 | Training-free efficient reasoning |
| 21 | **DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers** | 12 | Feb 21 | Dynamic scheduling for inference |
| 22 | **GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token** | 12 | Jan 13 | Selective token processing |
| 23 | **RelayGen: Intra-Generation Model Switching for Efficient Reasoning** | 11 | Feb 10 | Model switching during inference |
| 24 | **TERMINATOR: Learning Optimal Exit Points for Early Stopping of CoT** | 11 | Mar 17 | Early exit for chain-of-thought |
| 25 | **ThinkRouter: Efficient Reasoning via Routing between Latent and Discrete** | 8 | Feb 14 | Routing for efficient reasoning |
| 26 | **When Does Sparsity Mitigate the Curse of Depth in LLMs** | 7 | Mar 17 | Sparsity + depth interaction |
| 27 | **On the Limits of Layer Pruning for Generative Reasoning in LLMs** | 4 | Feb 3 | Layer pruning limits |
| 28 | **PACED: Distillation at the Frontier of Student Competence** | 4 | Mar 14 | Distillation efficiency |
| 29 | **Skip to the Good Part: Representation Structure & Layer Skipping** (Qualcomm) | 3 | Mar 10 | Layer skipping in LLMs |
| 30 | **RouteMoA: Dynamic Routing without Pre-Inference for MoA** | 3 | Jan 27 | Dynamic routing for efficiency |

## Core cluster (most directly relevant to early exit / skipping / adaptive depth):
- **Mixture-of-Depths Attention** (70 votes, Mar 17) — THE paper
- **TERMINATOR** (11 votes, Mar 17) — early exit for CoT
- **Skip to the Good Part** (3 votes, Mar 10) — layer skipping
- **On the Limits of Layer Pruning** (4 votes, Feb 3) — pruning limits
- **Elastic Attention** (17 votes, Jan 27) — adaptive sparsity
- **ConceptMoE** (42 votes, Jan 31) — token-level adaptive compute
- **GlimpRouter** (12 votes, Jan 13) — selective token processing
- **RelayGen** (11 votes, Feb 10) — model switching mid-generation
- **ThinkRouter** (8 votes, Feb 14) — routing reasoning paths

---

## SkipDecode Citation Tree — Most cited descendants

| # | Paper | Venue | Year | Citas ~  | PDF file |
|---|-------|-------|------|----------|----------|
| 1 | **SkipDecode** (Del Corro et al.) — ORIGINAL | arXiv | 2023 | 105 | SkipDecode_DelCorro_2023.pdf |
| 2 | **Medusa** (Cai et al.) — multi-head speculative | ICML | 2024 | 442 | Medusa_2024.pdf |
| 3 | **LayerSkip** (Meta) — layer dropout + early exit + self-spec | ACL | 2024 | ~80 | LayerSkip_Meta_2024.pdf |
| 4 | **FFN-SkipLLM** — adaptive FFN block skipping | EMNLP | 2024 | ~30 | FFN-SkipLLM_EMNLP2024.pdf |
| 5 | **Kangaroo** — double early exit self-spec | NeurIPS | 2024 | ~25 | Kangaroo_NeurIPS2024.pdf |
| 6 | **CLaSp** — in-context layer skip sin training | ACL | 2025 | new | CLaSp_ACL2025.pdf |
| 7 | **AdaSkip** — sublayer adaptive skipping | AAAI | 2025 | new | AdaSkip_AAAI2025.pdf |
| 8 | **DASH** — MDP-based skip/quantize decisions | arXiv | 2025 | new | DASH_2025.pdf |
| 9 | **SWIFT** — plug-and-play self-spec decoding | arXiv | 2024 | ~15 | SWIFT_2024.pdf |
| 10 | **Probe and Skip** — training-free, 2.46x speedup | arXiv | 2026 | new | ProbeAndSkip_2026.pdf |
| 11 | **TokenSkip** — CoT compression via token skip | arXiv | 2025 | new | TokenSkip_2025.pdf |
| 12 | **Mirror Speculative Decoding** — breaks serial barrier | ICLR | 2026 | new | MirrorSpecDec_ICLR2026.pdf |

## HuggingFace trending papers (PDFs also downloaded):
| Paper | PDF file |
|-------|----------|
| Mixture-of-Depths Attention (ByteDance) | MixtureOfDepths_Attention_ByteDance_2026.pdf |
| TERMINATOR | TERMINATOR_2026.pdf |
| Skip to the Good Part (Qualcomm) | SkipToTheGoodPart_Qualcomm_2026.pdf |
| Elastic Attention | ElasticAttention_2025.pdf |
| GlimpRouter | GlimpRouter_2025.pdf |
| RelayGen | RelayGen_2025.pdf |
| ThinkRouter | ThinkRouter_2025.pdf |
| On the Limits of Layer Pruning | OnTheLimitsOfLayerPruning_2025.pdf |
| ConceptMoE | ConceptMoE_2025.pdf |

**Total: 21 PDFs in papers_early_exit/**
