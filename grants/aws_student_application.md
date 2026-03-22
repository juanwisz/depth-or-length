# AWS Cloud Credit for Research — Student Application

## Form Fields

- **First Name:** Juan
- **Last Name:** Wisznia
- **Institution Name:** Universidad de San Andres
- **Institutional Email:** [FILL — @udesa.edu.ar]
- **Phone Number:** [FILL]
- **AWS Account ID:** [FILL]
- **Area of Research Study:** Computer & Information Sciences
- **Research Led By:** Faculty Member/PI
- **Requested Credit Amount in USD:** 5000
- **AWS Pricing Calculator URL:** [TODO — generate]
- **Project Title:** Inference Scaling Laws for Reasoning Language Models
- **Project Type:** Evaluation/Comparison
- **Additional external funding sources:** No
- **Country / Region:** Argentina
- **Industry:** Education
- **Job Role:** PhD Student
- **Job Title:** Graduate Research Assistant
- **Level of AWS Usage:** Evaluating/experimenting with AWS
- **Use Case:** AI & Machine Learning

---

## Project Proposal (draft text for the form)

### Problem Statement

Recent work on inference scaling laws (Wu et al., ICLR 2025; Snell et al., 2024) established that smaller language models combined with repeated sampling can outperform larger models at equal computational budgets. However, these findings were derived from standard (non-reasoning) models with predictable output lengths. The emergence of natively reasoning models — which produce long, variable-length chain-of-thought traces — challenges these conclusions. Preliminary evidence suggests that smaller reasoning models generate disproportionately more tokens per problem (3-5x), potentially inverting the established scaling laws. No systematic study has measured this with proper FLOP accounting across model families.

### Project Summary

We will establish the first inference scaling laws for reasoning language models by running controlled experiments across 4 model sizes in 3 model families on 5 diverse reasoning benchmarks. For each configuration, we measure accuracy under majority voting (k=1,2,4,8 samples) and total FLOPs including variable output length. The key deliverable is iso-FLOP scaling curves that reveal whether — and when — larger reasoning models become compute-optimal, contrary to established wisdom for non-reasoning models.

### AWS Services Required

- **Amazon SageMaker Batch Transform**: Primary compute. We deploy open-source HuggingFace models (DeepSeek-R1-Distill-Qwen 1.5B/7B/14B/32B, Qwen3, Qwen2.5-Instruct) on ml.g5.xlarge and ml.g5.2xlarge instances for batch inference across ~300,000 total requests.
- **Amazon S3**: Storage for input prompts, model outputs (JSONL), and intermediate results. Estimated ~50 GB total.
- **Amazon CloudWatch**: Monitoring job completion, instance utilization, and cost tracking.

### Budget Breakdown (estimated)

| Component | Instance Type | Hours | Cost |
|-----------|--------------|-------|------|
| 1.5B-class models (3 families) | ml.g5.xlarge ($1.41/hr) | ~240h | $338 |
| 7B-8B-class models (3 families) | ml.g5.xlarge ($1.41/hr) | ~240h | $338 |
| 14B-class models (3 families) | ml.g5.2xlarge ($1.89/hr) | ~240h | $454 |
| 32B-class models (3 families) | ml.g5.2xlarge ($1.89/hr) | ~360h | $680 |
| S3 storage + data transfer | — | — | $50 |
| Pilot experiments + debugging | ml.g5.xlarge | ~40h | $56 |
| Buffer for reruns + validation | — | — | $200 |
| **Total** | | | **$2,116** |

Note: Requesting $5,000 to allow for additional experiments (robustness checks, ablations, additional model families) and potential extension to a second paper.

### Timeline and Key Milestones

- **Week 1-2**: Infrastructure setup. Deploy models on SageMaker, validate batch pipeline, run pilot on MATH-500 with 2 model sizes.
- **Week 3-4**: Core experiments. Run full grid: 3 families × 4 sizes × 5 datasets × k={1,2,4,8}.
- **Week 5-6**: Analysis. Generate iso-FLOP scaling curves, compute crossover points, difficulty-stratified analysis.
- **Week 7-8**: Paper writing. Target: EMNLP 2026 main conference submission.
- **Week 9-12**: Robustness experiments, ablations, camera-ready revisions.

### Plan for Sharing Outcomes

1. **Peer-reviewed publication** at EMNLP 2026 (main conference), with all code and data released on GitHub under MIT license.
2. **Open-source benchmark** of reasoning model token counts and FLOP measurements across model families, released as a HuggingFace dataset for community use.
3. **Practical lookup tables** for compute-optimal reasoning model selection, published as a blog post and integrated into our group's open tools.
4. **All experiment code** released as a reproducible pipeline on GitHub, including SageMaker Batch Transform scripts that the community can reuse for similar scaling studies.

### Potential Future AWS Usage

Our research group (ELIAS Lab, Universidad de San Andres) is pursuing multiple lines of research in efficient LLM inference and cost-aware model selection. Successful completion of this project would lead to:
- Follow-up studies on inference scaling for multimodal reasoning models
- Extension to production routing systems that select optimal model sizes based on our scaling laws
- Training of lightweight routing classifiers that predict compute-optimal model size per-query
- These follow-up projects would continue using SageMaker for batch inference at larger scale.

### AWS Employee Contact

None at this time.

### AWS Public Data Sets

None required. All benchmarks are open-source and hosted on HuggingFace.

### Research Context

This work is led by Prof. Luciano Del Corro (Universidad de San Andres), former Senior Research Engineer at Microsoft Research (AI Frontiers group), co-author of Orca 2 (230+ citations), SkipDecode (105+ citations), and AgentInstruct (70+ citations). The student applicant (Juan Wisznia) is first author on an ACL 2025 paper on cost-aware LLM inference (40-90% cost reduction for ranking tasks) and co-author on an EMNLP 2024 paper. This project directly extends both lines of work: PI's expertise in efficient inference (SkipDecode) and student's expertise in cost-optimal model selection (ACL 2025).

### Publication Track Record

- Del Corro et al., "SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference" (2023, 105+ citations)
- Del Corro et al., "Orca 2: Teaching Small Language Models How to Reason" (2023, 230+ citations)
- Del Corro et al., "AgentInstruct: Toward Generative Teaching with Agentic Flows" (2024, 70+ citations)
- Wisznia et al., "Cost-Aware LLM Ranking" (ACL 2025)
- Wisznia & Del Corro, "The Greatest Good Benchmark" (EMNLP 2024)
