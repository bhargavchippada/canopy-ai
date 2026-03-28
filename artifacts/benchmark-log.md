# Benchmark Log

> All CDT benchmark results. Compare against paper stats: 10.4 avg nodes, 61 statements, 84.25 NLI.

## Paper Reference Stack (arxiv 2601.10080)

| Component | Paper Uses |
|-----------|-----------|
| CDT codifier (hypothesis gen) | GPT-4.1 |
| CDT validation (NLI in construction) | DeBERTa (CDT-Lite) or GPT-4.1-mini |
| RP gen model (generates actions in benchmark) | Llama-3.1-8B-Instruct |
| NLI eval judge (scores A/B/C) | GPT-4.1 |
| Embeddings (surface) | Qwen3-Embedding-8B |
| Embeddings (generator) | Qwen3-8B |
| Clustering | KMeans |
| Depth | 3 |
| θ_accept | 0.75 |
| θ_reject | 0.50 |

## Results

| # | Config | Hyp Gen | Eval | Surface Embed | Gen Embed | NLI Model | Cluster | Depth | θ_accept | θ_reject | Topics | Nodes | Stmts | Pairs | NLI Score | Date | PKL | Notes |
|---|--------|---------|------|---------------|-----------|-----------|---------|-------|----------|----------|--------|-------|-------|-------|-----------|------|-----|-------|
| 1 | Paper baseline | GPT-4.1 | GPT-4.1 | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | ~8 | 10.4 avg | 61 | ~167 | **84.25** | paper | paper | Reference target |
| 2 | GPT-4.1 CDT + Claude eval | Haiku | Sonnet | (paper CDT) | (paper CDT) | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | — | — | 167 | 61.98 | 2026-03-27 | Kasumi.gpt41.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | Gap from model swap in eval |
| 3 | Claude CDT 0.6B θ=0.80 WITH rel | Haiku | Sonnet | Qwen3-0.6B | Qwen3-0.6B | DeBERTa | kmeans | 3 | 0.80 | 0.50 | 8 | 85 | 194 | 167 | 43.11 | 2026-03-27 | Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl | Over-split: 85 nodes (8x paper). Verbose stmts. |
| 4 | Claude CDT 0.6B θ=0.80 NO rel | Haiku | Sonnet | Qwen3-0.6B | Qwen3-0.6B | DeBERTa | kmeans | 3 | 0.80 | 0.50 | 4 | — | — | 167 | 41.32 | 2026-03-27 | Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl | Without relationship CDTs |
| 5 | Claude CDT 8B θ=0.75 (paper-matched) | Haiku | Sonnet | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | 20 | 72 | 167 | 58.38 | 2026-03-28 | Kasumi.haiku.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | Two-phase arch. 20 nodes (2x paper). |
| 6 | Sonnet hyp gen, 8B θ=0.75 | Sonnet | Sonnet | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | 30 | 88 | 167 | 58.98 | 2026-03-28 | Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | +0.6 vs Haiku CDT. 30 nodes (3x paper). |

## Analysis

### Quality Gap Decomposition

| Factor | Impact | Evidence |
|--------|--------|----------|
| Embedding quality (0.6B→8B) | **+15 points** (43→58) | Run 3→5 |
| θ_accept alignment (0.80→0.75) | Included in above | Part of run 3→5 |
| Hypothesis gen model (Haiku→Sonnet) | **+0.6 points** (58.38→58.98) | Run 5→6 |
| Eval model gap (Claude vs GPT-4.1) | **~25 points** (58.98→84.25) | Run 6 vs paper — largest remaining gap |
| RP gen model (Claude vs Llama-8B) | Unknown | Not isolated yet |

### Key Observations

1. **CDT construction quality is NOT the bottleneck.** Sonnet vs Haiku for hypothesis gen makes <1 point difference. Paper confirms: "small gap between gpt-4.1-mini and gpt-4.1" for CDT construction.

2. **The gap is in eval model quality.** Run 2 shows GPT-4.1 CDT evaluated by Claude scores only 61.98 (vs paper's 84.25 with GPT-4.1 eval). The eval judge and RP generation model are the dominant factors.

3. **Over-splitting is resolved.** Paper-matched config produces 20-30 nodes (vs 85 with 0.6B/θ=0.80). Still 2-3x paper's 10.4, but much closer.

4. **Relationship CDTs help slightly.** Run 3 vs 4: 43.11 vs 41.32 (+1.8 with relationships).

### Pending Experiments

- [ ] Sonnet CDT with Sonnet eval (Sonnet for both gen+eval) — isolates eval quality
- [ ] Investigate RP gen model impact — is Claude generating different action quality than Llama-8B?
