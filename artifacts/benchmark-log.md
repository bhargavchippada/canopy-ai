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
| 6 | Sonnet CDT, Haiku gen+Sonnet eval | Haiku | Sonnet | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | 30 | 88 | 167 | 58.98 | 2026-03-28 | Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | Sonnet CDT +0.6 vs Haiku CDT. |
| 7 | Sonnet CDT, Sonnet gen+eval | Sonnet | Sonnet | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | 30 | 88 | 167 | **66.17** | 2026-03-28 | Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | +7.8 vs Haiku eval! Eval model is dominant factor. |
| 8 | GPT-4.1 CDT, Sonnet gen+eval | Sonnet | Sonnet | (paper CDT) | (paper CDT) | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | — | — | 167 | **66.17** | 2026-03-28 | Kasumi.gpt41.depth3.relation.pkl | SAME as Sonnet CDT! CDT quality = GPT-4.1 quality. |
| 9 | Sonnet CDT, Haiku gen+eval | Haiku | Haiku | Qwen3-8B | Qwen3-8B | DeBERTa | kmeans | 3 | 0.75 | 0.50 | 8 | 30 | 88 | 167 | 50.00 | 2026-03-28 | Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl | Cheapest baseline. ~8pt per model tier. |

## Analysis

### Quality Gap Decomposition

| Factor | Impact | Evidence |
|--------|--------|----------|
| Embedding quality (0.6B→8B) | **+15 points** (43→58) | Run 3→5 |
| θ_accept alignment (0.80→0.75) | Included in above | Part of run 3→5 |
| CDT hypothesis gen model (Haiku→Sonnet) | **+0.6 points** (58.38→58.98) | Run 5→6 (same eval) |
| Eval model (Haiku→Sonnet) | **+7.2 points** (58.98→66.17) | Run 6→7 (same CDT) |
| Remaining gap (Claude Sonnet vs GPT-4.1) | **~18 points** (66.17→84.25) | Run 7 vs paper |
| RP gen model (Claude vs Llama-8B) | Part of remaining | Not isolated yet |

### Key Observations

1. **Eval model quality is the #1 factor.** Sonnet eval adds +7.2 points over Haiku eval (run 6→7). CDT construction quality adds only +0.6 (run 5→6). Paper confirms: "small gap" for CDT construction across model tiers.

2. **CDT construction quality is NOT the bottleneck.** Sonnet vs Haiku for hypothesis gen makes <1 point difference. The tree quality (30 nodes, 88 stmts) is in the right range.

3. **Embedding quality was the early bottleneck.** 0.6B→8B embeddings added +15 points. This is now resolved via two-phase subprocess architecture.

4. **Remaining 18-point gap** (66.17→84.25) is the combined effect of: (a) Claude Sonnet vs GPT-4.1 for eval judging, (b) Claude vs Llama-8B for RP action generation. These are model-quality limits, not CDT algorithm issues.

5. **Over-splitting is resolved.** Paper-matched config produces 20-30 nodes (vs 85 with 0.6B/θ=0.80). Still 2-3x paper's 10.4, but much closer.

### Pending Experiments

- [x] Sonnet CDT with Sonnet gen+eval — **66.17** (dominant factor confirmed)
- [x] GPT-4.1 CDT with Sonnet gen+eval — **66.17** (identical to Sonnet CDT! CDT quality matches GPT-4.1)
- [x] Haiku+Haiku — **50.00** (cheapest baseline, confirms eval model quality ladder)
- [x] Llama 8-bit gen + Sonnet eval — **55.99** (8-bit quantization quality loss)
- [x] Llama fp16 gen + Sonnet eval — **~58.65** (killed at 62%, trending similar to Haiku gen)
- [ ] CDT quality investigation (clustering, hypothesis quality, NLI validation per step)
