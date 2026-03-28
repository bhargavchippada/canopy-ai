# Benchmark PRD — CDT Evaluation Framework

> 2026-03-28

## Objective

Establish a reproducible benchmark harness to measure CDT quality across configurations. The harness must produce comparable numbers against the paper's reported results and serve as the baseline measurement before T-CDT implementation.

## Success Criteria

- [ ] Vanilla baseline score measured (no CDT, character name + scene only)
- [ ] Our Claude-built CDT scored and compared to paper's 84.25
- [ ] Llama-3.1-8B benchmark scored for paper-closest comparison
- [ ] All results saved with full provenance (CDT config + eval config)
- [ ] Harness can re-run any configuration in <15 min (parallelized)

## Phase 1: Essential Metrics (NOW)

### 1.1 NLI Score (paper's primary metric)

Given a scene + CDT profile → generate RP response → score with DeBERTa NLI.

**Comparison matrix (Kasumi only):**

| Run | CDT Source | Gen Model | Eval Model | Status |
|-----|-----------|-----------|------------|--------|
| A | Vanilla (no CDT) | Claude Haiku | DeBERTa NLI | TODO |
| B | Our Claude CDT (with rels) | Claude Haiku | DeBERTa NLI | IN PROGRESS |
| C | Paper's GPT-4.1 CDT | Claude Haiku | DeBERTa NLI | DONE (61.98) |
| D | Our Claude CDT | Llama-3.1-8B | DeBERTa NLI | TODO (model ready) |
| E | Paper's GPT-4.1 CDT | Llama-3.1-8B | DeBERTa NLI | TODO (paper-closest) |

### 1.2 Tree Structure Stats

Per CDT package, report:
- Total nodes, statements, gates
- Per-topic breakdown (attribute + relationship)
- Max depth reached
- Average statement length (words)

### 1.3 Efficiency

- CDT construction wall time
- Benchmark evaluation wall time
- Total LLM calls (cost proxy)

## Phase 2: Ablation Studies (AFTER baseline)

### 2.1 Configuration Ablations

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Clustering | KMeans vs HDBSCAN | Our enhancement vs paper |
| Depth | 2 vs 3 | Impact of tree depth |
| Threshold accept | 0.75 vs 0.80 | Paper default vs ours |
| Gen model | Haiku vs Sonnet | Quality vs cost |
| Relationships | With vs without | Impact of relationship CDTs |

### 2.2 Behavioral Specificity

- Per-statement NLI: does statement entail for target character but NOT for others?
- Character discrimination score: can the CDT distinguish Kasumi from Arisa?

## Phase 3: T-CDT Metrics (AFTER T-CDT implementation)

### 3.1 Temporal Quality

- Profile drift rate: CDT delta after each new session batch
- Supersession rate: % of old patterns contradicted by new evidence
- Convergence: how many sessions until profile stabilizes

### 3.2 Bootstrap vs Evidence-Based

- CDT from rules only (zero sessions) vs CDT from N sessions
- Quality curve: score at 1, 5, 10, 25, 50, 100 sessions

### 3.3 Incremental vs Full Rebuild

- Score difference: incremental update vs full reconstruction
- Time savings: incremental update time vs full rebuild time

## Phase 4: Delulu-Specific Metrics (AFTER integration)

### 4.1 Profile Accuracy

- Human evaluation: user rates profile accuracy (1-5 scale)
- Wikified profile review: are the behavioral patterns real?

### 4.2 Prediction

- Steering prediction: given early interactions, can CDT predict user corrections?
- Preference prediction: given a task, can CDT predict user's approach preference?

## Result Artifact Format

```
results/
  Character.cdt_config.bench_config.json
```

Each result file contains:
```json
{
  "cdt_package": "Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl",
  "gen_model": "claude-haiku-4-5",
  "eval_method": "deberta-nli",
  "traversal": "gated",
  "relationships_included": true,
  "nli_score": 61.98,
  "n_pairs": 167,
  "wall_time_seconds": 2700,
  "per_pair_scores": [...],
  "timestamp": "2026-03-28T04:00:00Z"
}
```

## Implementation Notes

- Parallelized: 5-8 concurrent eval calls, target <15 min per benchmark run
- All results reproducible: same CDT + same config = same score (temperature=0)
- Harness is a CLI: `uv run python run_benchmark.py --character Kasumi --cdt_path packages/... --gen_model haiku`

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| B1 | Haiku for both gen + eval | Matches paper pattern (same model for both), cheaper, faster |
| B2 | DeBERTa NLI as primary metric | Paper's metric, reproducible, no LLM judge variance |
| B3 | Kasumi first, then expand | Smallest dataset (167 pairs), fastest iteration |
| B4 | Vanilla baseline required | Can't claim CDT helps without measuring "no CDT" |
| B5 | Temperature=0 for reproducibility | Same config = same result |
| B6 | Llama comparison for paper alignment | Paper used Llama+GPT-4.1, we need this data point (model downloaded) |
| B7 | Paper-exact config required | θ_accept=0.75, Qwen3-8B embeddings — our 0.80 + 0.6B produces bloated trees |
| B8 | Ablation: with/without relationships | Same CDT, --no-relationships flag, measures value of relationship CDTs |
| B9 | Relationship CDTs don't apply to delulu | Agents aren't relationship partners; project/task-type filters are the equivalent |
| B10 | Fix CDT quality before benchmarking | Cross-topic dedup, prune empty depth-3 nodes, compress verbose statements |

## CDT Quality Findings (2026-03-28)

Analysis of Kasumi.claude-haiku.depth3.relation.pkl (85 nodes, 194 stmts):

**Issues to fix before fair benchmarking:**
- 20 empty leaf nodes (17 at depth 3) — tree over-branches
- 9 near-duplicate statement pairs across topics (identity ≈ personality ≈ relationship)
- 23% of statements >200 chars (compound sentences)
- Arisa relationship: only 6 statements (should be richest)
- θ_accept=0.80 + Haiku verbosity → 3x more statements than paper's 61 avg

**Root causes:**
- 8 topics (paper uses ~4) → cross-topic duplication
- No post-construction dedup pass
- Depth-3 recurses even when evidence is thin
- 0.6B embeddings produce different cluster quality than paper's 8B

## Priority Fixes (before benchmark)

1. Rebuild with paper's exact config: θ_accept=0.75, Qwen3-8B
2. Embedding pre-processing refactor (E5 in canopy-design.md §16) — load each model once, not per topic
3. Integrate batch_generate() for silent drop tracking
4. Parallelize CDT construction with max_parallel=4 (LLM calls only after E5)

## Deferred Enhancements (after baseline — see canopy-design.md §16)

- E1: Hypothesis merge (cosine > 0.90 → LLM-combined statement)
- E2: Depth-3 pruning (wait for paper-config results first)
- E3: SOTA model exploration (one swap at a time, ablation style)
- E4: Configurable topic discovery (essential for delulu integration)

## Resolved Questions

1. **Kasumi only for Phase 1** — cheapest, fastest iteration. Expand after harness validated.
2. **CDT-Lite** — deferred to Phase 2 ablations
3. **Human Profile baseline** — deferred, use paper's profiles if needed later
4. **Head-to-head LLM judge** — deferred to Phase 2
5. **Relationship CDTs for delulu** — not person-relationships. Equivalent is project/task-type filtered CDTs. Deferred to delulu Phase 3.
