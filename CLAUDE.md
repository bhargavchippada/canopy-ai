# Canopy

> Evolving decision trees for behavioral profiling.

## Project Overview

Canopy extends Codified Decision Trees (CDT) with temporal dynamics, structured gates, and domain-agnostic profiling. Built on the CDT paper (arxiv 2601.10080).

## Current Phase

**Phase 0: COMPLETE** — 3 characters validated (Kasumi 13min, Arisa 45min, Yui 2hr).
**Phase 1: COMPLETE** — Package restructured into `src/canopy/`, 100% coverage on 7 core modules.
**Phase 2: COMPLETE** — Library API: BehavioralObservation, builder, wikify, cluster.
**Phase 3: COMPLETE** — Integration tests (11 GPU tests), examples, README.
**Phase 4: COMPLETE** — All legacy files migrated (zero OpenAI/exec/constant refs), batch LLM, parallel benchmark.
**Next:** Phase 5 (T-CDT + Advanced Features) or delulu integration.

## Design Reference

**Authoritative CDT design:** `artifacts/canopy-design.md` (design decisions D1-D30)

## Principles

- **Step by step** — Each phase gated by success criteria.
- **Branch per phase** — Merged only when complete.
- **Simple and clean** — Minimum complexity for current needs.
- **Research first** — Search before building.
- **Iterate** — Small commits, frequent validation.

## Tech Stack

- **Language:** Python 3.11+
- **Package manager:** uv (NEVER pip)
- **LLM:** Claude via claude-agent-sdk (Max subscription, no API key)
- **LLM Model:** claude-haiku-4-5 (hypothesis gen), claude-sonnet-4-6 (evaluation)
- **Embeddings:** Qwen3-0.6B (smoke test) / Qwen3-8B (full run)
- **NLI:** DeBERTa-v3-base-rp-nli
- **Local Models:** Llama-3.1-8B-Instruct (RP gen), Qwen3-8B/0.6B (embeddings), DeBERTa (NLI)
- **Testing:** pytest (unit + integration, run `pytest --co -q` for current count)
- **Linting:** ruff

## Library API

```python
from canopy import BehavioralObservation, CDTConfig, build_cdt, build_character_profile
from canopy.wikify import wikify_tree, wikify_profile
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.cluster import KMeansCluster, HDBSCANCluster

# 1. Configure LLM
set_adapter(ClaudeCodeAdapter(default_model="claude-haiku-4-5"))

# 2. Create observations
obs = [
    BehavioralObservation(scene="...", action="...", actor="Alice", participants=["Bob"]),
]

# 3. Build single CDT
tree = build_cdt(obs, character="Alice", topic="identity", config=CDTConfig(max_depth=2))

# 4. Or build full profile
topic2cdt, rel_topic2cdt = build_character_profile(obs, character="Alice")

# 5. Wikify to markdown
markdown = wikify_profile(topic2cdt, rel_topic2cdt, character="Alice")

# 6. Pre-compute embeddings with subprocess isolation (for 8B models)
from canopy.embeddings import precompute_embeddings
cache = precompute_embeddings("Alice", pairs, "/path/to/surface", "/path/to/gen")
indexed_pairs = [{**p, "_embed_idx": i} for i, p in enumerate(pairs)]
topic2cdt, rel_topic2cdt = build_character_profile(
    obs, character="Alice", embedding_cache=cache,
)

# 7. Batch LLM generation with drop tracking
from canopy.llm import batch_generate, BatchResult
result = batch_generate([("id1", "prompt1"), ("id2", "prompt2")])
print(result.successes, result.exhausted_ids, result.success_rate)
```

## Key Commands

```bash
uv sync                                           # Install dependencies
uv run python -m pytest                           # Unit tests (~10s)
uv run python -m pytest -m integration            # GPU integration tests
uv run python -m pytest --cov=canopy              # Coverage report
uv run ruff check src/canopy/                     # Lint

# CDT construction via CLI (outputs: packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl)
uv run python codified_decision_tree.py \
  --character Kasumi \
  --engine claude-haiku-4-5 \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --discriminator_path ~/models/deberta-v3-base-rp-nli \
  --cluster_method kmeans \
  --device_id 0

# Benchmark with parallel evaluation
uv run python run_benchmark.py \
  --character Kasumi \
  --method cdt_package \
  --cdt_path packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl \
  --engine claude-haiku-4-5 \
  --eval_engine claude-sonnet-4-6 \
  --max_parallel 6 \
  --device_id 0
```

## Model Paths

Models stored in `~/models/`:

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| DeBERTa NLI | `~/models/deberta-v3-base-rp-nli` | 715MB | NLI validation |
| Qwen3-Embedding-0.6B | `~/models/Qwen3-Embedding-0.6B` | ~1.2GB | Surface embedding (smoke test) |
| Qwen3-0.6B | `~/models/Qwen3-0.6B` | ~1.2GB | Generative embedding (smoke test) |
| Qwen3-Embedding-8B | `~/models/Qwen3-Embedding-8B` | ~16GB | Surface embedding (paper config) |
| Qwen3-8B | `~/models/Qwen3-8B` | ~16GB | Generative embedding (paper config) |
| Llama-3.1-8B-Instruct | `~/models/Llama-3.1-8B-Instruct` | ~16GB | RP generation (paper's gen model) |

**VRAM (RTX 5090, 32GB):** 0.6B models use ~3.1GB total. 8B models use two-phase subprocess isolation (each model loads once in a subprocess, exits to release VRAM). Llama fp16 + DeBERTa fits in ~21GB.

## Project Structure

```
canopy-ai/
├── src/canopy/                # Core package (12 modules)
│   ├── __init__.py            # Exports: CDTConfig, CDTNode, BehavioralObservation, EmbeddingCache
│   ├── core.py                # CDTNode, CDTConfig, build_character_cdts
│   ├── builder.py             # BehavioralObservation, build_cdt, build_character_profile
│   ├── wikify.py              # CDT → markdown (wikify_tree, wikify_profile)
│   ├── cluster.py             # KMeansCluster, HDBSCANCluster, ClusterStrategy Protocol
│   ├── embeddings.py          # EmbeddingCache, precompute_embeddings, select_cluster_centers
│   ├── _embed_worker.py       # Subprocess entry point for VRAM-isolated encoding
│   ├── validation.py          # NLI-based scene/statement checking (DeBERTa)
│   ├── prompts.py             # LLM hypothesis generation + summarization
│   ├── llm.py                 # LLM adapter (Protocol + ClaudeCodeAdapter + batch_generate)
│   ├── data.py                # HuggingFace dataset loading + caching
│   └── cli.py                 # CLI entry point
├── tests/                     # Unit + integration tests
│   ├── conftest.py            # GPU skip guard, integration deselection
│   ├── test_core.py           # CDTNode, CDTConfig, build_character_cdts, cache forwarding
│   ├── test_builder.py        # BehavioralObservation, build_cdt
│   ├── test_wikify.py         # wikify_tree, wikify_profile
│   ├── test_cluster.py        # KMeans, HDBSCAN, representative samples
│   ├── test_llm.py            # Adapter, extract_json, retry, session
│   ├── test_data.py           # Metadata loading, pair extraction
│   ├── test_embeddings.py     # EmbeddingCache, precompute, subprocess, cache path
│   ├── test_validation.py     # Guard tests
│   └── test_integration.py    # GPU tests (embeddings, validation, pipeline)
├── examples/                  # Usage examples
│   └── quickstart.py
├── codified_decision_tree.py  # Thin CLI wrapper
├── run_benchmark.py           # CDT benchmark with parallel eval (ThreadPoolExecutor)
├── verify_cdt.py              # CDT package inspector
├── artifacts/                 # Design docs, results
├── results/                   # Benchmark result JSON files (gitignored)
└── pyproject.toml
```

## Performance

| Metric | Before (Phase 0) | After (optimized) |
|--------|-------------------|-------------------|
| Single LLM call | 30-60s | 2-3s |
| 8 parallel calls | 4-8 min | ~3s |
| Kasumi CDT | ~90 min | ~13 min |

Optimizations: tools=[], setting_sources=[], Haiku model, asyncio.gather parallelism.
Phase 4 additions: `batch_generate()` with retry/drop tracking, `ThreadPoolExecutor` parallel benchmark (--max_parallel).
Phase 5 additions: Two-phase embedding architecture — each 8B model loads ONCE in a subprocess (OS reclaims VRAM on exit), then tree building uses pre-computed numpy arrays with zero GPU model loading. Eliminates OOM with 8B models on 32GB GPUs.

## Paper Hyperparameter Discrepancy

**Paper text** (Appendix B): θ_accept=0.75, θ_reject=0.50, θ_f=0.75, max_depth=4
**Paper code** (all .py files, README, build_cdt.sh): max_depth=3, threshold_accept=0.80, threshold_reject=0.50, threshold_filter=0.80

Three values disagree: θ_accept (0.75 vs 0.80), θ_f (0.75 vs 0.80), max_depth (4 vs 3). **Follow paper text as authoritative** — use θ_accept=0.75, θ_f=0.75, max_depth=4 as the reference configuration. Code defaults may reflect development/tuning values that diverged from the final published settings.

**Paper models:**
- Hypothesis gen: gpt-4.1
- Validation: gpt-4.1-mini (CDT) or DeBERTa (CDT-Lite, scores HIGHER: 88.38 vs 84.25)
- RP gen: llama-3.1-8b-instruct (64 tokens, newline stop, greedy)
- Eval: gpt-4.1 (never named in paper text, only in code defaults)
- Embeddings: qwen3-embedding-8b (surface), qwen3-8b (generative)
- Clustering: K-Means, max 8 clusters/node, min 16 samples/cluster

**Paper does NOT report:** A/B/C score distributions, eval model ablation, gen format comparison.

## Benchmark Results

| CDT Source | Gen Model | Eval Model | NLI Score | Notes |
|-----------|-----------|------------|-----------|-------|
| claude-haiku.d3.rel | Haiku | Haiku | 41.32 | Cheapest baseline |
| sonnet.qwen8b.d3.a75 | Haiku | Haiku | 50.00 | |
| haiku.qwen8b.d3.a75 | Haiku | Sonnet | 58.38 | |
| sonnet.qwen8b.d3.a75 | Haiku | Sonnet | 58.98 | |
| sonnet.qwen8b.d3.a75 | Llama | Sonnet | 55.99 | 64-token limit → 2x C rate |
| sonnet.qwen8b.d3.a80 | Sonnet | Sonnet | 64.07 | |
| paper-original | Sonnet | Sonnet | 65.87 | |
| gpt41.d3.rel | Sonnet | Sonnet | 66.17 | Our CDT = paper CDT quality |
| sonnet.qwen8b.d3.a75 | Sonnet | Sonnet | 66.17 | Narration format |
| sonnet.qwen8b.d3.a75 | Sonnet | Sonnet | **67.96** | Dialogue format (n=167) |
| **Paper (PoPiPa CDT)** | **Llama** | **GPT-4.1** | **84.25** | Paper Table 2, ~18% B rate |
| **Paper (PoPiPa CDT-Lite)** | **Llama** | **GPT-4.1** | **88.38** | DeBERTa validation path |

**Score gap explanation:** Sonnet eval gives ~35-40% B (neutral); GPT-4.1 eval gives ~18% B. The 66→84 gap is eval model calibration, not implementation quality. Human annotation (10 pairs) confirms Sonnet is well-calibrated.

### B-Score Analysis (artifacts/b-score-analysis.md)
- 40% B rate: 77.6% different-facet (gen caricature), 19.4% format-mismatch, 3% wrong
- 0% eval-too-strict — evaluator is correct
- 74.6% of B pairs in low-predictability scenes (structural ambiguity)
- Scene transitions 3.8x more common in B pairs (28.4%) than A pairs (7.5%)
- C pairs: 80% are enthusiasm-over-restraint (caricature failure on quiet moments)

### Caricature Bias (artifacts/caricature-bias-analysis.md)
- Gen model defaults to max-enthusiasm Kasumi for every prediction
- 100% ground truths are dialogue; 86% predictions are theatrical narration
- CDT grounding: 88 statements, ~15 mentions enthusiasm, ~0 restraint
- Dialogue format fix: +1.79 pts (66.17→67.96), B% increased to 40.1%

## CDT Quality Investigation (2026-03-28)

### Key Finding: Summarize Prompt Was the Bottleneck
Paper's `summarize_triggers` prompt has 50-line detailed instructions with Selection Principles, Dedup Rules, and explicit "single, concise sentence" constraints. Our minimal prompt produced broad hypotheses that passed NLI at any θ → flat trees.

**After fix:** Identity topic: 7 nodes/19 stmts/6 gates (was 1/8/0, paper: 8/22/7)

### CDT Structure Comparison (Kasumi)
| CDT | Attr Nodes | Attr Stmts | Rel Nodes | Rel Stmts | Total |
|-----|-----------|-----------|----------|----------|-------|
| GPT-4.1 (paper) | 23 | 60 | 13 | 23 | 36/83 |
| Sonnet (ours) | 6 | 32 | 24 | 56 | 30/88 |
| Haiku (ours) | 9 | 45 | 11 | 27 | 20/72 |

### Code Differences Found vs Paper (/home/turiya/projects/Codified_Decision_Tree/)
1. **Summarize prompt**: Paper has 50-line version with "single, concise sentence" constraint — FIXED
2. **Output format**: Paper uses Python code block, we use JSON — investigation pending
3. **θ_accept**: Paper text says 0.75, code defaults to 0.80 — follow paper text (0.75)
4. **θ_f**: Paper text says 0.75, code defaults to 0.80 — follow paper text (0.75)
5. **max_depth**: Paper text says 4, code defaults to 3 — follow paper text (4)
6. **Scene check**: Paper uses Llama (CDT) or DeBERTa (CDT-Lite, scores higher: 88.38)
7. **Eval model**: Paper uses gpt-4.1 (~18% B rate), never named in paper text
8. **Gen model**: Paper uses llama-3.1-8b-instruct with 64-token max, newline stop, greedy decoding
9. **Eval parse failure**: Paper crashes on parse failure; we default to B (may inflate B% by 1-2pts)

### Investigation Principle
Models (Haiku/Sonnet/Llama/GPT-4.1) are all capable enough for CDT tasks. Score gaps indicate code/config differences, not model limitations. Always investigate code first.

### per_pair_details Storage
`_save_benchmark_results()` now stores full per_pair_details: prediction text, ground_truth, scene, grounding, eval_reasoning, eval_model, gen_model. Previously only stored numeric per_pair_scores (prediction text was discarded).

## Known Issues

- Module-level mutable state in embeddings.py/validation.py (init_models pattern)

## CDT Artifact Conventions

- **Naming**: `packages/Character.gen_model.embed_model.nli_model.cluster.dN.aXX.rYY.relation.pkl`
  - Example: `Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl`
  - HDBSCAN: `Kasumi.haiku.qwen06b.deberta.hdbscan.d3.a80.r50.relation.pkl`
  - Paper originals: `Kasumi.gpt41.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl`
- **Metadata**: Every `.pkl` includes a `metadata` key with ALL construction config:
  - `gen_model`, `embed_model`, `nli_model` — models used
  - `max_depth`, `threshold_accept`, `threshold_reject`, `threshold_filter` — CDT config
  - `n_clusters`, `total_nodes`, `total_statements`, `has_relationships` — output stats
  - `built_at` — timestamp
- **Validation**: `run_benchmark.py` checks `topic2cdt` + `rel_topic2cdt` keys exist and are non-empty before running. Prints structure summary. Fails fast on wrong format.
- **Never benchmark with a CDT built by a different model** unless that's the explicit experiment
- **Extended metadata** (not in filename): `threshold_filter`, `hypotheses_per_cluster`, `n_training_pairs`, `temperature`
- **Benchmark results**: `results/Character.cdt_config_hash.bench_model.json` containing: CDT package used, eval model, gen model, traversal mode (gated vs all), relationship CDTs included, NLI score, per-pair results, timestamp
- **Never overwrite** paper originals — rename with provenance first

## Conventions

- Use uv for all Python operations
- Type hints on all public functions
- Immutable data structures (frozen dataclasses, tuples)
- No exec() or eval() on LLM output — structured JSON parsing only
- All LLM calls route through canopy.llm adapter
- Logging module (not print) in library code
- RuntimeError for precondition failures (not assert)

## Attribution

Based on Codified Decision Tree by Letian Peng et al. (MIT License).
