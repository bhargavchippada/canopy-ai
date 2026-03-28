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
- **Testing:** pytest (214 unit + 13 integration = 227 tests)
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
uv run python -m pytest                           # Unit tests (214, ~10s)
uv run python -m pytest -m integration            # GPU integration tests (13, ~23s)
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
| Qwen3-Embedding-0.6B | `~/models/Qwen3-Embedding-0.6B` | ~1.2GB | Surface embedding |
| Qwen3-0.6B | `~/models/Qwen3-0.6B` | ~1.2GB | Generative embedding |

**VRAM (RTX 5090, 32GB):** 0.6B models use ~3.1GB total. 8B models use two-phase subprocess isolation (each model loads once in a subprocess, exits to release VRAM).

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
├── tests/                     # 227 tests (214 unit + 13 integration)
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

## Benchmark Results

| CDT Source | Gen Model | Eval Model | NLI Score | Paper Score | Notes |
|-----------|-----------|------------|-----------|-------------|-------|
| Kasumi (GPT-4.1 CDT) | Haiku | Sonnet | 61.98 | 84.25 | Gap from model swap (Claude vs GPT-4.1/Llama) |

## Known Issues

- `prompts.py` at 73% coverage (needs real Claude for full coverage)
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
