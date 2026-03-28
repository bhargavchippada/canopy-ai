# Canopy

> Evolving decision trees for behavioral profiling.

## Project Overview

Canopy extends Codified Decision Trees (CDT) with temporal dynamics, structured gates, and domain-agnostic profiling. Built on the CDT paper (arxiv 2601.10080).

## Current Phase

**Phase 0: COMPLETE** — 3 characters validated (Kasumi 13min, Arisa 45min, Yui 2hr).
**Phase 1: COMPLETE** — Package restructured into `src/canopy/`, 100% coverage on 7 core modules.
**Phase 2: COMPLETE** — Library API: BehavioralObservation, builder, wikify, cluster.
**Phase 3: COMPLETE** — Integration tests (11 GPU tests), examples, README.
**Next:** Phase 4 (T-CDT + Advanced Features) or delulu integration.

## Design Reference

**Authoritative CDT design:** `artifacts/canopy-design.md` (25 design decisions D1-D25)

## Principles

- **Step by step** — Each phase gated by success criteria.
- **Branch per phase** — Merged only when complete.
- **Simple and clean** — Minimum complexity for current needs.
- **Research first** — Search before building.
- **Iterate** — Small commits, frequent validation.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv (NEVER pip)
- **LLM:** Claude via claude-agent-sdk (Max subscription, no API key)
- **LLM Model:** claude-haiku-4-5 (hypothesis gen), claude-sonnet-4-6 (evaluation)
- **Embeddings:** Qwen3-0.6B (smoke test) / Qwen3-8B (full run)
- **NLI:** DeBERTa-v3-base-rp-nli
- **Testing:** pytest (138 unit + 11 integration = 149 tests)
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
```

## Key Commands

```bash
uv sync                                           # Install dependencies
uv run python -m pytest                           # Unit tests (138, ~10s)
uv run python -m pytest -m integration            # GPU integration tests (11, ~23s)
uv run python -m pytest --cov=canopy              # Coverage report
uv run ruff check src/canopy/                     # Lint

# CDT construction via CLI
uv run python codified_decision_tree.py \
  --character Kasumi \
  --engine claude-haiku-4-5 \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --discriminator_path ~/models/deberta-v3-base-rp-nli \
  --device_id 0
```

## Model Paths

Models stored in `~/models/`:

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| DeBERTa NLI | `~/models/deberta-v3-base-rp-nli` | 715MB | NLI validation |
| Qwen3-Embedding-0.6B | `~/models/Qwen3-Embedding-0.6B` | ~1.2GB | Surface embedding |
| Qwen3-0.6B | `~/models/Qwen3-0.6B` | ~1.2GB | Generative embedding |

**VRAM (RTX 5090, 32GB):** 0.6B models use ~3.1GB total. 8B models need sequential loading.

## Project Structure

```
canopy-ai/
├── src/canopy/                # Core package (9 modules)
│   ├── __init__.py            # Exports: CDTConfig, CDTNode, BehavioralObservation, build_*
│   ├── core.py                # CDTNode, CDTConfig, build_character_cdts
│   ├── builder.py             # BehavioralObservation, build_cdt, build_character_profile
│   ├── wikify.py              # CDT → markdown (wikify_tree, wikify_profile)
│   ├── cluster.py             # KMeansCluster, HDBSCANCluster, ClusterStrategy Protocol
│   ├── embeddings.py          # Model loading, encoding (delegates to cluster.py)
│   ├── validation.py          # NLI-based scene/statement checking (DeBERTa)
│   ├── prompts.py             # LLM hypothesis generation + summarization
│   ├── llm.py                 # LLM adapter (Protocol + ClaudeCodeAdapter)
│   └── data.py                # HuggingFace dataset loading + caching
├── tests/                     # 149 tests (138 unit + 11 integration)
│   ├── conftest.py            # GPU skip guard, integration deselection
│   ├── test_core.py           # CDTNode, CDTConfig, build_character_cdts
│   ├── test_builder.py        # BehavioralObservation, build_cdt
│   ├── test_wikify.py         # wikify_tree, wikify_profile
│   ├── test_cluster.py        # KMeans, HDBSCAN, representative samples
│   ├── test_llm.py            # Adapter, extract_json, retry, session
│   ├── test_data.py           # Metadata loading, pair extraction
│   ├── test_embeddings.py     # Guard tests
│   ├── test_validation.py     # Guard tests
│   └── test_integration.py    # GPU tests (embeddings, validation, pipeline)
├── examples/                  # Usage examples
│   └── quickstart.py
├── codified_decision_tree.py  # Thin CLI wrapper
├── verify_cdt.py              # CDT package inspector
├── artifacts/                 # Design docs, results
└── pyproject.toml
```

## Performance

| Metric | Before (Phase 0) | After (optimized) |
|--------|-------------------|-------------------|
| Single LLM call | 30-60s | 2-3s |
| 8 parallel calls | 4-8 min | ~3s |
| Kasumi CDT | ~90 min | ~13 min |

Optimizations: tools=[], setting_sources=[], Haiku model, asyncio.gather parallelism.

## Known Issues

- `run_benchmark.py` and `cdt_profiling.py` not yet migrated (still import OpenAI)
- `cdt_profiling.py` still has `exec()` on LLM output
- `prompts.py` at 73% coverage (needs real Claude for full coverage)
- Module-level mutable state in embeddings.py/validation.py (init_models pattern)

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
