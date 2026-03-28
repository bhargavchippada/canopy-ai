# Canopy

> Evolving decision trees for behavioral profiling.

## Project Overview

Canopy extends Codified Decision Trees (CDT) with temporal dynamics, structured gates, and domain-agnostic profiling. Built on the CDT paper (arxiv 2601.10080).

## Current Phase

**Phase 0: COMPLETE** — 3 characters validated (Kasumi, Arisa, Yui).
**Phase 1: COMPLETE** — Package restructured into `src/canopy/`, 91% test coverage.
**Phase 2: Core Library API** — IN PROGRESS.
- BehavioralObservation as primary input type
- builder.py: build_cdt(), build_character_profile()
- wikify.py: CDT → markdown profiles
- cluster.py: KMeansCluster, HDBSCANCluster (with Protocol)
**Next:** Phase 3 (T-CDT + Advanced Features).

## Design Reference

**Authoritative CDT design:** `artifacts/canopy-design.md` (25 design decisions D1-D25)
- BehavioralObservation as primary input (not SceneActionPair)
- HDBSCAN clustering (discovers domains from data)
- Semantic gate conditions (embedding cosine similarity)
- Two-pass cross-cluster validation (follows CDT paper)
- T-CDT temporal weighting (configurable, fallback to equal weight)
- Unified hypothesis pipeline (any source: clusters, session cards, rules, docs, manual)
- Bootstrap mode (CDT from rules/docs with zero sessions)
- Computed confidence from evidence counts (not LLM self-assessed)

## Principles

- **Step by step** — Nail Phase 0-1 before planning further. No scope creep.
- **Branch per phase** — Each phase gets its own branch, merged only when complete.
- **Simple and clean** — No over-engineering. Minimum complexity for current needs.
- **Research first** — Always search online for tools, libraries, and patterns to reuse before building from scratch.
- **Iterate** — Small commits, frequent validation against real data.
- **Gate before proceeding** — Each phase has success criteria. Don't start next until current passes.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv (NEVER pip)
- **LLM:** Claude via claude-agent-sdk (Max subscription, no API key)
- **LLM Model:** claude-sonnet-4-6 (default)
- **Embeddings:** Qwen3-0.6B (Phase 0 smoke test) / all-MiniLM-L6-v2 (library default, see D24)
- **NLI:** DeBERTa-v3-base-rp-nli
- **Testing:** pytest
- **Linting:** ruff

## Library API

```python
from canopy import BehavioralObservation, CDTConfig, build_cdt, build_character_profile
from canopy.wikify import wikify_tree, wikify_profile
from canopy.llm import ClaudeCodeAdapter, set_adapter

# 1. Configure LLM
set_adapter(ClaudeCodeAdapter(default_model="claude-haiku-4-5"))

# 2. Create observations
obs = [
    BehavioralObservation(scene="...", action="...", actor="Alice", participants=["Bob"]),
]

# 3. Build CDT
tree = build_cdt(obs, character="Alice", topic="identity", config=CDTConfig(max_depth=2))

# 4. Or build full profile
topic2cdt, rel_topic2cdt = build_character_profile(obs, character="Alice")

# 5. Wikify to markdown
markdown = wikify_profile(topic2cdt, rel_topic2cdt, character="Alice")
```

See `artifacts/llm-adapters.md` for provider comparison.

## Key Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run ruff check .        # Lint

# CDT construction (smoke test with 0.6B models)
uv run python codified_decision_tree.py \
  --character Kasumi \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --device_id 0

# CDT construction (full run with 8B models — needs sequential loading)
# TODO: implement sequential model loading for 32GB VRAM
```

## Model Paths

Models stored in `~/models/`:

| Model | Path | Size | Status |
|-------|------|------|--------|
| DeBERTa NLI | `~/models/deberta-v3-base-rp-nli` | 715MB | Downloaded |
| Qwen3-Embedding-0.6B | `~/models/Qwen3-Embedding-0.6B` | ~1.2GB | Downloaded |
| Qwen3-0.6B | `~/models/Qwen3-0.6B` | ~1.2GB | Downloaded |
| Qwen3-Embedding-8B | `~/models/Qwen3-Embedding-8B` | ~16GB | Downloading |
| Qwen3-8B | `~/models/Qwen3-8B` | ~16GB | Downloading |
| Llama-3.1-8B-Instruct | `~/models/Llama-3.1-8B-Instruct` | ~16GB | Needs HF login (gated) |

**VRAM Note (RTX 5090, 32GB):** Cannot load DeBERTa + Qwen3-8B + Qwen3-Embedding-8B simultaneously (~35GB). Use 0.6B models for smoke testing. Full 8B run requires sequential model loading (not yet implemented).

## Project Structure

```
canopy-ai/
├── src/canopy/                # Core package
│   ├── __init__.py            # Exports: CDTConfig, CDTNode, BehavioralObservation, build_*
│   ├── core.py                # CDTNode, CDTConfig, build_character_cdts
│   ├── builder.py             # BehavioralObservation, build_cdt, build_character_profile
│   ├── wikify.py              # CDT → markdown profiles (wikify_tree, wikify_profile)
│   ├── cluster.py             # KMeansCluster, HDBSCANCluster, ClusterStrategy Protocol
│   ├── embeddings.py          # Model loading, encoding (uses cluster.py)
│   ├── validation.py          # NLI-based scene/statement validation (DeBERTa)
│   ├── prompts.py             # Hypothesis generation prompts + batch processing
│   ├── llm.py                 # LLM adapter (Protocol + ClaudeCodeAdapter)
│   ├── data.py                # HuggingFace dataset loading + caching
│   └── cli.py                 # CLI placeholder
├── codified_decision_tree.py  # Thin CLI wrapper (imports from canopy)
├── verify_cdt.py              # CDT package inspector
├── llm.py                     # Re-export shim (backwards compat)
├── run_benchmark.py           # NOT YET MIGRATED — Phase 1
├── cdt_profiling.py           # NOT YET MIGRATED — Phase 1
├── artifacts/                 # Design docs, results
├── tests/                     # Test suite
└── pyproject.toml
```

## Phase 0 Results

Baseline reproduction PASSED for 2 characters:

| Character | Nodes | Statements | Max Depth | Runtime |
|-----------|-------|------------|-----------|---------|
| Kasumi | 26 | 72 | 3 | ~1.5 hr |
| Arisa | 30 | 79 | 4 | ~3 hr |

Paper PoPiPa avg: 10.4 nodes, 61 statements (comparable given config differences).
See `artifacts/phase0-results.md` for full analysis.

## Performance Bottleneck

LLM calls via claude-code-sdk subprocess: **30-60s per call**.
CDT construction makes ~50-100+ calls per character.

**Planned optimizations:**
- Switch hypothesis gen to Haiku (faster, cheaper — Sonnet overkill)
- Add `extra_args` to skip settings/tools loading (~4.5s savings per call)
- Parallelize independent LLM calls with `asyncio.gather`
- Use `claude-code-sdk` (not `claude-agent-sdk`) matching delulu patterns

**Model allocation:**
- **DeBERTa NLI** — validation (GPU, 6 seconds for full dataset, keep as-is)
- **Haiku** — hypothesis generation (LLM, currently the bottleneck)
- **Sonnet** — reserve for evaluation/benchmarking where quality matters

## Known Issues

- `run_benchmark.py` and `cdt_profiling.py` still import from OpenAI — will crash until Phase 1 migration
- `cdt_profiling.py` still has `exec()` on LLM output — Phase 1 will fix
- Llama-3.1-8B-Instruct requires HF login for download (gated model)
- LLM call overhead: ~30-60s per call (subprocess + settings loading)

## Conventions

- Use uv for all Python operations
- Type hints everywhere
- Immutable data structures preferred
- No exec() or eval() on LLM output — structured JSON parsing only
- All LLM calls route through `llm.py` adapter (never direct API calls)
- All LLM calls use Claude via claude-agent-sdk (Max subscription)

## Attribution

Based on Codified Decision Tree by Letian Peng et al. (MIT License).
See artifacts/initial-plan.md for the full development plan.
