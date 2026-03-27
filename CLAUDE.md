# Canopy

> Evolving decision trees for behavioral profiling.

## Project Overview

Canopy extends Codified Decision Trees (CDT) with temporal dynamics, structured gates, and domain-agnostic profiling. Built on the CDT paper (arxiv 2601.10080).

## Current Phase

**Phase 0: Baseline Reproduction** — Reproducing original CDT benchmarks.

## Tech Stack

- **Language:** Python 3.11+
- **Package manager:** uv (NEVER pip)
- **LLM:** Anthropic Claude (replacing OpenAI from original CDT)
- **Embeddings:** Qwen3-8B (scene), Qwen3-Embedding-8B (action)
- **NLI:** DeBERTa-v3-base-rp-nli
- **Testing:** pytest
- **Linting:** ruff

## Key Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run ruff check .        # Lint
```

## Project Structure

```
canopy-ai/
├── src/canopy/            # Package source (target structure)
├── artifacts/             # Design docs, results
├── tests/                 # Test suite
├── codified_decision_tree.py  # Original CDT implementation (Phase 0)
├── run_benchmark.py       # Original benchmark runner (Phase 0)
├── cdt_profiling.py       # Original profiling script (Phase 0)
└── pyproject.toml
```

## Conventions

- Use uv for all Python operations
- Type hints everywhere
- Immutable data structures preferred
- No exec() or eval() on LLM output — structured JSON parsing only
- All LLM calls use Anthropic Claude (after Phase 1 migration)

## Attribution

Based on Codified Decision Tree by Letian Peng et al. (MIT License).
See artifacts/initial-plan.md for the full development plan.
