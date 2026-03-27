# Canopy — Initial Plan

> 2026-03-27

## Vision

Extend Codified Decision Trees (CDT) into a general-purpose, evolving behavioral profiling library. The first application is user behavior profiling for AI coding sessions (delulu project), but the library should be domain-agnostic.

## Principles

- **Step by step** — Nail Phase 0-1 before planning further. No scope creep.
- **Branch per phase** — Each phase gets its own branch, merged only when complete.
- **Simple and clean** — No over-engineering. Minimum complexity for current needs.
- **Research first** — Always search online for tools, libraries, and patterns to reuse before building from scratch.
- **Iterate** — Small commits, frequent validation against real data.
- **Gate before proceeding** — Each phase has success criteria. Don't start the next phase until the current one passes.

## Phase 0: Baseline Reproduction

**Goal:** Reproduce the original CDT paper's benchmarks to verify the implementation works and establish baselines.

### Tasks

1. **Environment setup**
   - Install dependencies with uv
   - Verify GPU access (CUDA)
   - Download required models: Qwen3-8B, Qwen3-Embedding-8B, DeBERTa-v3-base-rp-nli
   - Set up HuggingFace token access
   - Set up OpenAI API key (for initial reproduction; will swap to Claude later)

2. **Data download**
   - Download Fine-grained Fandom Benchmark action sequences from HuggingFace
   - Download Bandori Conversational Benchmark action sequences from HuggingFace
   - Verify data format matches code expectations

3. **CDT construction reproduction**
   - Run `build_cdt.sh` for at least 3 characters (e.g., Kasumi, Arisa, Yui)
   - Compare output CDT structure to paper's reported statistics (node counts, statement counts)
   - Save constructed CDTs as baseline packages

4. **Benchmark reproduction**
   - Run `run_benchmark.py` for the same characters
   - Compare accuracy metrics to paper's Table 1 and Table 2
   - Document any discrepancies

5. **Wikification reproduction**
   - Run wikification notebook on at least 1 character
   - Verify output quality matches paper's examples

6. **Document findings**
   - Write `artifacts/baseline-results.md` with all reproduction results
   - Note any issues, discrepancies, or gotchas discovered

### Success Criteria
- [ ] CDT construction runs end-to-end for 3+ characters
- [ ] Benchmark scores within 5% of paper's reported numbers
- [ ] Wikification produces readable profiles
- [ ] All findings documented

## Phase 1: Claude Migration

**Goal:** Replace OpenAI dependency with Anthropic Claude for all LLM calls.

### Tasks

1. **Swap hypothesis generation** — GPT-4.1 → Claude Sonnet 4.6
2. **Swap validation** — GPT-4.1-mini → Claude Haiku 4.5 (cheaper) or Sonnet
3. **Swap wikification** — GPT-4.1 → Claude Sonnet 4.6
4. **Swap evaluation** — GPT-4.1 → Claude Sonnet 4.6
5. **Remove exec()** — Replace `exec()` on LLM output with structured JSON parsing
6. **Re-run benchmarks** — Verify Claude-based CDTs match or exceed GPT-4.1 baselines
7. **Document findings** — `artifacts/claude-migration-results.md`

### Success Criteria
- [ ] No OpenAI dependency for core functionality (openai optional extra only)
- [ ] No exec() in codebase
- [ ] Benchmark scores within 5% of Phase 0 baselines
- [ ] All LLM calls use anthropic SDK or claude-code-sdk

## Phase 2: Code Modernization

**Goal:** Restructure into a proper installable Python package.

### Tasks

1. **Package structure** — Move core logic into `src/canopy/`
   - `src/canopy/core.py` — CDT_Node class (cleaned up)
   - `src/canopy/embeddings.py` — Embedding models
   - `src/canopy/validation.py` — NLI validation
   - `src/canopy/prompts.py` — All LLM prompts
   - `src/canopy/traverse.py` — CDT traversal
   - `src/canopy/wikify.py` — Wikification
   - `src/canopy/cli.py` — CLI commands
2. **Remove global state** — Models loaded at module level → dependency injection
3. **Add type hints** throughout
4. **Add tests** — Unit tests for core algorithm, traversal, validation
5. **Add CLI** — `canopy build`, `canopy traverse`, `canopy wikify`, `canopy benchmark`
6. **Verify installability** — `uv add canopy-ai` from local path works

### Success Criteria
- [ ] `pip install -e .` / `uv sync` works
- [ ] `import canopy` imports cleanly
- [ ] All existing functionality preserved
- [ ] Test coverage >= 80%

## Phase 3: Temporal CDT (T-CDT)

**Goal:** Add temporal dimension — newer evidence weighs more, superseded patterns tracked.

### Tasks

1. **Time-weighted validation** — Recent (scene, action) pairs weighted higher during NLI validation
2. **Supersession tracking** — When new evidence contradicts old patterns, mark old as superseded (not deleted)
3. **Recency-gated branches** — Gate conditions can include temporal predicates ("since date X")
4. **Incremental tree growth** — Add new data to existing CDT without full rebuild
5. **Benchmark against standard CDT** — Does T-CDT improve profile accuracy on evolving data?

### Success Criteria
- [ ] T-CDT produces measurably better profiles on temporally-evolving datasets
- [ ] Incremental update works without full reconstruction
- [ ] Superseded patterns preserved with timestamps and reasons

## Phase 4: Semantic Gates + Domain-Agnostic

**Goal:** Replace exec()-based gate conditions with semantic embedding gates. Make CDT work for any domain.

### Tasks

1. **Semantic gate format** — Natural language gates with pre-computed embeddings, evaluated via cosine similarity (see canopy-design.md Section 3)
2. **Deterministic traversal** — No LLM calls at inference time (embedding + cosine only)
3. **Domain adapters** — Plugin system for different data sources (AI sessions, social media, etc.)
4. **Delulu integration** — Use canopy as a library in the delulu project

### Success Criteria
- [ ] Traversal is synchronous, deterministic, no LLM calls
- [ ] delulu imports canopy and builds CDTs from session cards
- [ ] At least 2 domain adapters (character RP + user profiling)

## Phase 5: Paper

**Goal:** Write and submit a research paper on Temporal CDT.

### Tasks

1. **Experimental design** — Define evaluation protocol for T-CDT vs CDT
2. **Datasets** — Original CDT benchmarks + new temporal dataset (delulu sessions)
3. **Baselines** — CDT, PURE, PERSONAMEM approaches
4. **Writing** — Introduction, Related Work, Method, Experiments, Analysis, Conclusion
5. **Target venue** — ACL/EMNLP/NeurIPS workshop on personalization or user modeling

## Key Research Papers

| Paper | Relevance |
|-------|-----------|
| [CDT (arxiv 2601.10080)](https://arxiv.org/abs/2601.10080) | Base algorithm |
| [PERSONAMEM (arxiv 2504.14225)](https://arxiv.org/abs/2504.14225) | Dynamic profiling benchmark, temporal insights |
| [PURE (arxiv 2502.14541)](https://arxiv.org/abs/2502.14541) | Incremental profile management |
| [Zero-Shot DT (arxiv 2501.16247)](https://arxiv.org/abs/2501.16247) | LLM-based decision tree induction |
| [RL-LLM-DT (arxiv 2412.11417)](https://arxiv.org/abs/2412.11417) | Iterative DT improvement with LLM |

## Architecture (Target)

```
canopy-ai/
├── src/canopy/
│   ├── __init__.py
│   ├── cli.py              # CLI: canopy build, traverse, wikify, benchmark
│   ├── core.py             # CDTNode, CDTTree — core data structures
│   ├── builder.py          # CDT construction algorithm
│   ├── embeddings.py       # Embedding model management
│   ├── validation.py       # Evidence-based validation (NLI or LLM)
│   ├── prompts.py          # All LLM prompt templates
│   ├── traverse.py         # Deterministic CDT traversal
│   ├── wikify.py           # CDT → human-readable profile
│   ├── temporal.py         # T-CDT: time weighting, supersession
│   ├── gates.py            # Semantic gate conditions + embedding-based matching
│   └── adapters/           # Domain adapters
│       ├── __init__.py
│       ├── character.py    # Original CDT: character RP from storylines
│       └── user_profile.py # Canopy: user behavior from AI sessions
├── artifacts/              # Design docs, results, plans
├── tests/                  # Test suite
├── pyproject.toml          # Package config (uv/hatch)
└── README.md
```
