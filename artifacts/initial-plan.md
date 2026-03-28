# Canopy — Initial Plan

> 2026-03-27

**Note:** canopy-design.md is the authoritative design reference. This plan covers implementation phasing. The design doc (decisions D1-D30) supersedes any conflicting details here.

## Vision

Extend Codified Decision Trees (CDT) into a general-purpose, evolving behavioral profiling library. The first application is user behavior profiling for AI coding sessions (delulu project), but the library should be domain-agnostic.

## Principles

- **Step by step** — Nail Phase 0-1 before planning further. No scope creep.
- **Branch per phase** — Each phase gets its own branch, merged only when complete.
- **Simple and clean** — No over-engineering. Minimum complexity for current needs.
- **Research first** — Always search online for tools, libraries, and patterns to reuse before building from scratch.
- **Iterate** — Small commits, frequent validation against real data.
- **Gate before proceeding** — Each phase has success criteria. Don't start the next phase until the current one passes.

## Phase 0: Baseline Reproduction — COMPLETE

**Goal:** Reproduce the original CDT paper's benchmarks to verify the implementation works and establish baselines.

> **Status note:** We're using Claude from the start (not GPT), so Phase 0 and Phase 1 are partially merged — Claude is the LLM backend during reproduction, not a later migration target.

### Phase 0A: Smoke Test — COMPLETE

- Kasumi CDT constructed with Claude as LLM backend
- Results: 26 nodes, 72 statements
- Verified end-to-end pipeline works

### Phase 0B: Full Reproduction — COMPLETE

- 3 characters validated: Kasumi (13min), Arisa (45min), Yui (2hr)

### Success Criteria
- [x] CDT construction runs end-to-end for 1+ character (Kasumi done)
- [x] CDT construction runs end-to-end for 3+ characters
- [x] Benchmark scores compared to paper's reported numbers
- [x] Wikification produces readable profiles
- [x] All findings documented

## Phase 1: Code Modernization + Claude Migration — COMPLETE

**Goal:** Restructure into a proper installable Python package with Claude as the LLM backend.

### Tasks

1. **Package restructure** — Move core logic into `src/canopy/`
2. **Remove exec()** — Replace `exec()` on LLM output with structured JSON parsing
3. **LLM adapter pattern** — Abstract LLM calls behind adapter interface (already started)
4. **Proper installable package** — `uv add canopy-ai` from local path works, Python >=3.13
5. **Migrate run_benchmark.py and cdt_profiling.py** — Into package CLI or modules
6. **Tests** — Unit tests for core algorithm, traversal, validation

### Success Criteria
- [x] `uv sync` / `uv add canopy-ai` works
- [x] `import canopy` imports cleanly
- [x] No exec() in codebase
- [x] LLM calls go through adapter pattern
- [x] run_benchmark.py and cdt_profiling.py migrated
- [x] Test coverage >= 80%

## Phase 2: Core Library API — COMPLETE

**Goal:** Redesign the core API around behavioral observations and modern clustering.

### Tasks

1. **BehavioralObservation as primary input** — Replace SceneActionPair with BehavioralObservation (richer, domain-agnostic)
2. **HDBSCAN clustering** — Replace KMeans with HDBSCAN (density-based, no predefined k)
3. **Computed confidence** — Derive confidence from evidence counts, not LLM-assigned scores
4. **Unified hypothesis pipeline** — Any source (LLM, human, import) feeds the same pipeline
5. **Two-pass cross-cluster validation** — Validate hypotheses across clusters for consistency
6. **Traversal API** — Semantic, LLM, and hybrid traversal modes

### Success Criteria
- [x] BehavioralObservation is the primary input type
- [x] HDBSCAN replaces KMeans
- [x] Confidence computed from evidence, not LLM
- [x] Cross-cluster validation implemented
- [x] Traversal API supports all three modes
- [x] Test coverage >= 80%

## Phase 3: Integration Tests + Examples — COMPLETE

**Goal:** Integration tests, examples, README.

### Success Criteria
- [x] 11 GPU integration tests passing
- [x] Examples directory with quickstart.py
- [x] README with full API reference

## Phase 4: Legacy Migration + Batch LLM — COMPLETE

**Goal:** Migrate all legacy files, add batch LLM and parallel benchmark.

### Success Criteria
- [x] All legacy files migrated (zero OpenAI/exec/constant refs)
- [x] batch_generate() with retry/drop tracking
- [x] ThreadPoolExecutor parallel benchmark
- [x] 184 tests (173 unit + 11 integration)

## Key Research Papers

| Paper | Relevance |
|-------|-----------|
| [CDT (arxiv 2601.10080)](https://arxiv.org/abs/2601.10080) | Base algorithm |
| [PERSONAMEM (arxiv 2504.14225)](https://arxiv.org/abs/2504.14225) | Dynamic profiling benchmark, temporal insights |
| [PURE (arxiv 2502.14541)](https://arxiv.org/abs/2502.14541) | Incremental profile management |
| [Zero-Shot DT (arxiv 2501.16247)](https://arxiv.org/abs/2501.16247) | LLM-based decision tree induction |
| [RL-LLM-DT (arxiv 2412.11417)](https://arxiv.org/abs/2412.11417) | Iterative DT improvement with LLM |

## Architecture (Target)

See canopy-design.md Section 11 for the full target architecture and module layout.
