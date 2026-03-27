# Canopy — Initial Plan

> 2026-03-27

**Note:** canopy-design.md is the authoritative design reference. This plan covers implementation phasing. The design doc (25 decisions, D1-D25) supersedes any conflicting details here.

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

> **Status note:** We're using Claude from the start (not GPT), so Phase 0 and Phase 1 are partially merged — Claude is the LLM backend during reproduction, not a later migration target.

### Phase 0A: Smoke Test — COMPLETE

- Kasumi CDT constructed with Claude as LLM backend
- Results: 26 nodes, 72 statements
- Verified end-to-end pipeline works

### Phase 0B: Full Reproduction — IN PROGRESS

- Arisa CDT construction with depth=3
- Remaining characters TBD after Arisa completes

### Remaining Tasks

1. **Benchmark reproduction**
   - Run `run_benchmark.py` for completed characters
   - Compare accuracy metrics to paper's Table 1 and Table 2
   - Document any discrepancies

2. **Wikification reproduction**
   - Run wikification notebook on at least 1 character
   - Verify output quality matches paper's examples

3. **Document findings**
   - Write `artifacts/baseline-results.md` with all reproduction results
   - Note any issues, discrepancies, or gotchas discovered

### Success Criteria
- [x] CDT construction runs end-to-end for 1+ character (Kasumi done)
- [ ] CDT construction runs end-to-end for 3+ characters
- [ ] Benchmark scores within 5% of paper's reported numbers
- [ ] Wikification produces readable profiles
- [ ] All findings documented

## Phase 1: Code Modernization + Claude Migration

**Goal:** Restructure into a proper installable Python package with Claude as the LLM backend.

### Tasks

1. **Package restructure** — Move core logic into `src/canopy/`
2. **Remove exec()** — Replace `exec()` on LLM output with structured JSON parsing
3. **LLM adapter pattern** — Abstract LLM calls behind adapter interface (already started)
4. **Proper installable package** — `uv add canopy-ai` from local path works, Python >=3.13
5. **Migrate run_benchmark.py and cdt_profiling.py** — Into package CLI or modules
6. **Tests** — Unit tests for core algorithm, traversal, validation

### Success Criteria
- [ ] `uv sync` / `uv add canopy-ai` works
- [ ] `import canopy` imports cleanly
- [ ] No exec() in codebase
- [ ] LLM calls go through adapter pattern
- [ ] run_benchmark.py and cdt_profiling.py migrated
- [ ] Test coverage >= 80%

## Phase 2: Core Library API

**Goal:** Redesign the core API around behavioral observations and modern clustering.

### Tasks

1. **BehavioralObservation as primary input** — Replace SceneActionPair with BehavioralObservation (richer, domain-agnostic)
2. **HDBSCAN clustering** — Replace KMeans with HDBSCAN (density-based, no predefined k)
3. **Computed confidence** — Derive confidence from evidence counts, not LLM-assigned scores
4. **Unified hypothesis pipeline** — Any source (LLM, human, import) feeds the same pipeline
5. **Two-pass cross-cluster validation** — Validate hypotheses across clusters for consistency
6. **Traversal API** — Semantic, LLM, and hybrid traversal modes

### Success Criteria
- [ ] BehavioralObservation is the primary input type
- [ ] HDBSCAN replaces KMeans
- [ ] Confidence computed from evidence, not LLM
- [ ] Cross-cluster validation implemented
- [ ] Traversal API supports all three modes
- [ ] Test coverage >= 80%

## Phase 3: T-CDT + Advanced Features

**Goal:** Add temporal dimension, incremental growth, and calibration.

### Tasks

1. **Temporal weighting** — Recent evidence weighs more during validation
2. **Supersession tracking** — Contradicted patterns marked as superseded (not deleted), with timestamps and reasons
3. **Incremental tree growth** — 8-step algorithm for adding new data without full rebuild
4. **Gate calibration** — Calibrate gates using positive and negative examples
5. **Bootstrap mode** — Cold-start CDT construction from minimal data
6. **Evaluation framework** — Benchmarks comparing T-CDT vs standard CDT on evolving data

### Success Criteria
- [ ] T-CDT produces measurably better profiles on temporally-evolving datasets
- [ ] Incremental update works via 8-step algorithm without full reconstruction
- [ ] Superseded patterns preserved with timestamps and reasons
- [ ] Gate calibration from positive/negative examples
- [ ] Bootstrap mode functional
- [ ] Evaluation framework produces reproducible results

## Phase 4: Paper + Polish

**Goal:** Benchmark, write research paper, and publish the package.

### Tasks

1. **Benchmark evaluation** — Compare canopy T-CDT against original CDT paper results
2. **Research paper** — Introduction, Related Work, Method, Experiments, Analysis, Conclusion
3. **Datasets** — Original CDT benchmarks + new temporal dataset (delulu sessions)
4. **Baselines** — CDT, PURE, PERSONAMEM approaches
5. **Target venue** — ACL/EMNLP/NeurIPS workshop on personalization or user modeling
6. **Documentation** — Examples, tutorials, API reference
7. **PyPI publish** — `uv add canopy-ai` from PyPI

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
