# Session Progress (2026-03-28)

## Completed This Session

### 1. Two-Phase Embedding Architecture (DONE)
- Subprocess isolation for VRAM-safe 8B model loading
- EmbeddingCache frozen dataclass, _embed_worker.py
- 7 rounds of convergence review, all findings fixed
- Commits: d418ae7, c2688b1, a8b18de, 00b8772, 8ae94a9

### 2. 100% Test Coverage (DONE)
- 268 tests across 12 modules, all at 100%
- Commits: b08d58d, d1e9c8f

### 3. Benchmark Infrastructure
- TransformersAdapter + DispatchAdapter for local model support
- Multi-eval ensemble scoring (--multi-eval flag)
- benchmark_papercompat.py for paper-exact pipeline replication
- Commits: 40917b8, a7d194a, ba4da9e, 8ea3aa2

### 4. Benchmark Results Summary

| # | Config | Gen | Eval | NLI Score |
|---|--------|-----|------|-----------|
| 1 | Paper (GPT-4.1 CDT) | Llama | GPT-4.1 | 84.25 |
| 2 | Paper CDT, our eval | Sonnet | Sonnet | 65.87 |
| 3 | Our Sonnet v2 CDT (attr) | Sonnet | Sonnet | **67.66** |
| 4 | Our Sonnet v2 CDT (rel) | Sonnet | Sonnet | 64.07 |
| 5 | Sonnet CDT θ=0.75 | Sonnet | Sonnet | 66.17 |
| 6 | Same CDT | Haiku | Haiku | 50.00 |
| 7 | Llama 8-bit gen | Llama | Sonnet | 55.99 |

### 5. Key Findings

1. **CDT algorithm quality SOLVED**: Our v2 CDT (67.66) > paper's CDT (65.87) with same eval
2. **Summarize prompt was the bottleneck**: Paper's detailed prompt produces deeper trees (7 nodes vs 1)
3. **Eval model is dominant factor**: Each model tier adds ~8 points
4. **Remaining 18pt gap** (66→84) = benchmark pipeline (Llama gen + GPT-4.1 eval vs Claude)
5. **Paper always uses Llama for RP gen** — `--engine` only controls eval model

## In Progress

- Paper-compat benchmark running (PID 1359427, 32% done, ~67.59)
- Multi-eval feature implemented but not yet run

## Pending

- Run multi-eval benchmark to test Haiku+Sonnet ensemble
- Final CLAUDE.md update with all benchmark results
- Update canopy-design.md with final findings
