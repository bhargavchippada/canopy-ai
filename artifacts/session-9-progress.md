# Session 9 Progress (2026-03-30 → 2026-03-31)

## Critical Finding: CDT Non-Determinism

Same configuration rebuilt CDT scored **60.30** vs original **70.66** — 10-point variance from identical parameters. LLM hypothesis generation is inherently stochastic; tree structure cascades from early choices.

**Impact**: All 7 experiments appeared to regress (-5 to -9 pts) from 70.66, but the baseline was an unreproducible lucky build that got overwritten during the merge experiment. The "regressions" were within normal build variance.

## Experiments Run

| # | Experiment | Score | vs 70.66 | Notes |
|---|------------|-------|----------|-------|
| G2 | Contrastive d3 (4 topics) | 64.07 | -6.59 | Within build variance |
| H | discover_topics (4 extra) | 62.05 | -8.61 | Topics fragment pair pool |
| I | Hypothesis merge | 65.57 | -5.09 | Merge too aggressive (15 vs 69 nodes) |
| J | Baseline rebuild (same config) | **60.30** | **-10.36** | Proves non-determinism |
| K | dialogue_v3 (same CDT as J) | 60.18 | -0.12 vs J | Gen prompt has no effect |
| L | LifeChoice (Haiku, 13Q) | 15.4% | -30.8% | Inconclusive: tiny sample, wrong model |

## Code Changes

1. **Retry logic** in `build_character_cdts()` — failed topics retry sequentially after parallel phase
2. **`--no_merge` flag** — skip hypothesis merge LLM dedup during CDT construction
3. **`merge_fn` parameter** on `build_character_cdts()` — injectable merge function
4. **`scripts/lifechoice_benchmark.py`** — LifeChoice MCQ evaluation with LLM-generated profiles
5. **Contrastive instruction removed** from `make_hypothesis_prompt()` (reverted)

## Tests

- 361 passed, 13 skipped (GPU integration)
- New tests: `test_build_character_cdts_retries_failed_topics`, `test_build_character_cdts_handles_partial_failure` (updated)

## Files Modified
- `src/canopy/core.py` — retry logic, merge_fn parameter
- `src/canopy/prompts.py` — contrastive instruction removed
- `tests/test_core.py` — retry tests
- `codified_decision_tree.py` — --no_merge flag
- `results/experiment_log.md` — all experiment results
- `scripts/lifechoice_benchmark.py` — NEW
- `data/LifeChoice/` — cloned dataset

## What's Needed Next

1. **Fix non-determinism**: Seed KMeans, save/reuse CDT pickles for A/B testing
2. **LifeChoice with Sonnet**: 50+ questions for real signal
3. **Multi-build averaging**: Build N CDTs, benchmark each, report mean ± std
4. **Commit all changes**: Code changes uncommitted on master*
