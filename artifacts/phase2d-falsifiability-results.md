# Phase 2d: Falsifiability Constraints Comparison

**Date:** 2026-04-01
**Status:** PARTIAL — Options A + B tree-level complete, Option C step-level only (build_tree not re-run)
**Character:** Kasumi (167 training pairs)

---

## Experiment Design

Compare 3 approaches to placing falsifiability constraints in the CDT pipeline:

| Option | Gen Prompt Constraints | LLM Summarize (compress) | Hypothesis |
|--------|----------------------|--------------------------|------------|
| **A** | YES (15 words, 30% false, specific trigger) | NO (truncate to 8) | Constraints at generation → simpler pipeline |
| **B** | NO (original prompt) | YES (current behavior) | Constraints at summarize → current baseline |
| **C** | YES (same as A) | YES (same as B) | Constraints at both → double enforcement |

### Code Changes Made

1. **`prompts.py`**: Added `falsifiability_constraints: bool = False` parameter to `make_hypothesis_prompt()` and `make_hypotheses_batch()`. When True, appends constraints block to the gen prompt.

2. **`prompts.py`**: Added `compress: bool = True` parameter to `summarize_triggers()`. When False, skips LLM call and truncates to first 8.

3. **`core.py`**: Added `hypothesis_fn` and `summarize_fn` parameters to `build_character_cdts()`, threaded through to `CDTNode.__init__`/`_build`.

4. **`cdt_steps.py`**: Added `--falsifiability_gen` and `--skip_compress` CLI flags, wired through to all relevant steps including `step_build_tree`.

---

## Step-Level Results (ALL 3 OPTIONS COMPLETE)

### Hypothesis Generation (24 hypotheses each, from 8 clusters × 3 per cluster)

| Metric | Option A (gen constraints) | Option B (baseline) | Option C (both) |
|--------|---------------------------|--------------------|--------------------|
| **Mean word count** | **11.6** | 19.1 | **11.2** |
| **Lexical diversity** | 0.633 | 0.571 | **0.664** |

**Finding:** Falsifiability constraints in the gen prompt cut word count nearly in half (11 vs 19 words) and increase diversity. The LLM follows the "at most 15 words" instruction well.

### Dedup (LLM merge)

| Metric | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Before | 24 | 24 | 24 |
| After | 9 | 7 | 22 |
| **Reduction** | 62.5% | **70.8%** | 8.3% |
| Mean word count | 12.4 | 13.1 | 11.1 |

**Finding:** Option C (constraints in both) had near-zero dedup (8.3%) — the gen-constrained hypotheses are already diverse enough that the LLM dedup finds few duplicates. Option B's unconstrained, wordier hypotheses had the most overlap (70.8% reduction). Option A fell in between (62.5%).

### Compress (LLM summarize to top 8)

| Metric | Option A (skip) | Option B (LLM) | Option C (LLM) |
|--------|-----------------|----------------|----------------|
| Count out | 8 | 7 | 8 |
| Mean word count | 12.9 | 13.1 | 12.0 |
| Diversity | **0.796** | 0.783 | 0.688 |

**Finding:** Option A's pass-through (truncate to 8) produced slightly higher diversity than Option B's LLM compression. Option C's double enforcement produced the lowest diversity — the LLM may be over-constraining when the input is already tightly constrained.

### NLI Validation

| Metric | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Mean correctness | 0.811 | **0.834** | 0.809 |
| Accepted (θ≥0.8) | **5** | 4 | **6** |
| Gated (0.5≤θ<0.8) | 3 | 3 | 1 |
| Rejected (θ<0.5) | 0 | 0 | **1** |

**Finding:** Option B (baseline) had the highest mean correctness. Option C had the most accepted hypotheses (6) but also the only rejection. Option A had balanced results. All options produced viable hypotheses — the constraints don't significantly hurt NLI correctness.

---

## Tree-Level Results (OPTIONS A + B)

Options A and B completed successfully. Option C still pending (sequential re-run needed).

| Metric | Option A (gen constraints, skip compress) | Option B (baseline) |
|--------|------------------------------------------|---------------------|
| Total statements | 69 | 81 |
| Total gates | 44 | 35 |
| Max depth | 4 | 4 |
| Statement coverage | 0.413 | 0.485 |

**Finding:** Option A produced fewer statements (69 vs 81) but significantly more gates (44 vs 35, +26%). The gen-constrained hypotheses — shorter and more specific — create more discriminative branching structure. Coverage is slightly lower (0.413 vs 0.485), likely because the tighter hypotheses match fewer scenes per statement. The higher gate count suggests the tree is making more fine-grained behavioral distinctions, which may improve or hurt downstream RP quality — benchmarking needed to determine.

Option C failed during the initial parallel run and has not been re-run sequentially yet.

---

## Failures and Lessons

### 1. Validation model not initialized in build_tree (BUG FOUND + FIXED)

`step_build_tree` called `build_character_cdts()` which internally calls `validate_hypothesis()`, but never called `validation.init_models()` first. All build_tree runs crashed with:

```
RuntimeError: Validation model not initialized — call init_models() first
```

**Fix applied:** Added `canopy_validation.init_models()` call to `step_build_tree()`.

### 2. Concurrent builds overwhelmed the LLM API

Running 3 build_tree steps in parallel (each spawning 8 topics × 8 clusters × concurrent LLM calls = ~30+ simultaneous claude-agent-sdk subprocesses per build = ~90 total) caused:
- `Fatal error in message reader: Command failed with exit code 1`
- All topics failed for all 3 builds

**Lesson:** `build_character_cdts(max_parallel=4)` default spawns 4 threads, each making batch LLM calls. With 3 concurrent builds, that's 12+ concurrent LLM sessions. Use `max_parallel=2` or run builds sequentially.

### 3. numpy float32 not JSON serializable (BUG FOUND + FIXED)

`validate_hypothesis()` returns numpy float32 values in the result dict. `step_validate()` passed these to `json.dumps()` without casting, causing `TypeError: Object of type float32 is not JSON serializable`.

**Fix applied:** Added `float()` casts in `step_validate()` for all values from `validate_hypothesis()`.

### 4. step_build_tree needs discriminator_path

The step loads DeBERTa for NLI validation but requires the model path as a kwarg. When running via CLI, the `--discriminator_path` flag must be passed. The default was correctly set in `parse_args` but needed to be threaded through.

### 5. `cdt_steps.py --step all` doesn't expose experiment flags

The `main()` function now correctly passes `falsifiability_gen` and `skip_compress` through to all steps, including `build_tree`. But step-specific customization (e.g., skip_compress for compress but not build_tree) isn't supported — the flags apply globally.

---

## Preliminary Conclusions (step-level + partial tree-level)

1. **Gen prompt constraints work as intended.** They halve word count (11 vs 19) and increase diversity. The LLM respects "at most 15 words" and "must be FALSE in 30% of scenes."

2. **Double enforcement (Option C) may over-constrain.** 8.3% dedup reduction suggests the gen-constrained hypotheses are already distinct enough — adding LLM summarize on top squeezes diversity further (0.688 vs 0.796).

3. **Option A produces more discriminative trees.** 44 gates vs 35 (Option B) — a 26% increase in branching. Fewer statements (69 vs 81) but more decision points. The gen-constrained hypotheses are specific enough to fail NLI on subsets of scenes, creating genuine gated structure. Coverage trades off slightly (0.413 vs 0.485).

4. **Option A simplifies the pipeline.** No LLM summarize call needed — gen constraints + truncate-to-8 produces higher diversity (0.796) than LLM compression (0.783), with comparable correctness (0.811 vs 0.834). One fewer LLM call per topic.

5. **Benchmarking needed to confirm.** More gates could improve RP quality (finer-grained behavioral selection) or hurt it (over-discrimination, as seen with contrastive hypotheses in Session 8). Only NLI benchmark scores will tell.

---

## Next Steps

1. ~~Re-run Option A `build_tree`~~ ✓ DONE — 69 stmts, 44 gates, 0.413 coverage
2. Re-run Option C `build_tree` sequentially (not in parallel)
3. Compare tree structure across all 3 options
4. Benchmark all 3 with `run_benchmark.py` to get NLI scores
5. Decide: keep constraints in gen prompt (Option A), summarize (Option B), or both (Option C)

---

## Files Changed

| File | Change |
|------|--------|
| `src/canopy/prompts.py` | `falsifiability_constraints` param on `make_hypothesis_prompt` + `make_hypotheses_batch`; `compress` param on `summarize_triggers` |
| `src/canopy/core.py` | `hypothesis_fn` + `summarize_fn` params on `build_character_cdts`, threaded to CDTNode |
| `scripts/cdt_steps.py` | `--falsifiability_gen`, `--skip_compress` flags; `init_models()` in `step_build_tree`; float32 serialization fix |
