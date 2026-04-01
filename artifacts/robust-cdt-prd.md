# PRD: Robust CDT System — Reproducibility, Quality, and Sidekick Readiness

> "Measure twice, cut once. And make sure your ruler doesn't change between measurements."

**Date:** 2026-04-01
**Status:** DRAFT — awaiting review convergence
**Scope:** Kasumi (Rapori) dataset only — extend after validation

---

## 1. Problem Statement

### 1.1 The Reproducibility Crisis

CDT construction via LLM hypothesis generation is non-deterministic. Same config rebuilt produces 60.30 vs original 70.66 — a **10-point variance** from identical parameters. This means:

- No experiment with delta < 10 pts is statistically meaningful
- All 10 Session 9 experiments were within noise, not real signal
- We can't distinguish improvements from lucky draws
- The paper's results may also suffer from this (unreported)

**Sources of non-determinism:**
| Component | Deterministic? | Notes |
|-----------|---------------|-------|
| KMeans clustering | Yes | `seed=42` in `cluster.py` |
| Embedding models (Qwen) | Mostly | FP16 GPU ops have minor variance |
| DeBERTa NLI validation | Yes | Deterministic at inference |
| **LLM hypothesis generation** | **NO** | **Primary source of variance** |
| LLM merge/summarize | NO | Compounds hypothesis variance |
| LLM topic discovery labels | NO | Additional LLM call |

### 1.2 The Headroom Question

Across 19 Kasumi benchmark runs:
- **Score range (Sonnet eval):** 50.0 — 71.39
- **Best reproducible range:** ~60-70 (the 70.66 was likely an outlier)
- **Paper target (GPT-4.1 eval):** 84.25

The 70→84 gap is **eval model calibration** (GPT-4.1 gives ~18% B vs Sonnet's ~40% B). The B-score analysis proves our evaluator is correct — 77.6% of B pairs are genuinely different-facet responses, not eval errors.

**True headroom with Sonnet eval:** ~5-10 pts (from reducing caricature bias), not 15 pts.

### 1.3 The Sidekick Goal

The ultimate goal is NOT academic RP benchmarks. It's a CDT that:
1. Is human-readable and interpretable
2. Can guide a sidekick agent to take actions like the real user
3. Can be bootstrapped with existing rules/preferences
4. Produces consistent behavior regardless of which LLM builds it
5. Can be updated incrementally as new behavior is observed

---

## 2. Objectives & Success Criteria

### P0: Experiment Reproducibility
- [ ] Same CDT pickle + same benchmark config = same score (within 0.5 pts)
- [ ] Multi-build variance measured: N=5 CDT builds, report mean +/- std
- [ ] Experiment tracking with full provenance (git hash, CDT hash, config hash)
- [ ] No CDT overwrite without explicit versioning

### P1: CDT Quality Evaluation (Intrinsic Metrics)
- [ ] Intrinsic quality metrics defined and computed for every CDT build
- [ ] Quality metrics correlate with benchmark performance (validate via N=5 builds)
- [ ] Quality dashboard: one-command CDT quality report

### P2: RP Headroom Analysis
- [ ] Per-pair difficulty classification (easy/medium/hard/impossible)
- [ ] Theoretical ceiling computed for Sonnet eval
- [ ] Identify which test pairs are improvable vs structurally ambiguous

### P3: Interactive Debugging (Colab)
- [ ] Step-by-step CDT build notebook with intermediate outputs
- [ ] Each step independently runnable and inspectable
- [ ] Works for Kasumi; extensible to any character

### P4: Sidekick-Ready CDT Design
- [ ] Rule bootstrapping: inject known rules as hypothesis candidates
- [ ] Traversal optimization: reduce LLM calls at inference time
- [ ] Wikified context strategy for drift prevention

---

## 3. Architecture & Design

### 3.1 Reproducibility Layer

**Strategy: Cache-and-Replay**

The insight: we can't make LLM calls deterministic, but we can **cache every LLM response** and replay it. This gives us:
1. Exact reproducibility when replaying
2. Multi-build variance measurement when not replaying
3. Full provenance for every experiment

```
CDT Build Pipeline (with caching):

  pairs → [Embedding] → [Clustering] → [Hypothesis Gen] → [Merge] → [Summarize] → [NLI Validate] → [Tree Build]
              |               |              |                |            |              |
              v               v              v                v            v              v
         embed_cache     cluster_cache   hyp_cache       merge_cache  summ_cache    validate_cache
              |               |              |                |            |              |
              └───────────────┴──────────────┴────────────────┴────────────┴──────────────┘
                                                    |
                                               experiment_id/
                                                    ├── config.json
                                                    ├── embed_cache.npz
                                                    ├── cluster_labels.json
                                                    ├── hypotheses.json
                                                    ├── merged_hypotheses.json
                                                    ├── summarized_hypotheses.json
                                                    ├── validation_results.json
                                                    ├── cdt.pkl
                                                    └── metadata.json
```

**Implementation:**

1. **Experiment ID:** SHA256 hash of (config + data hash + git commit). Deterministic given same inputs.
2. **LLM Response Cache:** JSON file per step, keyed by prompt hash. If cache hit → return cached response. If miss → call LLM, cache response.
3. **Cache Modes:**
   - `fresh`: Always call LLM (for measuring variance)
   - `replay`: Only use cached responses (for exact reproducibility)
   - `fallback`: Use cache if available, call LLM if not (default)
4. **CDT Naming:** `{character}.{build_id}.{config_hash}.pkl` where `build_id` is monotonic per character (build_001, build_002, ...) and `config_hash` is first 8 chars of config SHA256.

**Key Design Decision:** Cache at the individual LLM call level, not the pipeline level. This lets us replay partial builds and inspect intermediate states.

### 3.2 Intrinsic CDT Quality Metrics

Benchmark score conflates CDT quality + gen quality + eval calibration. We need metrics that evaluate the CDT **in isolation**.

**Proposed Metrics:**

| Metric | Formula | What It Measures | Target |
|--------|---------|-----------------|--------|
| **Tree Depth** | max depth across all topics | Discriminative capacity | 3-4 |
| **Gate Count** | total gates across all topics | Conditional granularity | 15-25 |
| **Gate Discrimination Rate** | % of test scenes where gate=True vs False | Gate selectivity (50% = ideal) | 30-70% |
| **Statement Coverage** | % of test actions that match >= 1 statement (NLI True) | How much behavior the CDT explains | > 60% |
| **Statement Specificity** | avg % of test actions where each statement is True | How specific (not universal) each statement is | 40-70% |
| **Redundancy** | pairwise cosine similarity of all statements | Duplicate detection | < 0.85 |
| **Topic Balance** | std dev of nodes-per-topic | Even topic coverage | < 3.0 |
| **Gate Activation Variance** | std dev of gate True-rate across scenes | Gates distinguish different contexts | > 0.15 |

**Quality Score (composite):**
```
quality_score = (
    0.25 * normalize(gate_count, 10, 30) +       # Has discriminative structure
    0.25 * normalize(statement_coverage, 0.4, 0.8) +  # Explains behavior
    0.20 * normalize(gate_discrimination, 0.3, 0.7) +  # Gates are selective
    0.15 * (1 - normalize(redundancy, 0.5, 0.9)) +     # Not redundant
    0.15 * normalize(statement_specificity, 0.3, 0.7)   # Statements aren't universal
)
```

This gives us a CDT quality score (0-1) that's independent of generation and evaluation models. We can then validate that higher quality scores correlate with higher benchmark scores across N builds.

### 3.3 RP Headroom Analysis

**Approach:** Classify each of the 167 test pairs by difficulty:

| Difficulty | Criteria | Expected % | Can CDT Help? |
|-----------|----------|-----------|--------------|
| **EASY** | Scene strongly implies one response | ~25% | CDT should get A on these |
| **MEDIUM** | Scene narrows to 2-3 valid responses | ~30% | CDT helps by selecting correct facet |
| **HARD** | Scene allows 5+ valid responses, context-dependent | ~30% | CDT can help only with very specific gates |
| **IMPOSSIBLE** | Scene transition, no prior context, genuinely random | ~15% | No model can predict these |

**Method:**
1. For each test pair, check: does the scene contain a `[Scene: ...]` transition? → IMPOSSIBLE
2. Does the ground truth involve a rare/idiosyncratic behavior? → HARD
3. Does the CDT grounding include a statement matching the ground truth? → EASY/MEDIUM
4. Classify using a combination of heuristics + LLM classification

**Theoretical Ceiling (Sonnet eval):**
- EASY: 100% A → 25 pts
- MEDIUM: 70% A, 30% B → ~18 pts
- HARD: 30% A, 50% B, 20% C → ~11 pts
- IMPOSSIBLE: 10% A, 60% B, 30% C → ~4 pts
- **Total: ~58-68 pts** (this matches our observed range!)

If the theoretical ceiling is ~68-72, then 70.66 was near-optimal and the right strategy is not "improve RP score" but "make CDT quality reliable and extend to new tasks."

### 3.4 Reducing Non-Determinism (Beyond Caching)

**Strategy: More Deterministic Components, Less LLM Dependence**

| Current Step | Deterministic? | Proposed Change |
|-------------|---------------|-----------------|
| Hypothesis generation | NO (LLM) | Cache + multi-build averaging |
| Merge/dedup | NO (LLM) | Replace with **embedding-based dedup** (cosine > 0.85 → merge by picking longer statement). No LLM call. |
| Summarize/compress | NO (LLM) | Replace with **NLI-based ranking** (rank by avg NLI True score, keep top-8). No LLM call. |
| Topic discovery labels | NO (LLM) | Replace with **cluster centroid keywords** (TF-IDF of cluster texts). No LLM call. |
| Gate condition generation | NO (LLM) | Keep — this is the core creative step that needs LLM |

**Net effect:** Reduce LLM calls per CDT build from ~20+ to ~8 (one per cluster for hypothesis generation only). All other steps become deterministic given the same hypotheses.

### 3.5 Rule Bootstrapping

For the sidekick use case, users have existing rules/preferences that should seed the CDT.

**Approach:**
1. Accept rules as a list of `(statement, optional_gate)` pairs
2. Inject rules as Tier 2 hypotheses (source_type=RULE) alongside cluster-generated hypotheses
3. Rules go through the same NLI validation pipeline — they're validated against evidence, not blindly accepted
4. Rules that pass validation become CDT leaf nodes with high confidence
5. Rules that fail validation are flagged for review (the data contradicts the rule)

```python
def build_character_cdts(
    character: str,
    pairs: list[dict[str, Any]],
    ...
    seed_rules: list[tuple[str, str | None]] | None = None,  # NEW
) -> tuple[dict[str, CDTNode], dict[str, CDTNode]]:
```

**Why this works for the sidekick:** Bhargav's coding preferences (from delulu's session analysis) become validated CDT rules. The CDT confirms or challenges them against actual behavioral evidence. The result is a grounded profile, not just a list of rules.

### 3.6 Traversal Optimization

Current traversal is LLM-expensive:
```
For each scene:
  For each gate:
    DeBERTa NLI call (check_scene) → True/False
  Collect matching statements
```

**This is actually fast** — DeBERTa NLI is a local model call (~5ms per gate). The expensive part is during CDT **construction** (hypothesis generation, merge, summarize), not traversal.

**Optimization: Wikified Context as Cache**

Instead of traversing the CDT at inference time (which is already fast), pre-wikify the CDT into a markdown profile and use that as static context. This eliminates traversal entirely for the common case.

```
Strategy:
1. Build CDT → full tree with gates
2. Wikify → markdown profile (human-readable)
3. At inference: inject wikified profile as system context
4. Periodically: re-traverse to check for drift (compare wikified vs gate-traversed output)
```

**When to re-traverse (drift detection):**
- After N new observations are added
- When a user corrects the sidekick's behavior
- On a configurable schedule (e.g., every 50 interactions)

### 3.7 Interactive Colab Notebook Design

**Goal:** Step-by-step CDT build where each step outputs intermediate results for inspection and debugging.

**Notebook Structure:**

```
Cell 1: Setup & Data Loading
  - Load Kasumi data, show train/test split sizes
  - Display 5 sample pairs

Cell 2: Embedding
  - Precompute embeddings (or load from cache)
  - Visualize embedding space (t-SNE/UMAP)
  - Show cluster structure

Cell 3: Clustering
  - Run KMeans, show cluster assignments
  - Display representative samples per cluster
  - Interactive: change n_clusters, see effect

Cell 4: Hypothesis Generation (per cluster)
  - For ONE cluster: show prompt, show LLM response, show parsed hypotheses
  - Side-by-side: compare hypotheses from different clusters
  - Human evaluation: "Does this hypothesis make sense?"

Cell 5: Hypothesis Quality Check
  - Show all hypotheses across clusters
  - Cosine similarity matrix (find duplicates)
  - Statement length distribution
  - Specificity check: "Is this too universal?"

Cell 6: NLI Validation (per hypothesis)
  - For ONE hypothesis: show DeBERTa scores on each pair
  - Visualize True/False/Irrelevant distribution
  - Show which pairs support vs contradict
  - Threshold analysis: what happens at different θ_accept?

Cell 7: Tree Building
  - Build full tree with current config
  - Visualize tree structure (graphviz or text)
  - Compute intrinsic quality metrics
  - Compare: this build vs baseline metrics

Cell 8: Wikification
  - Show markdown profile
  - Human evaluation: "Does this read correctly?"

Cell 9: Benchmark (optional, expensive)
  - Run on first 10 test pairs (quick check)
  - Show per-pair predictions vs ground truth
  - Identify: which pairs got A, B, C and why

Cell 10: Multi-Build Comparison
  - Run N builds, compare quality metrics
  - Show variance across builds
  - Identify which steps cause the most variance
```

**Key Feature:** Each cell caches its output. Re-running cell 4 doesn't redo cells 1-3. The notebook becomes a debugging tool and an experiment workbench.

---

## 4. Implementation Phases

### Phase 1: Experiment Infrastructure (P0)
**Files:** New `src/canopy/experiment.py`, modify `codified_decision_tree.py`, `run_benchmark.py`

1. Implement `ExperimentConfig` dataclass with config hashing
2. Implement `LLMCache` (JSON file per step, keyed by prompt hash)
3. Add `--experiment_id` and `--cache_mode` flags to CLI
4. Add `--build_id` to CDT naming (monotonic)
5. Update CDT naming to include build_id: `Kasumi.build_003.a8f3c2d1.pkl`

### Phase 2: Intrinsic Quality Metrics (P1)
**Files:** New `src/canopy/quality.py`, modify `verify_cdt.py`

1. Implement all 8 quality metrics
2. Implement composite quality score
3. Add `--quality` flag to `verify_cdt.py`
4. Run quality metrics on all 15 existing Kasumi CDTs
5. Correlate quality scores with benchmark scores

### Phase 3: Deterministic Replacements (P0)
**Files:** Modify `src/canopy/prompts.py`, new `src/canopy/dedup.py`

1. Replace `merge_similar_hypotheses` LLM call with embedding-based cosine dedup
2. Replace `summarize_triggers` LLM call with NLI-based ranking (keep top-8 by avg True score)
3. Replace `discover_topics` LLM label call with TF-IDF keyword extraction
4. Test: same embeddings + same hypotheses → identical CDT (deterministic given hypotheses)

### Phase 4: Colab Notebook (P3)
**Files:** New `notebooks/cdt_step_by_step.ipynb`

1. Build notebook with cells 1-10 as designed above
2. Add caching between cells
3. Add visualization helpers (t-SNE, tree graphviz)
4. Test with Kasumi data end-to-end

### Phase 5: Rule Bootstrapping (P4)
**Files:** Modify `src/canopy/core.py`, `codified_decision_tree.py`

1. Add `seed_rules` parameter to `build_character_cdts`
2. Implement rule injection into hypothesis pipeline
3. Test with sample Kasumi rules (manually crafted)
4. Validate that rules pass/fail NLI correctly

### Phase 6: Multi-Build Variance Study
**Files:** New `scripts/variance_study.py`

1. Build N=5 CDTs with fresh LLM calls
2. Compute quality metrics for each
3. Benchmark each against same test set
4. Report: mean, std, min, max for both quality and benchmark scores
5. Identify: which quality metrics best predict benchmark score

---

## 5. Test Plan

### Unit Tests
- `ExperimentConfig` hashing is deterministic
- `LLMCache` hit/miss behavior
- Embedding-based dedup produces correct merges
- NLI-based ranking orders correctly
- Quality metrics compute correctly on known CDT structures
- Rule injection creates proper hypothesis entries

### Integration Tests
- Full CDT build with cache → replay produces identical CDT
- Quality metrics correlate with known-good/known-bad CDTs
- Colab notebook cells run without error

### Variance Study (the real test)
- 5 builds of same config: quality_score std < 0.1
- 5 builds of same config: benchmark_score std reported (establishes baseline)
- Deterministic replacements reduce variance vs current LLM-based steps

---

## 6. RP Dataset Analysis

### Current Data
- **167 test pairs** for Kasumi
- **Train/test split:** 50/50 by pair index (first half train, second half test)
- **Score range (Sonnet eval):** 50.0 — 71.40 across 19 benchmark runs

### Score Distribution (detailed from benchmark analysis)
| Score | Count | % | Interpretation |
|-------|-------|---|---------------|
| A (Entails) | 75 | 44.9% | Prediction matches ground truth |
| B (Neutral) | 69 | 41.3% | Both prediction & GT are valid responses |
| C (Contradicts) | 23 | 13.8% | Prediction contradicts ground truth |

### B-Score Breakdown (69 pairs)
| Category | Count | % | Fixable? |
|----------|-------|---|----------|
| DIFFERENT_FACET | 53 | 77.6% | Yes — CDT emotional rebalancing |
| FORMAT_MISMATCH | 13 | 19.4% | Partially — gen prompt tuning |
| GENUINELY_WRONG | 2 | 3.0% | Yes — CDT gate accuracy |
| EVAL_TOO_STRICT | 0 | 0.0% | N/A — evaluator is correct |

### Why We Can't Beat ~72 (Sonnet Eval)
1. **30% of B is structural ambiguity** — 74.6% of B pairs in LOW predictability scenes, scene transitions 3.8x more frequent in B vs A pairs. No model can predict these.
2. **50% of B is caricature** — CDT has ~15 enthusiasm statements, ~0 restraint statements. Fixable via CDT emotional rebalancing.
3. **20% of B is format mismatch** — same intent, different words. Only eval model change fully fixes this.

### Eval Model Calibration Comparison
| Config | A% | B% | C% | NLI |
|--------|----|----|----|----|
| Sonnet gen + GPT-4.1 eval | 49.7% | 32.9% | 17.4% | 66.17 |
| Sonnet gen + Sonnet eval | 44.9% | 41.3% | 13.8% | 65.57 |
| **Delta** | -4.8pp | **+8.4pp** | -3.6pp | -0.60 |

Sonnet eval is 8.4pp stricter than GPT-4.1 on the B→A boundary. Paper's 84.25 with GPT-4.1 eval benefits from this leniency.

### Improvement Headroom (Sonnet Eval)
| Scenario | NLI | Gain | Path |
|----------|-----|------|------|
| Current best (d4) | 71.40 | — | Already achieved |
| +CDT rebalance (fix 53 different-facet) | ~81.4 | +10.0 | Add quiet/vulnerable/practical to CDT grounding |
| +Format tuning (fix 13 format-mismatch) | ~86.2 | +14.8 | Perfect surface form matching |
| Theoretical hard ceiling | ~93.1 | +21.7 | Requires eval leniency — not recommended |

**Key finding:** Our best (71.40, Sonnet gen+eval) already **exceeds** the paper's architecture-equivalent score (70.66, Llama gen+Sonnet eval). We've matched paper parity.

### Actionable CDT Rebalancing (the +10 pt path)
The CDT grounding has ~88 statements with ~15 mentions of enthusiasm/energy and ~0 mentions of restraint/quietness. To fix:
1. Add hypotheses about contemplative behavior ("Kasumi hesitates when...")
2. Add hypotheses about vulnerability ("Kasumi shows uncertainty when...")
3. Add hypotheses about practical/specific actions (singing croquette prices, direct confrontation)
4. Ensure gate conditions distinguish high-energy vs quiet scenes

This is where **rule bootstrapping** (Section 3.5) directly helps — we can inject known quiet/vulnerable behavior patterns as seed rules.

### Theoretical Ceiling Calculation
```
EASY pairs (~25%):     100% A → 25.0 pts
MEDIUM pairs (~30%):    70% A, 30% B → 18.5 pts
HARD pairs (~30%):      30% A, 50% B, 20% C → 11.0 pts
IMPOSSIBLE (~15%):      10% A, 60% B, 30% C → 4.0 pts
─────────────────────────────────────────
Conservative ceiling:   ~58.5 pts (matches our observed range)
With caricature fix:    ~68.5-72 pts
With perfect gen:       ~81-86 pts (upper bound)
```

**Conclusion:** 70.66 was near the realistic ceiling without CDT rebalancing. The path to 80+ requires fixing the caricature problem, not hyperparameter tuning.

### LifeChoice Integration Plan

**Dataset Stats:**
- 1,533 MCQ decision points from 388 books, 1,462 characters
- 3-9 questions per character (avg ~4)
- 40,000+ chars of narrative context per character
- Paper SOTA: CharMap + GPT-4 = 67.95%, Human = 92.01%

**Cold-Start Analysis:**
| Aspect | Finding | CDT Suitability |
|--------|---------|----------------|
| Character-driven decisions | Stable across text length | Good fit |
| Plot-driven decisions | Require recent context | Harder for CDT |
| Genre: sci-fi/fantasy/romance | Stereotyped characters → strong | Good fit |
| Genre: mystery/crime | Complex logic chains → weak | Poor fit |
| Text length dependency | Need 50-70% of context for good performance | Sufficient in most cases |

**Approach for LifeChoice:**
1. NOT full CDT construction (insufficient observations per character)
2. LLM-generated behavioral profiles from narrative context (CharMap-equivalent)
3. Rule-based CDT-lite: extract personality patterns → validate against narrative examples → inject as grounding
4. **Test with Sonnet on 50+ questions** (current 13Q Haiku was inconclusive)
5. Focus on character-driven decisions in sci-fi/fantasy/romance (highest CDT fit)

---

## 7. Decisions

| # | Decision | Rationale | Alternatives Considered |
|---|----------|-----------|------------------------|
| D1 | Cache at LLM call level, not pipeline level | Enables partial replay and step inspection | Pipeline-level cache (coarser, less debuggable) |
| D2 | Replace merge/summarize LLM calls with deterministic alternatives | Reduces variance from 3 LLM steps to 1 (hypothesis gen only) | Keep LLM calls + cache (still non-deterministic on fresh builds) |
| D3 | Intrinsic quality metrics separate from benchmark | Decouples CDT quality from gen/eval model calibration | Only benchmark score (conflates 3 variables) |
| D4 | N=5 multi-build for variance measurement | Statistically meaningful variance estimate | N=3 (too few), N=10 (too expensive) |
| D5 | Focus on Kasumi only, extend after validation | Kasumi has 167 test pairs and rich experiment history | Multi-character (expensive, premature) |
| D6 | Wikified profile as primary context injection | Eliminates traversal at inference, human-readable | Live traversal (faster iteration but more LLM calls) |
| D7 | Rule bootstrapping via hypothesis pipeline | Rules are validated against evidence, not blindly accepted | Direct rule injection (no validation, could be wrong) |
| D8 | Embedding-based dedup over LLM merge | Deterministic, faster, no API cost | LLM merge (higher quality but non-deterministic) |

---

## 8. Open Questions

1. **Is the NLI-based ranking replacement for summarize_triggers sufficient?** The LLM summarize step also rewrites hypotheses for clarity. NLI ranking only selects, doesn't rewrite. Do we lose quality?

2. **What's the variance contribution of FP16 embedding computation?** Minor, but should we verify with a fixed-seed torch test?

3. **Should we explore deterministic hypothesis generation via constrained decoding?** e.g., provide a fixed template and fill in slots. This would eliminate LLM variance entirely but may reduce hypothesis quality.

4. **Is 5 the right N for multi-build?** Cost vs statistical power tradeoff. Each build is ~13 min + ~30 min benchmark = ~43 min. 5 builds = ~3.5 hours.

5. **Should the Colab also cover LifeChoice?** Adding cold-start profile generation is a separate workflow from CDT construction.

---

## 9. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Deterministic replacements reduce CDT quality | High | Compare quality metrics before/after; keep LLM path as fallback |
| Multi-build takes too long | Medium | Use 0.6B embeddings for quick builds, 8B for final validation |
| Quality metrics don't correlate with benchmark | High | If so, redesign metrics; the correlation validation IS the experiment |
| Rule bootstrapping adds noise | Medium | Rules are validated by NLI; failed rules are rejected, not injected |
| Colab becomes stale as codebase evolves | Low | Colab imports from canopy library; cells are thin wrappers |

---

## 10. Cost Estimate

| Phase | LLM Calls | GPU Time | Wall Clock |
|-------|-----------|----------|-----------|
| Phase 1 (Infrastructure) | 0 | 0 | 2-3 hours (code) |
| Phase 2 (Quality Metrics) | 0 | ~5 min (NLI on 15 CDTs) | 1-2 hours |
| Phase 3 (Deterministic) | 0 (replacing LLM calls) | 0 | 2-3 hours |
| Phase 4 (Colab) | ~20 (one build) | ~15 min | 3-4 hours |
| Phase 5 (Rules) | ~8 (one build) | ~13 min | 2-3 hours |
| Phase 6 (Variance) | ~40 (5 builds) | ~65 min | ~3.5 hours |
| **Total** | **~68** | **~100 min** | **~15-18 hours** |

All within Claude Max subscription limits. No API key costs.

---

## 11. Review Round 1 — Findings & Resolutions

### Review Sources
- **Architecture Review** (Turiya/Architect agent)
- **ML Research Review** (ML methodology agent)

### CRITICAL / HIGH Findings

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| **R1** | **Phase ordering wrong:** Need baseline variance BEFORE deterministic replacements | HIGH | **ACCEPTED.** Reorder: Phase 1 → 2 → 6a (baseline variance) → 3 → 6b (post-replacement variance) → 4 → 5. This is the Session 9 lesson — measure before changing. |
| **R2** | **Losing summarize rewriting will regress trees to 0 gates.** The falsifiability constraints (max 15 words, FALSE in 30%) were the key fix. NLI ranking doesn't rewrite. | HIGH | **ACCEPTED.** Move falsifiability constraints INTO `make_hypothesis_prompt` so raw hypotheses are already constrained. Then NLI-based selection of top-8 is safe. This is the correct architectural fix. |
| **R3** | **Composite quality score weights are arbitrary.** Should be fitted after variance study, not hardcoded. | HIGH | **ACCEPTED.** Phase 2 computes all 8 metrics independently. Composite formula becomes an output of Phase 6 (regression on metrics vs benchmark scores), not an input. |
| **R4** | **Ceiling calculation is circular.** Uses CDT quality to define the CDT's ceiling. | HIGH | **ACCEPTED.** Compute TWO ceilings: (a) CDT-independent (scene structure only: transitions, GT rarity, ambiguity), (b) CDT-dependent (current grounding coverage). The CDT-independent ceiling is the true upper bound. |
| **R5** | **Ceiling percentages are conjecture.** EASY/MEDIUM/HARD/IMPOSSIBLE distributions are not empirically derived. | HIGH | **ACCEPTED.** Phase 2 must include: empirically classify all 167 pairs using heuristics (scene transitions, GT rarity, CDT statement matching) and report actual distribution. No conjectured percentages. |
| **R6** | **Cache replay requires full upstream chain validation.** Missing `provenance_chain` verification. | HIGH | **ACCEPTED.** Add `provenance_chain` to metadata: hash of each upstream cache file. Replay mode verifies chain before proceeding, fails loudly if any link is missing. |
| **R7** | **70.66 "near-optimal" claim is unjustified.** Original is irreproducible; 60.30 is the reproducible score. | HIGH | **ACCEPTED.** Reframe: "Realistic reproducible range is 60-65. 70.66 was likely a lucky draw. The path forward is reliable 65+ with CDT rebalancing, not reproducing 70.66." |
| **R8** | **Thread safety of cache layer unaddressed.** ThreadPoolExecutor + file-based cache = race conditions. | HIGH | **ACCEPTED.** Use atomic writes (temp file + rename) and per-prompt-hash filenames (inherently collision-free for different prompts). NLI model lock already serializes validation; document this. |

### MEDIUM Findings

| # | Finding | Resolution |
|---|---------|------------|
| **R9** | N=5 underpowered for variance estimation | **Increase to N=10.** Cost is ~7 hours instead of 3.5. With N=10, 95% CI for σ narrows to [0.7σ, 1.5σ] vs [0.65σ, 1.8σ] at N=5. Still imperfect but much better. Define decision threshold upfront: "we need to know if σ < 5 pts." |
| **R10** | Quality metric validation is circular (independent of benchmark but validated against benchmark) | **Clarify:** The metrics measure CDT structure. Correlation with benchmark is a validation that the metrics capture something real, not a design criterion. If metrics don't correlate, they're measuring the wrong thing — that's useful information. |
| **R11** | Git commit hash invalidates cache on unrelated changes | Hash only pipeline-relevant files (prompts.py, core.py, validation.py, embeddings.py, cluster.py), not full git commit. |
| **R12** | Embedding model for statement dedup unspecified | Use DeBERTa NLI entailment (already loaded): if A entails B AND B entails A → duplicate. Reuses existing infrastructure, semantically appropriate. |
| **R13** | Gate discrimination metric: train vs test? | Training data only. Never leak test info into quality metrics. |
| **R14** | Missing grounding fidelity metric | **ACCEPTED.** Add metric: for each statement, compute NLI entailment against its source cluster's action texts. Measures whether LLM fabricated the hypothesis or grounded it. |
| **R15** | Rule bootstrapping may reject valid rules (NLI threshold too strict) | Add `min_rule_acceptance_threshold` (lower than `threshold_accept`). Rules check non-contradiction rather than entailment. |
| **R16** | "Pick longer statement" merge heuristic is naive | Pick statement with higher avg NLI True score against training data instead. |
| **R17** | No cache eviction or size management | Document expected cache size per build (~200MB). Add `--purge-cache` flag. |
| **R18** | Ensemble averaging at hypothesis level | **CONSIDERED.** Run hypothesis gen 3x per cluster, majority-vote. This is a good alternative to caching but 3x more expensive. Add as optional `--ensemble N` flag, default 1. |
| **R19** | Multi-build doesn't isolate variance sources | **ACCEPTED.** Add ablation: fresh hypothesis gen only (everything else cached) vs fresh clustering only vs all fresh. Identifies which step causes most variance. |
| **R20** | Colab prioritized over rule bootstrapping despite sidekick being the goal | **ACCEPTED.** Swap Phase 4 and 5: rules before Colab. |
| **R21** | Incremental CDT updates mentioned but not designed | **DEFERRED.** Explicitly out of scope for this PRD. Future PRD. |

### Updated Phase Ordering (Post-Review)

```
Phase 1: Experiment Infrastructure (caching, provenance, naming)
Phase 2: Intrinsic Quality Metrics (8 individual metrics + difficulty classification)
Phase 6a: Baseline Variance Study (N=10 builds with current LLM pipeline)
Phase 3: Deterministic Replacements (with falsifiability in prompt, NLI dedup)
Phase 6b: Post-Replacement Variance Study (N=10 builds, compare to 6a)
Phase 5: Rule Bootstrapping (seed rules, lower acceptance threshold)
Phase 4: Colab Notebook (debugging tool, extensible)
```

### Updated Decision Table

| # | Decision | Change from Round 1 |
|---|----------|-------------------|
| D4 | **N=10** multi-build (was N=5) | Increased for statistical power |
| D8 | **DeBERTa NLI entailment** for dedup (was embedding cosine) | More semantically appropriate, reuses existing model |
| D9 (NEW) | **Falsifiability constraints move to hypothesis prompt** | Prevents regression when removing LLM summarize step |
| D10 (NEW) | **Composite quality score is an output, not an input** | Fitted after variance study via regression |
| D11 (NEW) | **Two ceiling calculations** — CDT-independent and CDT-dependent | Prevents circular ceiling estimation |
| D12 (NEW) | **Variance ablation** — isolate hypothesis gen vs other sources | Identifies which pipeline step to focus on |
