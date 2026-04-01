# PRD v2: Step-Level CDT Quality Engineering

> "70.66 proves it's achievable. Now make every build that good."

**Date:** 2026-04-01
**Status:** DRAFT v2.1 — updated after full ambiguity resolution
**Scope:** Kasumi (Rapori) first — extensible to any character/domain
**Supersedes:** robust-cdt-prd.md (v1)

---

## 1. Vision

70.66 is not a ceiling — it's proof of what the CDT pipeline CAN produce. The problem is that we can't produce it reliably. The solution isn't caching luck — it's understanding each step well enough to engineer consistent quality.

### 1.1 Core Principles

1. **Step-level quality, not end-to-end.** The benchmark score is a lagging indicator. Each pipeline step has its own quality surface that we can measure and optimize independently.

2. **Provenance everywhere.** Every hypothesis, gate, and statement traces back to the raw observations that generated it. This enables debugging, trust, and richer inference.

3. **Both deterministic and non-deterministic methods.** Stochasticity is a feature when it explores the hypothesis space. Determinism is a feature when it provides reliability. Use each where it helps.

4. **Extensible, not hardcoded.** Nothing specific to Kasumi in the architecture. Different characters, domains, and data shapes should work without code changes.

5. **The step-by-step pipeline IS the iteration framework.** Built as both a Python script (for agent experimentation via CLI/tmux) and a Colab notebook (for Bhargav's final review and interactive exploration). The script is the primary tool; the Colab is the human-readable presentation layer.

---

## 2. Pipeline Architecture with Provenance

### 2.1 The Pipeline

```
Step 1: Data Loading
Step 2: Embedding
Step 3: Clustering
Step 4: Hypothesis Generation
Step 5: Dedup/Merge
Step 6: Summarize/Compress
Step 7: NLI Validation
Step 8: Tree Building
Step 9: Wikification + Grounding
```

### 2.2 Provenance Data Model

Every artifact in the pipeline carries provenance — a chain of references back to the raw observations.

```python
@dataclass(frozen=True)
class Provenance:
    """Traces any CDT artifact back to its source observations."""
    source_pair_indices: tuple[int, ...]   # Indices into the training pairs
    cluster_id: int | None = None          # Which cluster produced this
    hypothesis_id: str | None = None       # Unique ID for the hypothesis
    step: str = ""                         # Which pipeline step created this
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TrackedHypothesis:
    """A hypothesis with full provenance."""
    statement: str                         # The behavioral hypothesis text
    gate: str                              # The scene-check question
    provenance: Provenance                 # Where it came from
    quality: HypothesisQuality | None = None  # Step-level quality metrics

@dataclass(frozen=True)
class HypothesisQuality:
    """Per-hypothesis quality metrics computed during validation."""
    nli_true_rate: float       # % of relevant pairs where NLI says True
    nli_false_rate: float      # % where False
    nli_irrelevant_rate: float # % where Irrelevant
    specificity: float         # 1 - true_rate (lower = more specific)
    word_count: int            # Statement length
    grounding_fidelity: float  # NLI entailment against source cluster actions
```

### 2.3 CDTNode with Provenance

```python
class CDTNode:
    statements: list[str]
    gates: list[str]
    children: list[CDTNode]
    depth: int

    # NEW: provenance for every statement and gate
    statement_provenance: list[Provenance]   # 1:1 with statements
    gate_provenance: list[Provenance]        # 1:1 with gates
    hypothesis_quality: list[HypothesisQuality]  # Quality metrics per statement

    def get_evidence(self, statement_idx: int, pairs: list[dict]) -> list[dict]:
        """Retrieve the raw observations that support a statement."""
        prov = self.statement_provenance[statement_idx]
        return [pairs[i] for i in prov.source_pair_indices]

    def traverse_with_evidence(self, scene: str, pairs: list[dict]) -> list[dict]:
        """Traverse and return statements WITH their source observations."""
        results = []
        for i, stmt in enumerate(self.statements):
            evidence = self.get_evidence(i, pairs)
            results.append({
                "statement": stmt,
                "evidence": evidence[:3],  # Top 3 supporting observations
                "quality": self.hypothesis_quality[i] if self.hypothesis_quality else None,
            })
        for gate, child in zip(self.gates, self.children):
            from canopy.validation import check_scene
            if check_scene([scene], [gate])[0]:
                results.extend(child.traverse_with_evidence(scene, pairs))
        return results
```

### 2.4 Why Provenance Matters at Each Stage

| Stage | What provenance enables |
|-------|----------------------|
| **Debugging** | "This statement seems wrong" → look at the 5 observations it came from → "ah, the cluster mixed two different behaviors" |
| **Quality audit** | "Is this hypothesis grounded?" → check NLI entailment against source observations → grounding fidelity score |
| **Inference enrichment** | Gate fires → pull raw observations as additional context alongside the statement → richer grounding |
| **Caricature detection** | "Why all enthusiasm?" → trace statements to source clusters → all clusters have high-energy observations → need to rebalance input data or add contrastive rules |
| **Rule bootstrapping** | Injected rules have `source_type=RULE` provenance → distinguishable from data-derived hypotheses → can audit which rules survived validation |

---

## 3. Step-by-Step Quality Framework

### Step 1: Data Loading

**What:** Load scene-action pairs, split train/test.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Pair count | Data sufficiency | > 100 train |
| Scene length distribution | Context richness | Mean > 200 chars |
| Action length distribution | Response detail | Mean > 30 chars |
| Character mention rate | Character presence in scenes | > 50% |
| Behavioral register distribution | Emotional diversity (enthusiasm, quiet, practical, vulnerable) | No register > 60% |

**Iteration lever:** If behavioral registers are imbalanced, we know the CDT will be biased BEFORE building it. Can augment with rules or flag the imbalance.

**Extensibility:** `DataLoader` protocol — any data source that produces `list[dict]` with `scene` and `action` keys.

---

### Step 2: Embedding

**What:** Surface embedder encodes actions, generative embedder encodes scenes. Concatenated for clustering.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Embedding spread | Distribution coverage (std of L2 norms) | > 0.1 |
| Nearest-neighbor coherence | Semantically similar actions are close | Jaccard > 0.3 |
| Action-scene alignment | Matching pairs closer than random | Mean cosine gap > 0.05 |

**Deterministic:** Same model + same input = same embeddings.

**Non-deterministic option:** Ensemble N embedding models, average vectors. Different models capture different semantic aspects.

**Iteration:** Visualize with t-SNE/UMAP. If clusters don't form visually, the embedding model isn't capturing behavioral structure — swap models before proceeding.

---

### Step 3: Clustering

**What:** Group embeddings into behavioral themes. Each cluster gets representative samples for hypothesis generation.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Silhouette score | Cluster separation quality | > 0.1 |
| Intra-cluster coherence | Semantic similarity within cluster | Cosine > 0.5 |
| Cross-cluster distinctiveness | Clusters are different from each other | Mean cross-cosine < 0.7 |
| Cluster size balance | Std dev of cluster sizes | < 0.5 * mean size |
| Representative fidelity | Are samples close to centroid? | Mean rank < 3 |

**Deterministic:** KMeans with `seed=42`.

**Non-deterministic options:**
- HDBSCAN (auto-discovers K from data density)
- Multi-seed KMeans: run with seeds 1-10, consensus clustering
- Sweep K: run K=4,6,8 and pick best silhouette

**Iteration:** Show clusters with representative samples. "Does cluster 3 make sense as a behavioral theme?" If not, try different K or embeddings.

**Provenance:** Each pair gets a `cluster_id` assignment. The cluster's centroid and representative samples are stored.

---

### Step 4: Hypothesis Generation (THE VARIANCE ENGINE)

**What:** LLM generates `k` hypotheses per cluster — (statement, gate) pairs.

**This is where 70.66 vs 60.30 is decided.** Different hypothesis draws produce different trees with different benchmark scores.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Specificity | % of training actions matching (not universal) | 40-70% |
| Grounding fidelity | NLI entailment against source cluster actions | > 0.5 |
| Diversity | Cosine distance between hypotheses within cluster | > 0.3 |
| Falsifiability | % of scenes where hypothesis is FALSE | 30-70% |
| Word count | Statement brevity | 10-15 words |
| Emotional coverage | Behavioral registers represented | No register > 50% |

**Deterministic options:**
- Constrained templates: `"{character} tends to [VERB] when [CONDITION]"` — reduces variance
- Few-shot examples: 2-3 example hypotheses in prompt anchoring the quality bar
- Falsifiability constraints in the prompt (from the summarize_triggers breakthrough)

**Non-deterministic options (stochasticity as exploration):**
- Run hypothesis gen N times per cluster (N=3), keep UNION
- More hypotheses → more candidates → better selection downstream
- The LLM explores different parts of the behavioral space each run
- **Ensemble hypothesis generation:** the key non-deterministic lever

**Iteration:** For each cluster, show: prompt → response → parsed hypotheses → quality metrics. "Why did cluster 2 produce universal truths?" → look at prompt, look at cluster samples → "the samples are too diverse, need tighter cluster."

**Provenance:** Each `TrackedHypothesis` carries:
- `source_pair_indices`: the representative samples in the cluster
- `cluster_id`: which cluster
- `hypothesis_id`: unique ID for tracking through the pipeline

---

### Step 5: Dedup/Merge

**What:** Remove near-duplicate hypotheses. Currently via LLM call (too aggressive — 69→15 nodes in Experiment I).

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Merge ratio | input / output count | 0.7-0.95 (not 0.2) |
| Semantic diversity preserved | Cosine spread post-merge | > 80% of pre-merge spread |
| NLI score of kept vs discarded | Kept hypothesis has better NLI | > 70% of merges |

**Deterministic options:**
- DeBERTa NLI entailment: if A entails B (unidirectional), keep whichever has higher NLI True score against training data
- Threshold: only merge if entailment confidence > 0.8

**Non-deterministic option:**
- LLM merge with constrained output (lighter touch: "only merge if CLEARLY redundant")
- Multiple merge runs → consensus on which pairs are true duplicates

**Key learning from experiments:** No merge (Experiment J, 60.30) scored higher than aggressive merge (Experiment I, 65.57 with 15 nodes). But J had 29 gates vs I's 7. The merge was too aggressive. **Light-touch dedup is better than aggressive merge.**

**Provenance:** Merged hypotheses inherit provenance from both parents (union of source pair indices).

---

### Step 6: Summarize/Compress

**What:** If > 8 hypotheses remain, select top 8. Currently via LLM rewrite with falsifiability constraints.

**The breakthrough lived here:** falsifiability constraints (max 15 words, FALSE in 30%+, specific behavioral trigger) took trees from 0 gates → 22 gates.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Post-compress count | Number of surviving hypotheses | 6-10 |
| Falsifiability rate | % of scenes each hypothesis is FALSE | 30-70% |
| NLI True score | Hypothesis accuracy on relevant scenes | > 0.6 |
| Coverage | Distinct behavioral registers in top-8 | >= 3 registers |

**Strategy decision: Where do falsifiability constraints live?**

| Option | Pros | Cons |
|--------|------|------|
| **A: In hypothesis gen prompt (Step 4)** | Deterministic compress (just rank). Fewer pipeline steps. | LLM may struggle to generate constrained hypotheses. Lower yield. |
| **B: In summarize step (Step 6, current)** | Proven — this is what worked. LLM rewrites improve clarity. | Non-deterministic. Extra LLM call. |
| **C: Both** | Constraints in gen + light rewrite in summarize for polish. | Most expensive. But highest quality. |

**My recommendation: Option C for quality, Option A for determinism.** The Colab should test all three and let us compare. Option C uses stochasticity where it helps (rewriting for clarity) while ensuring the input is already constrained.

**Deterministic compress (if constraints in Step 4):**
- Rank by: NLI True score * (1 - universality) * grounding_fidelity
- Keep top-8
- Fully deterministic given the hypotheses and validation results

**Provenance:** Surviving hypotheses retain their full provenance chain.

---

### Step 7: NLI Validation

**What:** DeBERTa checks each hypothesis against ALL training pairs. True/False/Irrelevant. Thresholds determine: global statement vs gated statement vs rejected.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Acceptance rate | % hypotheses that become root statements | 20-40% |
| Gating rate | % that become conditional gates | 30-50% |
| Rejection rate | % discarded | 20-40% |
| Gate selectivity | % of pairs each gate filters | 30-70% |
| Human agreement (sample) | DeBERTa agrees with human on 20 verdicts | > 80% |

**Deterministic:** DeBERTa inference is deterministic.

**Non-deterministic option:** LLM-as-judge for borderline cases (DeBERTa confidence ∈ [0.45, 0.55]). Note: paper shows DeBERTa (CDT-Lite) outscores LLM validation (CDT), so DeBERTa is the better default.

**Iteration lever:** Threshold sweep — show tree structure at θ_accept = 0.65, 0.70, 0.75, 0.80, 0.85. Identify the sweet spot for this character's data.

**Provenance:** Each validation result stores:
- Per-pair verdicts (True/False/Irrelevant for each training pair)
- Filtered pair indices for gated hypotheses
- Correctness score and broadness score

---

### Step 8: Tree Building

**What:** Programmatic assembly of validated hypotheses into tree structure.

**Deterministic:** 100% deterministic given validation results.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Total nodes | Tree size | 20-40 |
| Total statements | Grounding richness | 60-120 |
| Total gates | Conditional granularity | 15-30 |
| Max depth | Discriminative capacity | 3-4 |
| Gate discrimination rate | Gates distinguish different scenes | 30-70% True rate |
| Topic balance | Std dev of nodes per topic | < 3.0 |
| Statement coverage | % of test actions matching >= 1 statement | > 60% |
| Grounding fidelity (aggregate) | Mean fidelity across all statements | > 0.5 |
| Redundancy | Max pairwise NLI entailment between statements | < 0.85 |

**Provenance:** Every statement and gate in the tree carries its full `Provenance` + `HypothesisQuality`.

---

### Step 9: Wikification + Grounding

**What:** Convert tree to markdown. At inference time, traverse tree for scene-specific grounding.

**Quality metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Profile readability | Human can understand the character from reading it | Subjective |
| Grounding length | Characters of grounding text per scene | 500-2000 |
| Evidence density | # of raw observations available per statement | >= 3 |

**NEW: Evidence-enriched grounding**

Current traversal returns bare statements. With provenance, traversal can return statements + supporting evidence:

```python
# Current (bare statements)
grounding = "Kasumi tends to rally the group when morale is low"

# NEW (with evidence)
grounding = """
Kasumi tends to rally the group when morale is low
  Evidence:
  - Scene: "Everyone looks dejected after the failed performance..."
    Action: "Kasumi jumps up: 'Come on everyone! We'll nail it next time!'"
  - Scene: "Arisa says she wants to quit the band..."
    Action: "Kasumi grabs her hand: 'No way! We need you, Arisa!'"
"""
```

This is **more grounding per statement** without more statements. The evidence is concrete, specific, and directly demonstrates the pattern. For the sidekick use case, this means the LLM sees not just "Bhargav prefers immutable patterns" but also the actual code examples where he chose immutability.

---

## 4. Experimentation Framework (Phase 1)

### 4.1 Two Artifacts, One Pipeline

The step-by-step pipeline is built as **two artifacts** from the same underlying code:

| Artifact | Primary user | Format | How it's used |
|----------|-------------|--------|---------------|
| **`scripts/cdt_steps.py`** | Turiya + agents (cody/ralph) | Python CLI script | Automated experimentation via tmux. Run individual steps, inspect outputs, iterate. Agents can run `uv run python scripts/cdt_steps.py --step clustering --character Kasumi` and read JSON output. |
| **`notebooks/cdt_step_by_step.ipynb`** | Bhargav (final review) | Jupyter/Colab notebook | Interactive exploration after experimentation is done. Visualizations, provenance explorer, threshold sliders. The human-readable presentation layer. |

Both import from the same `canopy` library. The script is the workhorse; the notebook is the review tool.

### 4.2 CLI Script Design (`scripts/cdt_steps.py`)

```bash
# Run a single step (reads upstream cache, writes output + quality metrics)
uv run python scripts/cdt_steps.py --step embedding --character Kasumi

# Run all steps end-to-end
uv run python scripts/cdt_steps.py --step all --character Kasumi

# Run hypothesis gen with ensemble
uv run python scripts/cdt_steps.py --step hypothesis_gen --character Kasumi --ensemble 3

# Run with self-correction enabled
uv run python scripts/cdt_steps.py --step all --character Kasumi --self-correct

# Inspect quality metrics for a specific step
uv run python scripts/cdt_steps.py --step clustering --character Kasumi --quality-only

# Compare two builds
uv run python scripts/cdt_steps.py --compare build_001 build_002
```

**Each step outputs:**
1. The step's artifacts (embeddings, clusters, hypotheses, etc.) → `cache/{character}/{build_id}/{step}/`
2. Quality metrics as JSON → `cache/{character}/{build_id}/{step}/quality.json`
3. Provenance records → `cache/{character}/{build_id}/{step}/provenance.json`
4. Human-readable summary → stdout (for agent monitoring via tmux)

**Agent workflow:**
```
cody: uv run python scripts/cdt_steps.py --step clustering --character Kasumi
cody: [reads quality.json] silhouette=0.08, register_diversity=2 → below threshold
cody: uv run python scripts/cdt_steps.py --step clustering --character Kasumi --k 12
cody: [reads quality.json] silhouette=0.15, register_diversity=4 → passes
cody: uv run python scripts/cdt_steps.py --step hypothesis_gen --character Kasumi
```

### 4.3 Steps Mapping

| Step name | Pipeline step | Key outputs | Quality metrics JSON |
|-----------|--------------|-------------|---------------------|
| `data` | Step 1 | Train/test pairs, register analysis | pair_count, register_distribution |
| `embedding` | Step 2 | Embedding arrays (.npz) | spread, coherence, alignment |
| `clustering` | Step 3 | Cluster assignments, representatives | silhouette, coherence, distinctiveness, balance |
| `hypothesis_gen` | Step 4 | TrackedHypothesis list (with provenance) | specificity, grounding_fidelity, diversity, falsifiability, coverage |
| `dedup` | Step 5 | Deduplicated hypothesis list | merge_ratio, diversity_preserved, nli_quality |
| `compress` | Step 6 | Top-N hypotheses per topic | count, falsifiability, nli_score, register_coverage |
| `validate` | Step 7 | Validation results per hypothesis | acceptance_rate, gating_rate, gate_selectivity |
| `build_tree` | Step 8 | CDTNode tree (with provenance) | nodes, statements, gates, depth, coverage, fidelity, redundancy |
| `wikify` | Step 9 | Markdown profile + evidence-enriched grounding | readability, grounding_length, evidence_density |
| `benchmark` | (optional) | Per-pair A/B/C results | nli_score, a_rate, b_rate, c_rate |

### 4.4 Colab Notebook Design (for Bhargav's review)

The notebook mirrors the CLI steps but adds visualizations and interactivity:

| Cell | Step | Visualizations | Interactive Controls |
|------|------|---------------|---------------------|
| 1 | Setup & Config | — | Character selector, config sliders |
| 2 | Data Loading | Behavioral register histogram | Filter by register, view samples |
| 3 | Embedding | t-SNE/UMAP scatter plot | Model selector, dimension toggle |
| 4 | Clustering | Colored scatter by cluster + quality gauges | K slider, method toggle |
| 5 | Cluster Inspection | Per-cluster sample cards | Browse clusters |
| 6 | Hypothesis Gen (single) | Prompt → response → quality table | Cluster selector, regenerate |
| 7 | Hypothesis Gen (all) | Cross-cluster diversity heatmap | Ensemble N toggle |
| 8 | Dedup | Before/after diff, merge reasoning | Threshold slider |
| 9 | Compress | Top-8 with falsifiability bars | Option A/B/C comparison |
| 10 | NLI Validation | Per-hypothesis verdict chart, threshold sensitivity curve | θ sweep slider |
| 11 | Tree + Quality | Tree visualization (graphviz), provenance explorer | Click statement → see evidence |
| 12 | Benchmark | Per-pair A/B/C table with prediction vs ground truth | Pair count slider |

### 4.5 Caching Strategy

Both script and notebook share the same cache at `cache/{character}/{build_id}/{step}/`.

- **Cache invalidation:** Re-running step N deletes cache for steps N+1 through wikify. Implemented via dependency graph.
- **Build isolation:** Each `build_id` (monotonic: build_001, build_002, ...) gets its own cache directory. Old builds are never overwritten.
- **Atomic writes:** Temp file + rename to prevent corruption from concurrent agents.

---

## 5. Automated Quality Self-Correction

The pipeline detects its own biases and corrects them — no manual intervention needed.

### 5.1 The Problem: Caricature Bias

The CDT produces ~15 enthusiasm statements and ~0 restraint/vulnerability. This isn't a CDT design flaw — it's a pipeline quality problem where majority behaviors dominate at every step:

```
Clustering: enthusiasm scenes dominate → most clusters are enthusiasm-flavored
Hypothesis gen: sees enthusiasm clusters → generates enthusiasm hypotheses
Validation: confirms them (they ARE true for most pairs) → root statements
Result: CDT only knows one behavioral register
```

### 5.2 Self-Correction Loop

After each step, quality metrics detect imbalance. If thresholds are violated, the pipeline automatically corrects before proceeding.

```
┌─────────────────────────────────────────────┐
│  For each step:                             │
│    1. Run step                              │
│    2. Compute step-level quality metrics    │
│    3. Check: any metric below threshold?    │
│       YES → apply correction → re-run step  │
│       NO  → proceed to next step            │
└─────────────────────────────────────────────┘
```

### 5.3 Per-Step Corrections

**Step 3 (Clustering) — Register Diversity Check:**
- After clustering, classify each cluster's behavioral register (enthusiasm, quiet, practical, vulnerable, etc.) via a lightweight LLM call on the representative samples
- **Trigger:** If any single register > 60% of clusters, or any expected register has 0 clusters
- **Correction:** Re-cluster with higher K (split dominant register) or use stratified clustering (ensure minority behaviors get their own cluster)

**Step 4 (Hypothesis Gen) — Emotional Coverage Check:**
- After generating all hypotheses, classify each hypothesis's behavioral register
- **Trigger:** If any register > 50% of hypotheses, or coverage < 3 distinct registers
- **Correction:** Run a targeted contrastive generation pass: "Generate hypotheses specifically about {character}'s QUIETER/less frequent behaviors — hesitation, vulnerability, uncertainty, practical actions. The following behavioral registers are NOT yet represented: [list]"
- The contrastive hypotheses join the pool alongside the standard ones — they compete on quality in validation, not inserted blindly

**Step 7 (Validation) — Coverage Gap Check:**
- After validation, check: what % of training actions are covered by at least one accepted statement?
- **Trigger:** If statement coverage < 60%, or specific B-score behavioral registers have 0 matching statements
- **Correction:** Lower θ_accept for underrepresented registers (a hypothesis covering quiet behavior that passes at 0.65 is more valuable than a 5th enthusiasm hypothesis at 0.80). Or re-run hypothesis gen with targeted prompts for uncovered behaviors.

### 5.4 Important Caveat: Contrastive Generation Previously Failed

Experiments G and G2 (Session 9) added contrastive instructions to ALL hypothesis generation and scored 63-64, well below the 70.66 baseline. The experiment log's analysis: "atypical behaviors dilute the CDT's core signal."

**Why this self-correction approach is different:**
- G/G2 added contrastive instructions to ALL clusters. This approach targets ONLY underrepresented registers.
- G/G2 added contrastive hypotheses to the pool indiscriminately. This approach makes them compete on NLI quality — weak contrastive hypotheses are rejected.
- G/G2 used broad instructions ("include atypical behavior"). This approach uses targeted prompts ("generate hypotheses specifically about vulnerability in [specific scene types]").

**However, this is an unvalidated hypothesis.** Phase 2d tests targeted contrastive generation before the self-correction loop is implemented. If it fails again — if ANY contrastive approach dilutes CDT quality — the self-correction loop is redesigned or removed entirely. We don't repeat Session 9's mistake of building on untested assumptions.

### 5.5 Subagent Review Gate

After self-correction, a review subagent evaluates:
- "Does this CDT capture the full behavioral range of the character?"
- "Are there obvious behavioral modes missing?"
- "Is the emotional coverage balanced relative to the training data distribution?"

The subagent sees: the wikified profile, the quality metrics dashboard, and 5 sample observations from each behavioral register. It flags gaps that automated metrics might miss.

This is NOT a human-in-the-loop — it's an automated quality gate. If the subagent finds gaps after self-correction has run, it reports them as warnings, not blocks.

### 5.6 Provenance for Self-Correction

When the pipeline self-corrects:
- Contrastive hypotheses carry `source_type=CONTRASTIVE` in provenance
- Re-clustered pairs carry `correction_round=N` metadata
- The Colab shows which hypotheses came from standard generation vs contrastive pass
- At inference, contrastive-sourced statements are indistinguishable from standard ones — they've passed the same validation

---

## 6. Implementation Phases (Revised)

### Phase 1: Step Pipeline + Provenance Foundation
**Goal:** Build the experimentation framework. Add provenance to CDTNode. Enable step-by-step iteration.
**Deliverables:**
- `Provenance`, `TrackedHypothesis`, `HypothesisQuality` dataclasses in `src/canopy/`
- `CDTNode` extended with `statement_provenance`, `gate_provenance`, `hypothesis_quality`
- `traverse_with_evidence()` method on CDTNode
- `scripts/cdt_steps.py` — CLI for running individual pipeline steps with quality metrics
- Step-level quality metrics computed and output as JSON at each stage
- Caching infrastructure: `cache/{character}/{build_id}/{step}/`
- `notebooks/cdt_step_by_step.ipynb` — Colab notebook for Bhargav's review (generated from same pipeline code)
**Test:** Run full Kasumi pipeline via `scripts/cdt_steps.py --step all`, verify provenance chain, inspect quality metrics at each step

### Phase 2: Baseline Measurement + Step-Level Validation
**Goal:** Establish the ground truth before changing anything. Validate step-level metrics. Test key hypotheses.
**Deliverables:**
- **2a: Baseline variance (FIRST).** 3 builds with current pipeline (no changes). Measure: benchmark score std, step-level quality metrics for each. This is the number to beat.
- **2b: Empirical difficulty classification** of all 167 test pairs (CDT-independent and CDT-dependent ceilings)
- **2c: Step metric correlation.** Across the 3 baseline builds + 15 existing Kasumi CDTs, correlate step-level metrics with benchmark scores. Which metrics actually matter?
- **2d: Key hypothesis tests** (each is a small experiment, not a full pipeline change):
  - Falsifiability constraints in gen prompt (Option A) vs current (Option B) vs both (Option C): build 1 CDT each, compare tree structure metrics
  - DeBERTa NLI dedup vs current LLM merge vs no merge: build 1 CDT each, compare merge ratio + tree metrics
  - Ensemble N=3 vs N=1: build 1 CDT each, compare quality metrics + benchmark score
  - Targeted contrastive gen (for underrepresented registers only) vs standard gen: build 1 CDT each, check register coverage
  - Evidence-enriched traversal A/B test: same CDT, benchmark with bare statements vs evidence-enriched (top-3)
- **2e: Threshold sweep** for NLI validation (θ_accept = 0.65, 0.70, 0.75, 0.80, 0.85) on 1 build
**Test:** Baseline variance measured. Step metrics correlated with benchmark. Each hypothesis test has a clear win/lose/inconclusive result. Only validated improvements proceed to Phase 3.
**Critical gate:** If baseline std < 2 pts, ensemble is unnecessary. If contrastive gen fails again (as in Experiments G/G2), self-correction loop is redesigned or removed.

### Phase 3: Validated Improvements
**Goal:** Apply ONLY the improvements that Phase 2 validated. Build the self-correction loop if contrastive gen passed. Build ensemble if N=3 beat N=1.
**Deliverables:**
- Implement whichever Phase 2d experiments won:
  - If Option C won → falsifiability in gen prompt + LLM polish
  - If NLI dedup won → replace LLM merge
  - If ensemble won → `--ensemble N` flag
  - If targeted contrastive won → self-correction loop (Section 5)
  - If evidence-enriched won → `traverse_with_evidence` as default for sidekick
- 3 builds with improved pipeline. Compare variance to Phase 2a baseline.
**Test:** Improved pipeline variance < baseline variance. Mean quality metrics >= baseline. No regressions on any validated step.

### Phase 4: Rule Bootstrapping (for sidekick use case)
**Goal:** Enable external rules to seed the hypothesis pipeline — validated, not blindly injected.
**Deliverables:**
- `seed_rules` parameter on `build_character_cdts`
- Rule injection as `TrackedHypothesis` with `source_type=RULE` provenance
- Lower acceptance threshold for rules (`min_rule_acceptance_threshold`)
- Rules compete with data-derived hypotheses on quality, not priority
**Test:** Injected rules pass/fail NLI correctly; surviving rules appear in tree with RULE provenance
**Note:** Caricature fix is handled by the automated self-correction loop (Section 5), not manual rules

### Phase 5: Experiment Infrastructure
**Goal:** Make experiments reproducible and trackable.
**Deliverables:**
- Experiment ID (config hash + pipeline-relevant file hashes + data hash)
- LLM response cache (per-step, keyed by prompt hash)
- Provenance chain verification (replay mode)
- CDT naming with build_id
- Cache invalidation rules
**Test:** Replay mode produces identical CDT; provenance chain verification catches stale caches

### Phase 6: Variance Study + Quality Correlation
**Goal:** Establish the real score distribution and validate quality metrics.
**Deliverables:**
- N=10 builds with current pipeline
- N=10 builds with ensemble (Phase 3)
- Step-level quality metrics for all 20 builds
- Regression: which metrics predict benchmark score?
- Variance ablation: isolate hypothesis gen vs other sources
**Test:** Report mean ± std for both pipeline variants; identify dominant variance source

---

## 7. Decisions (v2)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Colab is Phase 1, not Phase 4** | It's the iteration framework, not a debugging afterthought |
| D2 | **Provenance on every CDT artifact** | Enables debugging, trust, evidence-enriched grounding |
| D3 | **Both deterministic and non-deterministic options per step** | Not either/or — use each where it helps quality |
| D4 | **Ensemble hypothesis gen (N×k) as the quality lever** | Stochasticity explored the hypothesis space; selection makes it reliable |
| D5 | **Step-level quality metrics, not just end-to-end benchmark** | Each step has its own quality surface to optimize |
| D6 | **Evidence-enriched traversal** | At inference, pull raw observations alongside statements |
| D7 | **Falsifiability constraints in BOTH gen prompt AND summarize** | Gen for baseline quality, summarize for polish (Option C) |
| D8 | **Light-touch dedup via NLI, not aggressive LLM merge** | Experiment I showed aggressive merge halves the tree |
| D9 | **Threshold sweep, not fixed thresholds** | θ_accept may differ by character; the Colab finds the sweet spot |
| D10 | **Extensible protocols, not Kasumi-specific code** | DataLoader, EmbedderStrategy, ClusterStrategy protocols |
| D11 | **`cdt_steps.py` is the primary build tool; `codified_decision_tree.py` wraps it** | The existing CLI becomes a thin wrapper calling the step pipeline. No duplicate logic. |
| D12 | **Self-correction max 2 rounds** | If still imbalanced after 2 contrastive passes, the training data genuinely lacks that register. Flag as warning, don't loop forever. |
| D13 | **Evidence ranked by NLI entailment score** | `traverse_with_evidence` returns observations ranked by how strongly they support the statement (NLI True confidence). Most relevant evidence first. |
| D14 | **Ensemble dedup: keep one copy of convergent hypotheses** | If two independent runs produce the same hypothesis, that's convergence evidence. Store once with `convergence_count` in provenance metadata. |
| D15 | **Provenance for gated subtrees: filtered pairs only** | A gated hypothesis was validated against the filtered subset. Its provenance should reflect what it was actually tested on. |
| D16 | **Per-character threshold tuning via Colab sweep** | θ_accept may differ by character. The step pipeline recommends a threshold; the Colab visualizes the tradeoff. |

---

## 8. Resolved Ambiguities (from Discussion)

All 9 ambiguities raised during discussion have been resolved:

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | **70.66 baseline** | Achievable, not a lucky draw. Engineer the pipeline to produce it reliably. | 70.66 proves what CDT CAN produce. The fix is improving each step, not caching luck. |
| 2 | **Caricature fix** | Automated self-correction loop (Section 5). Pipeline detects register imbalance, generates contrastive hypotheses. No manual rules. | If the pipeline produces caricature, the pipeline is broken. Fix the pipeline, don't patch the output. |
| 3 | **Falsifiability constraints** | Option C — constraints in gen prompt + LLM polish in summarize. Colab tests all three options. | Deterministic floor (gen prompt) + non-deterministic polish (rewrite). Best of both. |
| 4 | **Variance study N** | Start N=5, escalate to N=10 only if ambiguous (std ≈ 4-6 pts). | Binary decision: "is ensemble needed?" N=5 sufficient for that. Save time unless result is ambiguous. |
| 5 | **Phase ordering** | Colab is Phase 1 — it's the iteration framework, not a debugging afterthought. | The Colab informs every other phase. You need to see each step before engineering infrastructure around it. |
| 6 | **Dedup method** | DeBERTa NLI unidirectional entailment. Keep whichever hypothesis has higher NLI True score against training data. | Reuses existing model. Semantically appropriate. Light-touch (not the aggressive LLM merge that halved the tree). |
| 7 | **Ceiling calculation** | Empirical pair classification in Phase 2. Two ceilings: CDT-independent (dataset upper bound) and CDT-dependent (current CDT's potential). | Not a limit to accept — a roadmap showing where remaining points are hiding. |
| 8 | **Composite quality score** | Compute 9 metrics independently. Fit composite via regression in Phase 6 after correlating with benchmark scores. | Weights are empirically derived, not guessed. If metrics don't correlate, that's useful information too. |
| 9 | **LifeChoice** | Defer until Kasumi is solid. Then test Option C: rule bootstrapping for cold-start characters. | Different problem (profile generation vs CDT construction). Don't split focus. |

---

## 9. Training Data Storage for Evidence-Enriched Traversal

### The Problem

`traverse_with_evidence()` needs access to the original training pairs to retrieve raw observations. Currently CDT pickles only store the tree structure — no training data.

### Design

The CDT package (`.pkl`) stores:
1. `topic2cdt` — attribute CDT trees (with provenance on every node)
2. `rel_topic2cdt` — relationship CDT trees (with provenance)
3. `metadata` — build config, timestamps, stats
4. **NEW: `training_pairs`** — the original scene-action pairs used for construction

**Size impact:** 167 training pairs × ~500 chars each ≈ ~80KB. Negligible compared to the tree structure. The provenance indices (tuples of ints) add even less.

**Alternative (reference-based):** Store only a hash + path to the training data file, load on demand. Smaller pickle but requires the data file to be available at inference time.

**Recommendation:** Store training pairs in the pickle for self-contained packages. The sidekick use case needs the CDT + evidence to be a single portable artifact.

### Inference Modes

| Mode | What's returned | Context size | Use case |
|------|----------------|-------------|----------|
| `traverse(scene)` | Bare statements only | ~500-1500 chars | RP benchmark (current) |
| `traverse_with_evidence(scene, top_k=3)` | Statements + top-3 supporting observations | ~2000-5000 chars | Sidekick grounding |
| `traverse_with_evidence(scene, top_k=0)` | Statements + ALL observations | ~5000-15000 chars | Deep debugging, full context |

The `top_k` parameter controls evidence depth. Default 3 balances richness with context length.

---

## 10. Colab Environment Considerations

### GPU Access

The Colab needs GPU for:
- Step 2 (Embedding): Qwen models, 0.6B or 8B
- Step 7 (NLI Validation): DeBERTa (~715MB)

**For Google Colab (cloud):** Use Colab's free/Pro GPU runtime. Install canopy as editable package (`pip install -e .`). Models download from HuggingFace on first run.

**For local Jupyter (RTX 5090):** Use local models at `~/models/`. No download needed. Faster, more VRAM.

**The notebook auto-detects:** Check for local model paths first, fall back to HuggingFace download. A `config` cell at the top sets model paths and device.

### Caching Between Cells

Each cell writes to `cache/{character}/{step_name}/`:
- `.pkl` for Python objects (embeddings, clusters, hypotheses)
- `.json` for human-readable intermediates (quality metrics, provenance)
- `.npz` for numpy arrays (embeddings)

Cache invalidation: re-running cell N deletes cache for cells N+1 through 12. Implemented via a simple dependency graph in the config cell.

---

## 11. Interaction Between Ensemble and Self-Correction

### The Question

If we do ensemble hypothesis generation (N=3 runs per cluster) AND contrastive generation (self-correction for missing registers), we could produce a LOT of hypotheses: 8 clusters × 3 hypotheses × 3 runs + contrastive = ~80+ candidates.

### The Design

```
Standard generation:     8 clusters × k=3 × N=3 runs = 72 hypotheses
Self-correction check:   Are all registers covered?
  YES → proceed to dedup with 72 candidates
  NO  → contrastive pass adds ~8-16 targeted hypotheses → ~80-88 total
Dedup (DeBERTa NLI):    Remove near-duplicates → ~40-60 survivors
Summarize (NLI rank):   Keep top-8 per topic → final tree
```

The funnel is: generate broadly (explore) → deduplicate (remove redundancy) → rank (select best) → validate (confirm quality). More input candidates means better selection at each stage.

**Cost:** N=3 ensemble triples hypothesis gen LLM calls (~24 → ~72). At ~3s each with Haiku, that's ~3.5 minutes extra. Acceptable.

---

## 12. Open Questions (v2.1)

All major ambiguities have been resolved (Section 7, Decisions D1-D16). Remaining questions are implementation-time decisions:

1. **Should training pairs in the pickle be compressed?** gzip could reduce size 5-10x but adds load time. Decide during Phase 1 implementation based on actual sizes.

2. **Behavioral register taxonomy:** What registers should the self-correction loop check for? Current list: enthusiasm, quiet/contemplative, vulnerability, practical, confrontational. Is this complete? Should it be character-specific or universal?

3. **Quality metric target values:** The targets in Section 3 (e.g., silhouette > 0.1, specificity 40-70%) are initial estimates. Should be calibrated against the N=5 variance study builds in Phase 6.

---

## 13. Review Round 2 Findings & Resolutions (v2.1)

### Architecture Review: 0 CRITICAL, 0 HIGH, 3 MEDIUM, 5 LOW

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| A1 | Notebook drift risk | LOW | **D17 added:** Notebook = zero pipeline logic, only imports + visualization |
| A2 | Provenance breaks pickle compat | MEDIUM | **Backward-compat:** Old pickles load with empty provenance (`getattr` fallback). Gate provenance behind `track_provenance=True` in CDTConfig. Default path stays lean. |
| A3 | Self-correction cascade unbounded across steps | MEDIUM | **D18 added:** Max 2 rounds per step, max 4 total per pipeline run. Budget exhausted → proceed with warnings. |
| A4 | Training pairs in pickle | LOW | No action — well-reasoned, self-contained artifact. |
| A5 | Step pipeline vs monolithic migration | MEDIUM | **D19 added:** `cdt_steps.py` orchestrates existing `canopy` functions with caching/metrics between calls. `build_character_cdts()` remains for simple use cases. Coexistence, not wrapping. |
| A6 | LLM call budget | LOW | Added: baseline ~24 calls, ensemble ~72, ensemble+correction ~110. At 3s/call: 1.2min → 5.5min. |
| A7 | `grounding_fidelity` undefined | LOW | **Defined:** fraction of source cluster actions where DeBERTa NLI(action, statement) = True with confidence > 0.5. |
| A8 | Duplicate section numbers | LOW | Fixed. |

### ML Methodology Review: 0 CRITICAL, 2 HIGH, 2 MEDIUM-HIGH, 1 MEDIUM

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| ML1 | Step-level quality metrics untested against benchmark | MEDIUM | **Phase 2c** already includes correlation study. Targets in Section 3 are explicitly stated as calibration bounds pending empirical validation. |
| ML2 | **Self-correction loop based on disproven hypothesis (G/G2 failed)** | MEDIUM-HIGH | **ACCEPTED.** Self-correction is NOT assumed to work. Phase 2d includes targeted contrastive gen as an experiment. If it fails again → loop is redesigned or removed. Section 5.4 caveat already added. The key difference from G/G2: targeted vs blanket contrastive, competitive NLI validation vs blind insertion. But this must be proven, not assumed. |
| ML3 | **Ensemble hypothesis gen (N×k) is unvalidated** | HIGH | **ACCEPTED.** Moved to Phase 2d as a validation experiment (N=1 vs N=3, 1 build each). Phase 3 only implements ensemble IF Phase 2d shows it beats N=1 on quality metrics or reduces variance. Added decision criterion: if N=3 best < N=1 best, abandon ensemble. |
| ML4 | **Evidence-enriched traversal may amplify caricature for RP** | MEDIUM-HIGH | **ACCEPTED.** Phase 2d includes A/B test: same CDT, benchmark with bare vs evidence-enriched. For RP, keep bare as default. For sidekick, enriched is default. Don't claim improvement without measurement. |
| ML5 | **Variance dominates; Phase 6 is a gating criterion, not refinement** | HIGH | **ACCEPTED.** Phase 2a front-loads 3-build baseline variance measurement BEFORE any changes. Phase 6 (post-improvement) must show reduced variance or the strategy has failed. Decision criterion: if improved pipeline std ≥ baseline std, the improvements didn't help reliability. |

### Updated Decisions (D17-D21)

| # | Decision | Rationale |
|---|----------|-----------|
| D17 | Notebook = zero pipeline logic | Prevents drift; all computation in canopy library |
| D18 | Self-correction budget: max 2/step, max 4/pipeline | Bounds worst-case LLM cost and prevents cascading corrections |
| D19 | Step pipeline coexists with `build_character_cdts()` | Step pipeline orchestrates existing functions; monolithic path remains for simple use |
| D20 | Ensemble and self-correction are Phase 2d experiments, not Phase 1 assumptions | Both are unvalidated. Build them only if Phase 2d proves they help. |
| D21 | Phase 2a baseline variance is a gating criterion | If we can't measure the problem, we can't claim to have fixed it |

---

## 14. Appendix: Review Round 1 Findings (from v1)

All 21 findings from v1 reviews are incorporated into this restructured design:

- **R1 (phase ordering):** Resolved — Colab first, then iterate, then infra
- **R2 (falsifiability):** Resolved — Option C (both gen + summarize)
- **R3 (arbitrary weights):** Resolved — composite is output of Phase 6 regression
- **R4-R5 (circular ceiling):** Resolved — empirical classification in Phase 2
- **R6 (provenance chain):** Resolved — provenance is now a core data model
- **R7 (70.66 reframing):** Resolved — "achievable, now make it reliable"
- **R8 (thread safety):** Carried forward — atomic writes for cache
- **R9 (N=10):** Updated to N=5 with escalation
- **R10-R21:** All incorporated into step-level quality framework

See `robust-cdt-prd.md` for full Round 1 review details.

---

## 15. Appendix: LifeChoice Deferred Plan

**Status:** Deferred until Kasumi robustness is established.

**Approach when ready:** Option C — Rule bootstrapping for cold-start characters.
1. Extract personality rules from book narrative context via LLM
2. Inject as `TrackedHypothesis` with `source_type=RULE` provenance
3. Validate rules against the character's MCQ questions (3-9 per character)
4. Use validated rules as CDT-lite grounding for MCQ answering

**Minimum evidence threshold:** A rule is considered validated if NLI True rate >= 50% AND validated against >= 5 observations. Characters with fewer than 5 questions use unvalidated profiles (with provenance marking them as UNVALIDATED).

**Test plan:** Sonnet on 50+ questions across 10+ characters. Compare: (a) no profile, (b) LLM-generated profile, (c) rule-bootstrapped CDT-lite profile. Report accuracy + per-question analysis.

**Paper baseline:** CharMap + GPT-4 = 67.95%. Human = 92.01%. Our target: match or exceed CharMap with Sonnet.
