# Canopy Design Document

> Authoritative reference for CDT algorithm design in canopy-ai.
> Extracted and generalized from delulu Phase 3 PRD (2026-03-26).
> Last updated: 2026-03-27.

---

## 1. Overview

Canopy is a **domain-agnostic Codified Decision Tree (CDT) library** with temporal dynamics and semantic gates. It takes flat behavioral data from any domain, discovers structure through embedding-based clustering, builds validated decision trees with natural language gate conditions, and produces both structured trees (for programmatic traversal) and human-readable markdown profiles (for context injection).

### Core Innovations Over Standard CDT

- **Temporal CDT (T-CDT):** Time-weighted validation where newer evidence weighs more. Superseded patterns are preserved with history, not deleted.
- **Semantic gate conditions:** Natural language gates with pre-computed embeddings. Traversal uses cosine similarity -- no LLM calls at inference time.
- **Computed confidence:** Programmatic `confidence = supporting / (supporting + contradicting)` from binary LLM verdicts. No LLM-generated confidence floats.
- **Two-layer data approach:** Pre-extracted summaries serve as hypotheses; raw data serves as validation evidence.
- **Domain-agnostic adapters:** Any domain that produces behavioral observations can use the CDT pipeline. A `SceneActionPair` adapter convenience converts (scene, action) pairs to `BehavioralObservation`.

### Research Foundations

| Paper | Key Contribution |
|-------|-----------------|
| **CDT** (arxiv 2601.10080) | Base algorithm: hypothesis extraction, NLI validation, tree construction |
| **PERSONAMEM** (arxiv 2504.14225) | Frontier models get ~50% on dynamic user profiling; explicit temporal state is mandatory |
| **PURE** (arxiv 2502.14541) | Extract-merge-compress pipeline for incremental profile management |
| **Zero-Shot DT** (arxiv 2501.16247) | LLM-based decision tree induction without labeled data |
| **RL-LLM-DT** (arxiv 2412.11417) | Iterative decision tree improvement via reinforcement learning |

---

## 2. Core Algorithm: CDT Construction

The CDT construction pipeline transforms a flat list of behavioral observations from any domain into a validated, hierarchical decision tree.

### Input

A flat list of `BehavioralObservation` items -- single text strings describing observed behavior in context. Canopy does not prescribe what the observation text contains -- domain adapters define the mapping.

```python
@dataclass(frozen=True)
class BehavioralObservation:
    text: str           # Combined scene+action behavioral description
    timestamp: datetime | None = None  # For T-CDT temporal weighting
    source_id: str | None = None       # Provenance reference
    metadata: dict[str, str] = field(default_factory=dict)  # Domain-specific metadata
```

`SceneActionPair` is available as an adapter convenience that converts to `BehavioralObservation`:

```python
@dataclass(frozen=True)
class SceneActionPair:
    scene: str          # Context / situation description
    action: str         # Observed behavior / decision in that context
    timestamp: datetime | None = None
    source_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_observation(self) -> BehavioralObservation:
        return BehavioralObservation(
            text=f"{self.scene} | {self.action}",
            timestamp=self.timestamp,
            source_id=self.source_id,
            metadata=self.metadata,
        )
```

### Step 1: Embedding-Based Clustering

Discover natural domains from the data. **No predefined tags. No predefined vocabulary.** The clustering finds structure organically.

```
Input:  N BehavioralObservation items
Process:
  1. Embed each observation: embed(observation.text)
  2. HDBSCAN clustering on embeddings (min_cluster_size, default 16)
     - HDBSCAN auto-discovers the number of clusters from data density — no k required
     - max_clusters serves as a safety cap: if HDBSCAN produces more, merge smallest clusters
     - Noise points (HDBSCAN label -1): configurable handling (see D11)
  3. Filter: discard clusters with fewer than min_cluster_size items
  4. Label each cluster via LLM: "Given these examples, what domain/theme do they represent?"
Output: K clusters, each with a label and list of member observations
```

The embedding model is configurable (see Section 10). HDBSCAN is the primary clustering algorithm, auto-discovering cluster count from data density. KMeans is available as a fallback for datasets where HDBSCAN produces too many or too few clusters (e.g., uniformly distributed embeddings). The clustering interface allows substitution of any algorithm.

### Step 2: Hypothesis Generation (Unified Pipeline)

Hypotheses come from multiple sources, all producing the same `Hypothesis` dataclass (see D12, D15). The CDT algorithm validates them all identically regardless of origin.

**Tier 1: Cluster hypotheses (CDT core).** For each cluster, the LLM generates candidate behavioral hypotheses -- testable statements about patterns in the data.

```
Input:  Cluster label + member BehavioralObservation items
Process:
  LLM prompt: "Given these behavioral observations from the domain '{label}',
  extract testable behavioral hypotheses. Each hypothesis should be specific
  enough to validate against new evidence."
Output: List of hypotheses per cluster (source_type = CLUSTER)
```

**Tier 2: Session-level hypotheses (see D10).** Pre-extracted summaries (e.g., session card `user_preferences`) serve as additional hypothesis candidates (source_type = SESSION_CARD).

**Additional sources (see D12):** Rules, documents, and manual hypotheses can also feed the pipeline (source_type = RULE, DOCUMENT, MANUAL). All sources merge, deduplicate (embedding cosine similarity > 0.90 = duplicate), then proceed to validation.

Each hypothesis includes:
- `statement`: The testable behavioral claim
- `source_type`: Origin of the hypothesis (CLUSTER, SESSION_CARD, RULE, DOCUMENT, MANUAL)
- `source_ref`: Provenance reference
- `cluster_id`: Which cluster it belongs to (if source_type is CLUSTER)
- `source_ids`: Which input pairs contributed to it
- `gate_condition`: Natural language condition (null if universal) -- see Section 3

### Step 3: Two-Pass Cross-Cluster Validation

Validation follows the CDT paper's two-pass approach to catch cross-cluster contradictions (see D9):

**Pass 1: Global validation.** Each hypothesis is tested against ALL observations (all clusters + noise). If confidence >= `cdt_accept_threshold` → ROOT (universal). If confidence <= `cdt_reject_threshold` → REJECTED. If between → Pass 2.

**Pass 2: Conditional validation.** Observations are filtered by gate semantic similarity to the hypothesis's gate condition. The hypothesis is validated against this filtered evidence set. If confidence >= `cdt_accept_threshold` → GATED rule. Otherwise → REJECTED.

The LLM classifies each evidence item as one of three binary verdicts:
- **supports** -- evidence FOR the hypothesis
- **contradicts** -- evidence AGAINST the hypothesis
- **irrelevant** -- not relevant to this hypothesis

Confidence is then computed **programmatically** (never by the LLM):

```python
def compute_confidence(verdicts: list[Verdict]) -> float:
    """Compute confidence from binary interaction verdicts.

    confidence = supporting / (supporting + contradicting)
    Irrelevant items are excluded from the calculation.
    """
    supporting = sum(1 for v in verdicts if v.verdict == "supports")
    contradicting = sum(1 for v in verdicts if v.verdict == "contradicts")
    if supporting + contradicting == 0:
        return 0.0  # No relevant evidence -> reject
    return supporting / (supporting + contradicting)
```

Validation status assignment based on thresholds:
- `confidence >= cdt_accept_threshold` (default 0.75) --> **accepted** (becomes a root-level or leaf statement)
- `confidence <= cdt_reject_threshold` (default 0.50) --> **rejected** (confidence of exactly 0.50 is rejected, not conditional)
- `confidence` in `(0.50, 0.75)` exclusive --> **conditional** (becomes a gated subtree with a gate condition)

### Step 4: Temporal Weighting (T-CDT)

When timestamps are available, apply time-weighted validation. See Section 5 for full T-CDT details.

- Recent evidence weighted higher via configurable half-life decay
- Contradicting newer evidence marks older patterns as **superseded** (not deleted)
- Superseded patterns preserved with: timestamp, reason, `superseded_by` reference

### Step 5: Tree Building

After validation, the tree is constructed **programmatically** (no LLM call needed):

```
Root (depth 0)
  |-- Cluster: "workflow" (depth 1)
  |     |-- [Accepted] "Always run tests before committing" (depth 2, leaf)
  |     |-- [Gate] "When blast radius is high" (depth 2)
  |     |     |-- "Run 4+ review rounds with specialists" (depth 3, leaf)
  |     |-- [Gate] "When blast radius is low" (depth 2)
  |           |-- "Single review pass is sufficient" (depth 3, leaf)
  |-- Cluster: "debugging" (depth 1)
  |     |-- [Accepted] "Add logging before guessing at fix" (depth 2, leaf)
  |     |-- [Gate] "When dealing with async code" (depth 2)
  |           |-- "Check for missing awaits first" (depth 3, leaf)
  ...
```

Structure (4 levels, depth 0-3):
- **Depth 0:** Root node (global container)
- **Depth 1:** Cluster nodes (one per discovered domain)
- **Depth 2:** Accepted leaf nodes (unconditional within cluster) and gate nodes (conditional)
- **Depth 3:** Gated leaf nodes (conditional hypotheses under gate nodes)
- **Rejected hypotheses** stored but marked rejected (for audit, not traversal)

The tree has 4 levels (depth 0-3) as shown in the example above. Deeper trees add complexity without proven value -- can be added later if needed.

### Step 6: Rules Augmentation (Optional)

External rule files (e.g., coding guidelines, team conventions) can be attached to relevant CDT nodes. An LLM identifies which rules are relevant to which nodes and extracts concise excerpts.

This step is optional -- the CDT is fully functional without it. Rules augmentation adds external context to reinforce or elaborate on learned patterns.

### Step 7: Wikification

Convert the CDT tree into a human-readable markdown profile. See Section 8 for details.

---

## 3. Semantic Gate Conditions

This is a **key design decision**. Gates are NOT JSON structured predicates. Gates are NOT exact field matching. Gates are **natural language conditions** with pre-computed embeddings.

### Why Natural Language Gates

Structured predicates like `{"field": "tags", "op": "contains", "value": "debugging"}` require:
- A predefined schema of fields and values
- Exact matching logic per operator
- Domain-specific field definitions

Natural language gates like `"When debugging unfamiliar async code"` are:
- Domain-agnostic -- no predefined fields needed
- Expressive -- capture nuance that structured predicates cannot
- Efficient at traversal time -- cosine similarity, no LLM calls

### Gate Data Structure

```python
@dataclass(frozen=True)
class GateCondition:
    text: str                      # "When blast radius is high"
    embedding: list[float]         # Pre-computed at CDT build time
    threshold: float = 0.75        # Cosine similarity threshold for activation
    gate_positive_examples: list[str] = field(default_factory=list)  # 5 positive context examples
    gate_negative_examples: list[str] = field(default_factory=list)  # 5 negative context examples
    positive_embeddings: list[list[float]] = field(default_factory=list)  # Pre-computed
    negative_embeddings: list[list[float]] = field(default_factory=list)  # Pre-computed
```

### How Gates Are Created

During CDT construction (Steps 2-3), the LLM generates gate conditions as natural language strings. At build time:

1. Each gate's text is embedded once and stored alongside the gate
2. **Gate calibration:** For each gate, the LLM generates 5 positive example contexts (should activate the gate) and 5 negative example contexts (should not activate the gate). These examples are embedded and stored with the gate.

### How Gates Are Evaluated

At traversal time (Section 4), the current context is embedded once. Each gate is evaluated using a weighted scoring approach:

1. Compute `gate_similarity`: cosine similarity between context embedding and gate embedding
2. Compute `best_positive_similarity`: max cosine similarity between context embedding and all positive example embeddings
3. Compute `score = 0.6 * gate_similarity + 0.4 * best_positive_similarity`
4. Gate activates if `score >= gate.threshold`

For borderline cases (score between 0.6 and 0.85), optional LLM verification can confirm the gate decision. This is configurable via `gate_llm_verification` (off by default for speed).

No LLM calls in the default path. O(nodes * examples) cosine comparisons.

```python
def evaluate_gate(gate: GateCondition, context_embedding: list[float]) -> bool:
    gate_sim = cosine_similarity(gate.embedding, context_embedding)
    best_positive_sim = max(
        (cosine_similarity(ex, context_embedding) for ex in gate.positive_embeddings),
        default=gate_sim,
    )
    score = 0.6 * gate_sim + 0.4 * best_positive_sim
    return score >= gate.threshold
```

### Expected Failure Modes

- **False activations:** Gates with broad/generic text (e.g., "When working on code") will match too many contexts. Mitigation: threshold tuning upward, more specific gate text.
- **Missed activations:** Gates with very specific text (e.g., "When debugging async race conditions in Python 3.12") may miss semantically similar but differently worded contexts. Mitigation: threshold tuning downward, positive examples cover phrasing variations.
- **Embedding model sensitivity:** Different embedding models produce different similarity distributions. Thresholds tuned for one model may not transfer. Recommendation: re-calibrate thresholds when changing embedding models.

### Gate Threshold Tuning

The default threshold of 0.75 can be overridden per gate. Higher thresholds make gates more selective (fewer false activations). Lower thresholds make gates more permissive (fewer missed activations). The threshold is set at CDT build time and stored with the gate. Recommended tuning approach: use the stored positive/negative examples as a validation set -- adjust the threshold until all positive examples activate and all negative examples do not.

---

## 4. CDT Traversal

Traversal is the inference-time operation: given the current context, which parts of the CDT are relevant?

### Input

```python
@dataclass(frozen=True)
class TraversalContext:
    description: str                    # Natural language context description
    metadata: dict[str, str] = field(default_factory=dict)  # Optional structured metadata
```

The `description` is the primary input. It can be a sentence, a paragraph, or a summary of the current situation. Examples:
- `"Debugging a race condition in an async Python pipeline"`
- `"Making a small config change to a single file"`
- `"Reviewing a schema migration that affects 16 files"`

### Process

```
1. Embed the context description once
2. Start at the root node
3. For each cluster node: always enter (clusters are not gated)
4. For each gate node: evaluate gate using calibrated scoring (see Section 3):
     score = 0.6 * gate_similarity + 0.4 * best_positive_similarity
   - If score >= gate.threshold: enter subtree, collect leaf statements
   - If score < gate.threshold: skip subtree
5. Collect all accepted leaf statements from entered subtrees
6. Root-level accepted statements are always included (no gate)
```

### Output

```python
@dataclass(frozen=True)
class TraversalResult:
    grounding_statements: list[str]     # Active statements from matched branches
    active_node_ids: list[int]          # CDT node IDs that were activated
    gate_conditions_evaluated: int      # Number of gates checked
    root_statements_count: int          # Global statements (always active)
    gated_statements_count: int         # Conditionally activated statements
```

### Key Properties

- **Multi-path:** Multiple branches can activate simultaneously. A context about "debugging async code in a Python CLI" might activate gates for debugging, async, and Python CLI simultaneously.
- **Synchronous:** No LLM calls at traversal time. One embedding call for the context, then O(nodes) cosine comparisons.
- **Root statements always included:** Accepted hypotheses at the root level (no gate condition) are always returned, regardless of context.
- **Superseded nodes skipped:** Nodes marked as superseded are never returned during traversal.
- **Rejected nodes skipped:** Nodes marked as rejected are never returned during traversal.

---

## 5. Temporal CDT (T-CDT)

T-CDT extends standard CDT with time awareness. This is motivated by PERSONAMEM's finding that frontier models get ~50% on dynamic user profiling -- explicit temporal state is mandatory for tracking evolving preferences.

### Time-Weighted Validation

During validation (Step 3), evidence items are weighted by recency:

```python
def temporal_weight(timestamp: datetime, half_life_days: int = 90) -> float:
    """Compute time decay weight. Half-life: weight = 0.5 at half_life_days age."""
    age_days = (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400
    return 0.5 ** (age_days / half_life_days)
```

When computing confidence, each verdict's contribution is scaled by its temporal weight:

```python
def compute_temporal_confidence(
    verdicts: list[TemporalVerdict],
    half_life_days: int = 90,
) -> float:
    """Compute time-weighted confidence.

    Each verdict is weighted by recency. More recent evidence has more influence.
    """
    weighted_supporting = sum(
        temporal_weight(v.timestamp, half_life_days)
        for v in verdicts if v.verdict == "supports"
    )
    weighted_contradicting = sum(
        temporal_weight(v.timestamp, half_life_days)
        for v in verdicts if v.verdict == "contradicts"
    )
    total = weighted_supporting + weighted_contradicting
    if total == 0:
        return 0.0
    return weighted_supporting / total
```

### Supersession Tracking

**Supersession trigger:** Supersession occurs when the temporal confidence of an existing accepted hypothesis drops below `cdt_reject_threshold` (0.50) due to new contradicting evidence added during incremental update. The system recomputes `temporal_confidence` for all existing hypotheses using the updated evidence set. If a previously-accepted hypothesis now falls below the reject threshold, it is marked as superseded.

When newer evidence contradicts an older accepted pattern:

1. The older pattern is marked as **superseded** (not deleted)
2. A new pattern is created as the active replacement
3. The superseded pattern retains:
   - `superseded_at`: timestamp of supersession
   - `superseded_reason`: why it was superseded
   - `superseded_by`: reference to the new pattern

```python
@dataclass(frozen=True)
class SupersessionRecord:
    original_node_id: int
    replacement_node_id: int
    superseded_at: datetime
    reason: str  # "Contradicted by newer evidence from sessions X, Y, Z"
```

### Why Preserve Superseded Patterns

- **Audit trail:** Understanding how preferences evolved over time
- **Rollback:** If the newer pattern is later invalidated, the old one can be restored
- **Wikification:** The profile document shows pattern evolution in a "Superseded Patterns" section
- **Analysis:** Temporal drift patterns reveal how behavior changes over time

### Incremental Tree Growth

New data does not require rebuilding the entire CDT. The 8-step incremental update algorithm (see D14):

1. Embed new observations
2. Assign to existing clusters by centroid similarity (centroid = mean of member embeddings, an approximation for HDBSCAN which does not produce centroids natively)
3. If distance > `new_cluster_threshold` (default: 2x avg intra-cluster distance) → flag as potential new cluster candidate
4. If 5+ flagged observations form a dense group (HDBSCAN on flagged observations with `min_cluster_size`) → create new cluster
5. Re-validate ALL hypotheses in affected clusters using the updated evidence set (including new observations)
6. Prescriptive hypotheses (from bootstrap mode, see D13) that now have evidence: confidence rises (confirmed) or drops (superseded)
7. Generate + validate new hypotheses from new/changed clusters
8. **Full rebuild trigger:** when > 30% of clusters are new, or > 50% of hypotheses have changed status, trigger a full CDT rebuild instead of incremental update

---

## 6. Two-Layer Data Approach

An optimization over standard CDT that separates hypothesis generation from validation.

### Standard CDT Flow

```
Raw data --> Extract hypotheses --> Validate hypotheses (against same raw data)
```

Standard CDT does both extraction and validation from the same raw data. This is expensive: the LLM must process all raw observations twice.

### Two-Layer Flow (Approach C — see D10)

```
Layer 1 (summaries) --> Seed hypotheses (cheap -- summaries are pre-extracted)
Layer 2 (raw data)  --> Validate ALL hypotheses (targeted -- only relevant evidence)
```

Session-level observations seed hypotheses; interaction-level observations are the evidence. Both tiers of hypotheses (cluster-derived + session-card-derived) are deduplicated then validated against interaction observations.

### How It Works

- **Layer 1 (hypothesis seeds):** Pre-extracted behavioral statements from upstream summarization (e.g., session card `user_preferences`). These serve as additional hypothesis candidates alongside cluster-derived hypotheses. In the CDT paper's terms, these are the "storyline summaries" from which character traits are hypothesized.
- **Layer 2 (validation evidence):** Raw evidence items (interaction observations) for validating or rejecting ALL hypotheses regardless of source. Each item is classified as supporting, contradicting, or irrelevant.

### Why This Is More Efficient

- Hypothesis extraction from summaries is cheap: summaries are already compressed, fewer tokens
- Validation is targeted: each cluster's hypotheses are validated against only matching evidence, not all evidence
- Hypothesis quality is higher: summaries have already been through an extraction pipeline, producing more focused behavioral statements

### Domain-Agnostic Application

Any system that produces both summaries and raw data can use this approach:
- **User profiling:** Session summaries (hypotheses) + raw session interactions (validation)
- **Character RP:** Story arc summaries (hypotheses) + raw dialogue (validation)
- **Code review:** PR summaries (hypotheses) + individual file diffs (validation)
- **Customer support:** Ticket summaries (hypotheses) + chat transcripts (validation)

When no summaries exist, Canopy falls back to standard single-layer CDT: hypotheses and validation both come from the raw observations.

---

## 7. Validation Mechanism

### LLM-Based Validation

For each hypothesis, the LLM classifies each evidence item:

```
Hypothesis: "User always requires tests before committing"

Evidence item 1: "User said: run the test suite before we commit"
  --> verdict: supports

Evidence item 2: "User committed a hotfix without running tests"
  --> verdict: contradicts

Evidence item 3: "User discussed database schema design"
  --> verdict: irrelevant
```

The LLM outputs **binary verdicts only** -- never confidence floats. This is a deliberate design choice: LLMs are unreliable at calibrated confidence estimation, but good at binary classification.

### Confidence Computation

Confidence is computed programmatically after collecting all verdicts:

```python
confidence = supporting_count / (supporting_count + contradicting_count)
```

Irrelevant items are excluded. If there is no relevant evidence (supporting + contradicting == 0), confidence is 0.0 (reject).

### Threshold-Based Status Assignment

| Confidence Range | Status | Tree Placement |
|-----------------|--------|---------------|
| >= `cdt_accept_threshold` (0.75) | Accepted | Root-level or leaf node |
| <= `cdt_reject_threshold` (0.50) | Rejected | Stored but not traversed (0.50 exactly is rejected) |
| `(0.50, 0.75)` exclusive | Conditional | Gated subtree |

### Two-Pass Cross-Cluster Validation (see D9)

Validation follows a global-then-conditional two-pass approach identical to the CDT paper's method:

- **Pass 1 (Global):** Each hypothesis is tested against ALL observations across all clusters and noise. High-confidence hypotheses become ROOT (universal). Low-confidence hypotheses are REJECTED. Intermediate hypotheses proceed to Pass 2.
- **Pass 2 (Conditional):** Observations are filtered by semantic similarity to the hypothesis's gate condition. The hypothesis is validated against this filtered evidence set. If it passes → GATED rule. Otherwise → REJECTED.

This catches cross-cluster contradictions: a hypothesis that looks valid within its cluster may be contradicted by evidence in other clusters.

For clusters with too many matching evidence items, the top N items (by quality or recency) are sampled. Default N = 500, configurable via `cdt_max_validation_items`.

### NLI-Based Validation (Alternative)

The original CDT paper uses Natural Language Inference (NLI) models (e.g., DeBERTa) instead of LLM calls for validation. Canopy supports both:

- **LLM validation:** More accurate for nuanced behavioral hypotheses, but more expensive
- **NLI validation:** Faster and cheaper, suitable for high-volume or latency-sensitive use cases

The validation interface is pluggable: implement `validate(hypothesis, evidence) -> Verdict` using either approach.

---

## 8. Wikification

Wikification converts the CDT tree into a human-readable markdown profile. This provides both: the CDT for programmatic traversal, and the wikified profile for human review and as a fallback context document.

### Profile Structure

```markdown
# CDT Profile: [Subject Name]

> Generated on [date] from [N] data items
> CDT: [K] clusters, [M] validated hypotheses, [R] rejected
> T-CDT: [S] superseded patterns tracked

## CORE VALUES (always active)

### [Pattern Name]
[Actionable statement]
- Confidence: 0.95 | Evidence: [N] items
- Rule: [augmented rule reference, if any]

...

## [CLUSTER LABEL] (gated patterns)

### When: [gate condition text]
[Actionable statement]
- Confidence: 0.88 | Evidence: [source references]

### When: [another gate condition]
[Actionable statement]
- Confidence: 0.85 | Evidence: [source references]

...

## SUPERSEDED PATTERNS (historical -- not active)

### [Date] [Original pattern]
~~[Original statement]~~
Superseded by: "[New pattern statement]"
Reason: [Why the pattern changed]
```

### Wikification Process

1. An LLM receives the CDT nodes (as JSON) and optionally rule augmentations
2. The LLM generates a narrative markdown document organized by the tree structure
3. Core values (root-level accepted, no gate condition) are listed first
4. Gated patterns are organized by cluster, each with its gate condition
5. Superseded patterns are shown at the end with history
6. The LLM is instructed to be specific, not generic -- include concrete details from evidence

### Token Budget

- Target: 10-20K tokens for typical profiles
- Ceiling: 50K tokens for data-rich profiles
- If output exceeds ceiling: log warning but save (the LLM is instructed on the budget)
- If output is suspiciously short (< 5K tokens): log warning (possible LLM failure)

---

## 9. Domain Adapters

Domain adapters translate domain-specific data into the universal `BehavioralObservation` format that the CDT pipeline consumes.

### Adapter Interface

```python
from abc import ABC, abstractmethod

class DomainAdapter(ABC):
    @abstractmethod
    def extract_observations(self, data: Any) -> list[BehavioralObservation]:
        """Extract behavioral observations from domain-specific data."""
        ...

    def extract_hypotheses(self, summaries: Any) -> list[str] | None:
        """Optional: extract pre-formed hypotheses from summary data.

        If implemented, enables the two-layer data approach (Section 6).
        If None is returned, standard single-layer CDT is used.
        """
        return None

    def additional_hypotheses(self) -> list[Hypothesis] | None:
        """Optional: provide additional hypotheses from domain-specific sources.

        Adapters can supply hypotheses from rules, documents, or other sources
        (see D12 unified hypothesis pipeline). These are merged with cluster
        and session-card hypotheses, deduplicated, and validated by the CDT.
        """
        return None
```

### Character RP Adapter (Original CDT Use Case)

The original CDT paper's use case: deriving character logic from storyline text. Uses `SceneActionPair` as a convenience that converts to `BehavioralObservation`.

```python
class CharacterAdapter(DomainAdapter):
    def extract_observations(self, data: list[str]) -> list[BehavioralObservation]:
        """Extract observations from storyline text via SceneActionPair.

        Scene = narrative context / situation description
        Action = character's response / behavior in that scene
        Converted to BehavioralObservation: "scene | action"
        """
        pairs = self._parse_scene_action_pairs(data)
        return [p.to_observation() for p in pairs]
```

### User Profiling Adapter (delulu Use Case)

Profiling a user's coding behavior from AI session data. Input is already combined scene+action in prose (classified interaction summaries), so maps directly to `BehavioralObservation`.

```python
class UserProfileAdapter(DomainAdapter):
    def extract_observations(self, data: SessionData) -> list[BehavioralObservation]:
        """Extract observations from classified interaction summaries.

        Summaries capture behavioral insight -- combined scene+action in prose.
        Raw fields (output_text, reaction_text) are too verbose/terse.
        """
        ...

    def extract_hypotheses(self, summaries: list[SessionCard]) -> list[str]:
        """Extract hypotheses from session card fields.

        user_preferences -> behavioral statements
        correction_patterns, steering_patterns -> behavioral observations
        key_decisions -> conditional rules with evidence
        """
        ...
```

### Adding a New Domain

To add a new domain adapter:

1. Implement `DomainAdapter.extract_observations()` to map your data to `BehavioralObservation`
2. Optionally implement `extract_hypotheses()` if you have pre-extracted summaries
3. Register the adapter with the CDT builder

The CDT pipeline is completely agnostic to the adapter's domain. Clustering, validation, tree building, and traversal all operate on the universal `BehavioralObservation` representation.

---

## 10. Configuration

All CDT-specific configuration values with defaults and valid ranges.

### Core CDT Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `cdt_accept_threshold` | 0.75 | 0.5-1.0 | Confidence threshold for accepting a hypothesis. **Note:** Code default is 0.80; paper config uses 0.75 (see D30). |
| `cdt_reject_threshold` | 0.50 | 0.0-0.75 | Confidence threshold for rejecting a hypothesis |
| `cdt_max_validation_items` | 500 | 50-5000 | Max evidence items per cluster for validation |

**Constraints:**
- `cdt_accept_threshold` must be strictly greater than `cdt_reject_threshold`
- `cdt_accept_threshold - cdt_reject_threshold >= 0.15` (minimum gap to avoid ambiguous conditional zone)

### Clustering Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_clusters` | 8 | 2-32 | Safety cap: max clusters (HDBSCAN may produce fewer) |
| `min_cluster_size` | 16 | 2-100 | HDBSCAN min_cluster_size parameter; also used as min items to form a valid cluster |
| `new_cluster_threshold` | (auto) | -- | Distance threshold for flagging new clusters during incremental growth (default: 2x average intra-cluster distance) |

### Noise Handling Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `noise_mode` | `"validate_only"` | `"discard"`, `"validate_only"`, `"appendix"` | How HDBSCAN noise points are handled (see D11) |

### Hypothesis Source Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_cluster_hypotheses` | true | -- | Generate hypotheses from clusters (CDT core) |
| `use_session_cards` | true | -- | Extract hypotheses from session card summaries (D10) |
| `use_rules` | true | -- | Extract hypotheses from rules files (D12) |
| `rules_paths` | [] | -- | Paths to rules files for hypothesis extraction |
| `use_documents` | false | -- | Extract hypotheses from documents (D12) |
| `document_paths` | [] | -- | Paths to documents for hypothesis extraction |
| `manual_hypotheses` | [] | -- | Manual hypothesis strings to inject |
| `bootstrap_confidence` | 0.5 | 0.0-1.0 | Confidence for unvalidated prescriptive hypotheses (D13) |
| `require_validation` | false | -- | If true, reject unvalidated hypotheses at traversal time |
| `dedup_similarity_threshold` | 0.90 | 0.8-1.0 | Cosine similarity threshold for hypothesis deduplication (D10) |

### Temporal Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `time_decay_enabled` | true | -- | Enable T-CDT temporal weighting. When false, standard CDT (equal weight). |
| `time_decay_half_life_days` | 90 | 7-365 | Half-life for temporal weight decay |

### Embedding Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `embedding_model` | (configurable) | -- | Model used for embedding observations and gate conditions |
| `gate_similarity_threshold` | 0.75 | 0.5-1.0 | Default cosine similarity threshold for gate activation |
| `gate_llm_verification` | false | -- | Use LLM to verify borderline gate decisions (score 0.6-0.85) |

### Configuration Dataclass

```python
@dataclass(frozen=True)
class CDTConfig:
    # Core CDT
    cdt_accept_threshold: float = 0.75
    cdt_reject_threshold: float = 0.50
    cdt_max_validation_items: int = 500

    # Clustering
    max_clusters: int = 8
    min_cluster_size: int = 16
    new_cluster_threshold: float = 2.0  # Multiplier on avg intra-cluster distance for new cluster detection
    noise_mode: str = "validate_only"  # "discard", "validate_only", "appendix" (D11)

    # Hypothesis sources (D12, D16)
    use_cluster_hypotheses: bool = True
    use_session_cards: bool = True
    use_rules: bool = True
    rules_paths: list[str] = field(default_factory=list)
    use_documents: bool = False
    document_paths: list[str] = field(default_factory=list)
    manual_hypotheses: list[str] = field(default_factory=list)
    bootstrap_confidence: float = 0.5  # For unvalidated prescriptive hypotheses (D13)
    require_validation: bool = False  # Reject unvalidated hypotheses at traversal time
    dedup_similarity_threshold: float = 0.90  # Cosine sim for hypothesis deduplication (D10)

    # Temporal
    time_decay_enabled: bool = True
    time_decay_half_life_days: int = 90

    # Embeddings
    embedding_model: str = ""  # Set per deployment
    gate_similarity_threshold: float = 0.75
    gate_llm_verification: bool = False

    def __post_init__(self) -> None:
        if self.cdt_accept_threshold <= self.cdt_reject_threshold:
            raise ValueError(
                f"cdt_accept_threshold ({self.cdt_accept_threshold}) must be "
                f"greater than cdt_reject_threshold ({self.cdt_reject_threshold})"
            )
        gap = self.cdt_accept_threshold - self.cdt_reject_threshold
        if gap < 0.15:
            raise ValueError(
                f"Minimum gap between thresholds must be >= 0.15, got {gap:.2f}. "
                f"accept={self.cdt_accept_threshold}, reject={self.cdt_reject_threshold}"
            )
        if self.noise_mode not in ("discard", "validate_only", "appendix"):
            raise ValueError(f"noise_mode must be 'discard', 'validate_only', or 'appendix', got '{self.noise_mode}'")
```

---

## 11. Architecture (Target Package Structure)

```
src/canopy/
├── __init__.py          # Public API exports
├── core.py              # CDTNode, CDTTree, BehavioralObservation, SceneActionPair, GateCondition dataclasses
├── builder.py           # CDT construction algorithm (Steps 1-5)
├── embeddings.py        # Embedding model management, cosine similarity
├── clustering.py        # HDBSCAN clustering (KMeans fallback), cluster labeling
├── validation.py        # Evidence-based validation (LLM + NLI)
├── gates.py             # Semantic gate conditions, embedding, matching
├── temporal.py          # T-CDT: time weighting, supersession tracking
├── traverse.py          # CDT traversal engine
├── wikify.py            # CDT -> markdown profile generation
├── prompts.py           # All LLM prompt templates
├── config.py            # CDTConfig dataclass, validation
├── adapters/            # Domain-specific data adapters
│   ├── __init__.py      # DomainAdapter ABC + registry
│   ├── character.py     # CDT paper: character RP (storyline -> pairs)
│   └── user_profile.py  # delulu: user behavior (sessions -> pairs)
├── cli.py               # CLI interface (build, traverse, wikify)
└── py.typed             # PEP 561 marker
```

### Module Responsibilities

| Module | Responsibility | LLM Calls? |
|--------|---------------|------------|
| `core.py` | Data structures only. No logic beyond validation. | No |
| `builder.py` | Orchestrates the full CDT construction pipeline (Steps 1-7). | Delegates to other modules |
| `embeddings.py` | Embed text, compute cosine similarity. Model loading and caching. | No (embedding model, not LLM) |
| `clustering.py` | HDBSCAN on embeddings (KMeans fallback), cluster filtering, LLM-based cluster labeling. | Yes (labeling only) |
| `validation.py` | Validate hypotheses against evidence. Compute confidence. | Yes (verdict classification) |
| `gates.py` | Create gate conditions, embed gate text, evaluate gates at traversal time. | No (embedding only) |
| `temporal.py` | Time decay computation, supersession detection and recording. | No |
| `traverse.py` | Walk the CDT tree, evaluate gates, collect grounding statements. | No (embedding + cosine only) |
| `wikify.py` | Convert CDT to markdown. | Yes (narrative generation) |
| `prompts.py` | Prompt templates for all LLM-calling modules. No logic. | N/A (templates only) |

### Dependency Graph

```
builder.py
  |-- clustering.py --> embeddings.py
  |-- validation.py
  |-- temporal.py
  |-- gates.py --> embeddings.py
  |-- wikify.py
  |-- prompts.py
  |-- adapters/*

traverse.py
  |-- gates.py --> embeddings.py
  |-- core.py
```

`traverse.py` is independent of `builder.py` -- it only needs the built CDT tree and the embedding model. This allows traversal to be used without the full build pipeline (e.g., loading a pre-built CDT from storage).

---

## 12. Bootstrap Mode (see D13)

Bootstrap mode enables CDT construction from rules, documents, and style guides when zero session history is available.

### How It Works

1. Prescriptive hypotheses are extracted from rules files, documents, and manual entries (source_type = RULE, DOCUMENT, MANUAL)
2. With no interaction evidence, hypotheses are accepted at `bootstrap_confidence` (default 0.5) with `validation_status: "unvalidated"`
3. As sessions accumulate, prescriptive hypotheses are incrementally validated against real evidence (D14 step 6)
4. Lifecycle: **prescriptive (day 0)** → **partially validated (day 7)** → **evidence-based (day 90)**

### Prescriptive vs Evidence-Based

| Aspect | Prescriptive (bootstrap) | Evidence-Based (validated) |
|--------|-------------------------|---------------------------|
| Source | Rules, docs, manual | Cluster observations, session cards |
| Initial confidence | `bootstrap_confidence` (0.5) | Computed from verdicts |
| validation_status | `"unvalidated"` | `"validated"` |
| Behavior | Applied but flagged as assumed | Applied with full confidence |
| Evolution | Confirmed or superseded by evidence | Standard T-CDT lifecycle |

### Configuration

- `bootstrap_confidence: float = 0.5` — confidence assigned to unvalidated prescriptive hypotheses
- `require_validation: bool = False` — if True, reject unvalidated hypotheses at traversal time

### Rejected Rules Are Valuable

When a rule is contradicted by observed behavior, the rejection itself is informative: "Rule says TDD, but user writes code first 65% of the time." These rejected prescriptive hypotheses are preserved in the CDT for audit.

### Paper Angle (see D17)

"From Guidelines to Behavior: How Prescriptive Rules Evolve into Evidence-Based Decision Trees Through Incremental User Observation." This prescriptive → validated lifecycle is novel — no existing system (PURE, PERSONAMEM, original CDT) supports this.

---

## 13. Design Decisions Log

Decisions made during design discussions, with rationale.

### D1: BehavioralObservation replaces SceneActionPair as primary input

**Decision:** The primary CDT input is `BehavioralObservation` (single text string), not `(scene, action)` pairs. `SceneActionPair` remains as an adapter convenience that converts to `BehavioralObservation` via `to_observation()`.

**Rationale:** Our data (classified interaction summaries from delulu) are combined scene+action in prose. Raw fields (`output_text`, `reaction_text`) are too verbose/terse. Summaries capture behavioral insight. CDT's clustering, hypothesis generation, and validation all work with single text embeddings -- the scene/action split adds no value to the algorithm. The original CDT paper's (scene, action) format is a domain-specific input structure, not a requirement of the CDT algorithm itself.

### D2: Three verdict classification (not four)

**Decision:** Validation uses three verdicts: `supports`, `contradicts`, `irrelevant`. No `unknown` option.

**Rationale:** "Unknown" and "irrelevant" serve the same purpose -- both are excluded from the confidence calculation `supporting / (supporting + contradicting)`. Adding "unknown" gives the LLM an escape hatch that reduces signal. When the LLM cannot determine relevance, `irrelevant` is the correct classification.

### D3: Traversal modes configurable

**Decision:** Three traversal modes: `SEMANTIC` (default, embedding cosine similarity), `LLM` (accurate but slow, LLM evaluates each gate), `HYBRID` (semantic pre-filter + LLM for borderline cases). Not locked to one approach.

**Rationale:** Different use cases have different latency/accuracy tradeoffs. Default semantic mode is fast (no LLM calls at inference). LLM mode is available for high-stakes decisions. Hybrid balances both. The `gate_llm_verification` config option in Section 3 implements the hybrid approach for borderline gate scores.

### D4: T-CDT configurable with equal-weight fallback

**Decision:** `time_decay_enabled: bool = True`. When `False`, standard CDT behavior (all evidence weighted equally). Easy A/B comparison.

**Rationale:** Allows benchmarking T-CDT against standard CDT on the same data. Essential for the research paper (Phase 5) and for domains where temporal dynamics are not relevant.

### D5: Validation batching configurable

**Decision:** `max_validation_items: int = 500` (mapped to `cdt_max_validation_items` in config). Set to 0 for unlimited (validate all evidence). Default 500 for cost control.

**Rationale:** Large clusters can have thousands of evidence items. Validating all of them is expensive and hits context window limits. Sampling the top N by quality/recency is semantically sound. The 0 option exists for small datasets or when accuracy is paramount.

### D6: No exec() -- JSON only

**Decision:** No support for `exec()` on LLM output, ever. Not even as an option or comparison mode. All LLM output is parsed as JSON.

**Rationale:** The original CDT paper uses `exec()` on LLM-generated Python predicates for gate evaluation. This is a security risk with zero benefit over JSON parsing + semantic embedding gates. Canopy uses natural language gates with pre-computed embeddings (Section 3) instead of executable code.

### D7: Rules augmentation is a later phase

**Decision:** Rules augmentation (Section 2, Step 6) is not in canopy Phase 0-1. It is a delulu-specific enrichment feature, not core CDT.

**Rationale:** Rules augmentation attaches external rule files to CDT nodes. This is domain-specific (delulu reads `~/.claude/rules/`). The core CDT algorithm does not depend on it. Canopy Phase 0-1 focuses on the algorithm; rules augmentation is deferred.

### D8: LLM confidence scores removed from delulu Phase 2

**Decision:** Both `confidence` and `source_confidence` fields removed from delulu Phase 2 classification output. Origin (human vs machine) is determined by tags, not LLM estimation.

**Rationale:** LLM-generated confidence floats are unreliable (PERSONAMEM finding). Computed evidence (`supporting / (supporting + contradicting)`) from binary verdicts is strictly more trustworthy. This reinforces the principle throughout the system: computed evidence > LLM self-assessment. Phase 3's quality filtering should use tag-based origin detection rather than `source_confidence`.

### D9: Cross-cluster validation follows CDT paper's two-pass approach

**Decision:** Validation uses a global-then-conditional two-pass approach identical to the CDT paper.

- **Pass 1 (Global):** Test each hypothesis against ALL observations (all clusters + noise). If confidence >= `cdt_accept_threshold` → ROOT (universal). If confidence <= `cdt_reject_threshold` → REJECTED. If between → Pass 2.
- **Pass 2 (Conditional):** Filter observations by gate semantic similarity, validate against filtered set. If confidence >= `cdt_accept_threshold` → GATED rule. Otherwise → REJECTED.

**Rationale:** Catches cross-cluster contradictions. A hypothesis that looks valid within its cluster may be contradicted by evidence in other clusters. This is identical to the CDT paper's approach.

### D10: Session-level observations as hypothesis seeds (Approach C)

**Decision:** Two tiers of hypothesis sources:
- **Tier 1:** Interaction observations → cluster → hypothesize (CDT core)
- **Tier 2:** Session card `user_preferences` → extract as additional hypothesis candidates

All hypotheses (Tier 1 + Tier 2) are deduplicated then validated against interaction observations. Session observations seed hypotheses; interaction observations are the evidence.

**Deduplication:** Embedding cosine similarity > 0.90 = duplicate. Merge and combine sources.

**Rationale:** Session cards capture high-level behavioral patterns that may not emerge from individual interaction clustering. Using them as hypothesis seeds (not evidence) leverages their compression while maintaining interaction-level validation rigor.

### D11: Noise observations used as validation evidence (VALIDATE_ONLY mode)

**Decision:** Three configurable modes for HDBSCAN noise points (label -1):
- `DISCARD` — noise observations are ignored entirely
- `VALIDATE_ONLY` (default) — noise observations validate/contradict hypotheses but don't generate new ones
- `APPENDIX` — noise observations are included in wikification as uncategorized data

**Rationale:** VALIDATE_ONLY provides contrarian evidence without pattern-matching on outliers. Noise observations may contain edge cases that contradict otherwise-accepted hypotheses, strengthening the validation signal.

### D12: Unified hypothesis pipeline — any source can produce hypotheses

**Decision:** All hypothesis sources produce the same `Hypothesis` dataclass with `source_type` (see D15). Sources: `CLUSTER` (CDT core), `SESSION_CARD`, `RULE`, `DOCUMENT`, `MANUAL`. All sources merge → deduplicate → two-pass validation → build tree. CDT doesn't care where hypotheses come from — it validates them all against evidence.

**Rationale:** Rules and third-party docs become hypotheses validated against actual user behavior. Rejected rules are valuable: "Rule says TDD, but user writes code first 65% of the time." This unification simplifies the pipeline — one validation path for all hypothesis types.

### D13: Bootstrap mode — CDT from rules/docs with zero sessions

**Decision:** New users with no session history can bootstrap CDT from rules, docs, and style guides. Prescriptive hypotheses are accepted at `bootstrap_confidence` (default 0.5) with `validation_status: "unvalidated"`. As sessions accumulate, prescriptive hypotheses get validated incrementally. See Section 12 for full details.

**Lifecycle:** prescriptive (day 0) → partially validated (day 7) → evidence-based (day 90).

**Configuration:**
- `bootstrap_confidence: float = 0.5`
- `require_validation: bool = False` — if True, reject unvalidated hypotheses at traversal time

**Rationale:** Cold start is a real problem. Users shouldn't wait for 50+ sessions before the CDT is useful. Bootstrap provides immediate value from existing rules while maintaining the evidence-based validation guarantee as data accumulates.

### D14: Incremental update algorithm (8 steps)

**Decision:** The 8-step incremental update algorithm (documented in Section 5):

1. Embed new observations
2. Assign to existing clusters (by centroid similarity — centroid = mean of member embeddings, approximation for HDBSCAN)
3. If distance > `new_cluster_threshold` (2x avg intra-cluster distance) → flag as potential new cluster
4. If 5+ flagged observations form dense group (HDBSCAN) → new cluster
5. Re-validate ALL hypotheses in affected clusters using updated evidence set
6. Prescriptive hypotheses that now have evidence: confidence rises (confirmed) or drops (superseded)
7. Generate + validate new hypotheses from new/changed clusters
8. Full rebuild trigger: >30% clusters new OR >50% hypotheses changed status

**Rationale:** Centroid-based assignment is an approximation (HDBSCAN doesn't produce centroids natively), but it's computationally efficient for incremental updates. Step 6 explicitly handles the bootstrap → evidence-based lifecycle from D13. The full rebuild trigger prevents drift from accumulating too many incremental patches.

### D15: HypothesisSource enum and Hypothesis dataclass

**Decision:** Standardized data structures for the unified hypothesis pipeline:

```python
class HypothesisSource(Enum):
    CLUSTER = "cluster"
    SESSION_CARD = "session_card"
    RULE = "rule"
    DOCUMENT = "document"
    MANUAL = "manual"

@dataclass
class Hypothesis:
    statement: str
    gate_condition: str | None
    source_type: HypothesisSource
    source_ref: str
    confidence: float  # computed from evidence, or bootstrap_confidence if unvalidated
    validation_status: str  # "validated" | "unvalidated" | "superseded"
    supporting_count: int
    contradicting_count: int
```

**Rationale:** All hypothesis sources produce the same structure. `source_type` tracks provenance for audit and lifecycle management. `validation_status` distinguishes bootstrap prescriptive hypotheses from evidence-based ones.

### D16: CDTBuilderConfig for hypothesis sources

**Decision:** Configuration for controlling which hypothesis sources are active:

```python
use_cluster_hypotheses: bool = True
use_session_cards: bool = True
use_rules: bool = True
rules_paths: list[str] = []
use_documents: bool = False
document_paths: list[str] = []
manual_hypotheses: list[str] = []
bootstrap_confidence: float = 0.5
require_validation: bool = False
```

**Rationale:** Granular control over hypothesis sources. Default configuration enables cluster + session card + rules (the most common setup). Documents and manual hypotheses are opt-in. `require_validation` provides a safety valve for users who don't want unvalidated prescriptive hypotheses in their CDT output.

### D17: Paper angle for bootstrap → evidence evolution

**Decision:** Research contribution: "From Guidelines to Behavior: How Prescriptive Rules Evolve into Evidence-Based Decision Trees Through Incremental User Observation."

**Rationale:** The lifecycle of prescriptive (day 0) → partially validated (day 7) → evidence-based (day 90) is novel. No existing system (PURE, PERSONAMEM, original CDT) supports bootstrapping from prescriptive rules and incrementally validating them against observed behavior. This is a publishable contribution.

### D18: Sidekick runtime — wikified profile as system prompt + active traversal

**Decision:** Two-phase delivery of CDT profile to the Sidekick:
- **v1:** Full wikified profile injected as system prompt. Simpler, more reliable.
- **v2:** Active traversal at key events (commits, schema changes, completion claims, error reports) for timely reinforcement of relevant rules.

Both needed long-term because LLMs drift from system prompt instructions as context grows.

**Rationale:** System prompt injection is the simplest reliable delivery mechanism — no infrastructure needed. But as conversations grow, LLMs attend less to system prompt content. Active traversal at key events re-surfaces the most relevant rules exactly when they matter, counteracting context drift.

### D19: Hybrid wikification — per-section generation + synthesis pass

**Decision:** Wikification uses a two-stage process:
1. **Per-section generation:** Each cluster's section is generated independently (parallelizable, individually evaluable).
2. **Synthesis pass:** One LLM call generates the intro summary + cross-cutting observations from the assembled sections.

Per-section failure can be retried without redoing the entire profile.

**Rationale:** Independent per-section generation enables parallel execution and granular retry. A single monolithic wikification call risks losing the entire profile if the LLM fails partway through. The synthesis pass adds coherence and cross-cutting observations that per-section generation cannot produce in isolation.

### D20: No observation quality filtering — all observations are behavioral evidence

**Decision:** Do not pre-filter observations before CDT construction. All observations — including "generic" ones (simple questions, terse approvals) — are fed to the pipeline.

**Rationale:** "Generic" observations represent the user's dominant behavior mode. The Sidekick needs to emulate routine behavior, not just dramatic moments. Filtering out routine interactions would bias the CDT toward exceptional behavior and miss the baseline patterns. HDBSCAN naturally handles true outliers via its noise label (-1). Pre-filtering adds a subjective quality judgment that undermines the evidence-based approach.

### D21: Keep gates specific — let embedding similarity handle soft matching

**Decision:** Do not generalize gate conditions after generation. Gates remain as specific as the LLM produces them.

**Rationale:** Specific gates that don't match a context = safe failure (the rule simply doesn't fire). Over-generalized gates = noisy false matches (the rule fires when it shouldn't). The embedding-based cosine similarity already provides soft matching — a gate like "When debugging async race conditions" will partially match "When debugging concurrency issues" without needing to be manually broadened. Specificity is the safe default.

### D22: Slim wikified profile for runtime, full evidence in DB

**Decision:** Two-tier profile architecture:
- **Runtime profile (~10-15K tokens):** Rules with confidence scores, no evidence links. Injected as system prompt.
- **Full evidence (DB):** Detailed profile with evidence chains, queryable on-demand ("why do you think I prefer X?").

**Rationale:** System prompt budget is finite. A 10-15K token profile leaves room for conversation context. Evidence links are valuable for human review and explainability but would bloat the runtime profile. The DB-backed full profile supports transparency ("show me the evidence") without runtime cost.

### D23: All evaluation modes implemented independently

**Decision:** Four evaluation modes, each producing independent scores:
- **Scenarios:** 6 automated test cases with known-correct outputs.
- **Holdout:** 80/20 prediction test — can the CDT predict behavior on held-out sessions?
- **Contradictions:** Adversarial detection — does the CDT contain self-contradicting rules?
- **Review:** Human annotation document for manual assessment.

Each mode can be run independently or in any combination. Canopy provides the evaluation logic; the CLI is implemented by the consuming application (e.g., `delulu eval --mode scenarios|holdout|contradictions|review`).

**Rationale:** Independent evaluation modes avoid coupling between assessment methods. A CDT that scores well on holdout prediction but poorly on contradiction detection reveals a different class of problem than one that fails both. Running modes independently also enables incremental evaluation during development.

### D24: Default embedding model all-MiniLM-L6-v2

**Decision:** Default embedding model is `all-MiniLM-L6-v2` (80MB, fast, offline). Configurable via `embedding_model` config.

**Note:** Actual deployment uses Qwen3 models — Qwen3-Embedding-0.6B for surface embeddings and Qwen3-0.6B for generative embeddings. The all-MiniLM-L6-v2 default in this design doc predates the migration to Qwen3.

**Rationale:** Good enough for natural language behavioral summaries. Small model size enables offline operation and fast embedding. Code-aware models (e.g., `code-search-ada`) are available as overrides for code-heavy domains but are not the default — most behavioral observations are natural language descriptions, not raw code.

### D25: Workflow sequences handled implicitly through gate conditions

**Decision:** No separate `WorkflowPattern` concept. Gates handle workflow-aware behavior through phase-aware conditions.

Examples:
- "When user has asked 3+ clarifying questions on same topic" → transition to planning
- "When in research phase" → expect exploratory questions
- "When user claims completion" → trigger verification

Traversal context includes session phase, interaction count, and recent interaction types.

**Rationale:** Introducing a separate workflow sequence concept adds a parallel system alongside gates. Gates are already capable of encoding phase awareness through natural language conditions. The traversal context provides the structured metadata (interaction count, phase) that gates can reference via embedding similarity. Adding a dedicated workflow system is premature — gates should be proven insufficient first.

### D26: Model allocation per pipeline step

**Decision:** Different models for different steps based on cost, speed, and quality requirements:

| Step | Model | Rationale |
|------|-------|-----------|
| Hypothesis generation | Haiku (or local model) | Creative but simple task — extract patterns from 8 observations. Haiku is fast and cheap. |
| Cluster labeling | Haiku | Simple naming task. |
| Validation | DeBERTa NLI (local GPU) | High-volume step (~1,200 inferences). DeBERTa is purpose-built for NLI, runs in ~6 seconds vs ~60 minutes with Sonnet. Free after model load. |
| Wikification | Sonnet/Opus | Quality matters — this is the final profile document the Sidekick reads. |
| Evaluation | Sonnet | Reasoning matters for accurate scoring. |

**Validation is the most expensive step** — O(hypotheses × observations) inferences. Using LLM calls for validation is ~25x more expensive than DeBERTa:
- DeBERTa: ~1,200 inferences × ~5ms = ~6 seconds (free, GPU)
- Haiku: ~1,200 calls × ~2s = ~40 minutes (~$0.10)
- Sonnet: ~1,200 calls × ~30s = ~10 hours (~$1.50)

**Alternatives for validation (configurable):**
- `DeBERTa NLI` (default) — `KomeijiForce/deberta-v3-base-rp-nli`, 715MB, fine-tuned for CDT
- `Haiku` — fallback when no GPU available, slower but works
- `Local Qwen NLI` — larger local model if DeBERTa quality is insufficient, explore Qwen3-based NLI
- `Sonnet` — highest quality, use only for debugging/evaluation, not production runs

**Rationale:** The CDT paper uses DeBERTa for validation precisely because of the volume. Replacing it with LLM calls would be 25x slower and add cost for marginal quality improvement on a binary classification task (support/contradict/irrelevant). Reserve expensive models for creative tasks (hypothesis gen, wikification) where quality variance matters.

---

## 14. Research References

| Paper | ArXiv ID | Key Contribution to Canopy |
|-------|----------|---------------------------|
| **CDT: Deriving Character Logic from Storyline as Codified Decision Trees** | 2601.10080 | Base algorithm: hypothesis extraction from (scene, action) pairs, NLI-based validation, tree construction. Canopy's core pipeline is an adaptation of this. |
| **PERSONAMEM: A Benchmark for Dynamic User Profiling** | 2504.14225 | Demonstrates that frontier models get ~50% on dynamic profiling tasks. Motivates T-CDT: explicit temporal state tracking is mandatory, not optional. |
| **PURE: Incremental Profile Management** | 2502.14541 | Extract-merge-compress pipeline for incremental profiles. Informs Canopy's incremental tree growth strategy and two-layer data approach. |
| **Zero-Shot Decision Trees via LLM** | 2501.16247 | LLM-based decision tree induction without labeled training data. Validates the approach of using LLMs for hypothesis generation and validation. |
| **RL-LLM-DT: Iterative Decision Tree Improvement** | 2412.11417 | Reinforcement learning for iterative DT refinement. Potential future extension for Canopy's CDT quality improvement loop. |

### Key Differences from Original CDT

| Aspect | Original CDT | Canopy |
|--------|-------------|--------|
| Domain | Character RP (storylines) | Domain-agnostic (any data source) |
| Gate conditions | `exec()` on Python predicates | Semantic embedding + cosine similarity |
| Validation | NLI model (DeBERTa) | LLM-based (primary) + NLI (alternative) |
| Confidence | NLI entailment scores | Programmatic: supporting/(supporting+contradicting) |
| Temporal awareness | None | T-CDT: time decay, supersession tracking |
| Input format | (scene, action) pairs | BehavioralObservation (single text); SceneActionPair as adapter convenience |
| Hypothesis source | Extracted from raw data | Unified pipeline: clusters + session cards + rules + docs + manual (D12) |
| Bootstrap | None | Prescriptive hypotheses from rules/docs, validated incrementally (D13) |
| Clustering | Predefined character traits | Embedding-based HDBSCAN (auto-discovers structure and cluster count) |
| Traversal | `exec()` gate evaluation | Cosine similarity (no code execution) |

---

## 15. CDT Quality Findings (2026-03-28)

Empirical analysis of Kasumi CDT built with Claude Haiku, Qwen3-0.6B embeddings, depth=3, θ_accept=0.80.

### Tree Statistics vs Paper

| Metric | Our CDT | Paper Avg | Ratio |
|--------|---------|-----------|-------|
| Nodes | 85 | 10.4 | 8.2x |
| Statements | 194 | 61 | 3.2x |
| Empty leaves | 17/25 depth-3 | not reported | — |
| Statement length | 168 chars avg | ~18 words | ~2x longer |

### Issues Found

1. **Cross-topic duplication**: identity/personality/relationship produce near-duplicate statements (9 pairs >0.55 similarity). No post-construction dedup pass exists.
2. **Hollow depth 3**: 25 nodes, only 8 statements (0.3 stmts/node). Tree over-branches when evidence is thin.
3. **Verbose statements**: 23% exceed 200 chars. Compound sentences that should be split.
4. **Imbalanced relationships**: Arisa interaction has only 6 statements despite being the richest dynamic in source material.
5. **Config mismatch with paper**: θ_accept=0.80 (paper: 0.75), Qwen3-0.6B (paper: 8B) — produces different tree shape.

### Paper-Matched Config Results (2026-03-28)

After implementing two-phase embedding architecture (subprocess isolation for VRAM-safe 8B model loading):

| Config | Embeddings | θ_accept | NLI Score | Nodes | Statements |
|--------|-----------|----------|-----------|-------|------------|
| 0.6B, θ=0.80 | Qwen3-0.6B | 0.80 | 43.11 | 85 | 194 |
| 8B, θ=0.75 (paper-matched) | Qwen3-8B | 0.75 | 58.38 | 20 | 72 |
| Paper (GPT-4.1 CDT) | Qwen3-8B | 0.75 | 84.25 | ~10 | ~61 |

**Key observations:**
- 8B embeddings + paper θ boosted score from 43.11 → 58.38 (+35%)
- Tree is much more compact: 20 nodes / 72 stmts vs 85 nodes / 194 stmts
- Remaining gap (58.38 vs 84.25) is likely gen model quality (Haiku vs GPT-4.1/Llama)
- Two-phase architecture eliminates OOM: peak VRAM ~16GB during Phase A, ~5GB during Phase B

### Planned Fixes

- **D27**: Cross-topic deduplication pass after tree construction (cosine similarity, merge threshold 0.55)
- **D28**: Minimum evidence threshold for depth-3 recursion (don't branch with <MIN_PAIRS_FOR_TREE pairs)
- **D29**: Statement compression pass (split compound sentences >200 chars)
- **D30**: Paper-exact config as default for benchmark reproduction (θ_accept=0.75, Qwen3-8B)

### Relationship CDTs for Delulu

Relationship CDTs (character × character) don't apply to delulu user profiling. The equivalent is **context-filtered CDTs**:
- Filter by project (Bhargav × delulu, Bhargav × canopy)
- Filter by task type (brainstorming, debugging, reviewing)
- Same algorithm: `CDTNode(character, goal_topic, filtered_pairs)` — just a different filter dimension

This is deferred to delulu Phase 3 integration. For MVP, a single global CDT from all sessions is sufficient.

---

## CDT Quality Findings (2026-03-27)

Observations from Kasumi Claude-built CDT (haiku, qwen06b, deberta, depth 3):

### F1: Cross-topic statement duplication
Identity/personality/relationship topics produce near-duplicate statements. The hypothesis generation prompt includes "other than established statements" but this only applies within a single topic's build — not across topics. Statements like "Kasumi values team cohesion" appear in identity, personality, AND relationship CDTs.

**Fix:** Post-construction deduplication pass. After `build_character_cdts()` returns, compute pairwise similarity between all statements across topics. Merge or remove duplicates above a cosine threshold (~0.85). This is a new step, not a change to the tree algorithm.

### F2: Over-branching at depth 3
17/25 depth-3 nodes are empty (no statements, no gates). The tree recurses into thin evidence slices where there's nothing meaningful to discover. This wastes LLM calls and produces noise.

**Fix:** Add a `min_evidence_ratio` check before recursing. If `len(filtered_pairs) / len(pairs) < threshold` (e.g., < 0.1), skip the subtree. The current `MIN_PAIRS_FOR_TREE = 8` filter catches absolute size but not relative thinness.

### F3: Compound statement splitting
Some generated statements exceed 200 characters and contain multiple claims joined by commas or "and". These are hard for the NLI validator to score accurately — one true clause and one false clause produces an ambiguous score.

**Fix:** Post-generation splitting pass in `make_hypotheses_batch()`. Split compound statements (>200 chars with coordinating conjunctions) into individual claims. Validate each claim separately.

### F4: Relationship CDT sparsity
Kasumi×Arisa (the richest dynamic in the source material) produces only 6 total statements despite having enough pairs. Investigation needed: the clustering may be producing clusters too similar to the attribute topics, causing deduplication at the hypothesis level.

**Fix:** Investigate whether the `gate_path` and `established_statements` parameters during relationship CDT construction are too aggressive in filtering. The relationship CDT should discover interaction-specific patterns, not just repeat attribute-level observations. Consider relationship-specific prompt tuning.

### F5: Missing topic-level quality gates
Currently no quality gate between CDT construction and output. A topic that produces 0 statements and 0 gates passes silently. There should be a minimum quality threshold: at least N statements per topic, otherwise log a warning or retry with different clustering.

**Fix:** Add post-construction quality check in `build_character_cdts()`. Flag topics with `total_statements < 3` or `total_nodes == 1` (leaf-only tree).

---

## 16. Future Enhancements (Post-Baseline)

Implement AFTER paper-exact baseline benchmark is established with Qwen3-8B + θ=0.75. One enhancement at a time, measure delta for each.

### E1: Hypothesis Merge (extends D27)

**Current**: Near-duplicate hypotheses across clusters are detected but one is simply dropped.
**Enhancement**: Merge similar hypotheses (cosine > 0.90) via LLM into a single combined statement that preserves nuance from both sources. Add as `merge_similar_hypotheses` step between `_hypothesize` and `_summarize` in core.py.
**Cost**: One LLM call per merged pair. Replaces two downstream validation calls with one — net neutral or cheaper.
**Example**: "Kasumi encourages others during difficulties" + "Kasumi offers emotional support when bandmates struggle" → "Kasumi proactively offers encouragement and emotional support when those around her face difficulties"

### E2: Depth-3 Pruning (extends D28)

**Current**: Any hypothesis between threshold_reject and threshold_accept triggers recursion. With Haiku, this over-branches — 17/25 depth-3 nodes are empty.
**Enhancement**: Wait for paper-config baseline first. If θ=0.75 + Qwen3-8B produces clean trees (~10 nodes avg like the paper), pruning is unnecessary. If still bloated, implement Option A (raise MIN_PAIRS_FOR_TREE at deeper levels) or Option C (confidence floor — only recurse if gated confidence > 0.60).
**Note**: The paper uses depth 3 successfully — the issue is our model configuration, not depth 3 itself.

### E3: SOTA Model Exploration

**Current**: Paper models — Qwen3-Embedding-8B (surface), Qwen3-8B (generative), DeBERTa-v3-base (NLI), Haiku (hypothesis gen).
**Enhancement**: After baseline, explore one model swap at a time:
- **Embedding**: GTE-Qwen2, NV-Embed-v2, Jina v3 — may improve clustering quality
- **NLI**: DeBERTa-v3-large, ModernBERT-based NLI — may improve validation accuracy
- **Hypothesis gen**: Sonnet, Qwen3-8B-Instruct — may reduce verbosity/duplication
**Approach**: Ablation style. Swap one model, keep rest constant, compare benchmark score.

### E4: Configurable Topic Discovery

**Current**: Topics are hardcoded as `ATTRIBUTE_TOPICS = ("identity", "personality", "ability", "relationship")` — designed for character RP profiling from the paper.
**Enhancement**: Support two modes:
- **Fixed topics** (current): Caller provides topic list. Good for known domains.
- **Discovered topics**: Clustering step discovers topics organically from data — first cluster all observations, label clusters as topics, then build CDTs per discovered topic.
**Motivation**: Essential for delulu integration where paper's character RP topics don't apply. Delulu needs workflow patterns, debugging approach, communication style, tool preferences, etc.

### E5: Embedding Pre-Processing Refactor (IN PROGRESS)

**Problem**: `select_cluster_centers()` loads/unloads 8B models per topic — 16 load cycles for 8 topics. PyTorch doesn't release VRAM on `del model`. OOM on 32GB GPU.
**Solution**: Two-phase architecture:
- **Phase A (Embedding)**: Load each model ONCE, encode ALL observations for ALL topics in one forward pass, save embeddings, unload. Use subprocess per model to guarantee VRAM release.
- **Phase B (Tree Building)**: Use pre-computed embeddings for clustering + LLM calls. No model loading. `max_parallel=4` works (LLM calls only, no GPU).
- **DeBERTa** (715MB) stays loaded in-process throughout — small enough.
**Status**: PRD in progress (cody). Design decision: attribute topics share the same pairs, so surface/generative encoding happens once for the full dataset, not per topic.

### E6: llama.cpp Model Serving (Future)

**Enhancement**: Serve embedding models via llama.cpp server instead of PyTorch in-process. Separate process handles VRAM, pipeline calls HTTP endpoints.
**Benefits**: Process isolation (no OOM cascade), better memory management (GGUF quantization), persistent loading (model stays up between runs), concurrent access.
**Deferred**: E5 (subprocess approach) solves the immediate VRAM issue. llama.cpp is the long-term architecture if we need both models loaded simultaneously or want quantized inference.

---

*This document is the authoritative reference for CDT algorithm design in canopy-ai. For implementation details specific to a particular domain integration (e.g., delulu user profiling), refer to that project's own PRD.*
