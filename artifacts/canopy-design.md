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
     - Noise points (HDBSCAN label -1) are discarded
  3. Filter: discard clusters with fewer than min_cluster_size items
  4. Label each cluster via LLM: "Given these examples, what domain/theme do they represent?"
Output: K clusters, each with a label and list of member observations
```

The embedding model is configurable (see Section 10). HDBSCAN is the primary clustering algorithm, auto-discovering cluster count from data density. KMeans is available as a fallback for datasets where HDBSCAN produces too many or too few clusters (e.g., uniformly distributed embeddings). The clustering interface allows substitution of any algorithm.

### Step 2: Per-Cluster Hypothesis Generation

For each cluster, the LLM generates candidate behavioral hypotheses -- testable statements about patterns in the data.

```
Input:  Cluster label + member BehavioralObservation items
Process:
  LLM prompt: "Given these behavioral observations from the domain '{label}',
  extract testable behavioral hypotheses. Each hypothesis should be specific
  enough to validate against new evidence."
Output: List of hypotheses per cluster
```

Each hypothesis includes:
- `statement`: The testable behavioral claim
- `cluster_id`: Which cluster it belongs to
- `source_ids`: Which input pairs contributed to it
- `gate_condition`: Natural language condition (null if universal) -- see Section 3

### Step 3: Evidence-Based Validation

Each hypothesis is validated against evidence items (the original observations from that cluster, or a sampled subset for large clusters).

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

New data does not require rebuilding the entire CDT. The incremental algorithm:

1. New observations are embedded
2. Compute cluster centroids as the mean of member embeddings (an approximation for HDBSCAN, which does not produce centroids natively). Assign each observation to the cluster with highest cosine similarity to its centroid.
3. If distance > `new_cluster_threshold` (default: 2x the average intra-cluster distance), flag as a potential new cluster candidate
4. If 5+ flagged observations form a dense group (HDBSCAN on flagged observations with `min_cluster_size`), create a new cluster
5. Re-validate ALL hypotheses in affected clusters using the updated evidence set (including new observations)
6. If `temporal_confidence` of any hypothesis drops below `cdt_reject_threshold` --> supersede (see Supersession Tracking above)
7. If new clusters were formed --> run hypothesis generation + validation on them
8. **Full rebuild trigger:** when > 30% of clusters are new, or > 50% of hypotheses have changed status, trigger a full CDT rebuild instead of incremental update

---

## 6. Two-Layer Data Approach

An optimization over standard CDT that separates hypothesis generation from validation.

### Standard CDT Flow

```
Raw data --> Extract hypotheses --> Validate hypotheses (against same raw data)
```

Standard CDT does both extraction and validation from the same raw data. This is expensive: the LLM must process all raw observations twice.

### Two-Layer Flow

```
Layer 1 (summaries) --> Extract hypotheses (cheap -- summaries are pre-extracted)
Layer 2 (raw data)  --> Validate hypotheses (targeted -- only relevant evidence)
```

### How It Works

- **Layer 1 (hypotheses):** Pre-extracted behavioral statements from upstream summarization. These serve as candidate hypotheses. In the CDT paper's terms, these are the "storyline summaries" from which character traits are hypothesized.
- **Layer 2 (validation):** Raw evidence items for validating or rejecting those hypotheses. Each item is classified as supporting, contradicting, or irrelevant.

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

### Per-Cluster Validation

Each cluster's hypotheses are validated against evidence matching that cluster's embedding neighborhood -- not against all evidence globally. This is semantically correct: debugging hypotheses should be validated against debugging evidence, not architecture evidence.

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
| `cdt_accept_threshold` | 0.75 | 0.5-1.0 | Confidence threshold for accepting a hypothesis |
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

## 12. Design Decisions Log

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

---

## 13. Research References

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
| Hypothesis source | Extracted from raw data | Two-layer: summaries (hypotheses) + raw (validation) |
| Clustering | Predefined character traits | Embedding-based HDBSCAN (auto-discovers structure and cluster count) |
| Traversal | `exec()` gate evaluation | Cosine similarity (no code execution) |

---

*This document is the authoritative reference for CDT algorithm design in canopy-ai. For implementation details specific to a particular domain integration (e.g., delulu user profiling), refer to that project's own PRD.*
