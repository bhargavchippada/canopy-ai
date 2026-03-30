# E7: Episodic Memory Layer — CDT-Guided RAG

> Enhancement to canopy-ai CDT pipeline.
> Goal: CDT should excel at BOTH behavioral prediction AND factual recall.

---

## 1. Problem Statement

CDT compresses specific facts into generalized behavioral patterns ("tends to prefer X when Y"). This is excellent for behavioral prediction (+4-5 pts on Bandori RP) but harmful for factual recall (-5 pts on PersonaMem), because the specific details get abstracted away.

**Current results:**
| Benchmark | Baseline (no CDT) | CDT-augmented | Delta |
|---|---|---|---|
| Bandori RP (Kasumi) | ~66 | 70.66 | **+4.66** |
| PersonaMem (32k) | 76.9% | 71% | **-5.9%** |

**Target:** CDT-augmented approach should match or beat baseline on BOTH benchmarks.

## 2. Architecture: CDT-Guided RAG

Inspired by CharMap (LifeChoice paper, +5.03%), Zep (3-tier KG, 94.8% DMR), and Mnemis (dual-route, 93.9 LoCoMo).

### Core Insight

CDT gates identify WHICH behavioral domain is relevant to a query. Use that domain as a retrieval filter over raw observations. Abstract profile guides retrieval of specific memories — don't replace them.

### Data Flow

```
Query arrives
  1. Embed query (reuse Phase A embeddings model)
  2. CDT Traverse → active statements + gate conditions
  3. Gate conditions → filter EpisodicIndex to relevant domain
  4. EpisodicIndex → top-k raw BehavioralObservation items (factual)
  5. Merge: CDT statements (behavioral) + RAG results (factual)
  6. Generate response with merged grounding
```

### Why CDT-Guided > Pure RAG

Pure RAG retrieves by similarity alone — noisy for behavioral prediction (returns similar scenes, not behavioral rules). CDT-guided RAG:
- Uses CDT gates as semantic filters → focused retrieval
- Preserves CDT's structured behavioral rules → behavioral prediction stays strong
- Falls back to pure similarity when no gates activate → factual recall covered

## 3. Components

### 3.1 EpisodicIndex

New class in `src/canopy/episodic.py`.

```python
@dataclass(frozen=True)
class EpisodicIndex:
    """Embedding index over raw BehavioralObservation items.

    Supports top-k retrieval with optional gate-condition filtering.
    Reuses Phase A pre-computed embeddings — no new GPU computation.
    """
    observations: list[BehavioralObservation]
    embeddings: np.ndarray  # shape (N, embed_dim), from Phase A cache

    @classmethod
    def from_embedding_cache(
        cls,
        observations: list[BehavioralObservation],
        embedding_cache: EmbeddingCache,
    ) -> EpisodicIndex:
        """Build index from existing Phase A embeddings."""
        ...

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        gate_filter: str | None = None,
        gate_embeddings: np.ndarray | None = None,
        gate_threshold: float = 0.3,
    ) -> list[tuple[BehavioralObservation, float]]:
        """Retrieve top-k observations by cosine similarity.

        If gate_filter provided, pre-filter observations to those
        semantically relevant to the gate condition (cosine > gate_threshold).
        This focuses retrieval on the behavioral domain CDT identified.

        Returns list of (observation, similarity_score) tuples.
        """
        ...
```

**Key design decisions:**
- Reuses Phase A embeddings — zero additional GPU cost
- Gate filtering is optional — falls back to pure similarity for factual queries
- Returns raw `BehavioralObservation` items — preserves full text, timestamps, source_ids
- Immutable dataclass — no mutation

### 3.2 HybridGrounding

New function in `src/canopy/episodic.py`.

```python
@dataclass(frozen=True)
class GroundingResult:
    """Combined CDT + RAG grounding for a query."""
    behavioral_statements: list[str]  # From CDT traverse
    factual_observations: list[BehavioralObservation]  # From RAG
    active_gates: list[str]  # Which CDT gates activated
    retrieval_scores: list[float]  # Similarity scores for RAG results

def hybrid_ground(
    query: str,
    topic2cdt: dict[str, CDTNode],
    episodic_index: EpisodicIndex,
    *,
    top_k: int = 10,
    gate_threshold: float = 0.3,
    embed_fn: Callable[[str], np.ndarray] | None = None,
) -> GroundingResult:
    """Ground a query using CDT behavioral rules + RAG factual retrieval.

    1. Traverse all CDT trees → collect statements + active gates
    2. Use active gates to filter EpisodicIndex retrieval
    3. Retrieve top-k relevant observations
    4. Return combined grounding
    """
    ...
```

### 3.3 Integration with Benchmarks

**PersonaMem integration:**
```python
# Current (CDT-only, hurts):
grounding = cdt.traverse(question_context)
prompt = f"Given this profile:\n{grounding}\n\nAnswer: {question}"

# New (hybrid):
result = hybrid_ground(question, topic2cdt, episodic_index)
prompt = f"""Given this behavioral profile:
{chr(10).join(result.behavioral_statements)}

Relevant conversation history:
{chr(10).join(obs.text for obs in result.factual_observations)}

Answer: {question}"""
```

**Bandori RP integration:**
```python
# Traverse still works as before for behavioral prediction
# Hybrid adds source observations as additional context
result = hybrid_ground(scene, topic2cdt, episodic_index)
prompt = f"""Character profile:
{chr(10).join(result.behavioral_statements)}

Similar past situations:
{chr(10).join(obs.text for obs in result.factual_observations[:5])}

Given this scene, what does {character} do next?"""
```

## 4. Implementation Plan

### Phase 1: EpisodicIndex (est. 200 lines + tests)
1. `EpisodicIndex` dataclass with `from_embedding_cache()` constructor
2. `retrieve()` with cosine similarity + optional gate filtering
3. Unit tests: index construction, retrieval accuracy, gate filtering
4. Integration test: build from real Phase A cache

### Phase 2: HybridGrounding (est. 150 lines + tests)
1. `GroundingResult` dataclass
2. `hybrid_ground()` function
3. `format_grounding()` helper — renders GroundingResult to prompt text
4. Unit tests: grounding merge, gate routing, edge cases
5. Integration test: full pipeline with real CDT + index

### Phase 3: Benchmark Evaluation
1. PersonaMem: hybrid_ground for all 589 questions, compare vs baseline (76.9%) and CDT-only (71%)
2. Bandori RP: hybrid_ground for Kasumi, compare vs CDT-only (70.66)
3. Run FULL datasets (never n=10)
4. Record per-question details for analysis

### Phase 4: Tuning
1. top_k sensitivity: 5, 10, 20
2. gate_threshold sensitivity: 0.2, 0.3, 0.5
3. With/without gate filtering (pure RAG vs CDT-guided RAG)
4. Measure: does CDT-guided RAG beat pure RAG on behavioral prediction?

## 5. Success Criteria

| Benchmark | Current Best | Target | Metric |
|---|---|---|---|
| PersonaMem 32k | 76.9% (baseline, no CDT) | ≥76.9% (hybrid matches or beats) | MCQ accuracy |
| Bandori RP Kasumi | 70.66 (CDT-only) | ≥70.0 (hybrid maintains) | NLI score |

**The hybrid approach succeeds if it matches baseline on factual recall while maintaining CDT's behavioral prediction advantage.** Ideally it beats both baselines.

## 6. Files Changed

| File | Change |
|---|---|
| `src/canopy/episodic.py` | NEW — EpisodicIndex, HybridGrounding |
| `tests/test_episodic.py` | NEW — unit + integration tests |
| `src/canopy/__init__.py` | Export new public API |
| `run_benchmark.py` | Add `--grounding hybrid` mode |
| `scripts/personamem_benchmark.py` | Add hybrid grounding path |

## 7. Non-Goals

- Full knowledge graph (Zep/Graphiti) — overkill for current needs
- Query-type router/classifier — start with always-hybrid, add routing if needed
- T-CDT temporal weighting — orthogonal enhancement, implement separately
- New embedding models — reuse existing Phase A embeddings

## 8. Research References

- CharMap/LifeChoice (2404.12138): Abstract description guides specific memory retrieval → +5.03%
- Zep/Graphiti (2501.13956): 3-tier KG with episodic+semantic+community, 94.8% DMR
- Mnemis (2602.15313): Dual-route System-1/System-2, 93.9 LoCoMo
- AriGraph (2407.04363): Semantic+episodic integration outperforms either alone
- PURE (2502.14541): Profile + raw data > profile alone
