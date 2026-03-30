# CDT Improvements PRD — Dual-Domain Excellence

> Goal: CDT should excel at BOTH behavioral prediction AND factual recall.
> Phase: Post-paper-parity iteration.

---

## 1. Problem Statement

CDT achieves paper parity on behavioral prediction (+4-5 pts on Bandori RP) but hurts factual recall (-5 pts on PersonaMem). For delulu's user profiling use case, both capabilities are needed — the Sidekick must predict what the user would do AND remember what they said.

### Current Scores

| Benchmark | Task Type | Baseline (no CDT) | CDT-augmented | Delta |
|---|---|---|---|---|
| Bandori RP (Kasumi) | Behavioral prediction | ~66 | 70.66 | **+4.66** |
| PersonaMem 32k | Factual recall | 76.9% | 71% | **-5.9%** |

### Target

| Benchmark | Target | Success Criterion |
|---|---|---|
| Bandori RP | ≥70.0 | Maintain behavioral prediction advantage |
| PersonaMem 32k | ≥76.9% | Match or beat full-context baseline |
| LifeChoice (new) | >67.95% (GPT-4 baseline) | Beat frontier on decision prediction |

## 2. Root Cause Analysis

CDT compresses specific facts into generalized behavioral patterns:
- **"User mentioned they love hiking on March 15"** → abstracted to **"tends to prefer outdoor activities"**
- The generalization helps behavioral prediction (correctly predicts outdoor preference in new scenarios)
- The generalization hurts factual recall (can't answer "when did the user mention hiking?")

This is the **semantic vs. episodic memory** divide (Tulving, 1972). CDT is pure semantic memory. We need to add episodic memory without losing the semantic advantage.

## 3. Architecture: CDT-Guided RAG (Hybrid)

### Research Foundation

| Paper | Key Finding | How We Apply It |
|---|---|---|
| CharMap/LifeChoice (2404.12138) | Abstract profile guides specific memory retrieval → +5.03% | CDT gates guide RAG retrieval |
| Zep/Graphiti (2501.13956) | 3-tier KG: episodic + semantic + community → 94.8% DMR | Two-tier: CDT (semantic) + RAG (episodic) |
| Mnemis (2602.15313) | Dual-route retrieval: fast similarity + deep traversal → 93.9% | CDT traverse as deep route, RAG as fast route |
| AriGraph (2407.04363) | Semantic + episodic integration outperforms either alone | Hybrid > CDT-only or RAG-only |
| PURE (2502.14541) | Profile + raw data > profile alone | CDT + observations > CDT alone |

### Core Architecture

```
Query → embed(query)
       ┌──────────────┐
       │  CDT Traverse │ → behavioral_statements + active_gates
       └──────┬───────┘
              │ gates as retrieval filters
       ┌──────▼───────┐
       │ EpisodicIndex │ → top-k raw observations (factual)
       └──────┬───────┘
              │
       ┌──────▼───────────┐
       │ Merged Grounding  │ → behavioral rules + specific facts
       └──────────────────┘
              │
       ┌──────▼──────┐
       │  LLM Prompt  │ → response grounded in both
       └─────────────┘
```

### Why CDT-Guided > Pure RAG

- Pure RAG retrieves by similarity alone → noisy, returns similar scenes not behavioral rules
- CDT gates focus retrieval on the relevant behavioral domain → precision
- CDT rules provide abstract behavioral scaffolding → consistency
- Raw observations provide specific facts → factual accuracy

## 4. Implementation Phases

### Phase 1: E7 — Episodic Memory Layer ✅ COMPLETE

**Status:** Implemented, 100% test coverage, 33 tests passing.

| Component | File | Lines | Status |
|---|---|---|---|
| EpisodicIndex | src/canopy/episodic.py | ~90 | ✅ Complete |
| HybridGrounding | src/canopy/episodic.py | ~50 | ✅ Complete |
| format_grounding | src/canopy/episodic.py | ~30 | ✅ Complete |
| _traverse_with_gates | src/canopy/episodic.py | ~20 | ✅ Complete |
| Tests | tests/test_episodic.py | ~330 | ✅ 33 tests, 100% coverage |

### Phase 2: Benchmark Validation — IN PROGRESS

Run hybrid grounding on both benchmark types:

1. **PersonaMem 32k** — all 589 questions
   - Baseline: 76.9% (full context, no CDT)
   - CDT-only: 71% (-5.9%)
   - Hybrid target: ≥76.9%
   - Strategy: CDT behavioral rules + RAG-retrieved conversation turns

2. **Bandori RP Kasumi** — full 167 pairs
   - CDT-only: 70.66
   - Hybrid target: ≥70.0
   - Strategy: CDT behavioral rules + similar past scenes

3. **Tuning parameters:**
   - top_k: {5, 10, 20}
   - gate_threshold: {0.2, 0.3, 0.5}
   - With/without gate filtering

### Phase 3: E1 — Hypothesis Merge

From canopy-design.md §16. Reduces redundancy in CDT statements.

- Near-duplicate hypotheses (cosine > 0.90) merged via LLM
- Single combined statement preserves nuance from both sources
- Added between `_hypothesize` and `_summarize` in core.py
- Expected impact: cleaner grounding, fewer redundant statements
- Measure delta on Kasumi RP and PersonaMem

### Phase 4: LifeChoice Benchmark Integration

New behavioral prediction benchmark (EMNLP 2025 Findings):
- 1,462 decision points from 388 books
- MCQ format (4 choices)
- Tests persona-driven decision prediction
- Frontier models: GPT-4 67.95%, Claude-3 67.13%
- CharMap (similar to CDT-guided RAG): +5.03%

Integration plan:
1. Add `canopy.datasets.lifechoice` dataset loader
2. Build CDT for book characters from LifeChoice's expert-written analyses
3. Run baseline (no CDT) + CDT-only + hybrid on full dataset
4. Compare against CharMap's results

### Phase 5: Cross-Character Validation on CDT Paper Benchmarks

Expand testing beyond Kasumi/Arisa/Haruhi:
- 85 pre-built CDT pickle files available
- 16 artifacts (8 Bandori + 8 Fandom)
- Test hybrid approach on best (PoPiPa: 88.38) and worst (FMA: 57.26)
- Verify CDT-guided RAG generalizes across character types

### Phase 6: Deferred Enhancements

| Enhancement | Source | Description | When |
|---|---|---|---|
| E2: Depth-3 Pruning | canopy-design §16 | Reduce over-branching | After Phase 2 results |
| E3: SOTA Models | canopy-design §16 | Explore better embedding/NLI/gen models | After Phase 4 |
| E4: Topic Discovery | canopy-design §16 | Auto-discover topics from data | Before delulu build-profile |
| E5: Embedding Refactor | canopy-design §16 | Two-phase subprocess architecture | IN PROGRESS |
| T-CDT | canopy-design §5 | Temporal weighting for evolving behavior | After baseline established |

## 5. Success Criteria

### Must Have (Phase 2)
- [ ] PersonaMem hybrid ≥76.9% (matches full-context baseline)
- [ ] Bandori RP hybrid ≥70.0 (maintains CDT advantage)
- [ ] All tests passing, episodic.py 100% coverage

### Should Have (Phase 3-4)
- [ ] E1 hypothesis merge implemented and measured
- [ ] LifeChoice benchmark integrated
- [ ] CDT-guided RAG beats pure CDT on LifeChoice

### Nice to Have (Phase 5-6)
- [ ] Cross-character validation on 10+ characters
- [ ] Topic discovery for delulu integration
- [ ] T-CDT prototype

## 6. Risks

| Risk | Mitigation |
|---|---|
| Hybrid grounding adds token cost | Limit top_k, gate filtering reduces retrieval scope |
| Gate filtering too aggressive | Fallback to unfiltered when all observations eliminated |
| n=10 results mislead | NEVER report without full dataset run |
| LifeChoice dataset format incompatible | Read dataset README first, check format before building |

## 7. Dependencies

- **canopy-ai**: Primary implementation target
- **delulu**: Consumer of CDT profiles (Phase 3 integration)
- **PersonaMem dataset**: HuggingFace download (already cached)
- **LifeChoice dataset**: GitHub download (new)
- **CDT paper repo**: Pre-built pickle files for cross-character testing

## 8. Open Questions

1. Should hybrid grounding always combine CDT + RAG, or should we add a query-type router? (Start with always-hybrid, add routing if needed based on Phase 2 results)
2. For PersonaMem, should the RAG index cover full conversation or just the end_index-truncated portion? (Truncated — matches the task constraint)
3. Should we build character CDTs from LifeChoice's expert analyses, or from the book text directly? (Expert analyses — they're high quality and readily available)
