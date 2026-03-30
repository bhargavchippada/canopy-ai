# CDT Quality Improvement PRD — Reaching 75+ on RP

## Problem
CDT RP score is stuck at 70.66. All experiments (T-CDT, hybrid RAG, gen prompt v2, relationships) made it worse. Analysis shows CDT over-represents enthusiasm, under-represents confusion/hesitation/mundane moments.

## Root Cause Analysis
- 38% B-rate: predictions capture right character logic but wrong specific action (generic enthusiasm vs scene-specific reaction)
- 10% C-rate: CDT pushes toward confidence in scenes where character is confused/hesitant
- CDT has ~30 statements about enthusiasm, ~0 about confusion/quiet moments
- Hardcoded topics (identity, personality, ability, relationship) all converge on dominant trait

## Solution: Three-Pronged Improvement

### 1. Contrastive Hypothesis Generation ✅ IMPLEMENTED
Add instruction to `make_hypothesis_prompt` requiring at least one atypical/non-dominant behavior per cluster. This ensures CDTs capture confusion, hesitation, quiet reactions alongside enthusiasm.

### 2. Topic Discovery (E4) ✅ IMPLEMENTED
`discover_topics()` clusters observations and labels clusters as topics organically. Surfaces behavioral modes the hardcoded 4 topics miss. `build_character_cdts(discover_extra_topics=True)` adds discovered topics alongside standard ones.

### 3. Scene-Aware Gen Prompt (dialogue_v3) ✅ IMPLEMENTED
Gen prompt instructs model to match scene's emotional state rather than defaulting to dominant behavior. "If the moment calls for confusion, hesitation, or a quiet reaction, respond accordingly."

## Experiment Plan (one variable at a time)

### Phase 1: Contrastive hypotheses only
- Build CDT with contrastive instruction (no other changes)
- Benchmark on Kasumi RP (full 167 pairs)
- Compare A/B/C rates vs baseline

### Phase 2: Topic discovery
- Build CDT with discover_extra_topics=True, n_extra_topics=4
- Benchmark on Kasumi RP
- Inspect discovered topics — what did the clustering find?

### Phase 3: Combined (contrastive + discovery)
- Both improvements together
- Benchmark on Kasumi RP

### Phase 4: Gen prompt v3
- Use dialogue_v3 with best CDT from Phase 1-3
- Benchmark on Kasumi RP

### Phase 5: Cross-benchmark validation
- Best config on Kasumi → test on Arisa (different character, same artifact)
- LifeChoice benchmark (new behavioral prediction dataset)
- PersonaMem 32k with hybrid grounding (check if CDT improvements help factual recall too)

## Success Criteria
- Kasumi RP: ≥73 (3+ pt improvement over 70.66)
- Arisa RP: ≥66 (maintain cross-character generalization)
- No regression on other benchmarks

## Implementation Status
All code changes committed. Ready for CDT rebuild + benchmark.
