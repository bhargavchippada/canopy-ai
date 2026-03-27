# Phase 0: Baseline Results

> 2026-03-27

## Smoke Test: Kasumi CDT (Claude + 0.6B models)

**Status: PASSED**

### Configuration
- **LLM:** claude-sonnet-4-6 via claude-agent-sdk (Max subscription)
- **Embeddings:** Qwen3-Embedding-0.6B (surface) + Qwen3-0.6B (generative)
- **NLI:** DeBERTa-v3-base-rp-nli
- **GPU:** RTX 5090 (32GB), ~3.1GB VRAM used
- **max_depth:** 2
- **Thresholds:** accept=0.8, reject=0.5, filter=0.8

### Output
- **File:** `packages/Kasumi.cdt.v3.1.package.relation.pkl` (19,157 bytes)
- **4 attribute topics:**
  - Kasumi's identity: 6 statements, 2 gates, 2 children
  - Kasumi's personality: 7 statements, 1 gate, 1 child
  - Kasumi's ability: 6 statements, 1 gate, 1 child
  - Kasumi's relationship: 7 statements, 1 gate, 1 child
- **4 relationship topics:**
  - Kasumi × Rimi: 1 statement, 4 gates
  - Kasumi × Tae: 4 statements, 0 gates
  - Kasumi × Saaya: 6 statements, 2 gates
  - Kasumi × Arisa: 5 statements, 1 gate

### Statement Quality (samples)
- "Kasumi tends to express her emotional states at full intensity and without apparent self-censorship"
- "When the group faces an obstacle, Kasumi tends to take initiative by proposing a proactive action"
- "Kasumi tends to respond to Arisa's tsundere behavior with warmth and affection"

Statements are non-assertive, general, and character-specific — matching the CDT paper's style.

### Timing
- **Total run:** ~1.5 hours
- Hypothesis generation: ~35-40s per cluster (Claude call), 8 clusters × 4 attributes
- Embedding: <1s per batch (0.6B model is fast)
- Validation: <1s per batch (DeBERTa on GPU)

### Comparison to Original (GPT-4.1 + 8B models)
- Original Kasumi package: 14,580 bytes
- Claude + 0.6B: 19,157 bytes (31% larger — more statements extracted)
- Quality appears comparable; Claude tends to generate slightly more detailed/nuanced hypotheses

### Issues Found
- `head -80` pipe kills process before pickle write (SIGPIPE)
- Branch switching by other sessions can revert working directory
- Module-level argparse prevents clean imports for testing
- VRAM budget confirmed: 8B models cannot load simultaneously (need sequential loading)

## Next Steps
- [ ] Run Kasumi with full depth (max_depth=3) to compare tree structure
- [ ] Run Arisa and Yui for 3-character validation
- [ ] Compare CDT node counts to paper's reported statistics
- [ ] Migrate run_benchmark.py for score comparison (Phase 1)
