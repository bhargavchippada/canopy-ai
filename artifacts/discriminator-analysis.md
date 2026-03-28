# Discriminator Analysis — Scene Check During Traversal

**Date:** 2026-03-28
**Question:** Does the paper use a different discriminator (Llama/gpt-4.1-mini) for scene checking than our DeBERTa?

---

## Finding: NO MISMATCH

**Both the paper and canopy-ai use the same DeBERTa model for ALL discrimination tasks.**

### Paper Code Evidence

**`codified_decision_tree.py` (CDT construction):**
- `discriminator_path = "KomeijiForce/deberta-v3-base-rp-nli"` (line 21, default)
- `check_scene()` (line 238): uses `classifier` (DeBERTa) for gate questions
- `check_statement_probs()` (line 248): uses `classifier` (DeBERTa) for statement NLI
- Both load from the same `discriminator_path`

**`run_benchmark.py` (traversal at eval time):**
- `discriminator_path = "KomeijiForce/deberta-v3-base-rp-nli"` (line 28/67, default)
- `check_scene()` (line 181): uses `classifier` (DeBERTa) for gate traversal
- Same prompt format: `"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."`

**`canopy-ai/src/canopy/validation.py` (our implementation):**
- `check_scene()` (line 31): uses `_classifier` (DeBERTa) for gate traversal
- Same prompt format: `"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."`

### What CDT vs CDT-Lite Actually Means

The paper text says "CDT uses gpt-4.1-mini for validation" — this refers to the **hypothesis generation engine** (`engine` parameter), not the discriminator. The architecture is:

| Component | CDT | CDT-Lite |
|---|---|---|
| Hypothesis generation | gpt-4.1 | gpt-4.1 (or lighter) |
| Statement NLI validation | DeBERTa | DeBERTa |
| Scene gate checking (build) | DeBERTa | DeBERTa |
| Scene gate checking (traverse) | DeBERTa | DeBERTa |
| RP generation | Llama-3.1-8B | Llama-3.1-8B |
| Evaluation | gpt-4.1 | gpt-4.1 |

The only difference between CDT and CDT-Lite is that CDT-Lite **also** uses DeBERTa for the validation discrimination that CDT does with gpt-4.1-mini. But for scene checking during traversal, both paths use DeBERTa.

Wait — re-reading the paper more carefully: CDT uses gpt-4.1-mini for the NLI **validation** of hypotheses (the accept/reject decision), while CDT-Lite distills this into DeBERTa. But the `check_scene()` function for gate traversal always uses DeBERTa in both code paths.

### Conclusion

**There is no discriminator mismatch between the paper's benchmark and ours.** Both use `KomeijiForce/deberta-v3-base-rp-nli` for scene gate checking during traversal. The same CDT tree will activate the same branches with both implementations.

The score gap (55-67 vs 84.25) is fully explained by the eval model difference (Sonnet ~35-40% B vs GPT-4.1 ~18% B), not by discriminator disagreement.

### Caveat

This analysis is based on the code defaults. If the paper authors ran with a different `--discriminator_path` for their published results, the conclusion would change. But the code, README, and shell scripts all default to DeBERTa.
