# Phase 0–1 PRD: Baseline CDT Reproduction + Claude Migration

> 2026-03-27 — v1.1 (post-review)

## Objective

Reproduce the original CDT paper's benchmarks using Claude instead of GPT, with an adapter-based LLM abstraction that allows easy swapping between providers.

## Success Criteria

### Phase 0: Baseline Reproduction
- [ ] CDT construction runs end-to-end for Kasumi (smoke test)
- [ ] CDT construction runs for 3+ characters (Kasumi, Arisa, Yui)
- [ ] No `exec()` on LLM output anywhere — JSON parsing only
- [ ] All HF models load and run on GPU (DeBERTa NLI, Qwen3-Embedding-8B, Qwen3-8B)
- [ ] Output CDT packages saved to `packages/`
- [ ] Results documented in `artifacts/baseline-results.md`

### Phase 1: Claude Migration (stretch — only after Phase 0 passes)
- [ ] No OpenAI dependency for core functionality
- [ ] Benchmark scores within 5% of GPT-4.1 baselines (from paper)
- [ ] All LLM calls routed through adapter interface
- [ ] Adapter supports Claude (SDK), OpenAI (optional), and subprocess/CLI backends

## Non-Goals

- No package restructuring (that's Phase 2)
- No temporal CDT features (Phase 3)
- No CLI tool (Phase 2)
- No test suite beyond smoke tests (Phase 2)

## Architecture

### LLM Adapter Interface

```
llm.py
├── LLMAdapter (Protocol)          — generate(prompt, **kwargs) -> str
├── ClaudeCodeAdapter              — claude-agent-sdk (Max subscription)
├── AnthropicAPIAdapter (future)   — anthropic SDK (API key)
├── OpenAIAdapter (future)         — openai SDK
├── generate()                     — module-level convenience using default adapter
└── DEFAULT_MODEL                  — configurable default
```

**Protocol:**
```python
class LLMAdapter(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str: ...
```

**Why adapter pattern:**
- Swap providers without touching CDT logic
- Test with cheaper/faster models during development
- Future: local model support, batch processing
- No over-engineering: just a Protocol + 1 concrete implementation for now

### File Changes

| File | Change |
|------|--------|
| `llm.py` | NEW — adapter interface + ClaudeCodeAdapter |
| `constant.py` | NEW — env var loader (gitignored) |
| `codified_decision_tree.py` | Replace OpenAI imports/calls with `llm.generate()`, replace `exec()` with JSON parsing |
| `run_benchmark.py` | Replace OpenAI imports/calls (Phase 1) |
| `cdt_profiling.py` | Replace OpenAI imports/calls (Phase 1) |
| `build_cdt.sh` | Update default engine to claude-sonnet-4-6 |
| `CLAUDE.md` | Document setup, model paths, adapter usage |

### exec() Elimination

The original code uses `exec()` on LLM output in `make_hypothesis()`:
```python
exec(code_str, globals(), local_vars)  # UNSAFE
```

**Replace with:** Change prompt to request JSON output, parse with `json.loads()`.

### Model Downloads

| Model | HF Path | Size | Purpose |
|-------|---------|------|---------|
| DeBERTa NLI | `KomeijiForce/deberta-v3-base-rp-nli` | ~400MB | NLI classification |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | ~16GB | Surface embedding |
| Qwen3-8B | `Qwen/Qwen3-8B` | ~16GB | Generative embedding |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | Response generation (benchmark) |

**Storage:** `~/models/` with symlinks or HF cache. RTX 5090 (32GB VRAM) can hold DeBERTa + one 8B model at a time. May need to sequence model loading.

### VRAM Budget (RTX 5090 — 32GB)

| Model | VRAM (fp16) | When Loaded |
|-------|-------------|-------------|
| DeBERTa NLI | ~0.8GB | Always (CDT construction + benchmark) |
| Qwen3-Embedding-8B | ~16GB | CDT construction only |
| Qwen3-8B | ~16GB | CDT construction only |
| Llama-3.1-8B-Instruct | ~16GB | Benchmark only |

CDT construction needs DeBERTa + Qwen3-Embedding + Qwen3-8B simultaneously = ~33GB. **Tight fit.** Options:
1. Use fp16 with aggressive memory management
2. Use smaller Qwen models (0.6B) for initial smoke test
3. Offload one model to CPU

**Decision:** Start with the 8B models on fp16. If OOM, fall back to smaller models for smoke test.

## Implementation Phases

### Phase 0A: Infrastructure (current)
1. ~~Create branch `phase-0-baseline`~~ ✅
2. ~~Install claude-agent-sdk~~ ✅
3. ~~Create `constant.py`~~ ✅
4. ~~Verify LLM call works~~ ✅
5. Build proper `llm.py` with adapter pattern
6. Replace OpenAI in `codified_decision_tree.py`
7. Replace `exec()` with JSON parsing
8. Update `build_cdt.sh`

### Phase 0B: Model Setup
1. Download DeBERTa NLI model
2. Download Qwen3-Embedding-8B
3. Download Qwen3-8B
4. Verify models load on GPU

### Phase 0C: Smoke Test
1. Run CDT construction for Kasumi (1 character, 1 attribute)
2. Verify output CDT structure
3. Compare to existing `packages/Kasumi.cdt.v3.1.package.relation.pkl`
4. Fix any issues

### Phase 0D: Full Reproduction
1. Run CDT for 3 characters (Kasumi, Arisa, Yui)
2. Document results in `artifacts/baseline-results.md`
3. Compare to paper's statistics

### Phase 1: Benchmark Migration (after Phase 0 passes)
1. Replace OpenAI in `run_benchmark.py`
2. Replace OpenAI in `cdt_profiling.py`
3. Run benchmarks, compare scores
4. Document in `artifacts/claude-migration-results.md`

## Test Plan

### Smoke Tests (Phase 0)
| # | Scenario | Input | Expected Output |
|---|----------|-------|-----------------|
| 1 | LLM adapter generates text | Simple prompt | Non-empty string response |
| 2 | JSON hypothesis parsing | Hypothesis prompt | Valid action_hypotheses + scene_check_hypotheses lists |
| 3 | DeBERTa NLI loads | Model path | Model on GPU, inference works |
| 4 | Qwen3 embedding loads | Model path | Embeddings computed |
| 5 | CDT construction Kasumi identity | Character=Kasumi, topic=identity | CDT_Node with statements |
| 6 | CDT package saves | Full Kasumi run | .pkl file in packages/ |

### Benchmark Tests (Phase 1)
| # | Scenario | Input | Expected Output |
|---|----------|-------|-----------------|
| 7 | Benchmark Kasumi | CDT package | Score within 5% of paper |
| 8 | Wikification Yui | CDT + storyline | Readable profile |

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| VRAM OOM with 8B models | HIGH | Fall back to smaller models, or load sequentially |
| Claude output format differs from GPT | MEDIUM | Adjust prompts, add format validation |
| claude-agent-sdk subprocess overhead | LOW | Acceptable for Phase 0; batch in Phase 2 |
| HF model download failures | LOW | Retry, use cached models |
| JSON parsing failures from LLM | MEDIUM | Add retry with format reminder, fallback regex |

## Open Questions (Resolved)

1. **VRAM:** ~~Can we fit DeBERTa + Qwen3-Embedding-8B + Qwen3-8B simultaneously in 32GB?~~ **NO.** ~35-39GB needed. **Decision:** Use 0.6B models for smoke test (same as original authors did in `cdt_profiling.py`). 8B models require sequential loading (Phase 2).
2. **Model paths:** ~~HF cache or explicit?~~ **Decision:** Explicit `~/models/` directory.
3. **Llama model:** ~~GGUF vs HF format.~~ **Decision:** Defer to Phase 1. Llama is only needed for benchmark response generation. The HF version requires gated access (login needed).

## Review Findings (4 iterations)

### Fixed
- [x] Adapter Protocol pattern in `llm.py`
- [x] `extract_json()` helper with fallback parsing
- [x] `traverse()` bug — passed strings to batched `check_scene()` (wraps in lists now)
- [x] Mutable default args in CDT_Node (`[] → None`)
- [x] File context managers (bare `open()` → `with open()`)
- [x] `bypassPermissions` removed from ClaudeCodeAdapter
- [x] Unused `hf_token` import removed
- [x] Duplicate `defaultdict` import removed
- [x] `build_cdt.sh` updated for Claude

### Deferred to Phase 1
- [ ] `exec()` in `cdt_profiling.py` (will be fixed when migrating that file)
- [ ] `run_benchmark.py` / `cdt_profiling.py` OpenAI imports (will crash until migrated)
- [ ] Adapter `chat()` method for multi-turn (needed by `cdt_profiling.py`)
- [ ] Type annotations on CDT functions
- [ ] `if __name__ == "__main__"` guard for importability
- [ ] Temperature parameter in adapter

### Acknowledged Risks
- Prompt injection via dataset content → inherent to research task, mitigated by removing `bypassPermissions`
- Subprocess overhead (~50-100 calls × 3-5s each) → acceptable for Phase 0 smoke test
- `constant.py` HF_TOKEN optional (empty string default) → gated models will fail with auth error
