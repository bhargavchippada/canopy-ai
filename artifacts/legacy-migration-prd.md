# Phase 4: Legacy Migration PRD

## Objective

Migrate all remaining legacy files to use the canopy package and Claude (via claude-agent-sdk). Eliminate every OpenAI reference, `exec()` call, and `from constant import` statement from the codebase.

## Success Criteria

- [ ] ZERO `from openai import` or `import openai` in any `.py` or `.ipynb` file
- [ ] ZERO `exec()` calls on LLM output
- [ ] ZERO `from constant import` statements
- [ ] ZERO `OpenAI(api_key=...)` client instantiation
- [ ] `constant.py` deleted
- [ ] `run_benchmark.py` uses canopy.llm for all LLM calls
- [ ] `run_benchmark.py` uses Claude for generation (replaces Llama-3.1-8B-Instruct)
- [ ] `cdt_profiling.py` uses canopy modules (llm, embeddings, validation, prompts, wikify)
- [ ] `Wikification.ipynb` recreated using canopy imports
- [ ] Kasumi benchmark runs successfully with migrated code
- [ ] Kasumi NLI score reported and compared against paper CDT=84.25 on PoPiPa
- [ ] 100% test coverage on new/changed code
- [ ] 3+ rounds of review before and after implementation

## Files Affected

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `run_benchmark.py` | 346 | Legacy | Migrate (highest priority) |
| `cdt_profiling.py` | 652 | Legacy + SECURITY ISSUE | Migrate (exec() on L290) |
| `Wikification.ipynb` | ~120 | Legacy | Recreate using canopy |
| `constant.py` | ~5 | Legacy secrets | Delete |
| `src/canopy/llm.py` | 242 | Current | May need EVAL_MODEL export |
| `src/canopy/validation.py` | - | Current | Already migrated |
| `src/canopy/embeddings.py` | - | Current | Already migrated |
| `src/canopy/prompts.py` | - | Current | Already migrated |
| `src/canopy/wikify.py` | - | Current | Already has wikify_profile |

## Migration Strategy

### Phase 4A: run_benchmark.py (HIGHEST PRIORITY)

**Current state:** Uses OpenAI for eval scoring, Llama-3.1-8B-Instruct for response generation, duplicate CDT_Node class, `from constant import openai_key`.

**Migration plan:**

1. **Remove imports:** Delete `openai`, `from constant import`, `from openai import OpenAI`, Llama model loading (AutoTokenizer, AutoModelForCausalLM for generator)
2. **Replace `generate()` function:** Use `canopy.llm.generate()` with `model="claude-haiku-4-5"` for eval scoring
3. **Replace `generate_llama()` function:** Use `canopy.llm.generate()` with `model="claude-haiku-4-5"` for response generation (Claude replaces Llama)
4. **Replace `check_scene()` function:** Note: canopy.validation.check_scene() takes lists, legacy takes singles. Use CDTNode.traverse() which handles wrapping internally. For direct calls, wrap: `check_scene([text], [q])[0]`.
5. **Delete duplicate `CDT_Node` class (L285-345):** Use `canopy.core.CDTNode` with custom `_LegacyUnpickler` (see Pickle Compatibility section). Do NOT use `verify_cdt._register_unpickle_classes()`.
6. **Replace `load_ar_pairs()`:** Note: canopy.data.load_ar_pairs() requires 3 args (character, character2artifact, band2members). Must call `load_character_metadata()` first to get mappings.
7. **Replace `client = OpenAI(api_key=openai_key)`:** Not needed — Claude Max has no API key
8. **Fix `tqdm.notebook` import:** Use `tqdm.auto` (works in both notebook and CLI)
9. **Keep argparse CLI interface** but update defaults (engine → claude-haiku-4-5, eval_engine → claude-sonnet-4-6)
10. **Keep DeBERTa loading** for NLI scoring — this is the discriminator, not the generator

**Key architectural decision:** Replace Llama generation with Claude. The paper uses a local LLM for generation and OpenAI for eval. We use Claude for both (Haiku for generation, Sonnet for eval). This changes the generation quality but eliminates the local LLM dependency (saves ~16GB VRAM).

**Eval model selection:**
- `claude-haiku-4-5` for response generation (fast, cheap — replaces Llama-3.1-8B)
- `claude-sonnet-4-6` for NLI eval scoring (quality — replaces GPT-4.1)

**Functions to keep (use canopy):**
- `evaluate()` — restructure to use canopy imports. Hoist pickle loading OUT of evaluate() into benchmark() (currently loads per-call, wasteful). Use `extract_json()` for score extraction instead of regex.
- `benchmark()` — keep loop logic, replace tqdm.notebook. Load CDT package once, pass to evaluate().

**Functions to delete:**
- `generate()` — replaced by canopy.llm.generate()
- `generate_llama()` — replaced by canopy.llm.generate()
- `check_scene()` — replaced by canopy.validation.check_scene()
- `load_ar_pairs()` — replaced by canopy.data.load_ar_pairs()
- `CDT_Node` class — replaced by canopy.core.CDTNode

### Phase 4B: cdt_profiling.py

**Current state:** Full CDT construction + wikification pipeline. Has `exec()` on L290 (CRITICAL security issue). Uses OpenAI for all LLM calls. Duplicate implementations of every canopy function.

**Migration plan:**

1. **Remove all OpenAI imports and client:** Replace with canopy.llm
2. **Replace `exec()` (L290):** The legacy code asks the LLM to output Python lists, then `exec()`s them. The canopy version (`prompts.py`) already asks for JSON and uses `extract_json()`. Use `canopy.prompts.make_hypotheses_batch()` instead.
3. **Replace duplicate functions with canopy imports:**
   - `generate()` → `canopy.llm.generate()`
   - `generative_encode()` → `canopy.embeddings.generative_encode()`
   - `surface_encode()` → `canopy.embeddings.surface_encode()`
   - `select_cluster_centers()` → `canopy.embeddings.select_cluster_centers()`
   - `check_scene()` → `canopy.validation.check_scene()`
   - `check_statement_probs()` → `canopy.validation.check_statement_probs()`
   - `make_hypothesis()` → `canopy.prompts.make_hypotheses_batch()`
   - `validate_hypothesis()` → `canopy.validation.validate_hypothesis()`
   - `summarize_triggers()` → `canopy.prompts.summarize_triggers()`
   - `CDT_Node` → `canopy.core.CDTNode`
   - `CDT_Node` → `canopy.core.CDTNode`
   - **NOTE:** `wikify()` is LLM-powered narrative generation (NOT equivalent to `canopy.wikify.wikify_tree()` which is deterministic markdown). Keep `wikify()` as local code, replacing only the `generate()` call inside it with `canopy.llm.generate()`.
4. **Keep:** argparse CLI, storyline parsing (`fill_in_instruction`, `parse_scene_to_actions`), ICL examples, LLM-powered `wikify()` function
5. **Storyline parsing LLM calls:** The legacy code uses multi-turn ICL via `messages=[*icl_turns, ...]`. Since `canopy.llm.generate()` only supports single-prompt, flatten ICL examples into a single prompt string (prepend examples as text blocks). Fix the reversed turn order (legacy has assistant before user — Claude requires strict alternation).
6. **Model initialization:** Use `canopy.embeddings.init_models()` and `canopy.validation.init_models()`

**Result:** cdt_profiling.py becomes a thin CLI that:
- Parses storylines into action series (unique functionality)
- Delegates CDT construction to `canopy.core.CDTNode`
- Keeps LLM-powered narrative wikification (unique functionality, uses canopy.llm.generate)
- Uses canopy for embeddings, validation, prompts

### Phase 4C: Wikification.ipynb

**Current state:** Jupyter notebook using OpenAI, constant imports. Loads pickled CDT packages and wikifies them.

**Migration plan:** Create new notebook using canopy imports:
- `from canopy.wikify import wikify_profile, wikify_tree`
- `from canopy.llm import set_adapter, ClaudeCodeAdapter, generate`
- `from canopy.core import CDTNode`
- Load pickled CDT packages (existing .pkl files in packages/)
- Wikify using canopy.wikify functions
- Keep the interactive notebook format for exploration

### Phase 4D: Delete constant.py

After all migrations complete, delete `constant.py`. Verify with grep that no file imports from it.

## Pickle Compatibility

The existing `.pkl` files in `packages/` contain serialized `CDT_Node` objects from the legacy code.

**CRITICAL (from review):** `verify_cdt._register_unpickle_classes()` destructively replaces `sys.modules["__main__"]`, which would break `run_benchmark.py`'s own module-level state. Do NOT import this function.

**Safe approach:** Use a custom `pickle.Unpickler` with `find_class()` override:

```python
class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if name == "CDT_Node":
            return CDTNode  # canopy.core.CDTNode
        return super().find_class(module, name)

def load_cdt_package(path: str) -> dict:
    with open(path, "rb") as f:
        return _LegacyUnpickler(f).load()
```

This safely maps legacy `CDT_Node` → `canopy.core.CDTNode` without clobbering `sys.modules`. Pickle calls `__new__` then sets attributes directly (bypasses `__init__`), so attribute names (`statements`, `gates`, `children`, `depth`) must match — they do.

## Test Plan

### Automated Tests
- Unit tests for any new helper functions in migrated files
- Integration test: load a CDT package, traverse, verify statements returned
- Integration test: run benchmark on 5 test pairs (mocked LLM for unit, real Claude for integration)

### Manual Verification
| # | Scenario | Expected |
|---|----------|----------|
| 1 | `uv run python run_benchmark.py --character Kasumi` | Runs to completion, reports NLI score |
| 2 | NLI score comparison | Score reported, compared to paper CDT=84.25 |
| 3 | `grep -r "from openai" *.py *.ipynb` | Zero matches |
| 4 | `grep -r "from constant import" *.py *.ipynb` | Zero matches |
| 5 | `grep -r "exec(" *.py` (excluding .venv) | Zero matches in project code |
| 6 | `uv run python cdt_profiling.py --help` | Shows CLI help, no import errors |
| 7 | Wikification.ipynb | Loads and runs cells without errors |

## Implementation Phases

### Phase 4A (this PR): run_benchmark.py
- **Deliverable:** Migrated benchmark script using canopy + Claude
- **Validation:** Kasumi benchmark runs, NLI score reported
- **Commit after:** review passes

### Phase 4B: cdt_profiling.py
- **Deliverable:** Migrated profiling script, zero exec() calls
- **Validation:** CLI runs with --help, no import errors
- **Commit after:** review passes

### Phase 4C: Wikification.ipynb + constant.py
- **Deliverable:** New notebook using canopy, constant.py deleted
- **Validation:** Zero legacy references in codebase
- **Commit after:** final review passes

## Review Findings (3 rounds completed)

Reviewers: Architect, Security, Code Quality. All findings addressed in PRD updates above.

Key resolutions:
- **Pickle:** Custom Unpickler instead of sys.modules clobber (CRITICAL → resolved)
- **wikify():** Keep LLM-powered version as local code, not canopy.wikify (HIGH → resolved)
- **ICL multi-turn:** Flatten into single prompt string (HIGH → resolved)
- **check_scene() signature:** Document list wrapping (HIGH → resolved)
- **load_ar_pairs() signature:** Document metadata loading step (HIGH → resolved)
- **exec() removal:** Confirmed correct via extract_json (CRITICAL → resolved by design)
- **Score extraction:** Use extract_json instead of regex (MEDIUM → resolved)
- **Perf:** Hoist pickle loading out of evaluate() (MEDIUM → resolved)

## Risks

1. **Pickle compatibility:** Old .pkl files serialize `__main__.CDT_Node`. Resolved via custom Unpickler.
2. **Claude vs Llama generation quality:** Different model may produce different NLI scores. This is expected — we're benchmarking Claude, not reproducing paper results exactly.
3. **Claude vs GPT-4.1 eval quality:** Sonnet 4.6 replaces GPT-4.1 for scoring. May affect score calibration.
4. **Rate limits:** Benchmark runs many sequential LLM calls. May hit rate limits with Claude Max. Mitigation: retry/backoff in canopy.llm already handles transient failures.

## Open Questions

None — proceeding with implementation. User can interrupt if any assumption is wrong.
