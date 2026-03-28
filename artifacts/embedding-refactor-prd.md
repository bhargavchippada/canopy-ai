# PRD: Two-Phase Embedding Architecture

> Separate embedding pre-computation from CDT tree building to eliminate GPU OOM failures.

## Problem

`select_cluster_centers()` loads/unloads 8B GPU models (~16GB each) **per CDTNode._build() call**. With 8+ topics (4 attribute + N relationship) and max_depth=3 recursion, that's 20+ model load/unload cycles per character build. PyTorch does NOT reliably release VRAM on `del model + gc.collect() + torch.cuda.empty_cache()`. Residual VRAM accumulates, causing OOM when loading the second model (surface 16GB + leaked VRAM > 32GB GPU).

## Solution

Two-phase architecture:
- **Phase A (Embedding):** Load each model ONCE in a subprocess, encode ALL pairs in one forward pass, save to disk, exit subprocess (OS guarantees VRAM release).
- **Phase B (Tree Building):** Use pre-computed embeddings for clustering. No GPU model loading. max_parallel=4 works (LLM API + DeBERTa only).

## Success Criteria

- [ ] Each 8B model loads exactly ONCE per character build (not per topic, not per depth)
- [ ] Embedding subprocesses exit cleanly, guaranteeing VRAM release between models
- [ ] `select_cluster_centers()` uses pre-computed embeddings when cache provided (zero GPU model loads)
- [ ] Attribute topics share the same embeddings (same pairs → same encodings, no redundant work)
- [ ] Relationship CDT pair filtering correctly indexes into pre-computed embeddings
- [ ] Recursive child nodes correctly index into pre-computed embeddings via `_embed_idx`
- [ ] DeBERTa (715MB) remains in the main process for validation (unchanged)
- [ ] Full CDT build completes without OOM on RTX 5090 (32GB) with 8B models
- [ ] No behavioral change in tree structure — same inputs produce same CDTs
- [ ] All existing tests pass unchanged (mocked `_embedder` path unaffected)
- [ ] 100% test coverage on new code
- [ ] 0.6B model path still works (backward compatible)
- [ ] max_parallel=4 tree building works — no lock contention, no model loading in Phase B

## Architecture

### Current Flow (per-node model loading)

```
codified_decision_tree.py
  ├── init_embedding_models(paths)          ← stores paths only
  ├── init_validation_models(path)          ← loads DeBERTa (~715MB)
  └── build_character_cdts(max_parallel=4)
        └── ThreadPoolExecutor(4 workers)
              ├── CDTNode("identity", ALL pairs)
              │     └── _build()
              │           ├── select_cluster_centers()     ← LOAD surface 16GB → encode → UNLOAD
              │           │                                ← LOAD generator 16GB → encode → UNLOAD
              │           ├── hypothesize + validate
              │           └── CDTNode(child, filtered_pairs)
              │                 └── select_cluster_centers()  ← LOAD/UNLOAD AGAIN (same pairs!)
              ├── CDTNode("personality", ALL pairs)       ← LOAD/UNLOAD AGAIN (same data!)
              ├── CDTNode("ability", ALL pairs)            ← LOAD/UNLOAD AGAIN
              └── CDTNode("relationship:Bob", filtered)   ← LOAD/UNLOAD AGAIN
```

**Problem:** 20+ load cycles. 4 attribute topics encode the SAME pairs redundantly.
PyTorch VRAM leaks accumulate → OOM.

### Proposed Flow (two-phase)

```
codified_decision_tree.py
  ├── init_validation_models(path)          ← loads DeBERTa (~715MB), unchanged
  │
  ├── PHASE A: precompute_embeddings(character, ALL pairs)
  │     ├── Subprocess 1: surface embedding
  │     │     ├── Load Qwen3-8B SentenceTransformer (~16GB)
  │     │     ├── Encode ALL actions in one batched pass → surface.npy
  │     │     └── Exit (OS reclaims ALL GPU memory)
  │     │
  │     └── Subprocess 2: generator embedding
  │           ├── Load Qwen3-8B CausalLM (~16GB)
  │           ├── Encode ALL scenes (with character suffix) → generator.npy
  │           └── Exit (OS reclaims ALL GPU memory)
  │
  │     Result: EmbeddingCache(surface=(N,D_s), generator=(N,D_g))
  │
  ├── Stamp _embed_idx on each pair
  │
  └── PHASE B: build_character_cdts(max_parallel=4, embedding_cache=cache)
        └── ThreadPoolExecutor(4 workers)  ← NO GPU model loading, NO lock contention
              ├── CDTNode("identity", ALL pairs)
              │     └── cached_select_cluster_centers()  ← numpy index lookup, NO GPU
              │           └── CDTNode(child, filtered_pairs)
              │                 └── cached_select_cluster_centers()  ← numpy subset, NO GPU
              ├── CDTNode("personality", ALL pairs)   ← same cache, zero cost
              ├── CDTNode("ability", ALL pairs)        ← same cache, zero cost
              └── CDTNode("relationship:Bob", filtered) ← cache.subset(indices)
```

**Result:** Each 8B model loads exactly ONCE. All 4 attribute topics share the same
pre-computed embeddings (zero redundancy). Subprocess exit guarantees VRAM release.
Phase B is pure LLM API + DeBERTa + numpy — max_parallel=4 runs without contention.

## API Design

### New: `EmbeddingCache` (frozen dataclass)

```python
# src/canopy/embeddings.py

@dataclass(frozen=True)
class EmbeddingCache:
    """Pre-computed embeddings for all training pairs.

    Arrays are made read-only on construction to prevent accidental mutation
    across concurrent threads in Phase B. _embed_idx is a reserved key on pair
    dicts — do not use it for other purposes.
    """
    surface: np.ndarray      # (N, D_surface) L2-normalized, read-only
    generator: np.ndarray    # (N, D_gen) L2-normalized, read-only
    _document: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Make arrays read-only to prevent cross-thread mutation
        object.__setattr__(self, 'surface', np.array(self.surface, copy=False))
        self.surface.flags.writeable = False
        object.__setattr__(self, 'generator', np.array(self.generator, copy=False))
        self.generator.flags.writeable = False

    @property
    def document(self) -> np.ndarray:
        """Combined embeddings for clustering (gen + surface concatenated).
        Cached on first access to avoid redundant concatenation across threads."""
        if self._document is None:
            doc = np.concatenate([self.generator, self.surface], axis=-1)
            doc.flags.writeable = False
            object.__setattr__(self, '_document', doc)
        return self._document

    def subset(self, indices: list[int] | np.ndarray) -> "EmbeddingCache":
        """Return cache containing only the specified pair indices.
        Always copies data (never returns a view) for thread safety."""
        return EmbeddingCache(
            surface=self.surface[indices].copy(),
            generator=self.generator[indices].copy(),
        )
```

### New: `precompute_embeddings()` (subprocess launcher)

```python
def precompute_embeddings(
    character: str,
    pairs: list[dict[str, Any]],
    surface_embedder_path: str,
    generator_embedder_path: str,
    device: str = "cuda:0",
    bs: int = 8,
    timeout: int = 600,
) -> EmbeddingCache:
    """Pre-compute all embeddings using subprocess isolation.

    Each model runs in a separate subprocess. On exit, OS reclaims all VRAM.
    Uses sys.executable for same Python/venv.

    Args:
        timeout: Max seconds per subprocess (default 600). Kills on timeout.

    Raises:
        RuntimeError: If subprocess fails, times out, or is OOM-killed (rc=-9).
        ValueError: If pairs is empty.
    """
```

**Subprocess error handling:**
- `rc == 0`: success, load .npy, validate shape matches `len(pairs)`.
- `rc == -9` (SIGKILL/OOM): raise `RuntimeError` with OOM-specific message suggesting smaller batch_size or model.
- `rc != 0`: raise `RuntimeError` with full stderr output.
- `TimeoutExpired`: kill subprocess, raise `RuntimeError` with timeout message.

**Temp file handling:**
- Use `tempfile.mkstemp()` (creates files with mode 0o600, unique names via random suffix).
- Clean up in `finally` block. Stale files from SIGKILL of parent are harmless — unique names prevent collision.
- After loading `.npy`, validate shape: `assert arr.shape[0] == len(pairs)`.

**Model paths:** Resolve to absolute paths via `Path(model_path).resolve()` before passing to subprocess. Log `sys.executable` at DEBUG level.

### New: `src/canopy/_embed_worker.py` (subprocess entry point)

Standalone script with `if __name__ == "__main__":` entry point:
1. Reads input texts from temp JSON file (path via CLI arg)
2. Loads specified model (surface=SentenceTransformer, generator=CausalLM)
3. For generator type: appends character suffix `"\n\nThus, {character} decides to"` to each scene (must match `embeddings.py:141` exactly)
4. Encodes all texts in batches, L2-normalizes
5. Saves embeddings as `.npy`
6. Exits → OS reclaims GPU memory

```
Usage: python -m canopy._embed_worker \
    --input /tmp/canopy_texts_a1b2c3.json \
    --output /tmp/canopy_embed_a1b2c3.npy \
    --model_path ~/models/Qwen3-8B \
    --model_type surface|generator \
    --character Kasumi \
    --device cuda:0 \
    --batch_size 8
```

**stdout/stderr:** Redirect stdout to `subprocess.PIPE` (discard or log at DEBUG). Only surface stderr on failure. This prevents worker tqdm/logging from interleaving with parent output.

### Changed: `select_cluster_centers()` — add cache parameter

```python
def select_cluster_centers(
    character: str,
    pairs: list[dict[str, Any]],
    ...,
    *,
    embedding_cache: EmbeddingCache | None = None,
) -> list[list[dict[str, Any]]]:
```

**Guard ordering (CRITICAL):** When `embedding_cache` is not None, short-circuit BEFORE the
`_device is None` guard. The cache path needs no initialized device/models.

```python
if embedding_cache is not None:
    # Validate all pairs have _embed_idx
    missing = [i for i, p in enumerate(pairs) if "_embed_idx" not in p]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs missing '_embed_idx'")
    indices = [p["_embed_idx"] for p in pairs]
    subset = embedding_cache.subset(indices)
    # Skip directly to clustering with cached embeddings
    ...
    return select_representative_samples(pairs, subset.document, centroids, ...)

if _device is None:
    raise RuntimeError("not initialized")
# ... existing model-loading path (backward compatible for 0.6B)
```

When `None`: fall back to current behavior (backward compatible for 0.6B models).

**Empty pairs guard:** Return `[]` immediately if `not pairs` (prevents `np.concatenate([])` crash).

### Changed: `CDTNode` — thread cache through recursion

Add `_embedding_cache: EmbeddingCache | None = None` parameter.
In `_build()`:
- Extract `_embed_idx` from pairs → get subset cache
- Pass `embedding_cache=subset_cache` to `select_cluster_centers()`
- Forward `_embedding_cache` to child CDTNode constructors
- `_embedder` override still takes priority (for test mocking)

### Changed: `build_character_cdts()` — accept and forward cache

```python
def build_character_cdts(
    ...,
    embedding_cache: EmbeddingCache | None = None,
) -> tuple[dict[str, CDTNode], dict[str, CDTNode]]:
```

### Pair Identity Tracking

Before Phase B, create COPIES of pairs with `_embed_idx` stamped (never mutate caller's dicts):
```python
indexed_pairs = [{**pair, "_embed_idx": idx} for idx, pair in enumerate(pairs)]
```

**Why copies, not mutation:** The `pairs` list comes from `load_ar_pairs()` or `BehavioralObservation.to_pair()`.
Mutating caller-owned dicts creates hidden side effects and shared mutable state across concurrent threads.
Copying is O(N) with ~748 pairs — negligible cost.

When `select_cluster_centers` receives pairs with `_embed_idx`, it extracts matching rows:
```python
indices = [p["_embed_idx"] for p in pairs]
subset_cache = embedding_cache.subset(indices)
```

**How filtering preserves indices:** Relationship CDTs filter pairs by character:
`[d for d in indexed_pairs if other in d.get("last_character", [])]`. Each filtered dict still
carries its original `_embed_idx`, which maps to the correct row in the full cache.
Recursive children (`filtered_pairs` from `validate_hypothesis()`) also preserve `_embed_idx`
because `validate_hypothesis` returns the same dict objects.

`_embed_idx` is a **reserved key** — do not use it for other purposes in pair dicts.

### CLI Changes (`codified_decision_tree.py`)

```python
# Phase A: Pre-compute (subprocess isolation)
cache = precompute_embeddings(
    character=args.character, pairs=pairs,
    surface_embedder_path=args.surface_embedder_path,
    generator_embedder_path=args.generator_embedder_path,
    device=f"cuda:{args.device_id}",
)

# Stamp indices on COPIES (never mutate caller's dicts)
indexed_pairs = [{**pair, "_embed_idx": idx} for idx, pair in enumerate(pairs)]

# Phase B: Build CDTs (no GPU model loading, max_parallel=4)
topic2cdt, rel_topic2cdt = build_character_cdts(
    args.character, indexed_pairs, other_characters, config,
    max_parallel=4, embedding_cache=cache,
)
```

`init_embedding_models()` call removed (models load in subprocesses now).

### Changed: `builder.py` — forward cache through library API

`build_cdt()` and `build_character_profile()` gain `embedding_cache` kwarg,
forwarding to `CDTNode` / `build_character_cdts()` respectively. This ensures
the library API path can also use pre-computed embeddings.

### Out of Scope

`cluster_method` parameter is not currently wired through `select_cluster_centers()`.
The CLI accepts `--cluster_method hdbscan` but `select_cluster_centers()` hardcodes KMeans.
This is a pre-existing gap — will address in a separate change.

## Implementation Phases

### Phase 1: EmbeddingCache + subprocess worker

**Files:** `src/canopy/embeddings.py`, `src/canopy/_embed_worker.py`, `tests/test_embeddings.py`

1. Add `EmbeddingCache` frozen dataclass
2. Create `_embed_worker.py` standalone script
3. Add `precompute_embeddings()` + `_run_embedding_subprocess()`
4. Unit tests: cache construction, subset, frozen, subprocess mock, temp file cleanup, error handling

**Verification:** `uv run pytest tests/test_embeddings.py` passes, 100% on new code.

### Phase 2: Wire cache through tree building

**Files:** `src/canopy/embeddings.py`, `src/canopy/core.py`, `src/canopy/builder.py`, `tests/test_core.py`, `tests/test_embeddings.py`

1. Add `embedding_cache` kwarg to `select_cluster_centers()` — short-circuit before `_device` guard when cache present
2. Add `_embedding_cache` to `CDTNode.__init__()` and `_build()` — extract `_embed_idx`, forward to children
3. Add `embedding_cache` to `build_character_cdts()` — pass to CDTNode constructors
4. Add `embedding_cache` to `build_cdt()` and `build_character_profile()` in `builder.py`
5. Add empty pairs guard to `select_cluster_centers()`: `if not pairs: return []`
6. Unit tests: cache skips loading, forwarded to children, `_embedder` override takes priority, missing `_embed_idx` raises, empty pairs returns []

**Verification:** All existing tests pass unchanged. New cache-path tests pass.

### Phase 3: CLI integration + GPU integration tests

**Files:** `codified_decision_tree.py`, `tests/test_integration.py`

1. Update CLI: remove `init_embedding_models()`, add `precompute_embeddings()`, stamp `_embed_idx`
2. GPU integration tests: subprocess encoding with 0.6B, cache-vs-direct consistency, end-to-end pipeline

**Verification:** `uv run pytest -m integration` passes. Full CLI build completes without OOM.

## Test Plan

### Unit Tests (no GPU)

| Test | Verifies |
|------|----------|
| `EmbeddingCache.test_construction` | Arrays stored correctly |
| `EmbeddingCache.test_frozen` | Immutable |
| `EmbeddingCache.test_document_property` | gen+surface concatenation |
| `EmbeddingCache.test_subset` | Correct row extraction |
| `EmbeddingCache.test_subset_preserves_order` | Index order maintained |
| `precompute_embeddings.test_launches_two_subprocesses` | surface then generator |
| `precompute_embeddings.test_subprocess_failure_raises` | RuntimeError with stderr |
| `precompute_embeddings.test_temp_file_cleanup` | Cleaned on success and failure |
| `select_cluster_centers.test_with_cache_skips_loading` | No model import when cache |
| `select_cluster_centers.test_without_cache_backward_compat` | Guard check still works |
| `CDTNode.test_cache_forwarded_to_children` | Recursive nodes get cache |
| `CDTNode.test_embedder_override_priority` | Mock _embedder > cache |
| `build_character_cdts.test_with_cache` | Cache passed to all topics |
| `_embed_worker.test_surface_encoding` | SentenceTransformer path |
| `_embed_worker.test_generator_encoding` | CausalLM + suffix path |
| `EmbeddingCache.test_arrays_readonly` | `flags.writeable == False` |
| `EmbeddingCache.test_document_cached` | Same object on repeated access |
| `EmbeddingCache.test_subset_copies` | Subset arrays are copies, not views |
| `select_cluster_centers.test_empty_pairs` | Returns `[]` immediately |
| `select_cluster_centers.test_missing_embed_idx_raises` | RuntimeError on missing key |
| `precompute_embeddings.test_oom_killed_subprocess` | Distinct error for rc=-9 |
| `precompute_embeddings.test_timeout` | Kills subprocess on timeout |
| `precompute_embeddings.test_shape_validation` | Shape mismatch raises |
| `build_character_cdts.test_cache_with_unstamped_pairs` | RuntimeError propagated |
| `build_cdt.test_with_cache` | builder.py forwards cache |
| `build_character_profile.test_with_cache` | builder.py forwards cache |

### Integration Tests (GPU required)

| Test | Verifies |
|------|----------|
| `test_precompute_subprocess` | Real subprocess with 0.6B models |
| `test_cache_matches_direct` | Cache clustering = direct clustering |
| `test_full_pipeline_with_cache` | End-to-end CDT build with cache |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Subprocess fails silently | Check rc, capture stderr. rc=-9 → OOM-specific error message |
| Subprocess hangs on corrupt model | `timeout` param (default 600s). `proc.communicate(timeout=timeout)` |
| Temp files not cleaned on SIGKILL | `tempfile.mkstemp()` unique names prevent collision. Stale files harmless |
| Truncated .npy from killed subprocess | Validate shape after load: `assert arr.shape[0] == len(pairs)` |
| Pair mutation across threads | Copy pairs before stamping: `{**pair, "_embed_idx": idx}` |
| Missing `_embed_idx` key | Guard in `select_cluster_centers()`: fail fast with clear RuntimeError |
| numpy array mutation across threads | `flags.writeable = False` on all EmbeddingCache arrays |
| `document` property recomputation | Cached on first access via `_document` field |
| `_embed_idx` key collision | Underscore-prefixed, documented as reserved, no existing usage |
| Subprocess env mismatch | `sys.executable` (same venv), resolve model paths to absolute |
| Worker stdout interleaves | Redirect stdout to PIPE, only surface stderr on failure |
| Remote model download in subprocess | Document: local paths required for 8B models |
| Large temp JSON | ~748 pairs = ~1-2MB, not a concern |
| Empty pairs list | Early return `[]` in `select_cluster_centers()` |

## Rollback Plan

Each phase independently reversible:
- **Phase 1:** Delete new files, no other code depends on them
- **Phase 2:** Remove `embedding_cache` params, existing mock tests unaffected
- **Phase 3:** Restore `init_embedding_models()` call, old path still functional

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| 8B model loads per build | 20+ (per node) | 2 (one surface, one generator) |
| VRAM peak (8B) | 32GB+ (OOM) | ~16GB (one model at a time) |
| Encoding redundancy | 4x for attribute topics | 0 (shared cache) |
| Phase B GPU usage | 16GB per select_cluster_centers | 0 (numpy only + DeBERTa 715MB) |
| max_parallel effectiveness | Limited by GPU locks | Full (no GPU contention) |

## Open Questions

None — design is fully constrained.

## Review Findings (Round 1) — All Addressed

Two review agents ran in parallel (architect + security). All findings incorporated above.

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| C1 | CRITICAL | In-place pair mutation → shared mutable state across threads | Copy pairs: `{**pair, "_embed_idx": idx}` |
| C2 | CRITICAL | SIGKILL bypasses `finally` cleanup; stale temp file collision | `mkstemp()` unique names, shape validation after load |
| H1 | HIGH | Guard ordering: `_device is None` fires before cache check | Cache path short-circuits before guard |
| H2 | HIGH | Missing `_embed_idx` → confusing KeyError | Guard in `select_cluster_centers()` with clear RuntimeError |
| H3 | HIGH | OOM-killed subprocess (rc=-9) gives empty stderr | Distinct error message for rc=-9 |
| H4 | HIGH | Empty pairs → `np.concatenate([])` crash | Early return `[]` guard |
| M1 | MEDIUM | `builder.py` not listed — API path can't use cache | Added to Phase 2 file list |
| M2 | MEDIUM | `cluster_method` not wired through | Documented as out of scope |
| M3 | MEDIUM | Generator suffix format unspecified | Explicitly documented: `"\n\nThus, {character} decides to"` |
| M4 | MEDIUM | numpy arrays mutable despite frozen dataclass | `flags.writeable = False` on all arrays |
| M5 | MEDIUM | No subprocess timeout | `timeout` param (default 600s), `TimeoutExpired` handling |
| M6 | MEDIUM | Truncated .npy loaded silently | Shape validation after load |
| M7 | MEDIUM | Worker stdout interleaves with parent | Redirect to PIPE |
| L1 | LOW | `_embed_idx` not documented as reserved | Added to docstrings |
| L2 | LOW | `document` property recomputes on every access | Cached via `_document` field |
| L3 | LOW | No test for unstamped pairs | Added `test_cache_with_unstamped_pairs` |
