# Canopy

> Evolving decision trees for behavioral profiling.

## Project Overview

Canopy extends Codified Decision Trees (CDT) with temporal dynamics, structured gates, and domain-agnostic profiling. Built on the CDT paper (arxiv 2601.10080).

## Current Phase

**Phase 0: COMPLETE** — 3 characters validated (Kasumi 13min, Arisa 45min, Yui 2hr).
**Phase 1: COMPLETE** — Package restructured into `src/canopy/`, 100% coverage on 7 core modules.
**Phase 2: COMPLETE** — Library API: BehavioralObservation, builder, wikify, cluster.
**Phase 3: COMPLETE** — Integration tests (11 GPU tests), examples, README.
**Phase 4: COMPLETE** — All legacy files migrated (zero OpenAI/exec/constant refs), batch LLM, parallel benchmark.
**Phase 5: IN PROGRESS** — T-CDT temporal weighting, E7 episodic memory (CDT-guided RAG), E1 hypothesis merge (LLM semantic dedup), E4 topic discovery, contrastive hypothesis generation, gen prompt templates.

### Key Findings (Session 8)
- **70.66 is a local maximum** for Kasumi RP with Sonnet eval — every additional signal (temporal, RAG, format constraints, more CDT text, relationships) hurts
- **Contrastive hypotheses** surface non-dominant behaviors (confusion, hesitation) but deeper trees over-discriminate
- **Hypothesis merge** now uses single LLM call for semantic dedup (not word-based matching)
- **PersonaMem 32k vs 128k**: our 76.9% is not comparable to paper's 50% (different splits)
- **T-CDT**: temporal weighting works but not for consistent characters (Bandori); good for evolving personas (delulu)

### Key Findings (Session 9)
- **CDT builds are NON-DETERMINISTIC**: same config rebuilt = 60.30 vs original 70.66 (10-pt variance). LLM hypothesis gen produces different trees each time. All A/B benchmark comparisons MUST use the SAME CDT pickle, not rebuilt CDTs.
- **--no_merge flag** added to `codified_decision_tree.py` to control hypothesis dedup (skip the LLM merge step)
- **Retry logic** added to `build_character_cdts()` for transient LLM failures during parallel topic construction

### Key Findings (Session 10)
- **Step pipeline solves reproducibility**: 3 identical builds → NLI scores 70.36, 68.86, 71.26 (mean=70.16, std=0.99). Caching deterministic steps isolates variance to hypothesis gen + dedup only. 10x improvement over session 9's 10-pt variance.
- **Tree structure varies 2x but NLI barely moves**: nodes=[36,39,61], stmts=[84,82,120] across 3 builds, yet NLI std=0.99. Bigger trees don't help — optimize hypothesis quality, not quantity.
- **Dedup is the key variance source**: LLM merge reduces 24→7 or 24→14 hypotheses depending on run. Deterministic dedup (DeBERTa entailment) would eliminate this variance.
- **Provenance system added**: Provenance, TrackedHypothesis, HypothesisQuality dataclasses in `canopy.provenance`. CDTNode extended with statement_provenance, gate_provenance, hypothesis_quality + get_evidence()/traverse_with_evidence(). Old pickles backward-compatible via __setstate__.
- **check_statement_pair_entailment()** added to validation.py — computes grounding fidelity (mean NLI True probability of statement against source actions)

### Conventions Discovered (Session 10 continued)
- **`step_build_tree` must init DeBERTa**: `build_character_cdts()` calls `validate_hypothesis()` internally, which requires `canopy.validation.init_models()`. Any step or script calling `build_character_cdts` must init the validation model first.
- **Never run 3+ `build_character_cdts` in parallel**: Each build spawns `max_parallel` threads, each making batch LLM calls. 3 concurrent builds = 90+ simultaneous claude-agent-sdk subprocesses → API rate limits and fatal errors. Run builds sequentially or use `max_parallel=1-2`.
- **Cast numpy float32 before JSON serialization**: `validate_hypothesis()` returns numpy float32 values. Always `float(v)` before passing to `json.dumps()`.
- **`build_character_cdts` now accepts `hypothesis_fn` and `summarize_fn`**: Custom hypothesis generation and summarization functions can be injected for experiments without modifying the default code path.

## Design Reference

**Authoritative CDT design:** `artifacts/canopy-design.md` (design decisions D1-D30)

## Principles

- **Step by step** — Each phase gated by success criteria.
- **Branch per phase** — Merged only when complete.
- **Simple and clean** — Minimum complexity for current needs.
- **Research first** — Search before building.
- **Iterate** — Small commits, frequent validation.

## Tech Stack

- **Language:** Python 3.11+
- **Package manager:** uv (NEVER pip)
- **LLM:** Claude via claude-agent-sdk (Max subscription, no API key)
- **LLM Model:** claude-haiku-4-5 (hypothesis gen), claude-sonnet-4-6 (evaluation)
- **Embeddings:** Qwen3-0.6B (smoke test) / Qwen3-8B (full run)
- **NLI:** DeBERTa-v3-base-rp-nli
- **Local Models:** Llama-3.1-8B-Instruct (RP gen), Qwen3-8B/0.6B (embeddings), DeBERTa (NLI)
- **Testing:** pytest (unit + integration, run `pytest --co -q` for current count)
- **Linting:** ruff

## Library API

```python
from canopy import BehavioralObservation, CDTConfig, build_cdt, build_character_profile
from canopy import EpisodicIndex, hybrid_ground, GroundingResult  # E7: CDT-guided RAG
from canopy.wikify import wikify_tree, wikify_profile
from canopy.episodic import format_grounding  # Prompt-ready grounding text
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.cluster import KMeansCluster, HDBSCANCluster
from canopy.validation import temporal_weight  # T-CDT time decay

# 1. Configure LLM
set_adapter(ClaudeCodeAdapter(default_model="claude-haiku-4-5"))

# 2. Create observations
obs = [
    BehavioralObservation(scene="...", action="...", actor="Alice", participants=["Bob"]),
]

# 3. Build single CDT
tree = build_cdt(obs, character="Alice", topic="identity", config=CDTConfig(max_depth=2))

# 4. Or build full profile
topic2cdt, rel_topic2cdt = build_character_profile(obs, character="Alice")

# 5. Wikify to markdown
markdown = wikify_profile(topic2cdt, rel_topic2cdt, character="Alice")

# 6. Pre-compute embeddings with subprocess isolation (for 8B models)
from canopy.embeddings import precompute_embeddings
cache = precompute_embeddings("Alice", pairs, "/path/to/surface", "/path/to/gen")
indexed_pairs = [{**p, "_embed_idx": i} for i, p in enumerate(pairs)]
topic2cdt, rel_topic2cdt = build_character_profile(
    obs, character="Alice", embedding_cache=cache,
)

# 7. Batch LLM generation with drop tracking
from canopy.llm import batch_generate, BatchResult
result = batch_generate([("id1", "prompt1"), ("id2", "prompt2")])
print(result.successes, result.exhausted_ids, result.success_rate)
```

## Key Commands

```bash
uv sync                                           # Install dependencies
uv run python -m pytest                           # Unit tests (~10s)
uv run python -m pytest -m integration            # GPU integration tests
uv run python -m pytest --cov=canopy              # Coverage report
uv run ruff check src/canopy/                     # Lint

# CDT construction via CLI (outputs: packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl)
uv run python codified_decision_tree.py \
  --character Kasumi \
  --engine claude-haiku-4-5 \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --discriminator_path ~/models/deberta-v3-base-rp-nli \
  --cluster_method kmeans \
  --device_id 0

# CDT construction without hypothesis merge (skip LLM dedup step)
uv run python codified_decision_tree.py \
  --character Kasumi \
  --engine claude-haiku-4-5 \
  --no_merge \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --discriminator_path ~/models/deberta-v3-base-rp-nli \
  --cluster_method kmeans \
  --device_id 0

# Benchmark with parallel evaluation
uv run python run_benchmark.py \
  --character Kasumi \
  --method cdt_package \
  --cdt_path packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl \
  --engine claude-haiku-4-5 \
  --eval_engine claude-sonnet-4-6 \
  --max_parallel 6 \
  --device_id 0
```

## Model Paths

Models stored in `~/models/`:

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| DeBERTa NLI | `~/models/deberta-v3-base-rp-nli` | 715MB | NLI validation |
| Qwen3-Embedding-0.6B | `~/models/Qwen3-Embedding-0.6B` | ~1.2GB | Surface embedding (smoke test) |
| Qwen3-0.6B | `~/models/Qwen3-0.6B` | ~1.2GB | Generative embedding (smoke test) |
| Qwen3-Embedding-8B | `~/models/Qwen3-Embedding-8B` | ~16GB | Surface embedding (paper config) |
| Qwen3-8B | `~/models/Qwen3-8B` | ~16GB | Generative embedding (paper config) |
| Llama-3.1-8B-Instruct | `~/models/Llama-3.1-8B-Instruct` | ~16GB | RP generation (paper's gen model) |

**VRAM (RTX 5090, 32GB):** 0.6B models use ~3.1GB total. 8B models use two-phase subprocess isolation (each model loads once in a subprocess, exits to release VRAM). Llama fp16 + DeBERTa fits in ~21GB.

## Project Structure

```
canopy-ai/
├── src/canopy/                # Core package (13 modules)
│   ├── __init__.py            # Exports: CDTConfig, CDTNode, BehavioralObservation, EmbeddingCache
│   ├── core.py                # CDTNode, CDTConfig, build_character_cdts
│   ├── builder.py             # BehavioralObservation, build_cdt, build_character_profile
│   ├── wikify.py              # CDT → markdown (wikify_tree, wikify_profile)
│   ├── cluster.py             # KMeansCluster, HDBSCANCluster, ClusterStrategy Protocol
│   ├── embeddings.py          # EmbeddingCache, precompute_embeddings, select_cluster_centers
│   ├── _embed_worker.py       # Subprocess entry point for VRAM-isolated encoding
│   ├── validation.py          # NLI-based scene/statement checking (DeBERTa)
│   ├── prompts.py             # LLM hypothesis generation + summarization
│   ├── quality.py             # Pure metric functions (no GPU/LLM): data, clustering, hypothesis, tree quality
│   ├── llm.py                 # LLM adapter (Protocol + ClaudeCodeAdapter + batch_generate)
│   ├── data.py                # HuggingFace dataset loading + caching
│   └── cli.py                 # CLI entry point
├── scripts/                   # CLI tools
│   └── cdt_steps.py           # Step-by-step CDT pipeline with caching + quality metrics
├── notebooks/                 # Interactive exploration
│   └── cdt_step_by_step.ipynb # 12-cell Colab: t-SNE, quality tables, tree viz (D17: zero pipeline logic)
├── tests/                     # Unit + integration tests
│   ├── conftest.py            # GPU skip guard, integration deselection
│   ├── test_core.py           # CDTNode, CDTConfig, build_character_cdts, cache forwarding
│   ├── test_builder.py        # BehavioralObservation, build_cdt
│   ├── test_wikify.py         # wikify_tree, wikify_profile
│   ├── test_cluster.py        # KMeans, HDBSCAN, representative samples
│   ├── test_llm.py            # Adapter, extract_json, retry, session
│   ├── test_data.py           # Metadata loading, pair extraction
│   ├── test_embeddings.py     # EmbeddingCache, precompute, subprocess, cache path
│   ├── test_validation.py     # Guard tests
│   ├── test_quality.py        # Pure metric function tests (100% coverage)
│   ├── test_cdt_steps.py      # Step pipeline CLI tests (99% coverage)
│   └── test_integration.py    # GPU tests (embeddings, validation, pipeline)
├── examples/                  # Usage examples
│   └── quickstart.py
├── codified_decision_tree.py  # Thin CLI wrapper
├── run_benchmark.py           # CDT benchmark with parallel eval (ThreadPoolExecutor)
├── verify_cdt.py              # CDT package inspector
├── artifacts/                 # Design docs, results
├── results/                   # Benchmark result JSON files (gitignored)
└── pyproject.toml
```

## Performance

| Metric | Before (Phase 0) | After (optimized) |
|--------|-------------------|-------------------|
| Single LLM call | 30-60s | 2-3s |
| 8 parallel calls | 4-8 min | ~3s |
| Kasumi CDT | ~90 min | ~13 min |

Optimizations: tools=[], setting_sources=[], Haiku model, asyncio.gather parallelism.
Phase 4 additions: `batch_generate()` with retry/drop tracking, `ThreadPoolExecutor` parallel benchmark (--max_parallel).
Phase 5 additions: Two-phase embedding architecture — each 8B model loads ONCE in a subprocess (OS reclaims VRAM on exit), then tree building uses pre-computed numpy arrays with zero GPU model loading. Eliminates OOM with 8B models on 32GB GPUs.

## Paper Hyperparameter Discrepancy

**Paper text** (Appendix B): θ_accept=0.75, θ_reject=0.50, θ_f=0.75, max_depth=4
**Paper code** (all .py files, README, build_cdt.sh): max_depth=3, threshold_accept=0.80, threshold_reject=0.50, threshold_filter=0.80

Three values disagree: θ_accept (0.75 vs 0.80), θ_f (0.75 vs 0.80), max_depth (4 vs 3). **Follow paper text as authoritative** — use θ_accept=0.75, θ_f=0.75, max_depth=4 as the reference configuration. Code defaults may reflect development/tuning values that diverged from the final published settings.

**Paper models:**
- Hypothesis gen: gpt-4.1
- Validation: gpt-4.1-mini (CDT) or DeBERTa (CDT-Lite, scores HIGHER: 88.38 vs 84.25)
- RP gen: llama-3.1-8b-instruct (64 tokens, newline stop, greedy)
- Eval: gpt-4.1 (never named in paper text, only in code defaults)
- Embeddings: qwen3-embedding-8b (surface), qwen3-8b (generative)
- Clustering: K-Means, max 8 clusters/node, min 16 samples/cluster

**Paper does NOT report:** A/B/C score distributions, eval model ablation, gen format comparison.

## Benchmark Results

### Paper Parity (Sonnet eval baseline)

| CDT Source | Gen Model | Eval Model | Rel | NLI Score | A% | B% | C% | Notes |
|-----------|-----------|------------|-----|-----------|----|----|----|----|
| **sonnet.d4.a75 (prompt fix)** | **Sonnet** | **Sonnet** | **No** | **70.66** | **51.5** | **38.3** | **10.2** | **Paper parity achieved** |
| Paper code (reference) | Llama | Sonnet | Yes | 70.96 | — | — | — | Paper's exact code + Sonnet eval |
| Paper CDT | Llama fp16 | Sonnet | Yes | 70.66 | — | — | — | Our code matching paper config |
| Paper CDT | Llama fp16 | Sonnet | No | 55.69 | 29.3 | 52.7 | 18.0 | Without relationships |
| sonnet.d3.a75 (pre-fix) | Sonnet | Sonnet | No | 67.96 | 47.9 | 40.1 | 12.0 | Before prompt fix |
| **Paper (GPT-4.1 eval)** | **Llama** | **GPT-4.1** | **No** | **84.25** | — | **~18** | — | **Paper Table 2 target** |
| **Paper CDT-Lite** | **Llama** | **GPT-4.1** | **No** | **88.38** | — | — | — | DeBERTa validation path |

### Extended Results

| CDT Source | Gen Model | Eval Model | NLI Score | Notes |
|-----------|-----------|------------|-----------|-------|
| claude-haiku.d3.rel | Haiku | Haiku | 41.32 | Cheapest baseline |
| sonnet.qwen8b.d3.a75 | Haiku | Haiku | 50.00 | |
| haiku.qwen8b.d3.a75 | Haiku | Sonnet | 58.38 | |
| sonnet.qwen8b.d3.a75 | Haiku | Sonnet | 58.98 | |
| paper-original | Sonnet | Sonnet | 65.87 | Paper CDT, Sonnet eval |
| Arisa paper CDT | Sonnet | Sonnet | 63.36 | Cross-character validation |
| Arisa paper CDT | Llama | Sonnet | 54.31 | Llama mode collapse on Arisa |

**Score gap explanation:** Sonnet eval gives ~38-40% B (neutral); GPT-4.1 eval gives ~18% B. The 70.66→84.25 gap (13.6 pts) is eval model calibration only. CDT structure and implementation verified identical to paper.

### PERSONAMEM Benchmark (Dynamic User Profiling)

| Method | Accuracy | Notes |
|--------|----------|-------|
| **Sonnet + full context** | **76.9% (453/589)** | **32k split, proper end_index truncation** |
| Paper frontier models (GPT-4.5, o1, Gemini) | ~50% | Direct prompting |
| RAG | ~55% | Retrieval-augmented |
| CDT alone | 20% (1/5) | CDT captures patterns, not specific facts |

Per-type: suggest_new_ideas weakest (43%), all others 74-91%.
Key insight: full conversation context + Sonnet is sufficient — CDT/RAG unnecessary for factual recall types.

### B-Score Analysis (artifacts/b-score-analysis.md)
- 40% B rate: 77.6% different-facet (gen caricature), 19.4% format-mismatch, 3% wrong
- 0% eval-too-strict — evaluator is correct
- 74.6% of B pairs in low-predictability scenes (structural ambiguity)
- Scene transitions 3.8x more common in B pairs (28.4%) than A pairs (7.5%)
- C pairs: 80% are enthusiasm-over-restraint (caricature failure on quiet moments)

### Caricature Bias (artifacts/caricature-bias-analysis.md)
- Gen model defaults to max-enthusiasm Kasumi for every prediction
- 100% ground truths are dialogue; 86% predictions are theatrical narration
- CDT grounding: 88 statements, ~15 mentions enthusiasm, ~0 restraint
- Dialogue format fix: +1.79 pts (66.17→67.96), B% increased to 40.1%

## CDT Quality Investigation (2026-03-28)

### Key Finding: Hypothesis Universality Was the Bottleneck
Claude models frame hypotheses as hedged universal truths ("Kasumi tends to openly affirm each bandmate's irreplaceable role...") that pass DeBERTa NLI against every scene → all hypotheses become global statements → flat trees with 0 gates.

**Root cause:** Paper's GPT-4.1 produces shorter, vaguer hypotheses (~15 words) that fail NLI on some scenes → gated tree structure.

**Fix (prompts.py):** Added three constraints to summarize_triggers:
1. Max 15 words per hypothesis (paper avg ~15w, ours was ~25w)
2. Must be FALSE in ≥30% of scenes (prevents universal truths)
3. Must reference specific behavioral trigger (prevents general traits)

**Before fix:** Identity: 1 node, 0 gates, depth 0 (flat)
**After fix:** Identity: 8 nodes, 7 gates, depth 3 (matches paper: 8/7/3)
**Full CDT:** 30 nodes, 113 stmts, 22 gates (paper: 29/103/21)

### CDT Structure Comparison (Kasumi)
| CDT | Attr Nodes | Attr Stmts | Rel Nodes | Rel Stmts | Total |
|-----|-----------|-----------|----------|----------|-------|
| GPT-4.1 (paper) | 23 | 60 | 13 | 23 | 36/83 |
| Sonnet (ours) | 6 | 32 | 24 | 56 | 30/88 |
| Haiku (ours) | 9 | 45 | 11 | 27 | 20/72 |

### Code Differences Found vs Paper (/home/turiya/projects/Codified_Decision_Tree/)
1. **Summarize prompt**: Paper has 50-line version with "single, concise sentence" constraint — FIXED
2. **Output format**: Paper uses Python code block, we use JSON — investigation pending
3. **θ_accept**: Paper text says 0.75, code defaults to 0.80 — follow paper text (0.75)
4. **θ_f**: Paper text says 0.75, code defaults to 0.80 — follow paper text (0.75)
5. **max_depth**: Paper text says 4, code defaults to 3 — follow paper text (4)
6. **Scene check**: Paper uses Llama (CDT) or DeBERTa (CDT-Lite, scores higher: 88.38)
7. **Eval model**: Paper uses gpt-4.1 (~18% B rate), never named in paper text
8. **Gen model**: Paper uses llama-3.1-8b-instruct with 64-token max, newline stop, greedy decoding
9. **Eval parse failure**: Paper crashes on parse failure; we default to B (may inflate B% by 1-2pts)

### Investigation Principle
Models (Haiku/Sonnet/Llama/GPT-4.1) are all capable enough for CDT tasks. Score gaps indicate code/config differences, not model limitations. Always investigate code first.

### per_pair_details Storage
`_save_benchmark_results()` now stores full per_pair_details: prediction text, ground_truth, scene, grounding, eval_reasoning, eval_model, gen_model. Previously only stored numeric per_pair_scores (prediction text was discarded).

## Known Issues

- Module-level mutable state in embeddings.py/validation.py (init_models pattern)

## CDT Artifact Conventions

- **Naming**: `packages/Character.gen_model.embed_model.nli_model.cluster.dN.aXX.rYY.relation.pkl`
  - Example: `Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl`
  - HDBSCAN: `Kasumi.haiku.qwen06b.deberta.hdbscan.d3.a80.r50.relation.pkl`
  - Paper originals: `Kasumi.gpt41.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl`
- **Metadata**: Every `.pkl` includes a `metadata` key with ALL construction config:
  - `gen_model`, `embed_model`, `nli_model` — models used
  - `max_depth`, `threshold_accept`, `threshold_reject`, `threshold_filter` — CDT config
  - `n_clusters`, `total_nodes`, `total_statements`, `has_relationships` — output stats
  - `built_at` — timestamp
- **Validation**: `run_benchmark.py` checks `topic2cdt` + `rel_topic2cdt` keys exist and are non-empty before running. Prints structure summary. Fails fast on wrong format.
- **Never benchmark with a CDT built by a different model** unless that's the explicit experiment
- **Extended metadata** (not in filename): `threshold_filter`, `hypotheses_per_cluster`, `n_training_pairs`, `temperature`
- **Benchmark results**: `results/Character.cdt_config_hash.bench_model.json` containing: CDT package used, eval model, gen model, traversal mode (gated vs all), relationship CDTs included, NLI score, per-pair results, timestamp
- **Never overwrite** paper originals — rename with provenance first

## Conventions

- Use uv for all Python operations
- Type hints on all public functions
- Immutable data structures (frozen dataclasses, tuples)
- No exec() or eval() on LLM output — structured JSON parsing only
- All LLM calls route through canopy.llm adapter
- Logging module (not print) in library code
- RuntimeError for precondition failures (not assert)

## Attribution

Based on Codified Decision Tree by Letian Peng et al. (MIT License).
