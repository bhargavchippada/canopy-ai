# Canopy

> The living crown of a decision tree — where behavioral intelligence grows.

**Canopy** extends [Codified Decision Trees (CDT)](https://arxiv.org/abs/2601.10080) with temporal dynamics, structured gate predicates, and domain-agnostic behavioral profiling. Build evolving decision trees that learn how users behave, validated against real evidence, and traversable at inference time without LLM calls.

## Installation

```bash
# Clone and install
git clone <repo-url> && cd canopy-ai
uv sync

# Download models (0.6B for quick testing)
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='~/models/Qwen3-Embedding-0.6B')"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='~/models/Qwen3-0.6B')"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('KomeijiForce/deberta-v3-base-rp-nli', local_dir='~/models/deberta-v3-base-rp-nli')"
```

Requires: Python 3.11+, CUDA GPU, Claude Max subscription (for hypothesis generation).

## Quickstart

```python
from canopy import BehavioralObservation, CDTConfig, build_cdt
from canopy.embeddings import init_models as init_embeddings
from canopy.validation import init_models as init_validation
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.wikify import wikify_tree
from pathlib import Path
import torch

# Setup
device = torch.device("cuda:0")
set_adapter(ClaudeCodeAdapter(default_model="claude-haiku-4-5"))
init_embeddings(
    str(Path.home() / "models/Qwen3-Embedding-0.6B"),
    str(Path.home() / "models/Qwen3-0.6B"),
    device,
)
init_validation(str(Path.home() / "models/deberta-v3-base-rp-nli"), device)

# Create observations
observations = [
    BehavioralObservation(scene="...", action="...", actor="Alice", participants=["Bob"]),
    # ... more observations
]

# Build and wikify
tree = build_cdt(observations, character="Alice", topic="identity", config=CDTConfig(max_depth=2))
print(wikify_tree(tree, title="Alice's identity"))
```

See `examples/quickstart.py` for a complete example.

## Features

- **Codified Decision Trees** — Hierarchical behavioral profiles with gated branches
- **Claude integration** — Hypothesis generation via claude-agent-sdk (Max subscription)
- **NLI validation** — DeBERTa-based natural language inference for hypothesis checking
- **Domain-agnostic** — Works for character profiling, user behavior, workflow patterns
- **Parallel hypothesis generation** — 8 clusters in ~3s via asyncio.gather
- **Resilient batch generation** — `batch_generate()` with retry and drop tracking
- **Parallel benchmarking** — ThreadPoolExecutor with configurable `--max_parallel`
- **Multiple clustering strategies** — KMeans (default) and HDBSCAN (density-based)
- **Markdown wikification** — Convert CDT trees to readable profile documents
- **CDT artifact provenance** — Descriptive filenames + embedded metadata in pickles
- **Benchmark result tracking** — JSON results saved with full provenance in `results/`

## API Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `BehavioralObservation` | `canopy.builder` | Frozen dataclass — scene, action, actor, participants, metadata |
| `CDTConfig` | `canopy.core` | Frozen dataclass — max_depth, threshold_accept/reject/filter |
| `CDTNode` | `canopy.core` | Tree node — statements, gates, children, traverse(), verbalize() |

### Builder Functions

| Function | Module | Description |
|----------|--------|-------------|
| `build_cdt(obs, character, topic, config)` | `canopy.builder` | Build one CDT from observations |
| `build_character_profile(obs, character, ...)` | `canopy.builder` | Build full profile (4 attrs + relationships) |
| `build_character_cdts(character, pairs, others, config)` | `canopy.core` | Low-level builder from raw pairs |

### Wikification

| Function | Module | Description |
|----------|--------|-------------|
| `wikify_tree(node, title)` | `canopy.wikify` | Single CDT → markdown section |
| `wikify_profile(topics, rels, character)` | `canopy.wikify` | Full profile → markdown document |

### Clustering

| Class | Module | Description |
|-------|--------|-------------|
| `KMeansCluster` | `canopy.cluster` | KMeans with auto-k (default) |
| `HDBSCANCluster` | `canopy.cluster` | Density-based, discovers k from data |

### LLM Adapter

| Function/Class | Module | Description |
|----------------|--------|-------------|
| `ClaudeCodeAdapter` | `canopy.llm` | Claude via claude-agent-sdk (Max sub) |
| `set_adapter(adapter)` | `canopy.llm` | Swap default LLM adapter |
| `generate(prompt, model)` | `canopy.llm` | Single LLM call |
| `generate_many(prompts, model)` | `canopy.llm` | Parallel LLM calls |
| `batch_generate(items, model, max_attempts)` | `canopy.llm` | Resilient batch with retry/drop tracking |
| `BatchResult` | `canopy.llm` | Frozen result: successes, dropped_ids, exhausted_ids |

## Testing

```bash
uv run python -m pytest                    # Unit tests (173 tests, ~10s)
uv run python -m pytest -m integration     # Integration tests (11 tests, ~23s, needs GPU)
uv run python -m pytest --cov=canopy       # Coverage report
```

## Status

**Phase 4: COMPLETE** — All legacy files migrated, batch LLM, parallel benchmark, 184 tests.

## Attribution

This project builds on the [Codified Decision Tree](https://github.com/KomeijiForce/Codified_Decision_Tree) implementation by Letian Peng et al. The original work is licensed under MIT.

```bibtex
@article{codified_decision_tree,
  title={Deriving Character Logic from Storyline as Codified Decision Trees},
  author={Letian Peng, Kun Zhou, Longfei Yun, Yupeng Hou, and Jingbo Shang},
  journal={arXiv preprint arXiv:2601.10080},
  year={2026}
}
```

---

## Original CDT Documentation

*The following documents the original CDT implementation included in this repository. It will be restructured as Canopy evolves.*

---

# Codified Decision Tree (CDT) [\<Link to Paper\>](https://arxiv.org/pdf/2601.10080)
**An algorithm to derive deep, validated, structured character behaviors from given storylines.**

This repo includes:
- Algorithm implementation: Constructing **Codified Decision Trees** based on scene-action pairs of characters
- Benchmarking: The performance of CDT-driven Role-playing
- Automatic profiling: Conversion script from CDT to reader-friendly wiki texts.

## How to use?
- **Initialization**

**Note:** The original setup required `constant.py` with OpenAI keys. Canopy has fully migrated to Claude via `claude-agent-sdk` — no `constant.py` or OpenAI dependency needed.

- **Construct CDTs**

```bash
uv run python codified_decision_tree.py \
  --character Kasumi \
  --engine claude-haiku-4-5 \
  --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \
  --generator_embedder_path ~/models/Qwen3-0.6B \
  --discriminator_path ~/models/deberta-v3-base-rp-nli \
  --cluster_method kmeans \
  --device_id 0
```

Output: `packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl`

- **Benchmark CDTs**

Benchmarks run against: [Fine-grained Fandom Benchmark](https://huggingface.co/datasets/KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences) and [Bandori Conversational Benchmark](https://huggingface.co/datasets/KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences).

```bash
uv run python run_benchmark.py \
  --character Kasumi \
  --method cdt_package \
  --cdt_path packages/Kasumi.haiku.qwen06b.deberta.kmeans.d3.a80.r50.relation.pkl \
  --engine claude-haiku-4-5 \
  --eval_engine claude-sonnet-4-6 \
  --max_parallel 6 \
  --device_id 0
```

Results saved as JSON in `results/` with full provenance.

- **Wikification**

Use `canopy.wikify` module or see `examples/quickstart.py`.

## Benchmark Results

See the [original paper](https://arxiv.org/abs/2601.10080) for full benchmark results.

## License

MIT
