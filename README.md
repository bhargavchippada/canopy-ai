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
- **Multiple clustering strategies** — KMeans (default) and HDBSCAN (density-based)
- **Markdown wikification** — Convert CDT trees to readable profile documents

## Testing

```bash
uv run python -m pytest                    # Unit tests (134 tests, ~10s)
uv run python -m pytest -m integration     # Integration tests (11 tests, ~23s, needs GPU)
uv run python -m pytest --cov=canopy       # With coverage report
```

## Status

**Phase 2: Core Library API** — Importable package with BehavioralObservation, builder, wikify, clustering.

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

You have to create a `constant.py` file at the root path, and put your openai and huggingface token there:
```python
openai_key = "..."
hf_token = "..."
```
You may also follow the configuration below:
```
torch: 2.7.1+cu126
transformers: 4.55.0
sentence_transformers: 5.1.0
sklearn: 1.7.1
openai: 2.14.0
```
- **Construct CDTs**

For characters involved in the paper's experiments, you can use the `build_cdt.sh` script to reproduce the CDTs:
```sh
python codified_decision_tree.py \
  --character "Kasumi" \
  --engine "gpt-4.1" \
  --max_depth 3 \
  --threshold_accept 0.8 \
  --threshold_reject 0.5 \
  --threshold_filter 0.8 \
  --device_id 1
```

You can build CDT for any character using the `CDT_Node` class given in `codified_decision_tree.py`:
```python
CDT_Node(character, goal_topic, pairs, built_statements, depth, established_statements, gate_path,
max_depth, threshold_accept, threshold_reject, threshold_filter)
```

Parameters:
- `character`: The name for your character;
- `goal_topic`: The goal (topic/aspect) you want the CDT to focus on;
- `pairs`: The training data for your CDT, in the format: `[{"scene": "...", "action": "..."}, {"scene": "...", "action": "..."}, ...]` where `character` takes the `action` in the `scene`;
- `built_statements`: Used for node growth, keep it `None`;
- `depth`: Used for depth-based termination, keep it `1`;
- `established_statements`: Used for diversification, keep `[]`;
- `gate_path`: Used for diversification, keep `[]`;
- `max_depth`: Used for depth-based termination, recommended to be set to `3`;
- `threshold_accept`: The parameter controlling the precision for statement acceptance;
- `threshold_reject`: The parameter controlling the precision for hypothesis abolishment;
- `threshold_filter`: The parameter controlling the filtering effect for gate acceptance;
- `device_id`: The GPU id you want to run the algorithm on.

- **Grounding**

With a constructed `CDT_Node` (e.g., `cdt_tree`), use `cdt_tree.traverse(scene)` to fetch grounding statements on the CDT for the input `scene`.

- **Benchmark CDTs**
The benchmarking is run by `run_benchmark.sh` on the two benchmarks: [Fine-grained Fandom Benchmark](https://huggingface.co/datasets/KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences) and [Bandori Conversational Benchmark](https://huggingface.co/datasets/KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences).

```python
python run_benchmark.py \
  --character "Kasumi" \
  --method "cdt_package" \
  --engine "gpt-4.1" \
  --eval_engine "gpt-4.1" \
  --generator_path "meta-llama/Llama-3.1-8B-Instruct" \
  --device_id 1
```

- **Wikification**

The example notebook to wikify CDTs into reader-friendly profiles is provided in `Wikification.ipynb`.

## Benchmark Results

See the [original paper](https://arxiv.org/abs/2601.10080) for full benchmark results.

## License

MIT
