# Canopy

> The living crown of a decision tree — where behavioral intelligence grows.

**Canopy** extends [Codified Decision Trees (CDT)](https://arxiv.org/abs/2601.10080) with temporal dynamics, structured gate predicates, and domain-agnostic behavioral profiling. Build evolving decision trees that learn how users behave, validated against real evidence, and traversable at inference time without LLM calls.

## Planned Features

- **Temporal CDT (T-CDT)** — Time-weighted validation where newer evidence takes precedence. Superseded patterns are preserved with history, not deleted.
- **Semantic gate conditions** — Embedding-based cosine similarity for fast, deterministic traversal at inference time.
- **Computed confidence** — Evidence-count based validation (supporting/contradicting/irrelevant), not LLM self-assessed scores.
- **Claude integration** — Uses Anthropic Claude for hypothesis generation (OpenAI replaced in Phase 0).
- **Domain-agnostic** — Works for user behavior profiling, character logic, workflow patterns, and more.
- **Installable package** — `uv add canopy-ai`

## Status

**Phase 0: Baseline Reproduction** — Reproducing the original CDT paper's benchmarks before extending.

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
