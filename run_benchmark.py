"""CDT benchmark — evaluate character response quality via NLI scoring.

Loads pre-built CDT packages, generates character responses using Claude,
and scores them against ground truth using NLI (DeBERTa) and LLM evaluation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from canopy.builder import BehavioralObservation
from canopy.core import CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.episodic import EpisodicIndex, format_grounding, hybrid_ground
from canopy.llm import EVAL_MODEL, HYPOTHESIS_MODEL, extract_json, generate
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------

SCORE_MAP: dict[str, int] = {"A": 100, "B": 50, "C": 0}


# ---------------------------------------------------------------------------
# Pickle compatibility for legacy CDT_Node → canopy CDTNode
# ---------------------------------------------------------------------------


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that maps legacy CDT_Node to canopy.core.CDTNode."""

    def find_class(self, module: str, name: str) -> type:
        if name == "CDT_Node" and module in ("__main__", "codified_decision_tree", "canopy.core"):
            return CDTNode
        return super().find_class(module, name)  # type: ignore[return-value]


def load_cdt_package(path: str) -> dict[str, Any]:
    """Load a pickled CDT package, mapping legacy CDT_Node → CDTNode."""
    with open(path, "rb") as f:
        return _LegacyUnpickler(f).load()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_GEN_SHORT: dict[str, str] = {
    "claude-haiku-4-5": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "gpt-4.1": "gpt41",
    "llama-local": "llama",
}


def _model_short_name(engine: str) -> str:
    """Convert model ID to short name for filenames."""
    return _GEN_SHORT.get(engine, engine.replace(".", "-"))


def _validate_cdt_package(cdts: dict[str, Any], character: str) -> None:
    """Validate CDT package structure before benchmarking."""
    if "topic2cdt" not in cdts:
        raise ValueError(f"CDT package missing 'topic2cdt' key. Keys found: {list(cdts.keys())}")
    if "rel_topic2cdt" not in cdts:
        raise ValueError(f"CDT package missing 'rel_topic2cdt' key. Keys found: {list(cdts.keys())}")

    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]

    if not topic2cdt:
        raise ValueError("CDT package has empty topic2cdt — no attribute CDTs found")

    # Print structure summary
    metadata = cdts.get("metadata", {})
    log.info("CDT package summary:")
    log.info("  Attribute topics: %d", len(topic2cdt))
    log.info("  Relationship topics: %d", len(rel_topic2cdt))
    if metadata:
        log.info("  Gen model: %s", metadata.get("gen_model", metadata.get("built_by", "unknown")))
        log.info("  Embed model: %s", metadata.get("embed_model", "unknown"))
        log.info("  NLI model: %s", metadata.get("nli_model", "unknown"))
        log.info("  Depth: %s", metadata.get("max_depth", metadata.get("depth", "unknown")))
        log.info("  Built at: %s", metadata.get("built_at", "unknown"))
        log.info("  Total nodes: %s", metadata.get("total_nodes", "unknown"))
        log.info("  Total statements: %s", metadata.get("total_statements", "unknown"))


def _save_benchmark_results(
    *,
    character: str,
    method: str,
    engine: str,
    eval_engine: str,
    cdt_path: str | None,
    cdt_metadata: dict[str, Any] | None,
    score: float,
    per_pair_results: list[Any],
    has_relationships: bool,
) -> None:
    """Save benchmark results as JSON with full provenance."""
    os.makedirs("results", exist_ok=True)

    gen_short = _model_short_name(engine)
    eval_short = _model_short_name(eval_engine)
    cdt_tag = os.path.basename(cdt_path).replace(".pkl", "") if cdt_path else "none"
    result_path = f"results/{character}.{cdt_tag}.{gen_short}+{eval_short}.json"

    result_data = {
        "character": character,
        "method": method,
        "score": round(score, 2),
        "n_pairs": len(per_pair_results),
        "n_succeeded": sum(1 for s in per_pair_results if s is not None),
        "n_failed": sum(1 for s in per_pair_results if s is None),
        "gen_model": engine,
        "eval_model": eval_engine,
        "cdt_package_used": cdt_path,
        "cdt_metadata": cdt_metadata,
        "traversal_config": {
            "mode": "gated",
            "relationships_included": has_relationships,
        },
        "per_pair_details": per_pair_results,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, default=str)

    log.info("Results saved to %s", result_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CDT-based RP generation benchmark with Claude")

    parser.add_argument("--character", type=str, default="Kasumi")
    parser.add_argument("--method", type=str, default="cdt_package")

    parser.add_argument(
        "--engine",
        type=str,
        default=HYPOTHESIS_MODEL,
        help="Model for response generation (default: claude-haiku-4-5). Only used when --gen_mode=claude.",
    )
    parser.add_argument(
        "--eval_engine",
        type=str,
        default=EVAL_MODEL,
        help="Model for eval scoring (default: claude-sonnet-4-6)",
    )

    parser.add_argument(
        "--gen_mode",
        type=str,
        choices=["claude", "llama"],
        default="claude",
        help="Generation mode: 'claude' (API via ClaudeCodeAdapter) or 'llama' (local Llama model). "
             "Default: claude.",
    )
    parser.add_argument(
        "--generator_path",
        type=str,
        default="~/models/Llama-3.1-8B-Instruct",
        help="Path to local HF model for RP generation (default: ~/models/Llama-3.1-8B-Instruct). "
             "Used when --gen_mode=llama.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load local generator model in 8-bit quantization (saves ~8GB VRAM).",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=6,
        help="Max concurrent evaluate() calls (default: 6)",
    )

    parser.add_argument(
        "--discriminator_path",
        type=str,
        default="~/models/deberta-v3-base-rp-nli",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--cdt_path",
        type=str,
        default=None,
        help="Path to CDT package pkl. Required when method is 'cdt_package'.",
    )
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        default=False,
        help="Skip relationship CDTs during traversal (attribute topics only).",
    )
    parser.add_argument(
        "--multi-eval",
        action="store_true",
        default=False,
        help="Run eval with both Haiku and Sonnet, output ensemble scores.",
    )
    parser.add_argument(
        "--narration",
        action="store_true",
        default=False,
        help="Use paper's original narration prompt instead of dialogue prompt.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Limit evaluation to first N test pairs (for quick testing).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k observations for hybrid RAG retrieval (default: 10).",
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.3,
        help="Gate similarity threshold for hybrid RAG filtering (default: 0.3).",
    )
    parser.add_argument(
        "--surface_embedder",
        type=str,
        default="~/models/Qwen3-Embedding-0.6B",
        help="Path to surface embedding model for hybrid mode.",
    )
    parser.add_argument(
        "--gen_prompt",
        type=str,
        choices=["narration", "dialogue", "dialogue_v2"],
        default="dialogue",
        help=(
            "Gen prompt style: 'narration' (paper original), "
            "'dialogue' (current default), "
            "'dialogue_v2' (format-matched with length constraint)."
        ),
    )

    return parser


# Gen prompt templates
GEN_PROMPTS: dict[str, str] = {
    "narration": "Answer a concise narration in one sentence.",
    "dialogue": "Answer in one short sentence of in-character dialogue.",
    "dialogue_v2": (
        'Write {character}\'s next line of dialogue. Output ONLY what {character} says, '
        'in the format "{character}: <dialogue>". Keep it under 15 words. '
        'Match the tone and energy level of the scene.'
    ),
}


# ---------------------------------------------------------------------------
# Evaluate a single (scene, action) pair
# ---------------------------------------------------------------------------


def evaluate(
    character: str,
    d: dict[str, Any],
    method: str,
    cdts: dict[str, Any],
    *,
    engine: str = HYPOTHESIS_MODEL,
    eval_engine: str = EVAL_MODEL,
    include_relationships: bool = True,
    narration: bool = False,
    episodic_index: EpisodicIndex | None = None,
    embed_fn: Any | None = None,
    top_k: int = 10,
    gate_threshold: float = 0.3,
    gen_prompt_style: str = "dialogue",
) -> dict[str, Any]:
    """Evaluate a single scene-action pair.

    Returns a dict with all data for analysis:
    - score: numeric (100/50/0)
    - score_letter: A/B/C
    - prediction: generated RP text
    - ground_truth: expected action
    - scene: input scene text
    - grounding: CDT statements used (or None)
    - eval_reasoning: evaluator's reasoning text
    - eval_model: which model scored
    - gen_model: which model generated
    """
    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d["last_character"]
    character = question.split("'s next action")[0].split("be ")[-1] if "'s next action" in question else "Character"

    if narration:
        prompt_suffix = GEN_PROMPTS["narration"]
    else:
        template = GEN_PROMPTS.get(gen_prompt_style, GEN_PROMPTS["dialogue"])
        prompt_suffix = template.format(character=character) if "{character}" in template else template
    prompt = f"""# Scene
{scene}

# Question
{question} {prompt_suffix}"""

    if method == "vanilla":
        grounding = None
    elif method == "extract_then_aggregate":
        with open(f"profiles/{character}.profile.eta.txt", encoding="utf-8") as f:
            grounding = f.read()
    elif method == "human_profile":
        with open(f"profiles/{character}.profile.txt", encoding="utf-8") as f:
            grounding = f.read()
    elif method == "cdt_package":
        grounding = _build_cdt_grounding(
            character,
            scene,
            last_character,
            cdts,
            include_relationships=include_relationships,
        )
    elif method == "hybrid":
        if episodic_index is None or embed_fn is None:
            raise ValueError("hybrid method requires episodic_index and embed_fn")
        grounding = _build_hybrid_grounding(
            character,
            scene,
            last_character,
            cdts,
            episodic_index,
            embed_fn,
            include_relationships=include_relationships,
            top_k=top_k,
            gate_threshold=gate_threshold,
        )
    else:
        grounding = None

    if grounding is not None:
        prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"

    prediction = generate(prompt, model=engine, max_tokens=100)

    score_instruction = f"""# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Score the action of {character} in the response based on the ground truth.
A: The ground truth entails the action of {character} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {character} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {character} in the response. (Following a contradicted character logic.)

Output in json:
```json
{{
"reasoning": "...",
"score": "A/B/C"
}}
```"""

    score_response = generate(score_instruction, model=eval_engine)

    reasoning = ""
    try:
        parsed = extract_json(score_response)
        score_letter = str(parsed.get("score", "B")).strip().upper()
        reasoning = str(parsed.get("reasoning", ""))
    except ValueError:
        log.warning("Failed to parse score JSON, defaulting to B. Response: %.200s", score_response)
        score_letter = "B"
    score = SCORE_MAP.get(score_letter, 50)

    return {
        "score": score,
        "score_letter": score_letter,
        "prediction": prediction,
        "ground_truth": action,
        "scene": scene,
        "grounding": grounding,
        "eval_reasoning": reasoning,
        "eval_model": eval_engine,
        "gen_model": engine,
    }


def evaluate_multi(
    character: str,
    d: dict[str, Any],
    method: str,
    cdts: dict[str, Any],
    *,
    engine: str = HYPOTHESIS_MODEL,
    eval_engines: list[str] | None = None,
    include_relationships: bool = True,
    narration: bool = False,
) -> dict[str, Any]:
    """Evaluate with multiple eval models, returning per-model and ensemble scores.

    Returns dict with keys: prediction, per_model (dict of model->score),
    ensemble_mean, ensemble_min, ensemble_max.
    """
    if eval_engines is None:
        eval_engines = [EVAL_MODEL]

    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d["last_character"]

    prompt_suffix = (
        "Answer a concise narration in one sentence." if narration
        else "Answer in one short sentence of in-character dialogue."
    )
    prompt = f"""# Scene
{scene}

# Question
{question} {prompt_suffix}"""

    if method == "cdt_package":
        grounding = _build_cdt_grounding(
            character, scene, last_character, cdts,
            include_relationships=include_relationships,
        )
    else:
        grounding = None

    if grounding is not None:
        prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"

    # Generate prediction ONCE
    prediction = generate(prompt, model=engine, max_tokens=100)

    score_instruction = f"""# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Score the action of {character} in the response based on the ground truth.
A: The ground truth entails the action of {character} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {character} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {character} in the response. (Following a contradicted character logic.)

Output in json:
```json
{{
"reasoning": "...",
"score": "A/B/C"
}}
```"""

    # Eval with each model
    per_model: dict[str, int] = {}
    for eval_model in eval_engines:
        resp = generate(score_instruction, model=eval_model)
        try:
            parsed = extract_json(resp)
            letter = str(parsed.get("score", "B")).strip().upper()
        except ValueError:
            letter = "B"
        per_model[eval_model] = SCORE_MAP.get(letter, 50)

    scores = list(per_model.values())
    return {
        "prediction": prediction,
        "per_model": per_model,
        "ensemble_mean": sum(scores) / len(scores),
        "ensemble_min": min(scores),
        "ensemble_max": max(scores),
    }


def _build_hybrid_grounding(
    character: str,
    scene: str,
    last_character: list[str],
    cdts: dict[str, Any],
    episodic_index: EpisodicIndex,
    embed_fn: Any,
    *,
    include_relationships: bool = True,
    top_k: int = 10,
    gate_threshold: float = 0.3,
) -> str:
    """Build grounding text using CDT-guided RAG (hybrid approach)."""
    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]

    # Combine attribute + relationship CDTs for traversal
    all_cdts: dict[str, CDTNode] = dict(topic2cdt)
    if include_relationships:
        for c in last_character:
            topic = f"{character}'s interaction with {c}"
            if topic in rel_topic2cdt:
                all_cdts[topic] = rel_topic2cdt[topic]

    result = hybrid_ground(
        scene,
        all_cdts,
        episodic_index,
        embed_fn=embed_fn,
        top_k=top_k,
        gate_threshold=gate_threshold,
    )

    return format_grounding(result, max_behavioral=30, max_factual=top_k)


def _build_cdt_grounding(
    character: str,
    scene: str,
    last_character: list[str],
    cdts: dict[str, Any],
    *,
    include_relationships: bool = True,
) -> str:
    """Build grounding text by traversing CDT trees for the given scene."""
    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]

    statements: list[str] = []
    for topic, cdt_tree in topic2cdt.items():
        statements.append(f"# {topic}")
        statements.extend(cdt_tree.traverse(scene))

    if not include_relationships:
        return "\n".join(statements)

    for c in last_character:
        topic = f"{character}'s interaction with {c}"
        if topic in rel_topic2cdt:
            cdt_tree = rel_topic2cdt[topic]
            statements.append(f"# {topic}")
            statements.extend(cdt_tree.traverse(scene))

    return "\n".join(statements)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def benchmark(
    character: str,
    method: str,
    *,
    engine: str = HYPOTHESIS_MODEL,
    eval_engine: str = EVAL_MODEL,
    discriminator_path: str = "~/models/deberta-v3-base-rp-nli",
    device_id: int = 0,
    max_parallel: int = 6,
    return_list: bool = False,
    cdt_path: str | None = None,
    include_relationships: bool = True,
    multi_eval: bool = False,
    max_pairs: int | None = None,
    narration: bool = False,
    top_k: int = 10,
    gate_threshold: float = 0.3,
    surface_embedder: str = "~/models/Qwen3-Embedding-0.6B",
    gen_prompt_style: str = "dialogue",
) -> list[int] | float:
    """Run the full benchmark for a character.

    Args:
        character: Character name.
        method: Evaluation method.
        engine: Model for response generation.
        eval_engine: Model for eval scoring.
        discriminator_path: Path to DeBERTa NLI model.
        device_id: CUDA device ID.
        max_parallel: Max concurrent evaluate() calls (default 6).
        return_list: If True, return list of scores instead of mean.
        cdt_path: Path to CDT package pkl. Required when method is 'cdt_package'.
        max_pairs: Limit to first N test pairs (for quick testing).

    Returns:
        Mean NLI score (float) or list of individual scores.
    """
    # Initialize DeBERTa for NLI validation (used by CDTNode.traverse)
    device = torch.device(f"cuda:{device_id}")
    resolved_path = os.path.expanduser(discriminator_path)
    init_validation_models(resolved_path, device)

    # Load character metadata and AR pairs
    _, character2artifact, band2members = load_character_metadata()
    test_pairs = load_ar_pairs(character, character2artifact, band2members)["test"]

    # Load CDT package once (hoisted out of evaluate)
    cdts: dict[str, Any] = {}
    pkg_path: str | None = None
    episodic_index: EpisodicIndex | None = None
    embed_fn: Any = None

    if method in ("cdt_package", "hybrid"):
        if cdt_path is None:
            raise ValueError(
                f"--cdt_path is required when method is '{method}'. Specify the path to the CDT package pkl file."
            )
        pkg_path = cdt_path
        cdts = load_cdt_package(pkg_path)
        _validate_cdt_package(cdts, character)
        log.info("Loaded CDT package from %s", pkg_path)

    if method == "hybrid":
        # Build EpisodicIndex from training pairs
        _, character2artifact, band2members = load_character_metadata()
        train_pairs = load_ar_pairs(character, character2artifact, band2members)["train"]
        log.info("Building EpisodicIndex from %d training pairs...", len(train_pairs))

        observations = [
            BehavioralObservation(
                scene=p["scene"],
                action=p["action"],
                actor=character,
            )
            for p in train_pairs
        ]

        # Pre-compute embeddings
        from canopy.embeddings import precompute_embeddings
        resolved_embedder = os.path.expanduser(surface_embedder)
        cache = precompute_embeddings(
            character=character,
            pairs=[o.to_pair() for o in observations],
            surface_embedder_path=resolved_embedder,
            generator_embedder_path=resolved_embedder,  # reuse surface for gen (both small)
            device=f"cuda:{device_id}",
        )
        episodic_index = EpisodicIndex.from_embedding_cache(observations, cache)
        log.info("EpisodicIndex ready: %d entries", len(episodic_index))

        # Build embed_fn for query embedding
        log.info("Loading surface embedder for query embedding...")
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(resolved_embedder)
        _embed_model = AutoModel.from_pretrained(resolved_embedder).to(device)
        _embed_model.eval()

        @torch.no_grad()
        def _embed_fn(text: str) -> np.ndarray:
            inputs = _tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(device)
            outputs = _embed_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1)
            vec = pooled[0].cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

        embed_fn = _embed_fn
        log.info("Hybrid grounding ready")

    # Build evaluation items upfront
    items: list[dict[str, Any]] = []
    for d in test_pairs:
        items.append(
            {
                "condition": d["scene"],
                "question": f"What'll be {character}'s next action in response to the current scene?",
                "action": d["action"],
                "last_character": d["last_character"],
            }
        )

    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError(f"--max_pairs must be a positive integer, got {max_pairs}")
        items = items[:max_pairs]
        log.info("Limiting to first %d test pairs", len(items))

    total = len(items)
    bar = tqdm(total=total, desc="Score=N/A")

    multi_results: list[dict[str, Any] | None] = []
    valid_results: list[dict[str, Any]] = []

    if multi_eval:
        eval_engines = [HYPOTHESIS_MODEL, EVAL_MODEL]
        log.info("Multi-eval mode: evaluating with %s", eval_engines)
        multi_results = [None] * total  # type: list[dict[str, Any] | None]

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for i, item in enumerate(items):
                future = executor.submit(
                    evaluate_multi,
                    character, item, method, cdts,
                    engine=engine, eval_engines=eval_engines,
                    include_relationships=include_relationships,
                    narration=narration,
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    multi_results[idx] = future.result()
                except Exception as exc:
                    log.error("evaluate_multi() failed for pair #%d: %s", idx + 1, exc)
                bar.update(1)
                valid = [r for r in multi_results if r is not None]
                if valid:
                    mean_ens = float(np.mean([r["ensemble_mean"] for r in valid]))
                    bar.set_description(f"Ensemble={mean_ens:.2f}")

        bar.close()
        valid = [r for r in multi_results if r is not None]
        if not valid:
            log.warning("No pairs evaluated")
            return [] if return_list else 0.0

        # Print per-model and ensemble results
        for model_name in eval_engines:
            model_scores = [r["per_model"][model_name] for r in valid]
            log.info("  %s mean: %.2f", model_name, float(np.mean(model_scores)))
        ensemble_scores = [r["ensemble_mean"] for r in valid]
        final_score = float(np.mean(ensemble_scores))
        log.info("  Ensemble mean: %.2f", final_score)
        log.info("  Ensemble min (conservative): %.2f", float(np.mean([r["ensemble_min"] for r in valid])))
        log.info("  Ensemble max (optimistic): %.2f", float(np.mean([r["ensemble_max"] for r in valid])))

        scores = [r["ensemble_mean"] for r in valid]
    else:
        eval_results: list[dict[str, Any] | None] = [None] * total

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for i, item in enumerate(items):
                future = executor.submit(
                    evaluate,
                    character, item, method, cdts,
                    engine=engine, eval_engine=eval_engine,
                    include_relationships=include_relationships,
                    narration=narration,
                    gen_prompt_style=gen_prompt_style,
                    episodic_index=episodic_index,
                    embed_fn=embed_fn,
                    top_k=top_k,
                    gate_threshold=gate_threshold,
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    eval_results[idx] = future.result()
                except Exception as exc:
                    log.error("evaluate() failed for pair #%d, skipping: %s", idx + 1, exc)
                bar.update(1)
                valid = [r for r in eval_results if r is not None]
                if valid:
                    mean_score = float(np.mean([r["score"] for r in valid]))
                    log.info("#%d/%d NLI Score: %.2f", len(valid), total, mean_score)
                    bar.set_description(f"Score={mean_score:.4f}")

        bar.close()
        valid_results = [r for r in eval_results if r is not None]
        scores = [r["score"] for r in valid_results]

    if not scores:
        log.warning("No test pairs evaluated for %s with method %s", character, method)
        return [] if return_list else 0.0

    if not multi_eval:
        final_score = float(np.mean(scores))

    # Save benchmark results with provenance (rich per-pair data when available)
    save_results: list[Any]
    if multi_eval:
        save_results = multi_results  # preserves None slots for n_failed
    else:
        save_results = valid_results
    try:
        _save_benchmark_results(
            character=character,
            method=method,
            engine=engine,
            eval_engine=eval_engine,
            cdt_path=pkg_path if method == "cdt_package" else None,
            cdt_metadata=cdts.get("metadata") if cdts else None,
            score=final_score,
            per_pair_results=save_results,
            has_relationships=(include_relationships and bool(cdts.get("rel_topic2cdt"))) if cdts else False,
        )
    except OSError as exc:
        log.error("Failed to save benchmark results: %s", exc)

    if return_list:
        return scores
    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = build_arg_parser().parse_args()

    # Set up adapter based on gen_mode
    gen_engine = args.engine
    if args.gen_mode == "llama":
        if args.engine != HYPOTHESIS_MODEL:
            log.warning("--engine %s is ignored in llama mode; using 'llama-local'", args.engine)
        from canopy.llm import (
            ClaudeCodeAdapter,
            DispatchAdapter,
            TransformersAdapter,
            set_adapter,
        )

        resolved_gen_path = os.path.expanduser(args.generator_path)
        local_adapter = TransformersAdapter(
            model_path=resolved_gen_path,
            device=f"cuda:{args.device_id}",
            load_in_8bit=args.load_in_8bit,
        )
        claude_adapter = ClaudeCodeAdapter()
        gen_engine = "llama-local"
        dispatch = DispatchAdapter(
            adapters={gen_engine: local_adapter},
            default=claude_adapter,
        )
        set_adapter(dispatch)
        log.info(
            "Using local Llama model %s for gen, Claude for eval (%s)",
            resolved_gen_path, args.eval_engine,
        )

    result = benchmark(
        args.character,
        args.method,
        engine=gen_engine,
        eval_engine=args.eval_engine,
        discriminator_path=args.discriminator_path,
        device_id=args.device_id,
        max_parallel=args.max_parallel,
        cdt_path=args.cdt_path,
        include_relationships=not args.no_relationships,
        multi_eval=args.multi_eval,
        max_pairs=args.max_pairs,
        narration=args.narration,
        top_k=args.top_k,
        gate_threshold=args.gate_threshold,
        surface_embedder=args.surface_embedder,
        gen_prompt_style=args.gen_prompt,
    )
    log.info("Final NLI Score: %.2f", result)


if __name__ == "__main__":
    main()
