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

from canopy.core import CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
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
    per_pair_results: list[int | None],
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
        "per_pair_scores": per_pair_results,
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
        help="Model for response generation (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--eval_engine",
        type=str,
        default=EVAL_MODEL,
        help="Model for eval scoring (default: claude-sonnet-4-6)",
    )

    parser.add_argument(
        "--generator_path",
        type=str,
        default=None,
        help="Path to local HF model for RP generation (e.g. ~/models/Llama-3.1-8B-Instruct). "
             "When set, --engine becomes a label and the local model is used for gen.",
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

    return parser


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
) -> int:
    """Evaluate a single scene-action pair.

    Requires ``init_validation_models()`` to have been called first
    (handled by ``benchmark()``). Direct callers must initialize models.

    Args:
        character: Character name.
        d: Dict with keys: condition, question, action, last_character.
        method: Evaluation method (cdt_package, vanilla, etc.).
        cdts: Pre-loaded CDT package dict with topic2cdt and rel_topic2cdt.
        engine: Model for response generation.
        eval_engine: Model for eval scoring.

    Returns:
        Score: 100 (entails), 50 (neutral), or 0 (contradicts).
    """
    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d["last_character"]

    prompt = f"""# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence."""

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
    else:
        grounding = None

    if grounding is not None:
        prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"

    prediction = generate(prompt, model=engine)

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

    log.debug("-" * 100)
    log.debug("Prediction: %s", prediction)
    log.debug("." * 100)
    log.debug("Ground truth: %s", action)
    log.debug("-" * 100)

    score_response = generate(score_instruction, model=eval_engine)

    try:
        parsed = extract_json(score_response)
        score_letter = str(parsed.get("score", "B")).strip().upper()
    except ValueError:
        log.warning("Failed to parse score JSON, defaulting to B. Response: %.200s", score_response)
        score_letter = "B"
    score = SCORE_MAP.get(score_letter, 50)

    return score


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
    if method == "cdt_package":
        if cdt_path is None:
            raise ValueError(
                "--cdt_path is required when method is 'cdt_package'. Specify the path to the CDT package pkl file."
            )
        pkg_path = cdt_path
        cdts = load_cdt_package(pkg_path)
        _validate_cdt_package(cdts, character)
        log.info("Loaded CDT package from %s", pkg_path)

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

    total = len(items)
    results: list[int | None] = [None] * total
    bar = tqdm(total=total, desc="Score=N/A")

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for i, item in enumerate(items):
            future = executor.submit(
                evaluate,
                character,
                item,
                method,
                cdts,
                engine=engine,
                eval_engine=eval_engine,
                include_relationships=include_relationships,
            )
            futures[future] = i

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log.error("evaluate() failed for pair #%d, skipping: %s", idx + 1, exc)

            completed += 1
            bar.update(1)
            valid_scores = [s for s in results if s is not None]
            if valid_scores:
                mean_score = float(np.mean(valid_scores))
                log.info(
                    "#%d/%d NLI Score: %.2f",
                    len(valid_scores),
                    total,
                    mean_score,
                )
                bar.set_description(f"Score={mean_score:.4f}")

    bar.close()

    scores = [s for s in results if s is not None]

    if not scores:
        log.warning("No test pairs evaluated for %s with method %s", character, method)
        return [] if return_list else 0.0

    final_score = float(np.mean(scores))

    # Save benchmark results with provenance
    try:
        _save_benchmark_results(
            character=character,
            method=method,
            engine=engine,
            eval_engine=eval_engine,
            cdt_path=pkg_path if method == "cdt_package" else None,
            cdt_metadata=cdts.get("metadata") if cdts else None,
            score=final_score,
            per_pair_results=results,
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

    # Set up adapter: DispatchAdapter when local generator is specified
    if args.generator_path:
        from canopy.llm import (
            ClaudeCodeAdapter,
            DispatchAdapter,
            TransformersAdapter,
            set_adapter,
        )

        local_adapter = TransformersAdapter(
            model_path=args.generator_path,
            device=f"cuda:{args.device_id}",
        )
        claude_adapter = ClaudeCodeAdapter()
        dispatch = DispatchAdapter(
            adapters={args.engine: local_adapter},
            default=claude_adapter,
        )
        set_adapter(dispatch)
        log.info(
            "Using local model %s for gen (%s), Claude for eval (%s)",
            args.generator_path, args.engine, args.eval_engine,
        )

    result = benchmark(
        args.character,
        args.method,
        engine=args.engine,
        eval_engine=args.eval_engine,
        discriminator_path=args.discriminator_path,
        device_id=args.device_id,
        max_parallel=args.max_parallel,
        cdt_path=args.cdt_path,
        include_relationships=not args.no_relationships,
    )
    log.info("Final NLI Score: %.2f", result)


if __name__ == "__main__":
    main()
