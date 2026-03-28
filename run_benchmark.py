"""CDT benchmark — evaluate character response quality via NLI scoring.

Loads pre-built CDT packages, generates character responses using Claude,
and scores them against ground truth using NLI (DeBERTa) and LLM evaluation.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
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
        "--discriminator_path",
        type=str,
        default="~/models/deberta-v3-base-rp-nli",
    )
    parser.add_argument("--device_id", type=int, default=0)

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
        grounding = _build_cdt_grounding(character, scene, last_character, cdts)
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
) -> str:
    """Build grounding text by traversing CDT trees for the given scene."""
    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]

    statements: list[str] = []
    for topic, cdt_tree in topic2cdt.items():
        statements.append(f"# {topic}")
        statements.extend(cdt_tree.traverse(scene))

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
    return_list: bool = False,
) -> list[int] | float:
    """Run the full benchmark for a character.

    Args:
        character: Character name.
        method: Evaluation method.
        engine: Model for response generation.
        eval_engine: Model for eval scoring.
        discriminator_path: Path to DeBERTa NLI model.
        device_id: CUDA device ID.
        return_list: If True, return list of scores instead of mean.

    Returns:
        Mean NLI score (float) or list of individual scores.
    """
    # Initialize DeBERTa for NLI validation (used by CDTNode.traverse)
    device = torch.device(f"cuda:{device_id}")
    resolved_path = os.path.expanduser(discriminator_path)
    init_validation_models(resolved_path, device)

    # Load character metadata and AR pairs
    _, character2artifact, band2members = load_character_metadata()

    if method == "cdt_package" and character not in character2artifact:
        raise ValueError(f"Unknown character '{character}'. Available: {sorted(character2artifact.keys())}")

    test_pairs = load_ar_pairs(character, character2artifact, band2members)["test"]

    # Load CDT package once (hoisted out of evaluate)
    cdts: dict[str, Any] = {}
    if method == "cdt_package":
        pkg_path = f"packages/{character}.cdt.v3.1.package.relation.pkl"
        cdts = load_cdt_package(pkg_path)
        log.info("Loaded CDT package from %s", pkg_path)

    scores: list[int] = []
    bar = tqdm(test_pairs)

    for d in bar:
        item = {
            "condition": d["scene"],
            "question": f"What'll be {character}'s next action in response to the current scene?",
            "action": d["action"],
            "last_character": d["last_character"],
        }
        try:
            score = evaluate(
                character,
                item,
                method,
                cdts,
                engine=engine,
                eval_engine=eval_engine,
            )
        except Exception as exc:
            log.error("evaluate() failed for pair #%d, skipping: %s", len(scores) + 1, exc)
            continue

        scores.append(score)
        mean_score = float(np.mean(scores))
        log.info("#%d # NLI Score: %.2f", len(scores), mean_score)
        bar.set_description(f"Score={mean_score:.4f}")

    if not scores:
        log.warning("No test pairs evaluated for %s with method %s", character, method)
        return [] if return_list else 0.0

    if return_list:
        return scores
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = build_arg_parser().parse_args()
    result = benchmark(
        args.character,
        args.method,
        engine=args.engine,
        eval_engine=args.eval_engine,
        discriminator_path=args.discriminator_path,
        device_id=args.device_id,
    )
    log.info("Final NLI Score: %.2f", result)


if __name__ == "__main__":
    main()
