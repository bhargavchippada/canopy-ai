"""Run the paper's EXACT benchmark pipeline with Claude, for gap analysis.

Replicates the paper's run_benchmark.py logic exactly:
- Same CDT loading (paper's pickle)
- Same traverse (DeBERTa check_scene)
- Same prompt format
- Same score parsing (strict regex, crash on failure)
- BUT uses Claude for gen+eval instead of Llama+GPT-4.1

Usage:
    uv run python benchmark_papercompat.py --cdt_path packages/Kasumi.paper-original.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from copy import deepcopy

import torch
from tqdm import tqdm

from canopy.data import load_ar_pairs, load_character_metadata
from canopy.llm import ClaudeCodeAdapter, generate, set_adapter
from canopy.validation import check_scene as check_scene_batch
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

CHARACTER = "Kasumi"


def check_scene_single(text: str, question: str) -> bool | None:
    """Match paper's scalar check_scene signature."""
    results = check_scene_batch([text], [question])
    return results[0]


def traverse(node: object, scene: str) -> list[str]:
    """Paper-compatible traverse using our DeBERTa check_scene."""
    statements = deepcopy(node.statements)  # type: ignore[attr-defined]
    for gate, child in zip(node.gates, node.children):  # type: ignore[attr-defined]
        if check_scene_single(scene, gate):
            statements.extend(traverse(child, scene))
    return statements


def evaluate(
    character: str,
    d: dict,
    cdts: dict,
    engine: str,
    eval_engine: str,
    strict_parse: bool = True,
    strip_first_line: bool = False,
) -> tuple[int, dict]:
    """Paper-compatible evaluate with configurable options."""
    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d.get("last_character", [])

    # Build grounding (exact paper logic)
    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]
    statements: list[str] = []
    for topic in topic2cdt:
        cdt_tree = topic2cdt[topic]
        statements.append(f"# {topic}")
        statements.extend(traverse(cdt_tree, scene))
    for c in last_character:
        topic = f"{character}'s interaction with {c}"
        if topic in rel_topic2cdt:
            cdt_tree = rel_topic2cdt[topic]
            statements.append(f"# {topic}")
            statements.extend(traverse(cdt_tree, scene))
    grounding = "\n".join(statements)

    # RP generation prompt (exact paper format)
    prompt = f"""# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence."""

    prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"

    # Generate RP response
    prediction = generate(prompt, model=engine)

    if strip_first_line:
        prediction = prediction.strip().split("\n")[0]

    # Eval scoring prompt (exact paper format)
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

    # Score parsing
    if strict_parse:
        # Paper's exact regex — crashes on failure
        matches = re.findall(r'"score": "(.*?)"', score_response, re.DOTALL)
        if not matches:
            return -1, {"prediction": prediction, "error": "no score match"}
        score_letter = matches[0].strip()
        score = {"A": 100, "B": 50, "C": 0}.get(score_letter, -1)
        if score == -1:
            return -1, {"prediction": prediction, "error": f"unknown score: {score_letter}"}
    else:
        # Our fallback parsing
        try:
            from canopy.llm import extract_json
            parsed = extract_json(score_response)
            score_letter = str(parsed.get("score", "B")).strip().upper()
        except ValueError:
            score_letter = "B"
        score = {"A": 100, "B": 50, "C": 0}.get(score_letter, 50)

    return score, {"prediction": prediction, "score_letter": score_letter, "action": action}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cdt_path", required=True)
    parser.add_argument("--engine", default="claude-sonnet-4-6")
    parser.add_argument("--eval_engine", default="claude-sonnet-4-6")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--strict_parse", action="store_true", default=True)
    parser.add_argument("--no_strict_parse", action="store_true")
    parser.add_argument("--strip_first_line", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=None, help="Limit pairs for quick test")
    args = parser.parse_args()

    strict = not args.no_strict_parse

    set_adapter(ClaudeCodeAdapter(default_model=args.engine))
    device = torch.device(f"cuda:{args.device_id}")
    from pathlib import Path
    disc_path = str(Path.home() / "models" / "deberta-v3-base-rp-nli")
    init_validation_models(disc_path, device)

    # Load CDT (with legacy class mapping)
    from canopy.core import CDTNode

    class _LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> type:
            if name == "CDT_Node":
                return CDTNode
            return super().find_class(module, name)

    with open(args.cdt_path, "rb") as f:
        cdts = _LegacyUnpickler(f).load()

    # Load test pairs (exact paper format)
    all_characters, character2artifact, band2members = load_character_metadata()
    pairs = load_ar_pairs(CHARACTER, character2artifact, band2members)["test"]

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    # Format like paper
    test_pairs = []
    for d in pairs:
        test_pairs.append({
            "condition": d["scene"],
            "question": f"What'll be {CHARACTER}'s next action in response to the current scene?",
            "action": d["action"],
            "last_character": d.get("last_character", []),
        })

    print(f"Running {len(test_pairs)} pairs | engine={args.engine} eval={args.eval_engine}")
    print(f"strict_parse={strict} strip_first_line={args.strip_first_line}")

    scores = []
    failures = 0
    details = []
    bar = tqdm(test_pairs)
    for d in bar:
        score, detail = evaluate(
            CHARACTER, d, cdts,
            engine=args.engine,
            eval_engine=args.eval_engine,
            strict_parse=strict,
            strip_first_line=args.strip_first_line,
        )
        if score == -1:
            failures += 1
            details.append(detail)
        else:
            scores.append(score)
            details.append(detail)
        if scores:
            bar.set_description(f"Score={sum(scores)/len(scores):.2f} (fail={failures})")

    if scores:
        final = sum(scores) / len(scores)
        print(f"\nFinal NLI Score: {final:.2f}")
        print(f"Pairs evaluated: {len(scores)}/{len(test_pairs)}")
        print(f"Parse failures: {failures}")
        print(f"Score distribution: A={sum(1 for d in details if d.get('score_letter')=='A')}, "
              f"B={sum(1 for d in details if d.get('score_letter')=='B')}, "
              f"C={sum(1 for d in details if d.get('score_letter')=='C')}")
    else:
        print("No pairs evaluated!")


if __name__ == "__main__":
    main()
