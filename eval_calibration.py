#!/usr/bin/env python3
"""Meta-evaluation: is Sonnet scoring B too often, or is GPT-4.1 too lenient?

Samples 20 pairs from the test set, runs generation + evaluation with full
output, and prints a human-readable table for manual calibration.

Usage:
    uv run python eval_calibration.py --cdt_path packages/Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a80.r50.relation.pkl
    uv run python eval_calibration.py --only-b   # Only show B-scored pairs
    uv run python eval_calibration.py --n 10      # Sample 10 pairs

Quick script — not production code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import textwrap
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from canopy.core import CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.llm import ClaudeCodeAdapter, extract_json, generate, set_adapter
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

SCORE_MAP = {"A": 100, "B": 50, "C": 0}
SCORE_LABEL = {100: "A (entails)", 50: "B (neutral)", 0: "C (contradicts)"}


def load_cdt_package(path: str) -> dict:
    """Load CDT with legacy class mapping."""
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "CDT_Node":
                return CDTNode
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def build_grounding(character, scene, last_character, cdts, include_rel=False):
    """Traverse CDT and build grounding text."""
    topic2cdt = cdts["topic2cdt"]
    rel_topic2cdt = cdts["rel_topic2cdt"]
    statements = []
    for topic, tree in topic2cdt.items():
        statements.append(f"# {topic}")
        statements.extend(tree.traverse(scene))
    if include_rel:
        for c in last_character:
            topic = f"{character}'s interaction with {c}"
            if topic in rel_topic2cdt:
                statements.append(f"# {topic}")
                statements.extend(rel_topic2cdt[topic].traverse(scene))
    return "\n".join(statements)


def evaluate_pair(character, d, cdts, engine, eval_engine):
    """Run full evaluate pipeline, return all intermediate data."""
    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d["last_character"]

    # Build prompt
    gen_prompt = f"""# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence."""

    grounding = build_grounding(character, scene, last_character, cdts)
    full_prompt = f"# Background Knowledge\n{grounding}\n\n{gen_prompt}"

    # Generate prediction
    prediction = generate(full_prompt, model=engine)

    # Build eval prompt
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

    try:
        parsed = extract_json(score_response)
        score_letter = str(parsed.get("score", "B")).strip().upper()
        reasoning = str(parsed.get("reasoning", ""))
    except ValueError:
        score_letter = "B"
        reasoning = f"[PARSE FAILED] Raw: {score_response[:200]}"

    score = SCORE_MAP.get(score_letter, 50)

    return {
        "scene": scene,
        "ground_truth": action,
        "prediction": prediction,
        "grounding_stmts": grounding.count("\n") + 1,
        "score_letter": score_letter,
        "score": score,
        "reasoning": reasoning,
    }


def main():
    parser = argparse.ArgumentParser(description="Meta-evaluation calibration")
    parser.add_argument("--character", default="Kasumi")
    parser.add_argument("--engine", default="claude-sonnet-4-6")
    parser.add_argument("--eval_engine", default="claude-sonnet-4-6")
    parser.add_argument("--cdt_path", required=True)
    parser.add_argument("--n", type=int, default=20, help="Number of pairs to sample")
    parser.add_argument("--only-b", action="store_true", help="Only show B-scored pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--discriminator_path", default="~/models/deberta-v3-base-rp-nli")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # Setup
    set_adapter(ClaudeCodeAdapter(default_model=args.engine))
    device = torch.device(f"cuda:{args.device_id}")
    init_validation_models(os.path.expanduser(args.discriminator_path), device)

    # Load data
    _, c2a, b2m = load_character_metadata()
    test_pairs = load_ar_pairs(args.character, c2a, b2m)["test"]
    cdts = load_cdt_package(args.cdt_path)

    # Build eval items
    items = []
    for d in test_pairs:
        items.append({
            "condition": d["scene"],
            "question": f"What'll be {args.character}'s next action in response to the current scene?",
            "action": d["action"],
            "last_character": d["last_character"],
        })

    # Sample
    random.seed(args.seed)
    sample = random.sample(range(len(items)), min(args.n, len(items)))

    print(f"Evaluating {len(sample)} pairs (seed={args.seed})")
    print(f"Gen: {args.engine}, Eval: {args.eval_engine}")
    print(f"CDT: {args.cdt_path}")
    print()

    results = []
    for idx, pair_idx in enumerate(sample):
        print(f"  [{idx+1}/{len(sample)}] Evaluating pair #{pair_idx}...", end=" ", flush=True)
        result = evaluate_pair(args.character, items[pair_idx], cdts, args.engine, args.eval_engine)
        result["pair_idx"] = pair_idx
        results.append(result)
        print(f"{result['score_letter']}")

    # Filter if --only-b
    if args.only_b:
        results = [r for r in results if r["score_letter"] == "B"]
        print(f"\nFiltered to {len(results)} B-scored pairs")

    # Print table
    w = textwrap.TextWrapper(width=90, subsequent_indent="    ")

    counts = {"A": 0, "B": 0, "C": 0}
    for r in results:
        counts[r["score_letter"]] = counts.get(r["score_letter"], 0) + 1

    print(f"\n{'='*95}")
    print(f"CALIBRATION RESULTS — {len(results)} pairs")
    print(f"Distribution: A={counts.get('A',0)} B={counts.get('B',0)} C={counts.get('C',0)}")
    print(f"{'='*95}")

    for i, r in enumerate(results):
        print(f"\n{'─'*95}")
        print(f"PAIR #{r['pair_idx']} — Sonnet scored: {r['score_letter']} ({r['score']})")
        print(f"{'─'*95}")

        # Scene (last 200 chars for readability)
        scene_short = r["scene"][-200:] if len(r["scene"]) > 200 else r["scene"]
        if len(r["scene"]) > 200:
            scene_short = "..." + scene_short
        print(f"\n  SCENE (last 200 chars):")
        for line in w.wrap(scene_short):
            print(f"    {line}")

        print(f"\n  GROUND TRUTH:")
        for line in w.wrap(r["ground_truth"]):
            print(f"    {line}")

        print(f"\n  PREDICTION:")
        for line in w.wrap(r["prediction"]):
            print(f"    {line}")

        print(f"\n  EVAL REASONING:")
        for line in w.wrap(r["reasoning"]):
            print(f"    {line}")

        print(f"\n  GROUNDING: {r['grounding_stmts']} statements from CDT")

    # Summary
    print(f"\n{'='*95}")
    print(f"MANUAL CALIBRATION CHECKLIST")
    print(f"{'='*95}")
    print(f"For each B-scored pair above, ask:")
    print(f"  1. Does the prediction follow the same CHARACTER LOGIC as ground truth? → should be A")
    print(f"  2. Does the prediction show a DIFFERENT behavioral facet? → correct B")
    print(f"  3. Does the prediction CONTRADICT the ground truth behavior? → should be C")
    print()
    print(f"If many Bs look like As: Sonnet is too conservative (the gap is evaluator bias)")
    print(f"If Bs look correct: GPT-4.1 is too lenient (our scores are more accurate)")


if __name__ == "__main__":
    main()
