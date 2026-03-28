#!/usr/bin/env python3
"""Build golden evaluation set: generate predictions for all 167 test pairs.

Generates Sonnet predictions + Sonnet eval scores, exports for human annotation.
Run once — multi_eval.py then re-evaluates the same predictions with other models.

Usage:
    uv run python build_golden_set.py \
      --cdt_path packages/Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a80.r50.relation.pkl

Output: artifacts/golden_set_kasumi.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from canopy.core import CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.llm import ClaudeCodeAdapter, extract_json, generate, set_adapter
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

SCORE_MAP = {"A": 100, "B": 50, "C": 0}


def load_cdt_package(path: str) -> dict:
    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "CDT_Node":
                return CDTNode
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return _U(f).load()


def build_grounding(character, scene, last_character, cdts):
    stmts = []
    for topic, tree in cdts["topic2cdt"].items():
        stmts.append(f"# {topic}")
        stmts.extend(tree.traverse(scene))
    return "\n".join(stmts)


def run_eval(scene, prediction, action, character, model):
    """Run A/B/C evaluation with a specific model."""
    prompt = f"""# Scene
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
    response = generate(prompt, model=model)
    try:
        parsed = extract_json(response)
        letter = str(parsed.get("score", "B")).strip().upper()
        reasoning = str(parsed.get("reasoning", ""))
    except ValueError:
        letter = "?"
        reasoning = "[PARSE FAILED]"
    return letter, reasoning


def process_pair(character, pair_idx, item, cdts, engine, eval_engine):
    scene = item["condition"]
    action = item["action"]
    last_character = item["last_character"]
    question = item["question"]

    # Generate prediction
    grounding = build_grounding(character, scene, last_character, cdts)
    gen_prompt = (
        f"# Background Knowledge\n{grounding}\n\n"
        f"# Scene\n{scene}\n\n"
        f"# Question\n{question} Answer a concise narration in one sentence."
    )
    prediction = generate(gen_prompt, model=engine)

    # Evaluate with Sonnet
    sonnet_letter, sonnet_reasoning = run_eval(scene, prediction, action, character, eval_engine)

    # Truncate scene
    scene_short = ("..." + scene[-300:]) if len(scene) > 300 else scene

    return {
        "pair_idx": pair_idx,
        "scene_truncated": scene_short,
        "ground_truth": action,
        "prediction": prediction,
        "sonnet_score": sonnet_letter,
        "sonnet_reasoning": sonnet_reasoning,
        "human_score": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", default="Kasumi")
    parser.add_argument("--engine", default="claude-sonnet-4-6")
    parser.add_argument("--eval_engine", default="claude-sonnet-4-6")
    parser.add_argument("--cdt_path", required=True)
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--discriminator_path", default="~/models/deberta-v3-base-rp-nli")
    parser.add_argument("--output", default="artifacts/golden_set_kasumi.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    set_adapter(ClaudeCodeAdapter(default_model=args.engine, max_concurrent=args.max_parallel))
    device = torch.device(f"cuda:{args.device_id}")
    init_validation_models(os.path.expanduser(args.discriminator_path), device)

    _, c2a, b2m = load_character_metadata()
    test_pairs = load_ar_pairs(args.character, c2a, b2m)["test"]
    cdts = load_cdt_package(args.cdt_path)

    items = [{
        "condition": d["scene"],
        "question": f"What'll be {args.character}'s next action in response to the current scene?",
        "action": d["action"],
        "last_character": d["last_character"],
    } for d in test_pairs]

    total = len(items)
    log.info("Building golden set: %d pairs", total)

    results = [None] * total
    bar = tqdm(total=total, desc="Golden set")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(process_pair, args.character, i, item, cdts, args.engine, args.eval_engine): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log.error("Pair #%d failed: %s", idx, exc)
                results[idx] = {"pair_idx": idx, "error": str(exc), "human_score": None}
            bar.update(1)
    bar.close()

    valid = [r for r in results if r and "sonnet_score" in r]
    dist = {"A": 0, "B": 0, "C": 0, "?": 0}
    for r in valid:
        dist[r["sonnet_score"]] = dist.get(r["sonnet_score"], 0) + 1

    golden = {
        "metadata": {
            "character": args.character,
            "gen_model": args.engine,
            "eval_model": args.eval_engine,
            "cdt_path": args.cdt_path,
            "total_pairs": total,
            "sonnet_distribution": dist,
        },
        "pairs": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    log.info("Saved to %s — A=%d B=%d C=%d", args.output, dist["A"], dist["B"], dist["C"])


if __name__ == "__main__":
    main()
