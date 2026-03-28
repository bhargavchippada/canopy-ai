#!/usr/bin/env python3
"""Re-evaluate golden set predictions with multiple evaluator models.

Takes the golden set (with predictions already generated) and runs A/B/C
evaluation with Haiku (and optionally other models). Adds columns to the
golden set for cross-evaluator comparison.

Usage:
    uv run python multi_eval.py \
      --golden artifacts/golden_set_kasumi.json \
      --eval_model claude-haiku-4-5

Output: Updates the golden set JSON in-place with haiku_score column.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from canopy.llm import ClaudeCodeAdapter, extract_json, generate, set_adapter

log = logging.getLogger(__name__)


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


def eval_pair(pair, character, eval_model, score_key):
    """Evaluate a single pair, return (pair_idx, score_letter, reasoning)."""
    if "error" in pair:
        return pair["pair_idx"], "?", "[SKIPPED — generation error]"

    # Use full scene from scene_truncated (it's all we have in golden set)
    scene = pair["scene_truncated"]
    prediction = pair["prediction"]
    action = pair["ground_truth"]

    letter, reasoning = run_eval(scene, prediction, action, character, eval_model)
    return pair["pair_idx"], letter, reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True, help="Path to golden_set_kasumi.json")
    parser.add_argument("--eval_model", default="claude-haiku-4-5", help="Evaluator model to add")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--character", default="Kasumi")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Derive the score key from model name
    model_short = args.eval_model.split("-")[1] if "-" in args.eval_model else args.eval_model
    score_key = f"{model_short}_score"
    reasoning_key = f"{model_short}_reasoning"

    log.info("Re-evaluating with %s → keys: %s, %s", args.eval_model, score_key, reasoning_key)

    set_adapter(ClaudeCodeAdapter(default_model=args.eval_model, max_concurrent=args.max_parallel))

    # Load golden set
    with open(args.golden) as f:
        golden = json.load(f)

    pairs = golden["pairs"]
    total = len(pairs)

    # Skip pairs that already have this evaluator's score
    to_eval = [(i, p) for i, p in enumerate(pairs) if score_key not in p]
    if not to_eval:
        log.info("All pairs already have %s — nothing to do", score_key)
        return

    log.info("Evaluating %d/%d pairs with %s", len(to_eval), total, args.eval_model)

    bar = tqdm(total=len(to_eval), desc=f"Eval ({model_short})")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(eval_pair, pair, args.character, args.eval_model, score_key): idx
            for idx, pair in to_eval
        }
        for future in as_completed(futures):
            list_idx = futures[future]
            try:
                pair_idx, letter, reasoning = future.result()
                pairs[list_idx][score_key] = letter
                pairs[list_idx][reasoning_key] = reasoning
            except Exception as exc:
                log.error("Pair #%d failed: %s", list_idx, exc)
                pairs[list_idx][score_key] = "?"
                pairs[list_idx][reasoning_key] = f"[ERROR] {exc}"
            bar.update(1)

    bar.close()

    # Compute distribution
    dist = {"A": 0, "B": 0, "C": 0, "?": 0}
    for p in pairs:
        s = p.get(score_key, "?")
        dist[s] = dist.get(s, 0) + 1

    # Update metadata
    golden["metadata"][f"{model_short}_distribution"] = dist

    # Cross-evaluator comparison
    if "sonnet_score" in pairs[0]:
        agree = sum(1 for p in pairs if p.get("sonnet_score") == p.get(score_key))
        sonnet_a_other_b = sum(1 for p in pairs
                               if p.get("sonnet_score") == "B" and p.get(score_key) == "A")
        sonnet_b_other_a = sum(1 for p in pairs
                               if p.get("sonnet_score") == "A" and p.get(score_key) == "B")
        golden["metadata"]["cross_eval"] = {
            "agreement_rate": f"{agree}/{total} ({agree/total*100:.1f}%)",
            "sonnet_B_other_A": sonnet_a_other_b,
            "sonnet_A_other_B": sonnet_b_other_a,
        }
        log.info("Agreement: %d/%d (%.1f%%)", agree, total, agree/total*100)
        log.info("Sonnet=B but %s=A: %d (Sonnet too conservative)", model_short, sonnet_a_other_b)
        log.info("Sonnet=A but %s=B: %d (Sonnet too lenient)", model_short, sonnet_b_other_a)

    # Save
    with open(args.golden, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    log.info("Updated %s — %s distribution: A=%d B=%d C=%d",
             args.golden, model_short, dist["A"], dist["B"], dist["C"])

    # Print summary table
    print(f"\n{'='*60}")
    print(f"CROSS-EVALUATOR COMPARISON")
    print(f"{'='*60}")
    print(f"{'Pair':>5} {'Sonnet':>8} {model_short:>8} {'Match':>6}")
    print(f"{'-'*35}")

    mismatches = 0
    for p in pairs:
        s = p.get("sonnet_score", "?")
        o = p.get(score_key, "?")
        if s != o:
            mismatches += 1
            print(f"  {p['pair_idx']:>3}   {s:>5}    {o:>5}    {'':>4}✗")

    print(f"\nMismatches: {mismatches}/{total} ({mismatches/total*100:.1f}%)")


if __name__ == "__main__":
    main()
