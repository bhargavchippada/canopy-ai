"""Quick test: compare gen output formats and their impact on eval scores.

Tests 10 pairs with different gen prompt styles to find the best format.
"""

from __future__ import annotations

import logging
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from canopy.core import CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.llm import ClaudeCodeAdapter, extract_json, generate, set_adapter
from canopy.validation import check_scene as check_scene_batch
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)
CHARACTER = "Kasumi"
SCORE_MAP = {"A": 100, "B": 50, "C": 0}


def traverse(node: object, scene: str) -> list[str]:
    statements = deepcopy(node.statements)  # type: ignore[attr-defined]
    for gate, child in zip(node.gates, node.children):  # type: ignore[attr-defined]
        results = check_scene_batch([scene], [gate])
        if results[0]:
            statements.extend(traverse(child, scene))
    return statements


def build_grounding(cdts: dict, scene: str, last_character: list[str]) -> str:
    statements: list[str] = []
    for topic, tree in cdts["topic2cdt"].items():
        statements.append(f"# {topic}")
        statements.extend(traverse(tree, scene))
    return "\n".join(statements)


GEN_PROMPTS = {
    "default": "Answer a concise narration in one sentence.",
    "dialogue": "Answer in character as a single line of dialogue, not narration.",
    "action": "Answer with a brief action description in one sentence, like a screenplay direction.",
    "first_person": "Answer as the character would speak, in first person, one sentence.",
}


def evaluate_with_format(
    d: dict, cdts: dict, gen_style: str, engine: str, eval_engine: str,
) -> dict:
    scene = d["condition"]
    question = d["question"]
    action = d["action"]
    last_character = d.get("last_character", [])

    grounding = build_grounding(cdts, scene, last_character)
    suffix = GEN_PROMPTS[gen_style]

    prompt = f"# Background Knowledge\n{grounding}\n\n# Scene\n{scene}\n\n# Question\n{question} {suffix}"

    prediction = generate(prompt, model=engine)

    score_instruction = f"""# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Score the action of {CHARACTER} in the response based on the ground truth.
A: The ground truth entails the action of {CHARACTER} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {CHARACTER} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {CHARACTER} in the response. (Following a contradicted character logic.)

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
        reasoning = ""

    return {
        "style": gen_style,
        "score": SCORE_MAP.get(score_letter, 50),
        "letter": score_letter,
        "prediction": prediction,
        "ground_truth": action,
        "reasoning": reasoning,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    set_adapter(ClaudeCodeAdapter(default_model="claude-sonnet-4-6"))
    device = torch.device("cuda:0")
    disc_path = str(Path.home() / "models" / "deberta-v3-base-rp-nli")
    init_validation_models(disc_path, device)

    class _LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> type:
            if name == "CDT_Node":
                return CDTNode
            return super().find_class(module, name)

    with open("packages/Kasumi.paper-original.pkl", "rb") as f:
        cdts = _LegacyUnpickler(f).load()

    _, character2artifact, band2members = load_character_metadata()
    pairs = load_ar_pairs(CHARACTER, character2artifact, band2members)["test"]

    test_pairs = [
        {
            "condition": d["scene"],
            "question": f"What'll be {CHARACTER}'s next action in response to the current scene?",
            "action": d["action"],
            "last_character": d.get("last_character", []),
        }
        for d in pairs[:10]
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"Testing {len(test_pairs)} pairs x {len(GEN_PROMPTS)} styles = {len(test_pairs) * len(GEN_PROMPTS)} evals")

    all_results: dict[str, list[dict]] = {style: [] for style in GEN_PROMPTS}

    for style in GEN_PROMPTS:
        print(f"\n--- Style: {style} ---")
        futures_map = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            for i, d in enumerate(test_pairs):
                future = executor.submit(
                    evaluate_with_format, d, cdts, style,
                    "claude-sonnet-4-6", "claude-sonnet-4-6",
                )
                futures_map[future] = i

            for future in as_completed(futures_map):
                try:
                    result = future.result()
                    all_results[style].append(result)
                except Exception as exc:
                    log.error("Failed: %s", exc)

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    for style, results in all_results.items():
        scores = [r["score"] for r in results]
        letters = [r["letter"] for r in results]
        mean = np.mean(scores) if scores else 0
        a_count = letters.count("A")
        b_count = letters.count("B")
        c_count = letters.count("C")
        print(f"\n{style:15s}: mean={mean:.1f}  A={a_count} B={b_count} C={c_count}")
        for r in results[:3]:
            pred_short = r["prediction"][:80].replace("\n", " ")
            gt_short = r["ground_truth"][:60].replace("\n", " ")
            print(f"  [{r['letter']}] pred: {pred_short}...")
            print(f"       gt:   {gt_short}")


if __name__ == "__main__":
    main()
