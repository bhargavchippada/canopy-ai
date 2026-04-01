"""LifeChoice benchmark: CDT-augmented decision prediction.

Compares baseline (direct prompting) vs CDT-augmented accuracy
on the LifeChoice MCQ dataset (Character is Destiny, arxiv 2404.12138).

Usage:
    uv run python scripts/lifechoice_benchmark.py --max_characters 5 --model claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from canopy.llm import ClaudeCodeAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/LifeChoice/data/data.json")


def load_data() -> list[list[dict]]:
    """Load LifeChoice dataset, filtering empty groups."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [group for group in data if group]


def build_mcq_prompt(
    character_name: str,
    memory: str,
    scenario: str,
    question: str,
    options: list[str],
    grounding: str | None = None,
) -> str:
    """Build MCQ prompt following LifeChoice paper format."""
    grounding_section = ""
    if grounding:
        grounding_section = f"""
1.3. Behavioral Profile (CDT Analysis)
{grounding}
"""

    options_text = "\n".join(
        f"{chr(65 + i)}. {opt.lstrip('ABCD. ')}" for i, opt in enumerate(options)
    )

    return f"""Please play the role of {character_name} based on the Profile \
and make your life choice under the Scenario regarding Question. Return ONLY \
the option letter (A, B, C, or D) that your character should most \
appropriately choose in the current scenario.

# Inputs:
1. Profile:
1.1. Description
{character_name}

1.2. Memory
{memory}
{grounding_section}
2. Scenario:
{scenario}

3. Question:
{question}

4. Options:
{options_text}

# Output:
Your choice (A, B, C, or D):"""


def extract_answer(response: str) -> str | None:
    """Extract single letter answer from LLM response."""
    response = response.strip()
    if len(response) == 1 and response.upper() in "ABCD":
        return response.upper()
    match = re.search(r"\b([ABCD])\b", response)
    if match:
        return match.group(1).upper()
    return None


async def build_llm_profile(
    character_name: str,
    input_text: str,
    adapter: ClaudeCodeAdapter,
    model: str,
) -> str | None:
    """Use LLM to generate a behavioral profile from input_text."""
    if len(input_text) < 1000:
        return None

    # Truncate to fit context
    excerpt = input_text[:8000]

    prompt = f"""Analyze the following text about {character_name} and produce \
a concise behavioral profile. Focus on:
1. Core values and motivations
2. Decision-making patterns (impulsive vs cautious, emotional vs rational)
3. Key relationships and how they influence decisions
4. Behavioral tendencies under pressure

Text:
{excerpt}

Output a concise profile (under 500 words) as bullet points:"""

    try:
        profile = await adapter._async_generate(prompt, model)
        return profile.strip()
    except Exception as e:
        logger.warning("Profile gen failed for %s: %s", character_name, e)
        return None


async def evaluate_character(
    group: list[dict],
    adapter: ClaudeCodeAdapter,
    model: str,
    use_cdt: bool = False,
) -> list[dict]:
    """Evaluate all questions for a character."""
    character_name = group[0]["character_name"]
    input_text = group[0]["input_text"]

    grounding = None
    if use_cdt:
        grounding = await build_llm_profile(
            character_name, input_text, adapter, model,
        )
        if grounding:
            logger.info(
                "Profile for %s: %d chars", character_name, len(grounding)
            )
        else:
            logger.info("No profile for %s (text too short)", character_name)

    results = []
    for q_item in group:
        mcq = q_item["Multiple Choice Question"]
        correct = mcq["Correct Answer"].strip().upper()
        memory = input_text[:12000]

        prompt = build_mcq_prompt(
            character_name=character_name,
            memory=memory,
            scenario=mcq["Scenario"],
            question=mcq["Question"],
            options=mcq["Options"],
            grounding=grounding,
        )

        try:
            response = await adapter._async_generate(prompt, model)
            predicted = extract_answer(response)
        except Exception as e:
            logger.warning("LLM error for %s: %s", character_name, e)
            predicted = None

        is_correct = predicted == correct if predicted else False
        results.append({
            "character": character_name,
            "correct_answer": correct,
            "predicted": predicted,
            "is_correct": is_correct,
            "mode": "cdt" if use_cdt else "baseline",
            "had_grounding": grounding is not None if use_cdt else False,
        })
        logger.info(
            "%s [%s] pred=%s correct=%s %s",
            character_name,
            "CDT" if use_cdt else "BASE",
            predicted,
            correct,
            "✓" if is_correct else "✗",
        )

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="LifeChoice CDT benchmark")
    parser.add_argument(
        "--max_characters", type=int, default=5,
        help="Max characters to evaluate",
    )
    parser.add_argument(
        "--model", type=str, default="claude-haiku-4-5",
        help="Model for MCQ answering",
    )
    parser.add_argument(
        "--skip_cdt", action="store_true",
        help="Skip CDT evaluation (baseline only)",
    )
    parser.add_argument(
        "--min_text_length", type=int, default=2000,
        help="Minimum input_text length for CDT (chars)",
    )
    args = parser.parse_args()

    adapter = ClaudeCodeAdapter(default_model=args.model)

    data = load_data()
    logger.info(
        "Loaded %d character groups, %d total questions",
        len(data), sum(len(g) for g in data),
    )

    eligible = [g for g in data if len(g[0]["input_text"]) >= args.min_text_length]
    eligible.sort(key=lambda g: len(g[0]["input_text"]))

    selected = eligible[:args.max_characters]
    logger.info(
        "Selected %d characters (%d questions)",
        len(selected), sum(len(g) for g in selected),
    )

    all_baseline: list[dict] = []
    all_cdt: list[dict] = []

    for i, group in enumerate(selected):
        char_name = group[0]["character_name"]
        logger.info(
            "=== Character %d/%d: %s (%d questions, %d chars) ===",
            i + 1, len(selected), char_name, len(group),
            len(group[0]["input_text"]),
        )

        baseline_results = await evaluate_character(
            group, adapter, args.model, use_cdt=False,
        )
        all_baseline.extend(baseline_results)

        if not args.skip_cdt:
            cdt_results = await evaluate_character(
                group, adapter, args.model, use_cdt=True,
            )
            all_cdt.extend(cdt_results)

    # Results
    print("\n" + "=" * 60)
    print("LIFECHOICE BENCHMARK RESULTS")
    print("=" * 60)

    baseline_correct = sum(1 for r in all_baseline if r["is_correct"])
    baseline_total = len(all_baseline)
    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total else 0

    print(f"\nBaseline: {baseline_correct}/{baseline_total} = {baseline_acc:.1f}%")
    print(f"  Model: {args.model}")

    if all_cdt:
        cdt_correct = sum(1 for r in all_cdt if r["is_correct"])
        cdt_total = len(all_cdt)
        cdt_acc = cdt_correct / cdt_total * 100 if cdt_total else 0
        delta = cdt_acc - baseline_acc

        cdt_with_grounding = [r for r in all_cdt if r["had_grounding"]]
        grounded_correct = sum(1 for r in cdt_with_grounding if r["is_correct"])
        grounded_total = len(cdt_with_grounding)
        grounded_acc = (
            grounded_correct / grounded_total * 100 if grounded_total else 0
        )

        print(f"\nCDT-augmented: {cdt_correct}/{cdt_total} = {cdt_acc:.1f}%")
        print(f"  Delta: {delta:+.1f}%")
        print(
            f"  With grounding: {grounded_correct}/{grounded_total}"
            f" = {grounded_acc:.1f}%"
        )

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "n_characters": len(selected),
        "n_questions": baseline_total,
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_results": all_baseline,
    }
    if all_cdt:
        output["cdt_accuracy"] = round(cdt_acc, 2)
        output["cdt_delta"] = round(delta, 2)
        output["cdt_results"] = all_cdt

    out_path = Path("results/lifechoice_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
