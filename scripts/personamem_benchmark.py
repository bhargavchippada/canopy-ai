"""PersonaMem 32k benchmark — Sonnet baseline with full context.

Evaluates Sonnet's MCQ accuracy on ALL PersonaMem 32k questions using
the full conversation context (truncated at end_index to prevent leakage).

Usage:
    uv run python scripts/personamem_benchmark.py
    uv run python scripts/personamem_benchmark.py --max_parallel 4
    uv run python scripts/personamem_benchmark.py --max_questions 50  # quick test
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SIZE = "32k"
MODEL = "claude-sonnet-4-6"
RESULTS_DIR = "results"


def build_mcq_prompt(
    context_messages: list[dict[str, str]],
    user_message: str,
    options: tuple[str, ...],
) -> str:
    """Build the MCQ selection prompt from full context + question.

    Uses the full conversation history (already truncated at end_index by caller)
    formatted as a dialogue transcript, followed by the MCQ question.
    """
    # Format conversation history as transcript
    transcript_parts: list[str] = []
    for msg in context_messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            # Include system messages as context markers (topic transitions, etc.)
            if content.strip():
                transcript_parts.append(f"[System: {content[:500]}]")
        elif role == "user":
            transcript_parts.append(f"User: {content}")
        elif role == "assistant":
            transcript_parts.append(f"Assistant: {content}")

    transcript = "\n\n".join(transcript_parts)

    options_text = "\n".join(f"  {opt}" for opt in options)

    return f"""Below is the conversation history between a user and their AI assistant. Based on this history, answer the user's question by selecting the most appropriate response.

# Conversation History
{transcript}

# User's Current Question
{user_message}

# Response Options
{options_text}

Select the response that BEST matches this specific user's current preferences, situation, and history. Pay attention to how their preferences may have evolved over time.

Output ONLY the option letter in parentheses, e.g. (a), (b), (c), or (d). Nothing else."""


def evaluate_single(
    question: Any,
    context_messages: list[dict[str, str]],
    model: str,
) -> dict[str, Any]:
    """Evaluate a single MCQ question. Returns result dict."""
    from canopy.llm import generate

    # Truncate context at end_index (prevents data leakage)
    truncated = context_messages[:question.end_index]

    prompt = build_mcq_prompt(truncated, question.user_message, question.all_options)

    try:
        answer = generate(prompt, model=model, max_tokens=10)
        answer = answer.strip().lower()

        # Extract letter — look for (a), (b), (c), (d) pattern
        model_letter = ""
        match = re.search(r"\(([abcd])\)", answer)
        if match:
            model_letter = f"({match.group(1)})"
        else:
            # Fallback: look for bare letter
            for char in answer:
                if char in "abcd":
                    model_letter = f"({char})"
                    break

        is_correct = model_letter == question.correct_answer

        return {
            "persona_id": question.persona_id,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "topic": question.topic,
            "user_message": question.user_message[:200],
            "model_answer": model_letter,
            "correct_answer": question.correct_answer,
            "is_correct": is_correct,
            "raw_response": answer[:50],
            "context_length": len(truncated),
            "distance_to_ref_tokens": question.distance_to_ref_tokens,
        }
    except Exception as exc:
        log.error("Failed on question %s: %s", question.question_id, exc)
        return {
            "persona_id": question.persona_id,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "topic": question.topic,
            "user_message": question.user_message[:200],
            "model_answer": "",
            "correct_answer": question.correct_answer,
            "is_correct": False,
            "raw_response": f"ERROR: {exc}",
            "context_length": 0,
            "distance_to_ref_tokens": question.distance_to_ref_tokens,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaMem 32k benchmark")
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit to first N questions (for quick testing)")
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    from canopy.datasets.personamem import (
        _download_shared_contexts,
        load_personamem_questions,
    )
    from canopy.llm import ClaudeCodeAdapter, set_adapter

    # Set up LLM adapter
    set_adapter(ClaudeCodeAdapter(default_model=args.model))

    # Load data
    log.info("Loading shared contexts (size=%s)...", SIZE)
    shared_contexts = _download_shared_contexts(SIZE)
    log.info("Loaded %d shared contexts", len(shared_contexts))

    log.info("Loading questions (size=%s)...", SIZE)
    questions = load_personamem_questions(size=SIZE)
    log.info("Loaded %d questions", len(questions))

    if args.max_questions is not None:
        questions = questions[:args.max_questions]
        log.info("Limited to %d questions", len(questions))

    # Map questions to their contexts
    missing_ctx = 0
    valid_items: list[tuple[Any, list[dict[str, str]]]] = []
    for q in questions:
        ctx = shared_contexts.get(q.shared_context_id)
        if ctx is None:
            missing_ctx += 1
            continue
        valid_items.append((q, ctx))

    if missing_ctx > 0:
        log.warning("%d questions have missing shared contexts — skipped", missing_ctx)

    log.info("Evaluating %d questions with %d parallel workers...", len(valid_items), args.max_parallel)

    # Parallel evaluation
    results: list[dict[str, Any]] = [{}] * len(valid_items)
    correct_count = 0
    total_done = 0

    bar = tqdm(total=len(valid_items), desc="Accuracy=N/A")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for i, (q, ctx) in enumerate(valid_items):
            future = executor.submit(evaluate_single, q, ctx, args.model)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                if result["is_correct"]:
                    correct_count += 1
                total_done += 1
                acc = 100 * correct_count / total_done
                bar.set_description(f"Accuracy={acc:.1f}%")
            except Exception as exc:
                log.error("Future failed for item %d: %s", idx, exc)
                total_done += 1
            bar.update(1)

    bar.close()

    # Filter out empty results
    valid_results = [r for r in results if r]

    # Compute per-type accuracy
    type_correct: Counter[str] = Counter()
    type_total: Counter[str] = Counter()
    for r in valid_results:
        qtype = r["question_type"]
        type_total[qtype] += 1
        if r["is_correct"]:
            type_correct[qtype] += 1

    # Compute per-persona accuracy
    persona_correct: Counter[int] = Counter()
    persona_total: Counter[int] = Counter()
    for r in valid_results:
        pid = r["persona_id"]
        persona_total[pid] += 1
        if r["is_correct"]:
            persona_correct[pid] += 1

    # Overall
    total_correct = sum(1 for r in valid_results if r["is_correct"])
    overall_acc = 100 * total_correct / len(valid_results) if valid_results else 0

    # Print results
    print(f"\n{'=' * 60}")
    print(f"PERSONAMEM {SIZE} BENCHMARK — {args.model}")
    print(f"{'=' * 60}")
    print(f"Overall: {total_correct}/{len(valid_results)} ({overall_acc:.1f}%)")
    print(f"Chance baseline: 25.0%")

    print(f"\n{'─' * 60}")
    print("Per-type accuracy:")
    for qtype in sorted(type_total.keys()):
        c = type_correct[qtype]
        t = type_total[qtype]
        print(f"  {qtype:<40s} {c:>3}/{t:<3} ({100 * c / t:.1f}%)")

    print(f"\n{'─' * 60}")
    print("Per-persona accuracy:")
    for pid in sorted(persona_total.keys()):
        c = persona_correct[pid]
        t = persona_total[pid]
        print(f"  persona_{pid:<3d} {c:>3}/{t:<3} ({100 * c / t:.1f}%)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "benchmark": "personamem",
        "size": SIZE,
        "model": args.model,
        "n_questions": len(valid_results),
        "n_correct": total_correct,
        "accuracy": overall_acc,
        "per_type": {
            qtype: {
                "correct": type_correct[qtype],
                "total": type_total[qtype],
                "accuracy": 100 * type_correct[qtype] / type_total[qtype],
            }
            for qtype in sorted(type_total.keys())
        },
        "per_persona": {
            str(pid): {
                "correct": persona_correct[pid],
                "total": persona_total[pid],
                "accuracy": 100 * persona_correct[pid] / persona_total[pid],
            }
            for pid in sorted(persona_total.keys())
        },
        "per_question": valid_results,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    output_path = f"{RESULTS_DIR}/personamem_{SIZE}_{args.model.split('-')[-1]}_baseline.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
