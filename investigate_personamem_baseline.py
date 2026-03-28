"""PersonaMem baseline — Sonnet + full context, no grounding. Parallel execution."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from canopy.datasets.personamem import (
    load_personamem_questions,
    _download_shared_contexts,
)
from canopy.llm import ClaudeCodeAdapter, generate, set_adapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SIZE = "32k"


def main() -> None:
    set_adapter(ClaudeCodeAdapter(default_model="claude-sonnet-4-6", system_prompt=None))

    questions = load_personamem_questions(size=SIZE)  # All personas
    contexts = _download_shared_contexts(SIZE)
    log.info("Loaded %d questions, %d contexts", len(questions), len(contexts))

    test_qs = questions  # ALL questions (full benchmark)
    total = len(test_qs)
    log.info("Running %d questions with %d workers", total, 6)

    def evaluate_one(idx: int, q):
        ctx_messages = contexts.get(q.shared_context_id, [])
        truncated = ctx_messages[:q.end_index] if q.end_index < len(ctx_messages) else ctx_messages

        context_parts = []
        for msg in truncated:
            if msg["role"] == "system":
                continue
            context_parts.append(f"{msg['role'].capitalize()}: {msg.get('content', '')}")
        context_str = "\n".join(context_parts)

        options_text = "\n".join(q.all_options)
        prompt = f"""# Conversation History
{context_str}

# User's Current Message
{q.user_message}

# Question: Which response best matches the user's known preferences and history?
{options_text}

Select ONLY the letter of the best answer: (a), (b), (c), or (d). Output just the letter."""

        response = generate(prompt, model="claude-sonnet-4-6")
        answer_match = re.search(r"\(([a-d])\)", response.lower())
        answer = f"({answer_match.group(1)})" if answer_match else "?"
        return idx, q, answer

    results: list[tuple[int, object, str]] = [None] * total  # type: ignore[list-item]
    correct_count = 0
    results_by_type: dict[str, list[bool]] = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(evaluate_one, i, q): i for i, q in enumerate(test_qs)}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                idx, q, answer = future.result()
                is_correct = answer == q.correct_answer
                if is_correct:
                    correct_count += 1
                results_by_type.setdefault(q.question_type, []).append(is_correct)
                if done % 50 == 0 or done == total:
                    log.info("%d/%d done, running accuracy: %.1f%%", done, total, 100 * correct_count / done)
            except Exception as exc:
                log.error("Q%d failed: %s", futures[future], exc)
                done_q = test_qs[futures[future]]
                results_by_type.setdefault(done_q.question_type, []).append(False)

    print(f"\nOverall: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print("\nBy type:")
    for qtype, type_results in sorted(results_by_type.items()):
        n_correct = sum(type_results)
        print(f"  {qtype[:40]:40s} {n_correct}/{len(type_results)} ({100*n_correct/len(type_results):.0f}%)")


if __name__ == "__main__":
    main()
