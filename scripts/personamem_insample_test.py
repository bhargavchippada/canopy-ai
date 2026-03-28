"""PersonaMem in-sample quality check — CDT grounding for MCQ selection.

Loads persona_id=0 from the 32k split, builds a CDT from their observations,
then tests CDT-grounded MCQ selection on 5 questions.

Usage:
    uv run python scripts/personamem_insample_test.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PERSONA_ID = 0
SIZE = "32k"
N_QUESTIONS = 5
CDT_MODEL = "claude-sonnet-4-6"
EVAL_MODEL = "claude-sonnet-4-6"
SURFACE_EMBEDDER = "~/models/Qwen3-Embedding-0.6B"
GENERATOR_EMBEDDER = "~/models/Qwen3-0.6B"
DISCRIMINATOR = "~/models/deberta-v3-base-rp-nli"
DEVICE = "cuda:0"


def main() -> None:
    from canopy import CDTConfig
    from canopy.core import build_character_cdts
    from canopy.datasets.personamem import (
        PersonaMemQuestion,
        _download_shared_contexts,
        load_personamem_observations,
        load_personamem_questions,
    )
    from canopy.embeddings import precompute_embeddings
    from canopy.llm import ClaudeCodeAdapter, generate, set_adapter
    from canopy.validation import init_models as init_validation_models

    # ---------------------------------------------------------------
    # Step 1: Load observations
    # ---------------------------------------------------------------
    log.info("Loading persona %d observations (size=%s)...", PERSONA_ID, SIZE)
    obs_by_actor = load_personamem_observations(
        size=SIZE, persona_id=PERSONA_ID, scene_window=6,
    )
    actor_key = f"user_{PERSONA_ID}"
    observations = obs_by_actor.get(actor_key, [])
    log.info("Loaded %d observations for %s", len(observations), actor_key)

    if not observations:
        log.error("No observations found for persona %d", PERSONA_ID)
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Build CDT
    # ---------------------------------------------------------------
    log.info("Building CDT (model=%s, 0.6B embeddings for speed)...", CDT_MODEL)

    # Configure LLM adapter — no system prompt (paper-matching)
    set_adapter(ClaudeCodeAdapter(default_model=CDT_MODEL, system_prompt=None))

    # Load DeBERTa for NLI validation
    device = torch.device(DEVICE)
    init_validation_models(os.path.expanduser(DISCRIMINATOR), device)

    # Convert observations to scene-action pairs for canopy
    pairs = [obs.to_pair() for obs in observations]

    # Pre-compute embeddings (0.6B for speed — subprocess isolation)
    cache = precompute_embeddings(
        character=actor_key,
        pairs=pairs,
        surface_embedder_path=os.path.expanduser(SURFACE_EMBEDDER),
        generator_embedder_path=os.path.expanduser(GENERATOR_EMBEDDER),
        device=DEVICE,
    )
    log.info("Embeddings: surface=%s, generator=%s", cache.surface.shape, cache.generator.shape)

    # Stamp _embed_idx on pairs
    indexed_pairs = [{**pair, "_embed_idx": idx} for idx, pair in enumerate(pairs)]

    config = CDTConfig(
        max_depth=4,
        threshold_accept=0.75,
        threshold_reject=0.50,
        threshold_filter=0.75,
    )

    # Build CDTs — no relationship targets for user personas
    topic2cdt, rel_topic2cdt = build_character_cdts(
        actor_key, indexed_pairs, [],  # no other characters for relationship CDTs
        config, max_parallel=4, embedding_cache=cache,
    )

    # ---------------------------------------------------------------
    # Step 3: Print CDT structure
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CDT STRUCTURE — {actor_key}")
    print("=" * 60)

    total_nodes = 0
    total_stmts = 0
    total_gates = 0

    def count_tree(node: object, depth: int = 0) -> tuple[int, int, int, int]:
        n, s, g, md = 1, len(node.statements), len(node.gates), depth
        for child in node.children:
            cn, cs, cg, cd = count_tree(child, depth + 1)
            n += cn; s += cs; g += cg; md = max(md, cd)
        return n, s, g, md

    for topic, tree in [*topic2cdt.items(), *rel_topic2cdt.items()]:
        n, s, g, d = count_tree(tree)
        total_nodes += n; total_stmts += s; total_gates += g
        print(f"  {topic}: {n}n/{s}s/{g}g depth={d}")
        # Print root statements
        for stmt in tree.statements[:3]:
            print(f"    - {stmt}")
        if len(tree.statements) > 3:
            print(f"    ... (+{len(tree.statements) - 3} more)")

    print(f"\n  TOTAL: {total_nodes}n/{total_stmts}s/{total_gates}g")

    # ---------------------------------------------------------------
    # Step 4: Load MCQ questions
    # ---------------------------------------------------------------
    log.info("Loading MCQ questions for persona %d...", PERSONA_ID)
    questions = load_personamem_questions(size=SIZE, persona_id=PERSONA_ID)
    log.info("Loaded %d questions", len(questions))

    if len(questions) < N_QUESTIONS:
        log.warning("Only %d questions available (requested %d)", len(questions), N_QUESTIONS)

    test_questions = questions[:N_QUESTIONS]

    # Load shared contexts for truncation
    shared_contexts = _download_shared_contexts(SIZE)

    # ---------------------------------------------------------------
    # Step 5-7: Evaluate each question with CDT grounding
    # ---------------------------------------------------------------
    # Switch adapter to eval model (with default system prompt for eval)
    set_adapter(ClaudeCodeAdapter(default_model=EVAL_MODEL))

    correct = 0
    print("\n" + "=" * 60)
    print(f"MCQ EVALUATION — {len(test_questions)} questions")
    print("=" * 60)

    for i, q in enumerate(test_questions):
        print(f"\n{'─' * 60}")
        print(f"Q{i + 1} [{q.question_type}] (topic: {q.topic})")
        print(f"  User: {q.user_message[:200]}")

        # Step 5: Truncate context at end_index
        full_context = shared_contexts.get(q.shared_context_id, [])
        truncated_context = full_context[:q.end_index]

        # Build scene from last few messages of truncated context
        scene_parts: list[str] = []
        for msg in truncated_context[-6:]:
            if msg["role"] == "system":
                continue
            text = msg.get("content", "")[:200]
            scene_parts.append(f"{msg['role'].capitalize()}: {text}")
        scene = "\n".join(scene_parts)

        # Traverse CDT with scene context
        grounding_statements: list[str] = []
        for topic, tree in topic2cdt.items():
            stmts = tree.traverse(scene)
            if stmts:
                grounding_statements.append(f"# {topic}")
                grounding_statements.extend(stmts)

        grounding = "\n".join(grounding_statements) if grounding_statements else "(no grounding activated)"
        print(f"  Grounding: {len(grounding_statements)} lines")
        for line in grounding_statements[:5]:
            print(f"    {line[:120]}")
        if len(grounding_statements) > 5:
            print(f"    ... (+{len(grounding_statements) - 5} more)")

        # Step 6: Construct MCQ prompt with CDT grounding
        options_text = "\n".join(
            f"  {opt}" for opt in q.all_options
        )

        prompt = f"""# User Preference Knowledge
{grounding}

# Recent Conversation Context
{scene}

# User's Question
{q.user_message}

# Response Options
{options_text}

Based on the user preference knowledge and conversation context, select the response option that BEST matches this user's current preferences and situation. Consider preference changes over time.

Output ONLY the option letter in parentheses, e.g. (a), (b), (c), or (d)."""

        # Step 7: Get model answer
        answer = generate(prompt, model=EVAL_MODEL, max_tokens=10)
        answer = answer.strip()

        # Extract letter from answer
        model_letter = ""
        for char in answer:
            if char in "abcd":
                model_letter = f"({char})"
                break

        is_correct = model_letter == q.correct_answer
        if is_correct:
            correct += 1

        print(f"  Options: {[opt[:60] + '...' for opt in q.all_options]}")
        print(f"  Model: {model_letter}  Correct: {q.correct_answer}  {'✓' if is_correct else '✗'}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {correct}/{len(test_questions)} correct ({100 * correct / len(test_questions):.0f}%)")
    print(f"Chance baseline: 25%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
