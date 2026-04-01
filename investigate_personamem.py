"""PersonaMem in-sample CDT quality check.

Builds a CDT from persona 0's observations, then tests it on MCQ questions.
"""

from __future__ import annotations

import logging
import os
import re

import torch

from canopy.builder import BehavioralObservation, build_cdt
from canopy.core import CDTConfig
from canopy.datasets.personamem import (
    load_personamem_observations,
    load_personamem_questions,
    _download_shared_contexts,
)
from canopy.embeddings import precompute_embeddings
from canopy.llm import ClaudeCodeAdapter, generate, set_adapter
from canopy.validation import init_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PERSONA_ID = 0
SIZE = "32k"
TOPIC = "musicRecommendation"


def main() -> None:
    # --- Setup ---
    device = torch.device("cuda:0")
    init_models(os.path.expanduser("~/models/deberta-v3-base-rp-nli"), device)
    set_adapter(ClaudeCodeAdapter(default_model="claude-sonnet-4-6", system_prompt=None))

    # --- Step 1: Build CDT ---
    log.info("Loading observations for persona %d...", PERSONA_ID)
    all_obs = load_personamem_observations(size=SIZE, persona_id=PERSONA_ID)
    actor_key = f"user_{PERSONA_ID}"
    obs = all_obs[actor_key]
    log.info("Loaded %d observations (topic=%s)", len(obs), TOPIC)

    # Filter to topic
    topic_obs = [o for o in obs if o.metadata.get("topic") == TOPIC]
    log.info("Filtered to %d observations for topic=%s", len(topic_obs), TOPIC)

    # Pre-compute embeddings (0.6B for speed)
    pairs = [o.to_pair() for o in topic_obs]
    log.info("Pre-computing embeddings (0.6B)...")
    cache = precompute_embeddings(
        character=actor_key,
        pairs=pairs,
        surface_embedder_path=os.path.expanduser("~/models/Qwen3-Embedding-0.6B"),
        generator_embedder_path=os.path.expanduser("~/models/Qwen3-0.6B"),
        device="cuda:0",
    )

    # Stamp pairs with _embed_idx
    indexed_obs = [
        BehavioralObservation(
            scene=o.scene, action=o.action, actor=o.actor,
            participants=o.participants,
            metadata={**o.metadata, "_embed_idx": i},
        )
        for i, o in enumerate(topic_obs)
    ]

    log.info("Building CDT (Sonnet, d4, theta=0.75)...")
    config = CDTConfig(
        max_depth=4,
        threshold_accept=0.75,
        threshold_reject=0.50,
        threshold_filter=0.75,
    )
    tree = build_cdt(
        indexed_obs, character=actor_key, topic=TOPIC,
        config=config, embedding_cache=cache,
    )

    # --- Step 2: Inspect CDT ---
    print("\n" + "=" * 60)
    print(f"CDT for {actor_key} / {TOPIC}")
    print("=" * 60)

    def print_tree(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}Statements ({len(node.statements)}):")
        for s in node.statements:
            print(f"{prefix}  - {s}")
        for gate, child in zip(node.gates, node.children):
            print(f"{prefix}Gate: {gate}")
            print_tree(child, indent + 1)

    stats = tree.count_stats()
    print(f"Nodes: {stats['total_nodes']}, Stmts: {stats['total_statements']}, "
          f"Gates: {stats['total_gates']}, Depth: {stats['max_depth']}")
    print()
    print_tree(tree)

    # --- Step 3: Load questions and traverse ---
    print("\n" + "=" * 60)
    print("MCQ Evaluation")
    print("=" * 60)

    questions = load_personamem_questions(size=SIZE, persona_id=PERSONA_ID)
    log.info("Loaded %d questions for persona %d", len(questions), PERSONA_ID)

    # Load shared contexts for truncation
    contexts = _download_shared_contexts(SIZE)

    # Take first 5 questions
    test_qs = questions[:5]
    correct_count = 0

    for i, q in enumerate(test_qs):
        print(f"\n--- Question {i+1} (type={q.question_type}) ---")
        print(f"Topic: {q.topic}")
        print(f"User: {q.user_message[:100]}")
        print(f"Correct: {q.correct_answer}")

        # Get truncated context (no data leakage)
        ctx_messages = contexts.get(q.shared_context_id, [])
        truncated = ctx_messages[:q.end_index] if q.end_index < len(ctx_messages) else ctx_messages

        # Build scene from last few messages
        scene_parts = []
        for msg in truncated[-6:]:
            if msg["role"] == "system":
                continue
            content = msg.get("content", "")
            for prefix in ("User: ", "Assistant: "):
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
            scene_parts.append(f"{msg['role'].capitalize()}: {content[:200]}")
        scene = "\n".join(scene_parts)

        # Traverse CDT
        grounding_stmts = tree.traverse(scene)
        grounding = "\n".join(grounding_stmts) if grounding_stmts else "(no grounding activated)"
        print(f"Grounding ({len(grounding_stmts)} stmts): {grounding[:200]}")

        # Format options
        options_text = "\n".join(q.all_options)

        # Ask Sonnet with CDT grounding
        prompt = f"""# User Preference Profile
{grounding}

# Recent Conversation
{scene}

# User's Current Message
{q.user_message}

# Question: Which response best matches the user's known preferences?
{options_text}

Select ONLY the letter of the best answer: (a), (b), (c), or (d). Output just the letter."""

        response = generate(prompt, model="claude-sonnet-4-6")

        # Extract answer letter
        answer_match = re.search(r"\(([a-d])\)", response.lower())
        answer = f"({answer_match.group(1)})" if answer_match else response.strip()[:5]

        is_correct = answer == q.correct_answer
        if is_correct:
            correct_count += 1
        print(f"Model answer: {answer} | Correct: {q.correct_answer} | {'✓' if is_correct else '✗'}")

    print(f"\n{'=' * 60}")
    print(f"Score: {correct_count}/{len(test_qs)} ({100*correct_count/len(test_qs):.0f}%)")


if __name__ == "__main__":
    main()
