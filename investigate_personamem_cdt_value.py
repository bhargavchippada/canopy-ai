"""PersonaMem CDT value test — does CDT grounding improve over raw context?

Tests 3 approaches on persona 0 (17 questions):
A) Baseline: full context only (established: 14/17 = 82%)
B) Wikified CDT profile prepended to context
C) CDT traversal grounding per question
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
from canopy.wikify import wikify_tree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PERSONA_ID = 0
SIZE = "32k"
TOPIC = "musicRecommendation"


def _answer(prompt: str) -> str:
    resp = generate(prompt, model="claude-sonnet-4-6")
    m = re.search(r"\(([a-d])\)", resp.lower())
    return f"({m.group(1)})" if m else "?"


def main() -> None:
    device = torch.device("cuda:0")
    init_models(os.path.expanduser("~/models/deberta-v3-base-rp-nli"), device)
    set_adapter(ClaudeCodeAdapter(default_model="claude-sonnet-4-6", system_prompt=None))

    # --- Build CDT ---
    all_obs = load_personamem_observations(size=SIZE, persona_id=PERSONA_ID)
    topic_obs = all_obs[f"user_{PERSONA_ID}"]

    pairs = [o.to_pair() for o in topic_obs]
    cache = precompute_embeddings(
        character=f"user_{PERSONA_ID}", pairs=pairs,
        surface_embedder_path=os.path.expanduser("~/models/Qwen3-Embedding-0.6B"),
        generator_embedder_path=os.path.expanduser("~/models/Qwen3-0.6B"),
        device="cuda:0",
    )
    indexed_obs = [
        BehavioralObservation(
            scene=o.scene, action=o.action, actor=o.actor,
            participants=o.participants,
            metadata={**o.metadata, "_embed_idx": i},
        )
        for i, o in enumerate(topic_obs)
    ]
    config = CDTConfig(max_depth=4, threshold_accept=0.75, threshold_reject=0.50, threshold_filter=0.75)
    tree = build_cdt(
        indexed_obs, character=f"user_{PERSONA_ID}", topic=TOPIC,
        config=config, embedding_cache=cache,
    )
    stats = tree.count_stats()
    log.info("CDT: %d nodes, %d stmts, %d gates, depth %d",
             stats["total_nodes"], stats["total_statements"], stats["total_gates"], stats["max_depth"])

    # --- Wikify CDT ---
    wiki = wikify_tree(tree, title=f"user_{PERSONA_ID} / {TOPIC}")
    log.info("Wiki profile: %d chars", len(wiki))

    # --- Load questions + contexts ---
    questions = load_personamem_questions(size=SIZE, persona_id=PERSONA_ID)
    contexts = _download_shared_contexts(SIZE)
    log.info("Testing %d questions", len(questions))

    # --- Run all 3 approaches ---
    results: dict[str, dict[str, list[bool]]] = {
        "baseline": {},
        "wiki_cdt": {},
        "traverse_cdt": {},
    }

    for i, q in enumerate(questions):
        ctx_messages = contexts.get(q.shared_context_id, [])
        truncated = ctx_messages[:q.end_index] if q.end_index < len(ctx_messages) else ctx_messages

        # Full context
        context_parts = []
        for msg in truncated:
            if msg["role"] == "system":
                continue
            context_parts.append(f"{msg['role'].capitalize()}: {msg.get('content', '')}")
        context_str = "\n".join(context_parts)
        options_text = "\n".join(q.all_options)

        mcq_block = f"""# User's Current Message
{q.user_message}

# Which response best matches the user's known preferences and history?
{options_text}

Select ONLY the letter: (a), (b), (c), or (d)."""

        # A) Baseline
        ans_a = _answer(f"# Conversation History\n{context_str}\n\n{mcq_block}")

        # B) Wiki CDT + context
        ans_b = _answer(f"# User Behavioral Profile\n{wiki}\n\n# Conversation History\n{context_str}\n\n{mcq_block}")

        # C) Traversal CDT + context
        scene_parts = []
        for msg in truncated[-6:]:
            if msg["role"] == "system":
                continue
            scene_parts.append(f"{msg['role'].capitalize()}: {msg.get('content', '')[:200]}")
        scene = "\n".join(scene_parts)
        grounding_stmts = tree.traverse(scene)
        grounding = "\n".join(f"- {s}" for s in grounding_stmts) if grounding_stmts else "(no patterns activated)"
        ans_c = _answer(f"# Behavioral Patterns for This Context\n{grounding}\n\n# Conversation History\n{context_str}\n\n{mcq_block}")

        ok_a = ans_a == q.correct_answer
        ok_b = ans_b == q.correct_answer
        ok_c = ans_c == q.correct_answer

        for approach, ok in [("baseline", ok_a), ("wiki_cdt", ok_b), ("traverse_cdt", ok_c)]:
            results[approach].setdefault(q.question_type, []).append(ok)

        tag = q.question_type[:15]
        print(f"Q{i+1:2d} [{tag:15s}] base={ans_a}{'✓' if ok_a else '✗'} wiki={ans_b}{'✓' if ok_b else '✗'} trav={ans_c}{'✓' if ok_c else '✗'} correct={q.correct_answer}")

    # --- Summary ---
    print("\n" + "=" * 70)
    for approach in ["baseline", "wiki_cdt", "traverse_cdt"]:
        total_correct = sum(sum(v) for v in results[approach].values())
        total_qs = sum(len(v) for v in results[approach].values())
        print(f"\n{approach.upper()} ({total_correct}/{total_qs} = {100*total_correct/total_qs:.0f}%)")
        for qtype, vals in sorted(results[approach].items()):
            n = sum(vals)
            print(f"  {qtype[:40]:40s} {n}/{len(vals)} ({100*n/len(vals):.0f}%)")


if __name__ == "__main__":
    main()
