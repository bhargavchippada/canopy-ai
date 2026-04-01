"""PersonaMem hybrid test — full context + CDT grounding for suggest_new_ideas."""

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
    device = torch.device("cuda:0")
    init_models(os.path.expanduser("~/models/deberta-v3-base-rp-nli"), device)
    set_adapter(ClaudeCodeAdapter(default_model="claude-sonnet-4-6", system_prompt=None))

    # Build CDT (reuse from earlier — same config)
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
    tree = build_cdt(
        indexed_obs, character=f"user_{PERSONA_ID}", topic=TOPIC,
        config=CDTConfig(max_depth=4, threshold_accept=0.75, threshold_reject=0.50, threshold_filter=0.75),
        embedding_cache=cache,
    )
    log.info("CDT built: %d nodes, %d gates", tree.count_stats()["total_nodes"], tree.count_stats()["total_gates"])

    # Load questions + contexts
    questions = load_personamem_questions(size=SIZE, persona_id=PERSONA_ID)
    contexts = _download_shared_contexts(SIZE)

    # Filter to suggest_new_ideas only
    suggest_qs = [q for q in questions if q.question_type == "suggest_new_ideas"]
    log.info("Testing %d suggest_new_ideas questions", len(suggest_qs))

    correct_baseline = 0
    correct_hybrid = 0

    for i, q in enumerate(suggest_qs):
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

        # Scene for CDT traversal (last few messages)
        scene_parts = []
        for msg in truncated[-6:]:
            if msg["role"] == "system":
                continue
            scene_parts.append(f"{msg['role'].capitalize()}: {msg.get('content', '')[:200]}")
        scene = "\n".join(scene_parts)

        grounding_stmts = tree.traverse(scene)
        grounding = "\n".join(f"- {s}" for s in grounding_stmts)

        # --- Baseline (full context only) ---
        prompt_baseline = f"""# Conversation History
{context_str}

# User's Current Message
{q.user_message}

# Question: Which response best matches the user's known preferences?
{options_text}

Select ONLY the letter: (a), (b), (c), or (d)."""

        resp_b = generate(prompt_baseline, model="claude-sonnet-4-6")
        match_b = re.search(r"\(([a-d])\)", resp_b.lower())
        ans_b = f"({match_b.group(1)})" if match_b else "?"

        # --- Hybrid (CDT + full context) ---
        prompt_hybrid = f"""# User Behavioral Profile (from conversation analysis)
{grounding if grounding else "(no specific patterns activated)"}

# Conversation History
{context_str}

# User's Current Message
{q.user_message}

# Question: Which response best matches the user's known preferences and behavioral patterns?
{options_text}

Select ONLY the letter: (a), (b), (c), or (d)."""

        resp_h = generate(prompt_hybrid, model="claude-sonnet-4-6")
        match_h = re.search(r"\(([a-d])\)", resp_h.lower())
        ans_h = f"({match_h.group(1)})" if match_h else "?"

        b_ok = ans_b == q.correct_answer
        h_ok = ans_h == q.correct_answer
        if b_ok: correct_baseline += 1
        if h_ok: correct_hybrid += 1

        print(f"Q{i+1} correct={q.correct_answer} | baseline={ans_b}{'✓' if b_ok else '✗'} | hybrid={ans_h}{'✓' if h_ok else '✗'} | grounding={len(grounding_stmts)}stmts")

    print(f"\nBaseline: {correct_baseline}/{len(suggest_qs)} ({100*correct_baseline/len(suggest_qs):.0f}%)")
    print(f"Hybrid:   {correct_hybrid}/{len(suggest_qs)} ({100*correct_hybrid/len(suggest_qs):.0f}%)")


if __name__ == "__main__":
    main()
