"""PersonaMem hybrid benchmark — CDT-guided RAG vs baseline vs CDT-only.

Compares three approaches on PersonaMem 32k (persona_id=0):
  A) Baseline: full context only (established: 76.9%)
  B) CDT-only: CDT traverse grounding + context
  C) Hybrid: CDT behavioral statements + RAG factual observations + context

Usage:
    uv run python scripts/personamem_hybrid_benchmark.py
    uv run python scripts/personamem_hybrid_benchmark.py --max_questions 10
    uv run python scripts/personamem_hybrid_benchmark.py --top_k 5 --gate_threshold 0.2
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from canopy.builder import BehavioralObservation, build_cdt
from canopy.core import CDTConfig, CDTNode
from canopy.datasets.personamem import (
    _download_shared_contexts,
    load_personamem_observations,
    load_personamem_questions,
)
from canopy.embeddings import precompute_embeddings
from canopy.episodic import EmbedFn, EpisodicIndex, format_grounding, hybrid_ground
from canopy.llm import ClaudeCodeAdapter, generate, set_adapter
from canopy.validation import init_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SIZE = "32k"
MODEL = "claude-sonnet-4-6"
PERSONA_ID = 0
RESULTS_DIR = "results"


def _extract_answer(response: str) -> str:
    """Extract MCQ letter from model response."""
    answer = response.strip().lower()
    match = re.search(r"\(([abcd])\)", answer)
    if match:
        return f"({match.group(1)})"
    for char in answer:
        if char in "abcd":
            return f"({char})"
    return "?"


def _build_context_transcript(
    context_messages: list[dict[str, str]],
    end_index: int,
) -> str:
    """Build formatted transcript from context messages, truncated at end_index."""
    truncated = context_messages[:end_index]
    parts: list[str] = []
    for msg in truncated:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            if content.strip():
                parts.append(f"[System: {content[:500]}]")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def _build_mcq_block(user_message: str, options: tuple[str, ...]) -> str:
    """Build the MCQ question block (shared across all approaches)."""
    options_text = "\n".join(f"  {opt}" for opt in options)
    return f"""# User's Current Question
{user_message}

# Response Options
{options_text}

Select the response that BEST matches this specific user's current preferences, \
situation, and history. Pay attention to how their preferences may have evolved over time.

Output ONLY the option letter in parentheses, e.g. (a), (b), (c), or (d). Nothing else."""


def evaluate_baseline(
    transcript: str,
    user_message: str,
    options: tuple[str, ...],
    model: str,
) -> str:
    """Approach A: Full context only."""
    mcq = _build_mcq_block(user_message, options)
    prompt = f"""Below is the conversation history between a user and their AI assistant. \
Based on this history, answer the user's question by selecting the most appropriate response.

# Conversation History
{transcript}

{mcq}"""
    return _extract_answer(generate(prompt, model=model, max_tokens=10))


def evaluate_cdt_only(
    transcript: str,
    user_message: str,
    options: tuple[str, ...],
    model: str,
    topic2cdt: dict[str, CDTNode],
    context_messages: list[dict[str, str]],
    end_index: int,
) -> str:
    """Approach B: CDT traverse grounding + context."""
    # Build scene from last few messages for traversal
    truncated = context_messages[:end_index]
    scene_parts: list[str] = []
    for msg in truncated[-6:]:
        if msg["role"] == "system":
            continue
        scene_parts.append(
            f"{msg['role'].capitalize()}: {msg.get('content', '')[:200]}"
        )
    scene = "\n".join(scene_parts)

    # Traverse all CDTs
    all_stmts: list[str] = []
    for _topic, cdt in topic2cdt.items():
        stmts = cdt.traverse(scene)
        all_stmts.extend(stmts)

    grounding = "\n".join(f"- {s}" for s in all_stmts) if all_stmts else "(no patterns activated)"

    mcq = _build_mcq_block(user_message, options)
    prompt = f"""Below is the conversation history between a user and their AI assistant. \
A behavioral profile summarizes the user's known patterns. Based on BOTH, answer the user's question.

# Behavioral Profile
{grounding}

# Conversation History
{transcript}

{mcq}"""
    return _extract_answer(generate(prompt, model=model, max_tokens=10))


def evaluate_hybrid(
    transcript: str,
    user_message: str,
    options: tuple[str, ...],
    model: str,
    topic2cdt: dict[str, CDTNode],
    episodic_index: EpisodicIndex,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    gate_threshold: float = 0.3,
) -> str:
    """Approach C: CDT behavioral + RAG factual + context."""
    query = user_message

    result = hybrid_ground(
        query,
        topic2cdt,
        episodic_index,
        embed_fn=embed_fn,
        top_k=top_k,
        gate_threshold=gate_threshold,
    )

    grounding_text = format_grounding(result)
    if not grounding_text.strip():
        grounding_text = "(no grounding activated)"

    mcq = _build_mcq_block(user_message, options)
    prompt = f"""Below is the conversation history between a user and their AI assistant. \
A behavioral profile and relevant past interactions are provided. \
Based on ALL available context, answer the user's question.

# User Grounding
{grounding_text}

# Conversation History
{transcript}

{mcq}"""
    return _extract_answer(generate(prompt, model=model, max_tokens=10))


def _build_embed_fn(
    surface_embedder_path: str,
    device: str,
) -> EmbedFn:
    """Build an embedding function for query/gate embedding.

    Loads the surface embedder model and returns a callable that embeds
    a single text string into a normalized vector.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(surface_embedder_path)
    embed_model = AutoModel.from_pretrained(surface_embedder_path).to(device)
    embed_model.eval()

    @torch.no_grad()
    def embed_fn(text: str) -> np.ndarray:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(device)
        outputs = embed_model(**inputs)
        # Mean pooling over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        hidden = outputs.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1)
        # L2 normalize
        vec = pooled[0].cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    return embed_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaMem hybrid benchmark")
    parser.add_argument("--persona_id", type=int, default=PERSONA_ID)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gate_threshold", type=float, default=0.3)
    parser.add_argument(
        "--surface_embedder",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-Embedding-0.6B"),
    )
    parser.add_argument(
        "--generator_embedder",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--discriminator",
        type=str,
        default=os.path.expanduser("~/models/deberta-v3-base-rp-nli"),
    )
    args = parser.parse_args()

    if not 0 <= args.persona_id <= 19:
        parser.error(f"persona_id must be 0-19, got {args.persona_id}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Initialize models ---
    log.info("Initializing NLI model...")
    init_models(args.discriminator, device)
    set_adapter(ClaudeCodeAdapter(default_model=args.model, system_prompt=None))

    # --- Step 2: Load PersonaMem data ---
    log.info("Loading PersonaMem data (persona=%d, size=%s)...", args.persona_id, SIZE)
    all_obs = load_personamem_observations(size=SIZE, persona_id=args.persona_id)
    actor_key = f"user_{args.persona_id}"
    topic_obs = all_obs.get(actor_key, [])
    log.info("Loaded %d observations for %s", len(topic_obs), actor_key)

    questions = load_personamem_questions(size=SIZE, persona_id=args.persona_id)
    shared_contexts = _download_shared_contexts(SIZE)
    log.info("Loaded %d questions, %d shared contexts", len(questions), len(shared_contexts))

    if args.max_questions is not None:
        questions = questions[: args.max_questions]
        log.info("Limited to %d questions", len(questions))

    # --- Step 3: Pre-compute embeddings ---
    log.info("Pre-computing embeddings...")
    pairs = [o.to_pair() for o in topic_obs]
    cache = precompute_embeddings(
        character=actor_key,
        pairs=pairs,
        surface_embedder_path=args.surface_embedder,
        generator_embedder_path=args.generator_embedder,
        device=str(device),
    )

    # --- Step 4: Build CDTs (all topics) ---
    log.info("Building CDTs for %s...", actor_key)
    indexed_obs = [
        BehavioralObservation(
            scene=o.scene,
            action=o.action,
            actor=o.actor,
            participants=o.participants,
            metadata={**o.metadata, "_embed_idx": i},
        )
        for i, o in enumerate(topic_obs)
    ]
    config = CDTConfig(
        max_depth=4,
        threshold_accept=0.75,
        threshold_reject=0.50,
        threshold_filter=0.75,
    )

    # Build per-topic CDTs
    topic2cdt: dict[str, CDTNode] = {}
    topics_in_data: set[str] = set()
    for o in topic_obs:
        t = o.metadata.get("topic")
        if t:
            topics_in_data.add(t)

    for topic in sorted(topics_in_data):
        topic_specific = [o for o in indexed_obs if o.metadata.get("topic") == topic]
        if len(topic_specific) < 3:
            log.info("Skipping topic %s (only %d obs)", topic, len(topic_specific))
            continue
        log.info("Building CDT for topic=%s (%d obs)...", topic, len(topic_specific))
        tree = build_cdt(
            topic_specific,
            character=actor_key,
            topic=topic,
            config=config,
            embedding_cache=cache,
        )
        topic2cdt[topic] = tree
        stats = tree.count_stats()
        log.info(
            "  %s: %d nodes, %d stmts, %d gates",
            topic,
            stats["total_nodes"],
            stats["total_statements"],
            stats["total_gates"],
        )

    log.info("Built %d topic CDTs", len(topic2cdt))

    # --- Step 5: Build EpisodicIndex ---
    log.info("Building EpisodicIndex from %d observations...", len(indexed_obs))
    episodic_index = EpisodicIndex.from_embedding_cache(indexed_obs, cache)
    log.info("EpisodicIndex ready: %d entries", len(episodic_index))

    # --- Step 6: Build embed_fn for hybrid queries ---
    log.info("Loading surface embedder for query embedding...")
    embed_fn = _build_embed_fn(args.surface_embedder, str(device))

    # --- Step 7: Evaluate all three approaches ---
    # Filter to questions with available contexts
    valid_items: list[tuple] = []
    for q in questions:
        ctx = shared_contexts.get(q.shared_context_id)
        if ctx is not None:
            valid_items.append((q, ctx))

    log.info("Evaluating %d questions × 3 approaches...", len(valid_items))

    # Results storage
    approach_names = ("baseline", "cdt_only", "hybrid")
    results_per_q: list[dict] = []

    bar = tqdm(total=len(valid_items), desc="Progress")

    errors = 0
    for i, (q, ctx) in enumerate(valid_items):
        transcript = _build_context_transcript(ctx, q.end_index)

        try:
            # A) Baseline
            ans_baseline = evaluate_baseline(
                transcript, q.user_message, q.all_options, args.model,
            )
        except Exception as exc:
            log.error("Baseline failed on Q%d: %s", i, exc)
            ans_baseline = "?"
            errors += 1

        try:
            # B) CDT-only
            ans_cdt = evaluate_cdt_only(
                transcript, q.user_message, q.all_options, args.model,
                topic2cdt, ctx, q.end_index,
            )
        except Exception as exc:
            log.error("CDT-only failed on Q%d: %s", i, exc)
            ans_cdt = "?"
            errors += 1

        try:
            # C) Hybrid
            ans_hybrid = evaluate_hybrid(
                transcript, q.user_message, q.all_options, args.model,
                topic2cdt, episodic_index, embed_fn,
                top_k=args.top_k,
                gate_threshold=args.gate_threshold,
            )
        except Exception as exc:
            log.error("Hybrid failed on Q%d: %s", i, exc)
            ans_hybrid = "?"
            errors += 1

        ok_b = ans_baseline == q.correct_answer
        ok_c = ans_cdt == q.correct_answer
        ok_h = ans_hybrid == q.correct_answer

        results_per_q.append({
            "question_id": q.question_id,
            "question_type": q.question_type,
            "topic": q.topic,
            "correct_answer": q.correct_answer,
            "baseline_answer": ans_baseline,
            "cdt_only_answer": ans_cdt,
            "hybrid_answer": ans_hybrid,
            "baseline_correct": ok_b,
            "cdt_only_correct": ok_c,
            "hybrid_correct": ok_h,
        })

        tag = q.question_type[:15]
        bar.set_postfix_str(
            f"B={'Y' if ok_b else 'N'} C={'Y' if ok_c else 'N'} H={'Y' if ok_h else 'N'} [{tag}]"
        )
        bar.update(1)

    bar.close()

    if errors > 0:
        log.warning("%d evaluation errors occurred (recorded as '?')", errors)

    # --- Step 8: Compute and display results ---
    print(f"\n{'=' * 70}")
    print(f"PERSONAMEM HYBRID BENCHMARK — persona_{args.persona_id} ({SIZE})")
    print(f"{'=' * 70}")

    for approach in approach_names:
        key = f"{approach}_correct"
        total_correct = sum(1 for r in results_per_q if r[key])
        total = len(results_per_q)
        acc = 100 * total_correct / total if total else 0
        print(f"\n{approach.upper()}: {total_correct}/{total} ({acc:.1f}%)")

        # Per-type breakdown
        type_correct: Counter[str] = Counter()
        type_total: Counter[str] = Counter()
        for r in results_per_q:
            qtype = r["question_type"]
            type_total[qtype] += 1
            if r[key]:
                type_correct[qtype] += 1

        for qtype in sorted(type_total.keys()):
            c = type_correct[qtype]
            t = type_total[qtype]
            print(f"  {qtype:<40s} {c:>3}/{t:<3} ({100 * c / t:.1f}%)")

    # --- Step 9: Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "benchmark": "personamem_hybrid",
        "size": SIZE,
        "persona_id": args.persona_id,
        "model": args.model,
        "top_k": args.top_k,
        "gate_threshold": args.gate_threshold,
        "n_questions": len(results_per_q),
        "summary": {},
        "per_question": results_per_q,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    for approach in approach_names:
        key = f"{approach}_correct"
        total_correct = sum(1 for r in results_per_q if r[key])
        total = len(results_per_q)
        output["summary"][approach] = {
            "correct": total_correct,
            "total": total,
            "accuracy": 100 * total_correct / total if total else 0,
        }

    output_path = f"{RESULTS_DIR}/personamem_{SIZE}_hybrid_p{args.persona_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
