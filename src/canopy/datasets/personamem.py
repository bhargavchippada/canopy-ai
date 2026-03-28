"""PersonaMem dataset loader — long-context user memory benchmark.

Loads conversation contexts and MCQ evaluation pairs from the PersonaMem
benchmark (COLM 2025, bowen-upenn/PersonaMem on HuggingFace).

Conversations are multi-session user–chatbot dialogues with evolving
preferences across time periods (init → week → month → year).

Contexts are OpenAI message-list format with system/user/assistant roles.
Questions are MCQ pairs testing whether an LLM remembers user preferences.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from canopy.builder import BehavioralObservation

log = logging.getLogger(__name__)

# PersonaMem topic categories (13 total)
TOPICS: tuple[str, ...] = (
    "bookRecommendation",
    "datingConsultation",
    "familyRelations",
    "financialConsultation",
    "foodRecommendation",
    "homeDecoration",
    "legalConsultation",
    "medicalConsultation",
    "movieRecommendation",
    "musicRecommendation",
    "studyConsultation",
    "therapy",
    "travelPlanning",
)


@dataclass(frozen=True)
class PersonaMemQuestion:
    """A single MCQ evaluation question from PersonaMem.

    Attributes:
        persona_id: Integer persona identifier (0-19).
        question_id: Unique question UUID.
        question_type: Category of memory being tested.
        topic: Conversation topic the question relates to.
        user_message: The user's message/question text.
        correct_answer: Letter of correct option, e.g. '(c)'.
        all_options: List of MCQ option strings.
        shared_context_id: Hash linking to the conversation context.
        end_index: Position in shared context where relevant info ends.
        distance_to_ref_tokens: How far back the reference is in tokens.
    """

    persona_id: int
    question_id: str
    question_type: str
    topic: str
    user_message: str
    correct_answer: str
    all_options: tuple[str, ...]
    shared_context_id: str
    end_index: int
    distance_to_ref_tokens: int


def _download_shared_contexts(size: str) -> dict[str, list[dict[str, str]]]:
    """Download and parse shared context JSONL from HuggingFace.

    Returns:
        Dict mapping context_id → list of OpenAI-format message dicts.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "bowen-upenn/PersonaMem",
        f"shared_contexts_{size}.jsonl",
        repo_type="dataset",
    )

    contexts: dict[str, list[dict[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            for ctx_id, messages in entry.items():
                contexts[ctx_id] = messages

    log.info("Loaded %d shared contexts for size=%s", len(contexts), size)
    return contexts


def _download_questions(size: str) -> list[PersonaMemQuestion]:
    """Download and parse questions from HuggingFace datasets library.

    Returns:
        List of PersonaMemQuestion dataclasses.
    """
    from datasets import load_dataset

    ds = load_dataset("bowen-upenn/PersonaMem", split=size)

    questions: list[PersonaMemQuestion] = []
    for row in ds:
        options = row["all_options"]
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                options = [options]

        questions.append(
            PersonaMemQuestion(
                persona_id=row["persona_id"],
                question_id=row["question_id"],
                question_type=row["question_type"],
                topic=row["topic"],
                user_message=row["user_question_or_message"],
                correct_answer=row["correct_answer"],
                all_options=tuple(options),
                shared_context_id=row["shared_context_id"],
                end_index=row["end_index_in_shared_context"],
                distance_to_ref_tokens=row["distance_to_ref_in_tokens"],
            )
        )

    log.info("Loaded %d questions for size=%s", len(questions), size)
    return questions


def _extract_topic_from_system(system_content: str) -> str | None:
    """Extract conversation topic from a system message.

    PersonaMem system messages contain topic transitions like:
    'Topic: musicRecommendation' or 'current_topic: therapy'
    """
    for line in system_content.split("\n"):
        lower = line.lower().strip()
        if "topic:" in lower:
            # Extract topic value after colon
            parts = line.split(":", 1)
            if len(parts) == 2:
                candidate = parts[1].strip().strip('"').strip("'")
                if candidate in TOPICS:
                    return candidate
    return None


def _messages_to_observations(
    messages: list[dict[str, str]],
    persona_id: int,
    scene_window: int = 6,
    default_topic: str | None = None,
) -> list[BehavioralObservation]:
    """Convert OpenAI message list to BehavioralObservation list.

    Each user message becomes an observation with preceding turns as scene.
    System messages are used to track topic transitions.

    Args:
        messages: OpenAI-format message list (role/content dicts).
        persona_id: Integer persona identifier.
        scene_window: Number of preceding messages for scene context.

    Returns:
        List of BehavioralObservation for user messages.
    """
    actor = f"user_{persona_id}"
    observations: list[BehavioralObservation] = []
    current_topic: str | None = default_topic

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content", "")

        # Track topic from system messages
        if role == "system":
            detected = _extract_topic_from_system(content)
            if detected is not None:
                current_topic = detected
            continue

        # Only user messages become observations
        if role != "user":
            continue

        # Build scene from preceding messages (skip system messages)
        scene_parts: list[str] = []
        start = max(0, i - scene_window)
        for j in range(start, i):
            prev = messages[j]
            if prev["role"] == "system":
                continue
            prev_content = prev.get("content", "")
            # Strip "User: " / "Assistant: " prefixes if present
            text = prev_content
            for prefix in ("User: ", "Assistant: "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            scene_parts.append(f"{prev['role'].capitalize()}: {text}")

        scene = "\n".join(scene_parts)

        # Strip "User: " prefix from action if present
        action = content
        if action.startswith("User: "):
            action = action[6:]

        observations.append(
            BehavioralObservation(
                scene=scene,
                action=action,
                actor=actor,
                participants=("assistant",),
                metadata={
                    "dataset": "personamem",
                    "persona_id": persona_id,
                    "topic": current_topic,
                    "message_index": i,
                },
            )
        )

    return observations


def load_personamem_observations(
    size: str = "32k",
    persona_id: int | None = None,
    scene_window: int = 6,
) -> dict[str, list[BehavioralObservation]]:
    """Load PersonaMem conversations as BehavioralObservation lists.

    Args:
        size: Context size split ('32k', '128k', '1M').
        persona_id: If set, load only this persona. Otherwise load all.
        scene_window: Number of preceding messages for scene context.

    Returns:
        Dict mapping 'user_{persona_id}' → list of observations.
    """
    if size not in ("32k", "128k", "1M"):
        raise ValueError(f"Invalid size: {size!r}. Must be '32k', '128k', or '1M'.")

    contexts = _download_shared_contexts(size)
    questions = _download_questions(size)

    # Map shared_context_id → (persona_id, topics) from questions
    ctx_to_persona: dict[str, int] = {}
    ctx_to_topics: dict[str, set[str]] = {}
    for q in questions:
        ctx_to_persona[q.shared_context_id] = q.persona_id
        ctx_to_topics.setdefault(q.shared_context_id, set()).add(q.topic)

    # Convert each context to observations
    result: dict[str, list[BehavioralObservation]] = {}
    for ctx_id, messages in contexts.items():
        pid = ctx_to_persona.get(ctx_id)
        if pid is None:
            log.warning("Context %s has no matching questions, skipping", ctx_id[:20])
            continue

        if persona_id is not None and pid != persona_id:
            continue

        # Infer primary topic from questions referencing this context
        topics_for_ctx = ctx_to_topics.get(ctx_id, set())
        primary_topic = sorted(topics_for_ctx)[0] if topics_for_ctx else None

        actor_key = f"user_{pid}"
        obs = _messages_to_observations(
            messages, pid, scene_window=scene_window, default_topic=primary_topic,
        )

        if actor_key in result:
            result[actor_key].extend(obs)
        else:
            result[actor_key] = obs

    for actor_key, obs_list in result.items():
        log.info("Persona %s: %d observations", actor_key, len(obs_list))

    return result


def load_personamem_questions(
    size: str = "32k",
    persona_id: int | None = None,
) -> list[PersonaMemQuestion]:
    """Load PersonaMem MCQ questions for evaluation.

    Args:
        size: Context size split ('32k', '128k', '1M').
        persona_id: If set, filter to this persona only.

    Returns:
        List of PersonaMemQuestion dataclasses.
    """
    questions = _download_questions(size)
    if persona_id is not None:
        questions = [q for q in questions if q.persona_id == persona_id]
    return questions
