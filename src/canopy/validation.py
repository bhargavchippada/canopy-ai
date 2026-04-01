"""NLI-based validation for CDT hypothesis checking."""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Module-level model references — set by init_models()
_classifier: Any = None
_classifier_tokenizer: Any = None
_device: torch.device | None = None
_model_lock = threading.Lock()


def init_models(discriminator_path: str, device: torch.device) -> None:
    """Load the NLI classifier model onto the given device."""
    global _classifier, _classifier_tokenizer, _device

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _device = device
    _classifier_tokenizer = AutoTokenizer.from_pretrained(discriminator_path)
    _classifier = AutoModelForSequenceClassification.from_pretrained(discriminator_path).to(device)


def check_scene(texts: list[str], questions: list[str]) -> list[bool | None]:
    """Check whether scenes satisfy given questions via NLI.

    Returns a list of True/None/False per (text, question) pair.
    """
    if _classifier is None:
        raise RuntimeError("Validation model not initialized — call init_models() first")
    prompts = [
        f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."
        for text, question in zip(texts, questions)
    ]

    with _model_lock:
        with torch.no_grad():
            inputs = _classifier_tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(_device)
            logits = _classifier(**inputs).logits
            choices = logits.argmax(-1)

    return [[False, None, True][choice.item()] for choice in choices]


def check_statement_probs(
    character: str,
    actions: list[str],
    statements: list[str],
) -> np.ndarray:
    """Check statement-action NLI probabilities.

    Returns:
        numpy array of shape (3,) with [false_score, none_score, true_score].
    """
    if _classifier is None:
        raise RuntimeError("Validation model not initialized — call init_models() first")
    prompts = [
        f"""Character: {character}

Action: {action}

Statement: {statement}

Question: Does the statement provide correct grounding, which directly supports the character to take the action?

yes: the action involves direct information from the statement.

no: the action indicates the statement's assertion is not always correctly.

unknown: the action is irrelevant to the statement or the causal relationship cannot be determined

Directly answer only yes/no/unknown."""
        for action, statement in zip(actions, statements)
    ]

    with _model_lock:
        with torch.no_grad():
            logits = _classifier(**_classifier_tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(_device)).logits
            probs = logits.softmax(-1).sum(0).detach().cpu().numpy()

    return probs


def temporal_weight(
    timestamp: datetime,
    half_life_days: int = 90,
    *,
    reference_time: datetime | None = None,
) -> float:
    """Compute time decay weight relative to a reference time.

    More recent evidence gets higher weight. Evidence exactly half_life_days
    before reference_time gets weight 0.5.

    Args:
        timestamp: When this evidence was observed.
        half_life_days: Days until weight drops to 0.5.
        reference_time: The "now" for computing age. Defaults to the current
            UTC time. For datasets with synthetic timestamps, pass the most
            recent timestamp in the dataset so weighting is relative.
    """
    ref = reference_time or datetime.now(timezone.utc)
    age_days = (ref - timestamp).total_seconds() / 86400
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def check_statement_probs_per_pair(
    character: str,
    actions: list[str],
    statements: list[str],
    bs: int = 64,
) -> np.ndarray:
    """Check statement-action NLI probabilities per pair.

    Returns:
        numpy array of shape (N, 3) with [false_score, none_score, true_score] per pair.
    """
    if _classifier is None:
        raise RuntimeError("Validation model not initialized — call init_models() first")

    all_probs: list[np.ndarray] = []
    for idx in range(0, len(actions), bs):
        batch_actions = actions[idx:idx + bs]
        batch_stmts = statements[idx:idx + bs]
        prompts = [
            f"""Character: {character}

Action: {action}

Statement: {statement}

Question: Does the statement provide correct grounding, which directly supports the character to take the action?

yes: the action involves direct information from the statement.

no: the action indicates the statement's assertion is not always correctly.

unknown: the action is irrelevant to the statement or the causal relationship cannot be determined

Directly answer only yes/no/unknown."""
            for action, statement in zip(batch_actions, batch_stmts)
        ]

        with _model_lock:
            with torch.no_grad():
                logits = _classifier(**_classifier_tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(_device)).logits
                probs = logits.softmax(-1).detach().cpu().numpy()  # (batch, 3)
                all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (N, 3)


def check_statement_pair_entailment(
    character: str,
    statement: str,
    actions: list[str],
    bs: int = 64,
) -> float:
    """Check how well a statement is entailed by a set of actions.

    Computes the mean NLI True probability across all (statement, action) pairs.
    This measures grounding fidelity: how strongly the source observations
    support the hypothesis.

    Args:
        character: Character name for the NLI prompt.
        statement: The hypothesis statement to check.
        actions: Source actions from the cluster that generated this hypothesis.
        bs: Batch size for NLI inference.

    Returns:
        Mean NLI True probability (0.0 to 1.0). Higher = better grounding.
    """
    if not actions:
        return 0.0
    per_pair = check_statement_probs_per_pair(
        character, actions, [statement] * len(actions), bs=bs,
    )
    # per_pair shape: (N, 3) with [false, none, true]
    true_probs = per_pair[:, 2]
    return float(true_probs.mean())


def validate_hypothesis(
    character: str,
    pairs: list[dict[str, Any]],
    hypothesized_question: str | None,
    hypothesized_action: str,
    bs: int = 64,
    *,
    time_decay_enabled: bool = False,
    time_decay_half_life_days: int = 90,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Validate a hypothesis against all pairs.

    When time_decay_enabled=True and pairs have '_timestamp' keys,
    each pair's NLI verdict is weighted by temporal recency (T-CDT).
    More recent pairs contribute more to the final True/False/None scores.

    Returns (result_counts, filtered_pairs).
    """
    res: defaultdict[str, float] = defaultdict(int)
    filtered_pairs: list[dict[str, Any]] = []
    relevance_all: list[bool | None] = []

    for idx in tqdm(range(0, len(pairs), bs), desc="Filtering Scenes...", leave=True):
        pairs_batch = pairs[idx : idx + bs]
        scenes = [pair["scene"] for pair in pairs_batch]

        if hypothesized_question is None:
            relevance: list[bool | None] = [True for _ in pairs_batch]
        else:
            relevance = check_scene(scenes, [hypothesized_question] * len(pairs_batch))
        relevance_all.extend(relevance)

    for pair, rel in zip(pairs, relevance_all):
        if rel:
            filtered_pairs.append(pair)
        else:
            res["Irrelevant"] += 1.0

    if not filtered_pairs:
        return dict(res), filtered_pairs

    if time_decay_enabled:
        # T-CDT: per-pair NLI scoring with temporal weights
        actions = [pair["action"] for pair in filtered_pairs]
        per_pair_probs = check_statement_probs_per_pair(
            character, actions, [hypothesized_action] * len(actions), bs=bs,
        )  # (N, 3) — [false, none, true] per pair

        # Compute reference time from the dataset (most recent timestamp)
        # so temporal weighting is relative within the dataset, not absolute
        timestamps = [p.get("_timestamp") for p in filtered_pairs if p.get("_timestamp")]
        ref_time = max(timestamps) if timestamps else None

        for i, pair in enumerate(filtered_pairs):
            ts = pair.get("_timestamp")
            weight = (
                temporal_weight(ts, time_decay_half_life_days, reference_time=ref_time)
                if ts and ref_time
                else 1.0
            )
            res["False"] += float(per_pair_probs[i, 0]) * weight
            res["None"] += float(per_pair_probs[i, 1]) * weight
            res["True"] += float(per_pair_probs[i, 2]) * weight
    else:
        # Standard CDT: aggregate scoring (original behavior)
        for idx in tqdm(range(0, len(filtered_pairs), bs), desc="Validating Statements...", leave=True):
            pairs_batch = filtered_pairs[idx : idx + bs]
            actions = [pair["action"] for pair in pairs_batch]
            score = check_statement_probs(character, actions, [hypothesized_action] * len(pairs_batch))
            res["False"] += score[0]
            res["None"] += score[1]
            res["True"] += score[2]

    return dict(res), filtered_pairs
