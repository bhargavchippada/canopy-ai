"""NLI-based validation for CDT hypothesis checking."""

from __future__ import annotations

import threading
from collections import defaultdict
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
            logits = _classifier(**_classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(_device)).logits
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
            logits = _classifier(**_classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(_device)).logits
            probs = logits.softmax(-1).sum(0).detach().cpu().numpy()

    return probs


def validate_hypothesis(
    character: str,
    pairs: list[dict[str, Any]],
    hypothesized_question: str | None,
    hypothesized_action: str,
    bs: int = 64,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Validate a hypothesis against all pairs.

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

    for idx in tqdm(range(0, len(filtered_pairs), bs), desc="Validating Statements...", leave=True):
        pairs_batch = filtered_pairs[idx : idx + bs]
        actions = [pair["action"] for pair in pairs_batch]
        score = check_statement_probs(character, actions, [hypothesized_action] * len(pairs_batch))
        res["False"] += score[0]
        res["None"] += score[1]
        res["True"] += score[2]

    return dict(res), filtered_pairs
