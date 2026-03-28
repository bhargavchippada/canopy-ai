"""NLI-based validation for CDT hypothesis checking."""

from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

# Module-level model references — set by init_models()
_classifier = None
_classifier_tokenizer = None
_device = None


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
    assert _classifier is not None, "Call init_models() first"
    prompts = [
        f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."
        for text, question in zip(texts, questions)
    ]

    with torch.no_grad():
        logits = _classifier(**_classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(_device)).logits
        choices = logits.argmax(-1)

    return [[False, None, True][choice.item()] for choice in choices]


def check_statement_probs(
    character: str,
    actions: list[str],
    statements: list[str],
) -> tuple[float, float, float]:
    """Check statement-action NLI probabilities.

    Returns numpy array of [false_score, none_score, true_score].
    """
    assert _classifier is not None, "Call init_models() first"
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

    with torch.no_grad():
        logits = _classifier(**_classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(_device)).logits
        probs = logits.softmax(-1).sum(0).detach().cpu().numpy()

    return probs


def validate_hypothesis(
    character: str,
    pairs: list[dict],
    hypothesized_question: str | None,
    hypothesized_action: str,
    bs: int = 64,
) -> tuple[dict[str, float], list[dict]]:
    """Validate a hypothesis against all pairs.

    Returns (result_counts, filtered_pairs).
    """
    res: dict[str, float] = defaultdict(int)
    filtered_pairs: list[dict] = []
    relevance_all: list[bool | None] = []

    for idx in tqdm(range(0, len(pairs), bs), desc="Filtering Scenes...", leave=True):
        pairs_batch = pairs[idx : idx + bs]
        scenes = [pair["scene"] for pair in pairs_batch]

        if hypothesized_question is None:
            relevance = [True for _ in pairs_batch]
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

    return res, filtered_pairs
