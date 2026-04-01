"""Provenance data model for CDT artifacts.

Every hypothesis, gate, and statement traces back to the raw observations
that generated it. This enables debugging, trust, and richer inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HypothesisQuality:
    """Per-hypothesis quality metrics computed during validation.

    Attributes:
        nli_true_rate: Fraction of relevant pairs where NLI says True.
        nli_false_rate: Fraction where NLI says False.
        nli_irrelevant_rate: Fraction where NLI says Irrelevant.
        specificity: 1 - nli_true_rate (higher = more specific).
        word_count: Statement length in words.
        grounding_fidelity: NLI entailment score against source cluster actions.
    """

    nli_true_rate: float
    nli_false_rate: float
    nli_irrelevant_rate: float
    specificity: float
    word_count: int
    grounding_fidelity: float


@dataclass(frozen=True)
class Provenance:
    """Traces any CDT artifact back to its source observations.

    Attributes:
        source_pair_indices: Indices into the training pairs.
        cluster_id: Which cluster produced this artifact.
        hypothesis_id: Unique ID for the hypothesis.
        step: Which pipeline step created this artifact.
        metadata: Additional metadata (e.g. convergence_count).
    """

    source_pair_indices: tuple[int, ...]
    cluster_id: int | None = None
    hypothesis_id: str | None = None
    step: str = ""
    metadata: tuple[tuple[str, Any], ...] = ()

    def metadata_dict(self) -> dict[str, Any]:
        """Return metadata as a mutable dict for convenience."""
        return dict(self.metadata)


@dataclass(frozen=True)
class TrackedHypothesis:
    """A hypothesis with full provenance.

    Attributes:
        statement: The behavioral hypothesis text.
        gate: The scene-check question.
        provenance: Where it came from.
        quality: Step-level quality metrics (None until computed).
    """

    statement: str
    gate: str
    provenance: Provenance
    quality: HypothesisQuality | None = None
