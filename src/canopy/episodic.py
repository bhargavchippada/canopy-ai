"""Episodic memory layer — CDT-guided RAG for hybrid behavioral + factual grounding.

The CDT captures abstract behavioral patterns ("tends to prefer X when Y").
The EpisodicIndex preserves raw observations for factual retrieval.
HybridGrounding combines both: CDT gates identify the relevant behavioral domain,
then EpisodicIndex retrieves specific facts from that domain.

Usage::

    from canopy.episodic import EpisodicIndex, hybrid_ground, format_grounding

    # Build index from Phase A embeddings
    index = EpisodicIndex.from_embedding_cache(observations, embedding_cache)

    # Ground a query using CDT + RAG
    result = hybrid_ground(query_text, topic2cdt, index, embed_fn=my_embed_fn)

    # Format for prompt injection
    prompt_text = format_grounding(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from canopy.builder import BehavioralObservation
from canopy.core import CDTNode
from canopy.embeddings import EmbeddingCache

log = logging.getLogger(__name__)


class EmbedFn(Protocol):
    """Callable that embeds a single text string into a vector."""

    def __call__(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievedObservation:
    """A retrieved observation with its similarity score."""

    observation: BehavioralObservation
    score: float


@dataclass(frozen=True)
class GroundingResult:
    """Combined CDT + RAG grounding for a query.

    Attributes:
        behavioral_statements: Abstract behavioral rules from CDT traversal.
        factual_observations: Specific raw observations from RAG retrieval.
        active_gates: CDT gate conditions that activated for this query.
    """

    behavioral_statements: tuple[str, ...]
    factual_observations: tuple[RetrievedObservation, ...]
    active_gates: tuple[str, ...]


@dataclass(frozen=True)
class EpisodicIndex:
    """Embedding index over raw BehavioralObservation items.

    Reuses Phase A pre-computed embeddings — no new GPU computation needed.
    Supports top-k cosine similarity retrieval with optional gate filtering.

    Attributes:
        observations: The raw behavioral observations.
        embeddings: L2-normalized embedding matrix, shape (N, D).
    """

    observations: tuple[BehavioralObservation, ...]
    embeddings: np.ndarray  # (N, D), L2-normalized, read-only

    def __post_init__(self) -> None:
        # Coerce to canonical types first, then validate invariants
        if isinstance(self.observations, list):
            object.__setattr__(self, "observations", tuple(self.observations))
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D array")
        if len(self.observations) != len(self.embeddings):
            raise ValueError(
                f"observations ({len(self.observations)}) != "
                f"embeddings ({len(self.embeddings)})"
            )
        # Make read-only copy
        arr = np.array(self.embeddings, copy=True)
        arr.flags.writeable = False
        object.__setattr__(self, "embeddings", arr)

    @classmethod
    def from_embedding_cache(
        cls,
        observations: list[BehavioralObservation],
        embedding_cache: EmbeddingCache,
    ) -> EpisodicIndex:
        """Build index from existing Phase A embeddings.

        Uses the surface embeddings (action-level) for retrieval, as they
        capture what the actor did — most relevant for both behavioral
        prediction and factual recall.

        Args:
            observations: The raw behavioral observations.
            embedding_cache: Pre-computed embeddings from Phase A.

        Returns:
            An EpisodicIndex ready for retrieval.
        """
        if len(observations) != len(embedding_cache.surface):
            raise ValueError(
                f"observations ({len(observations)}) != "
                f"cache rows ({len(embedding_cache.surface)})"
            )
        return cls(
            observations=tuple(observations),
            embeddings=embedding_cache.surface,
        )

    @classmethod
    def from_arrays(
        cls,
        observations: list[BehavioralObservation],
        embeddings: np.ndarray,
    ) -> EpisodicIndex:
        """Build index from pre-computed embedding arrays.

        Args:
            observations: The raw behavioral observations.
            embeddings: L2-normalized embedding matrix, shape (N, D).

        Returns:
            An EpisodicIndex ready for retrieval.
        """
        return cls(observations=tuple(observations), embeddings=embeddings)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        gate_embeddings: np.ndarray | None = None,
        gate_threshold: float = 0.3,
    ) -> list[RetrievedObservation]:
        """Retrieve top-k observations by cosine similarity.

        If gate_embeddings provided, pre-filter observations to those
        semantically relevant to the active CDT gates. This focuses
        retrieval on the behavioral domain CDT identified as relevant.

        Args:
            query_embedding: Embedded query vector, shape (D,).
            top_k: Maximum observations to retrieve.
            gate_embeddings: Embedded gate condition vectors, shape (G, D).
                When provided, only observations with max gate similarity
                above gate_threshold are considered.
            gate_threshold: Minimum cosine similarity to an active gate
                for an observation to be eligible.

        Returns:
            List of RetrievedObservation sorted by descending similarity.
        """
        if len(self.observations) == 0:
            return []

        query = query_embedding.flatten()
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"query dim ({query.shape[0]}) != index dim ({self.embeddings.shape[1]})"
            )
        if not np.isfinite(query).all():
            raise ValueError("query_embedding contains NaN or Inf values")

        # Compute cosine similarities (embeddings are L2-normalized)
        similarities = self.embeddings @ query

        # Apply gate filtering if active
        if gate_embeddings is not None and len(gate_embeddings) > 0:
            if gate_embeddings.ndim != 2:
                raise ValueError(
                    f"gate_embeddings must be 2-D, got {gate_embeddings.ndim}-D"
                )
            if gate_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    f"gate_embeddings dim ({gate_embeddings.shape[1]}) != "
                    f"index dim ({self.embeddings.shape[1]})"
                )
            if not np.isfinite(gate_embeddings).all():
                raise ValueError("gate_embeddings contains NaN or Inf values")
            # For each observation, compute max similarity to any active gate
            gate_sims = self.embeddings @ gate_embeddings.T  # (N, G)
            max_gate_sim = gate_sims.max(axis=1)  # (N,)
            # Mask out observations below threshold
            mask = max_gate_sim >= gate_threshold
            if mask.sum() == 0:
                # No observations pass gate filter — fall back to unfiltered
                log.debug("Gate filter eliminated all observations, falling back to unfiltered")
            else:
                similarities = np.where(mask, similarities, -np.inf)

        # Get top-k indices
        k = min(top_k, len(self.observations))
        if k >= len(self.observations):
            # argpartition requires k < len; just argsort when k == len
            top_indices = np.argsort(-similarities)[:k]
        else:
            top_indices = np.argpartition(-similarities, k)[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Filter out -inf scores (from gate filtering)
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if np.isfinite(score):
                results.append(
                    RetrievedObservation(
                        observation=self.observations[idx],
                        score=score,
                    )
                )

        return results

    def __len__(self) -> int:
        return len(self.observations)


def hybrid_ground(
    query: str,
    topic2cdt: dict[str, CDTNode],
    episodic_index: EpisodicIndex,
    *,
    embed_fn: EmbedFn,
    top_k: int = 10,
    gate_threshold: float = 0.3,
) -> GroundingResult:
    """Ground a query using CDT behavioral rules + RAG factual retrieval.

    1. Traverse all CDT trees for the query → collect behavioral statements + active gates
    2. Embed the query and active gate conditions
    3. Use active gates to filter EpisodicIndex retrieval (CDT-guided RAG)
    4. Return combined grounding

    Args:
        query: The input text to ground (scene, question, etc.).
        topic2cdt: Attribute CDT trees (identity, personality, etc.).
        episodic_index: Index over raw observations.
        embed_fn: Function to embed text into vectors matching the index dimension.
        top_k: Maximum observations to retrieve from the index.
        gate_threshold: Minimum gate similarity for observation filtering.

    Returns:
        GroundingResult with behavioral statements, factual observations, and active gates.
    """
    # Step 1: Traverse CDTs
    all_statements: list[str] = []
    all_gates: list[str] = []

    for _topic, cdt in topic2cdt.items():
        statements, gates = _traverse_with_gates(cdt, query)
        all_statements.extend(statements)
        all_gates.extend(gates)

    # Step 2: Embed query and gates
    query_embedding = embed_fn(query)

    gate_embeddings = None
    if all_gates:
        gate_vecs = [embed_fn(gate) for gate in all_gates]
        gate_embeddings = np.stack(gate_vecs)

    # Step 3: CDT-guided RAG retrieval
    retrieved = episodic_index.retrieve(
        query_embedding,
        top_k=top_k,
        gate_embeddings=gate_embeddings,
        gate_threshold=gate_threshold,
    )

    return GroundingResult(
        behavioral_statements=tuple(all_statements),
        factual_observations=tuple(retrieved),
        active_gates=tuple(all_gates),
    )


_MAX_TRAVERSE_DEPTH = 20


def _traverse_with_gates(
    node: CDTNode, scene: str, *, _depth: int = 0
) -> tuple[list[str], list[str]]:
    """Traverse a CDT tree and collect both statements and activated gates.

    Unlike CDTNode.traverse() which only returns statements, this also
    tracks which gate conditions were activated — needed for CDT-guided RAG.

    Note: check_scene may return None (uncertain). None is falsy in Python,
    so uncertain gates are treated as not activated — consistent with
    CDTNode.traverse() behavior.
    """
    if _depth > _MAX_TRAVERSE_DEPTH:
        log.warning("CDT traversal depth limit reached (%d) — possible cycle", _depth)
        return [], []

    from canopy.validation import check_scene

    statements = list(node.statements)
    activated_gates: list[str] = []

    for gate, child in zip(node.gates, node.children):
        results = check_scene([scene], [gate])
        if results[0]:
            activated_gates.append(gate)
            child_stmts, child_gates = _traverse_with_gates(
                child, scene, _depth=_depth + 1
            )
            statements.extend(child_stmts)
            activated_gates.extend(child_gates)

    return statements, activated_gates


def format_grounding(
    result: GroundingResult,
    *,
    max_behavioral: int = 20,
    max_factual: int = 10,
) -> str:
    """Format a GroundingResult as text for prompt injection.

    Args:
        result: The grounding result to format.
        max_behavioral: Maximum behavioral statements to include.
        max_factual: Maximum factual observations to include.

    Returns:
        Formatted text ready for inclusion in a prompt.
    """
    parts: list[str] = []

    behavioral = result.behavioral_statements[:max_behavioral]
    if behavioral:
        parts.append("## Behavioral Profile")
        for stmt in behavioral:
            parts.append(f"- {stmt}")

    factual = result.factual_observations[:max_factual]
    if factual:
        if parts:
            parts.append("")
        parts.append("## Relevant Context")
        for retrieved in factual:
            obs = retrieved.observation
            action_text = obs.action[:500]
            scene_text = obs.scene[:200]
            parts.append(f"- {action_text} (context: {scene_text})")

    return "\n".join(parts)
