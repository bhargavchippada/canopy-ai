"""Core CDT data structures — CDTNode, CDTConfig, and build helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CDTConfig:
    """Configuration for CDT construction.

    Attributes:
        max_depth: Maximum tree depth for recursive subtree growth.
        threshold_accept: Accuracy above which a statement is accepted globally.
        threshold_reject: Accuracy below which a gated statement is discarded.
        threshold_filter: Broadness below which a gated statement triggers recursion.
    """

    max_depth: int = 3
    threshold_accept: float = 0.8
    threshold_reject: float = 0.5
    threshold_filter: float = 0.8


# Type alias for dependency-injected callables
ClusterFn = Callable[..., list[list[dict[str, Any]]]]
ValidateFn = Callable[..., tuple[dict[str, float], list[dict[str, Any]]]]
HypothesisFn = Callable[..., tuple[list[str], list[str]]]
SummarizeFn = Callable[..., tuple[list[str], list[str]]]

# Minimum pairs required for tree construction
MIN_PAIRS_FOR_TREE = 8


class CDTNode:
    """A node in a Codified Decision Tree.

    Each node holds:
    - statements: globally applicable behavioral statements
    - gates: questions that filter to child nodes
    - children: child CDTNode instances (1 gate → 1 child)

    Use ``build_character_cdts()`` for the standard construction workflow,
    or instantiate directly for custom topologies.
    """

    def __init__(
        self,
        character: str,
        goal_topic: str,
        pairs: list[dict[str, Any]] | None,
        *,
        built_statements: list[str] | None = None,
        depth: int = 1,
        established_statements: list[str] | None = None,
        gate_path: list[str] | None = None,
        config: CDTConfig | None = None,
        _embedder: ClusterFn | None = None,
        _validator: ValidateFn | None = None,
        _hypothesis_fn: HypothesisFn | None = None,
        _summarize_fn: SummarizeFn | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.gates: list[str] = []
        self.children: list[CDTNode] = []
        self.depth = depth

        cfg = config or CDTConfig()
        established_statements = established_statements or []
        gate_path = gate_path or []

        if built_statements is not None:
            if pairs is not None:
                raise ValueError("Cannot specify both built_statements and pairs")
            self.statements = list(built_statements)
        elif pairs is None or len(pairs) <= MIN_PAIRS_FOR_TREE or self.depth > cfg.max_depth:
            pass
        else:
            self._build(
                character, goal_topic, pairs,
                established_statements, gate_path, cfg,
                _embedder, _validator, _hypothesis_fn, _summarize_fn,
            )

    def _build(
        self,
        character: str,
        goal_topic: str,
        pairs: list[dict[str, Any]],
        established_statements: list[str],
        gate_path: list[str],
        cfg: CDTConfig,
        embedder: ClusterFn | None,
        validator: ValidateFn | None,
        hypothesis_fn: HypothesisFn | None,
        summarize_fn: SummarizeFn | None,
    ) -> None:
        """Build the tree node via hypothesis generation + validation."""
        from canopy.embeddings import select_cluster_centers
        from canopy.prompts import make_hypotheses_batch, summarize_triggers
        from canopy.validation import validate_hypothesis

        _select_clusters = embedder or select_cluster_centers
        _validate = validator or validate_hypothesis
        _hypothesize = hypothesis_fn or make_hypotheses_batch
        _summarize = summarize_fn or summarize_triggers

        clusters = _select_clusters(
            character, pairs,
            n_in_cluster_case=16, n_in_cluster_sample=8, n_max_cluster=8, bs=8,
        )
        log.info("Making hypotheses for %d clusters in parallel...", len(clusters))
        statement_candidates, gates = _hypothesize(
            clusters, character, goal_topic,
            established_statements + self.statements, gate_path,
        )

        gates, statement_candidates = _summarize(character, gates, statement_candidates)
        global_statements: list[str] = []
        gated_statements: list[str] = []
        remained_gates: list[str] = []

        for gate, stmt in zip(gates, statement_candidates):
            res, _ = _validate(character, pairs, None, stmt)
            correctness = res["True"] / (res["True"] + res["False"] + 1e-8) + 1e-8
            if correctness >= cfg.threshold_accept:
                global_statements.append(stmt)
            else:
                gated_statements.append(stmt)
                remained_gates.append(gate)

        self.statements.extend(global_statements)

        deps = dict(_embedder=_select_clusters, _validator=_validate,
                     _hypothesis_fn=_hypothesize, _summarize_fn=_summarize)

        for gate, stmt in zip(remained_gates, gated_statements):
            res, filtered_pairs = _validate(character, pairs, gate, stmt)
            correctness = res["True"] / (res["True"] + res["False"] + 1e-8) + 1e-8
            broadness = 1 - res["Irrelevant"] / sum(res.values())
            if broadness <= cfg.threshold_filter:
                if correctness <= cfg.threshold_reject:
                    continue
                elif correctness >= cfg.threshold_accept:
                    self.gates.append(gate)
                    self.children.append(CDTNode(
                        character, goal_topic, None,
                        built_statements=[stmt],
                        depth=self.depth + 1,
                        established_statements=established_statements + self.statements,
                        gate_path=gate_path + [gate],
                        config=cfg,
                        **deps,
                    ))
                else:
                    self.gates.append(gate)
                    self.children.append(CDTNode(
                        character, goal_topic, filtered_pairs,
                        depth=self.depth + 1,
                        established_statements=established_statements + self.statements,
                        gate_path=gate_path + [gate],
                        config=cfg,
                        **deps,
                    ))

    def traverse(self, scene: str) -> list[str]:
        """Traverse the tree and collect applicable statements for a scene.

        Args:
            scene: The scene text to evaluate against gate conditions.

        Returns:
            List of statements from all nodes whose gates match the scene.
        """
        from canopy.validation import check_scene

        statements = deepcopy(self.statements)
        for gate, child in zip(self.gates, self.children):
            results = check_scene([scene], [gate])
            if results[0]:
                statements.extend(child.traverse(scene))
        return statements

    def verbalize(self, indent: int = 0) -> str:
        """Convert tree to human-readable indented text."""
        prefix = "  " * indent
        lines: list[str] = []
        for s in self.statements:
            lines.append(f"{prefix}- {s}")
        for gate, child in zip(self.gates, self.children):
            lines.append(f'{prefix}IF "{gate}":')
            lines.append(child.verbalize(indent + 1))
        return "\n".join(lines) if lines else f"{prefix}(empty)"

    def count_stats(self) -> dict[str, int]:
        """Count tree statistics recursively.

        Returns:
            Dict with keys: statements, gates, max_depth, total_nodes,
            total_statements, total_gates.
        """
        stats = {
            "statements": len(self.statements),
            "gates": len(self.gates),
            "max_depth": self.depth,
            "total_nodes": 1,
            "total_statements": len(self.statements),
            "total_gates": len(self.gates),
        }
        for child in self.children:
            child_stats = child.count_stats()
            stats["total_nodes"] += child_stats["total_nodes"]
            stats["total_statements"] += child_stats["total_statements"]
            stats["total_gates"] += child_stats["total_gates"]
            stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
        return stats


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

ATTRIBUTE_TOPICS = ("identity", "personality", "ability", "relationship")
MIN_RELATION_PAIRS = 16


def build_character_cdts(
    character: str,
    pairs: list[dict[str, Any]],
    other_characters: list[str],
    config: CDTConfig | None = None,
) -> tuple[dict[str, CDTNode], dict[str, CDTNode]]:
    """Build attribute and relationship CDTs for a character.

    Args:
        character: Character name (e.g. "Kasumi").
        pairs: Training scene-action pairs.
        other_characters: Other characters for relationship CDTs.
        config: CDT construction config. Uses defaults if None.

    Returns:
        (topic2cdt, rel_topic2cdt) — attribute and relationship CDT dicts.
    """
    cfg = config or CDTConfig()

    topic2cdt: dict[str, CDTNode] = {}
    for attribute in ATTRIBUTE_TOPICS:
        goal_topic = f"{character}'s {attribute}"
        log.info("Building CDT: %s", goal_topic)
        topic2cdt[goal_topic] = CDTNode(character, goal_topic, pairs, config=cfg)

    rel_topic2cdt: dict[str, CDTNode] = {}
    for other in other_characters:
        goal_topic = f"{character}'s interaction with {other}"
        relation_pairs = [d for d in pairs if other in d["last_character"]]
        if len(relation_pairs) >= MIN_RELATION_PAIRS:
            log.info("Building CDT: %s (%d pairs)", goal_topic, len(relation_pairs))
            rel_topic2cdt[goal_topic] = CDTNode(
                character, goal_topic, relation_pairs, config=cfg,
            )

    return topic2cdt, rel_topic2cdt
