"""Core CDT data structures — CDTNode and CDTConfig."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from tqdm import tqdm


@dataclass(frozen=True)
class CDTConfig:
    """Configuration for CDT construction."""

    max_depth: int = 3
    threshold_accept: float = 0.8
    threshold_reject: float = 0.5
    threshold_filter: float = 0.8


class CDTNode:
    """A node in a Codified Decision Tree.

    Each node holds:
    - statements: globally applicable behavioral statements
    - gates: questions that filter to child nodes
    - children: child CDTNode instances (1 gate → 1 child)
    """

    def __init__(
        self,
        character: str,
        goal_topic: str,
        pairs: list[dict] | None,
        *,
        built_statements: list[str] | None = None,
        depth: int = 1,
        established_statements: list[str] | None = None,
        gate_path: list[str] | None = None,
        config: CDTConfig | None = None,
        # Injected dependencies — set by builder, not by caller
        _embedder: object | None = None,
        _validator: object | None = None,
        _hypothesis_fn: object | None = None,
        _summarize_fn: object | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.gates: list[str] = []
        self.children: list[CDTNode] = []
        self.depth = depth

        cfg = config or CDTConfig()
        established_statements = established_statements or []
        gate_path = gate_path or []

        if built_statements is not None:
            assert pairs is None
            self.statements = built_statements
        elif pairs is None or len(pairs) <= 8 or self.depth > cfg.max_depth:
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
        pairs: list[dict],
        established_statements: list[str],
        gate_path: list[str],
        cfg: CDTConfig,
        embedder: object | None,
        validator: object | None,
        hypothesis_fn: object | None,
        summarize_fn: object | None,
    ) -> None:
        """Build the tree node (extracted from __init__ for clarity)."""
        from canopy.embeddings import select_cluster_centers
        from canopy.prompts import make_hypotheses_batch, summarize_triggers
        from canopy.validation import validate_hypothesis

        # Use injected functions or module-level defaults
        _select_clusters = embedder or select_cluster_centers
        _validate = validator or validate_hypothesis
        _hypothesize = hypothesis_fn or make_hypotheses_batch
        _summarize = summarize_fn or summarize_triggers

        clusters = _select_clusters(character, pairs, n_in_cluster_case=16, n_in_cluster_sample=8, n_max_cluster=8, bs=8)
        print(f"  Making hypotheses for {len(clusters)} clusters in parallel...")
        statement_candidates, gates = _hypothesize(clusters, character, goal_topic, established_statements + self.statements, gate_path)

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
                    ))
                else:
                    self.gates.append(gate)
                    self.children.append(CDTNode(
                        character, goal_topic, filtered_pairs,
                        depth=self.depth + 1,
                        established_statements=established_statements + self.statements,
                        gate_path=gate_path + [gate],
                        config=cfg,
                        _embedder=_select_clusters,
                        _validator=_validate,
                        _hypothesis_fn=_hypothesize,
                        _summarize_fn=_summarize,
                    ))

    def traverse(self, scene: str) -> list[str]:
        """Traverse the tree and collect applicable statements for a scene."""
        from canopy.validation import check_scene

        statements = deepcopy(self.statements)
        for gate, child in zip(self.gates, self.children):
            results = check_scene([scene], [gate])
            if results[0]:
                statements.extend(child.traverse(scene))
        return statements

    def verbalize(self, indent: int = 0) -> str:
        """Convert tree to human-readable text."""
        prefix = "  " * indent
        lines: list[str] = []
        for s in self.statements:
            lines.append(f"{prefix}- {s}")
        for gate, child in zip(self.gates, self.children):
            lines.append(f'{prefix}IF "{gate}":')
            lines.append(child.verbalize(indent + 1))
        return "\n".join(lines) if lines else f"{prefix}(empty)"
