"""Core CDT data structures — CDTNode, CDTConfig, and build helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        time_decay_enabled: Enable T-CDT temporal weighting in validation.
        time_decay_half_life_days: Half-life for temporal weight decay.
    """

    max_depth: int = 3
    threshold_accept: float = 0.8
    threshold_reject: float = 0.5
    threshold_filter: float = 0.8
    time_decay_enabled: bool = False
    time_decay_half_life_days: int = 90


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
        _embedding_cache: Any | None = None,
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
                character,
                goal_topic,
                pairs,
                established_statements,
                gate_path,
                cfg,
                _embedder,
                _validator,
                _hypothesis_fn,
                _summarize_fn,
                _embedding_cache,
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
        embedding_cache: Any | None = None,
    ) -> None:
        """Build the tree node via hypothesis generation + validation."""
        from canopy.embeddings import select_cluster_centers
        from canopy.prompts import make_hypotheses_batch, merge_similar_hypotheses, summarize_triggers
        from canopy.validation import validate_hypothesis

        _validate = validator or validate_hypothesis
        _hypothesize = hypothesis_fn or make_hypotheses_batch
        _summarize = summarize_fn or summarize_triggers

        # When embedding_cache is provided and no custom embedder override,
        # pass cache through to select_cluster_centers for zero-GPU clustering
        if embedder is not None:
            _select_clusters = embedder
            clusters = _select_clusters(
                character,
                pairs,
                n_in_cluster_case=16,
                n_in_cluster_sample=8,
                n_max_cluster=8,
                bs=8,
            )
        else:
            clusters = select_cluster_centers(
                character,
                pairs,
                n_in_cluster_case=16,
                n_in_cluster_sample=8,
                n_max_cluster=8,
                bs=8,
                embedding_cache=embedding_cache,
            )
        log.info("Making hypotheses for %d clusters in parallel...", len(clusters))
        statement_candidates, gates = _hypothesize(
            clusters,
            character,
            goal_topic,
            established_statements + self.statements,
            gate_path,
        )

        # E1: Merge near-duplicate hypotheses before summarization
        gates, statement_candidates = merge_similar_hypotheses(gates, statement_candidates)
        gates, statement_candidates = _summarize(character, gates, statement_candidates)
        global_statements: list[str] = []
        gated_statements: list[str] = []
        remained_gates: list[str] = []

        temporal_kwargs: dict[str, Any] = {}
        if cfg.time_decay_enabled:
            temporal_kwargs = {
                "time_decay_enabled": True,
                "time_decay_half_life_days": cfg.time_decay_half_life_days,
            }

        for gate, stmt in zip(gates, statement_candidates):
            res, _ = _validate(character, pairs, None, stmt, **temporal_kwargs)
            correctness = res.get("True", 0) / (res.get("True", 0) + res.get("False", 0) + 1e-8) + 1e-8
            if correctness >= cfg.threshold_accept:
                global_statements.append(stmt)
            else:
                gated_statements.append(stmt)
                remained_gates.append(gate)

        self.statements.extend(global_statements)

        for gate, stmt in zip(remained_gates, gated_statements):
            res, filtered_pairs = _validate(character, pairs, gate, stmt, **temporal_kwargs)
            correctness = res.get("True", 0) / (res.get("True", 0) + res.get("False", 0) + 1e-8) + 1e-8
            broadness = 1 - res.get("Irrelevant", 0) / (sum(res.values()) + 1e-8)
            if broadness <= cfg.threshold_filter:
                if correctness <= cfg.threshold_reject:
                    continue
                elif correctness >= cfg.threshold_accept:
                    self.gates.append(gate)
                    self.children.append(
                        CDTNode(
                            character,
                            goal_topic,
                            None,
                            built_statements=[stmt],
                            depth=self.depth + 1,
                            established_statements=established_statements + self.statements,
                            gate_path=gate_path + [gate],
                            config=cfg,
                            _embedder=embedder,
                            _validator=_validate,
                            _hypothesis_fn=_hypothesize,
                            _summarize_fn=_summarize,
                            _embedding_cache=embedding_cache,
                        )
                    )
                else:
                    self.gates.append(gate)
                    self.children.append(
                        CDTNode(
                            character,
                            goal_topic,
                            filtered_pairs,
                            depth=self.depth + 1,
                            established_statements=established_statements + self.statements,
                            gate_path=gate_path + [gate],
                            config=cfg,
                            _embedder=embedder,
                            _validator=_validate,
                            _hypothesis_fn=_hypothesize,
                            _summarize_fn=_summarize,
                            _embedding_cache=embedding_cache,
                        )
                    )

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
MIN_TOPIC_PAIRS = 8  # Minimum pairs for a discovered topic to build a CDT


def build_character_cdts(
    character: str,
    pairs: list[dict[str, Any]],
    other_characters: list[str],
    config: CDTConfig | None = None,
    *,
    max_parallel: int = 4,
    embedding_cache: Any | None = None,
    discover_extra_topics: bool = False,
    n_extra_topics: int = 4,
) -> tuple[dict[str, CDTNode], dict[str, CDTNode]]:
    """Build attribute and relationship CDTs for a character.

    Topics are built concurrently using a thread pool. Default ``max_parallel=4``
    overlaps LLM API round-trips across topics.

    When ``discover_extra_topics=True``, the standard 4 attribute topics are
    built first, then additional topics are discovered organically from the data
    via clustering. This surfaces non-dominant behavioral patterns that the
    fixed topics may miss.

    Args:
        character: Character name (e.g. "Kasumi").
        pairs: Training scene-action pairs. Must have ``_embed_idx`` keys when
            ``embedding_cache`` is provided.
        other_characters: Other characters for relationship CDTs.
        config: CDT construction config. Uses defaults if None.
        max_parallel: Maximum concurrent CDT builds. Default 4.
        embedding_cache: Pre-computed embeddings from ``precompute_embeddings()``.
        discover_extra_topics: If True, discover additional topics from data.
        n_extra_topics: Number of extra topics to discover (default 4).

    Returns:
        (topic2cdt, rel_topic2cdt) — attribute and relationship CDT dicts.
    """
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    cfg = config or CDTConfig()

    # Collect all tasks: (kind, goal_topic, topic_pairs)
    tasks: list[tuple[str, str, list[dict[str, Any]]]] = []

    # Standard attribute topics (always included)
    for attribute in ATTRIBUTE_TOPICS:
        goal_topic = f"{character}'s {attribute}"
        tasks.append(("attr", goal_topic, pairs))

    # Discovered topics (additional, from data clustering)
    if discover_extra_topics:
        try:
            discovered = discover_topics(
                character, pairs,
                embedding_cache=embedding_cache,
                n_max_topics=n_extra_topics,
            )
            for topic_label, topic_pairs in discovered:
                tasks.append(("attr", topic_label, topic_pairs))
                log.info("Added discovered topic: %s (%d pairs)", topic_label, len(topic_pairs))
        except Exception:
            log.exception("Topic discovery failed — continuing with standard topics only")

    for other in other_characters:
        goal_topic = f"{character}'s interaction with {other}"
        relation_pairs = [d for d in pairs if other in d.get("last_character", [])]
        if len(relation_pairs) >= MIN_RELATION_PAIRS:
            tasks.append(("rel", goal_topic, relation_pairs))

    topic2cdt: dict[str, CDTNode] = {}
    rel_topic2cdt: dict[str, CDTNode] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for kind, goal_topic, topic_pairs in tasks:
            log.info("Building CDT: %s (%d pairs)", goal_topic, len(topic_pairs))
            future = executor.submit(
                CDTNode,
                character,
                goal_topic,
                topic_pairs,
                config=cfg,
                _embedding_cache=embedding_cache,
            )
            futures[future] = (kind, goal_topic)

        for future in as_completed(futures):
            kind, goal_topic = futures[future]
            try:
                node = future.result()
                if kind == "attr":
                    topic2cdt[goal_topic] = node
                else:
                    rel_topic2cdt[goal_topic] = node
                log.info("Completed CDT: %s", goal_topic)
            except Exception:
                log.exception("Failed to build CDT: %s", goal_topic)
                failed.append(goal_topic)

    if failed:
        raise RuntimeError(f"CDT build failed for {len(failed)} topic(s): {failed}")

    return topic2cdt, rel_topic2cdt


def discover_topics(
    character: str,
    pairs: list[dict[str, Any]],
    *,
    embedding_cache: Any | None = None,
    n_max_topics: int = 8,
    min_topic_pairs: int = MIN_TOPIC_PAIRS,
    model: str | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Discover behavioral topics organically from the data.

    Instead of using hardcoded topics (identity, personality, etc.),
    clusters all observations and labels each cluster as a topic.
    This surfaces non-dominant behavioral patterns that fixed topics miss.

    Args:
        character: Character name.
        pairs: All training pairs (must have _embed_idx if embedding_cache provided).
        embedding_cache: Pre-computed embeddings from precompute_embeddings().
        n_max_topics: Maximum number of topics to discover.
        min_topic_pairs: Minimum pairs per topic (smaller clusters are dropped).
        model: LLM model for topic labeling.

    Returns:
        List of (topic_label, topic_pairs) tuples.
    """
    from canopy.embeddings import select_cluster_centers
    from canopy.llm import generate

    if not pairs:
        return []

    # Step 1: Cluster ALL pairs into behavioral groups
    # Use larger clusters for topic discovery (not the small samples used in CDT building)
    if embedding_cache is not None:
        from canopy.cluster import KMeansCluster

        indices = [p["_embed_idx"] for p in pairs]
        subset = embedding_cache.subset(indices)
        clusterer = KMeansCluster(
            n_in_cluster_case=max(min_topic_pairs, len(pairs) // n_max_topics),
            n_max_cluster=n_max_topics,
        )
        labels, _ = clusterer.fit_predict(subset.document)

        # Group pairs by cluster label
        cluster_groups: dict[int, list[dict[str, Any]]] = {}
        for pair, label in zip(pairs, labels):
            cluster_groups.setdefault(label, []).append(pair)
    else:
        # Fallback: use select_cluster_centers (loads models)
        clusters = select_cluster_centers(
            character, pairs,
            n_in_cluster_case=max(min_topic_pairs, len(pairs) // n_max_topics),
            n_in_cluster_sample=min(8, len(pairs)),
            n_max_cluster=n_max_topics,
            bs=8,
        )
        cluster_groups = {i: cluster for i, cluster in enumerate(clusters)}

    # Step 2: Filter small clusters
    valid_clusters = {
        k: v for k, v in cluster_groups.items()
        if len(v) >= min_topic_pairs
    }
    log.info("Discovered %d clusters (%d valid with >=%d pairs)",
             len(cluster_groups), len(valid_clusters), min_topic_pairs)

    # Step 3: Label each cluster via LLM
    topics: list[tuple[str, list[dict[str, Any]]]] = []
    for cluster_id, cluster_pairs in sorted(valid_clusters.items()):
        # Sample a few examples for labeling
        sample = cluster_pairs[:5]
        examples = "\n\n".join(
            f"Scene: {p['scene'][:200]}\nAction: {p['action'][:200]}"
            for p in sample
        )
        label_prompt = (
            f"Given these examples of {character}'s behavior, what specific behavioral "
            f"theme or pattern do they represent? Answer with a short label (2-5 words) "
            f"that captures the distinct behavioral mode shown.\n\n"
            f"Examples:\n{examples}\n\n"
            f"Label (2-5 words):"
        )
        try:
            raw_label = generate(label_prompt, model=model, max_tokens=20)
            # Sanitize: strip whitespace/quotes/newlines, limit length
            label = raw_label.strip().strip('"\'.\n\r')
            label = " ".join(label.split())[:60]  # collapse whitespace, cap length
            if not label:
                label = f"behavioral_cluster_{cluster_id}"
        except Exception:
            label = f"behavioral_cluster_{cluster_id}"
            log.warning("Topic labeling failed for cluster %d, using default", cluster_id)

        topic_name = "{character}'s {label}".replace("{character}", character).replace("{label}", label.lower())
        topics.append((topic_name, cluster_pairs))
        log.info("  Topic: %s (%d pairs)", topic_name, len(cluster_pairs))

    return topics
