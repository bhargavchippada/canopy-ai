"""Pure metric functions for CDT pipeline quality assessment.

No GPU or LLM dependencies. Only numpy and sklearn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import silhouette_score as _silhouette_score

if TYPE_CHECKING:
    from canopy.core import CDTNode

logger = logging.getLogger(__name__)


def compute_data_quality(pairs: list[dict]) -> dict:
    """Compute basic statistics about observation pairs.

    Args:
        pairs: List of dicts with "scene" and "action" string keys.

    Returns:
        Dict with pair_count, scene_length_mean, action_length_mean.
    """
    if not pairs:
        return {
            "pair_count": 0,
            "scene_length_mean": 0.0,
            "action_length_mean": 0.0,
        }

    scene_lengths = [len(p.get("scene", "")) for p in pairs]
    action_lengths = [len(p.get("action", "")) for p in pairs]

    return {
        "pair_count": len(pairs),
        "scene_length_mean": float(np.mean(scene_lengths)),
        "action_length_mean": float(np.mean(action_lengths)),
    }


def compute_clustering_quality(
    labels: np.ndarray,
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> dict:
    """Compute clustering quality metrics.

    Args:
        labels: Cluster assignment per sample (shape: [n_samples]).
        embeddings: Sample embeddings (shape: [n_samples, dim]).
        centroids: Cluster centroids (shape: [n_clusters, dim]).

    Returns:
        Dict with silhouette_score, cluster_balance, n_clusters.
    """
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = int(len(unique_labels))

    # Silhouette requires >= 2 clusters and more samples than clusters.
    # Mask out noise labels (label=-1 from HDBSCAN) before computing.
    mask = labels >= 0
    n_valid = int(mask.sum())
    if n_clusters <= 1 or n_valid <= n_clusters:
        sil_score = -1.0
    else:
        sil_score = float(_silhouette_score(embeddings[mask], labels[mask]))

    # Cluster balance: min_size / max_size (1.0 = perfect)
    if n_clusters == 0:
        balance = 0.0
    else:
        cluster_sizes = np.array(
            [int(np.sum(labels == lbl)) for lbl in unique_labels]
        )
        max_size = int(cluster_sizes.max())
        min_size = int(cluster_sizes.min())
        balance = float(min_size / max_size) if max_size > 0 else 0.0

    return {
        "silhouette_score": sil_score,
        "cluster_balance": balance,
        "n_clusters": n_clusters,
    }


def compute_hypothesis_quality(
    hypotheses: list[str],
    _pairs: list[dict] | None = None,
) -> dict:
    """Compute hypothesis set quality metrics.

    Args:
        hypotheses: List of hypothesis strings.
        pairs: Observation pairs (reserved for future coverage metrics).

    Returns:
        Dict with count, mean_word_count, diversity.
    """
    if not hypotheses:
        return {
            "count": 0,
            "mean_word_count": 0.0,
            "diversity": 0.0,
        }

    word_counts = [len(h.split()) for h in hypotheses]
    all_words = [w.lower() for h in hypotheses for w in h.split()]
    total_words = len(all_words)
    unique_words = len(set(all_words))
    diversity = float(unique_words / total_words) if total_words > 0 else 0.0

    return {
        "count": len(hypotheses),
        "mean_word_count": float(np.mean(word_counts)),
        "diversity": diversity,
    }


def compute_tree_quality(
    topic2cdt: dict[str, CDTNode],
    pairs: list[dict],
) -> dict:
    """Compute CDT tree structure quality metrics.

    Args:
        topic2cdt: Mapping of topic name to root CDTNode.
        pairs: Observation pairs used to build the tree.

    Returns:
        Dict with total_nodes, total_statements, total_gates,
        max_depth, statement_coverage.
    """
    total_nodes = 0
    total_statements = 0
    total_gates = 0
    max_depth = 0

    for node in topic2cdt.values():
        stats = node.count_stats()
        total_nodes += stats["total_nodes"]
        total_statements += stats["total_statements"]
        total_gates += stats["total_gates"]
        max_depth = max(max_depth, stats["max_depth"])

    pair_count = len(pairs)
    coverage = float(total_statements / pair_count) if pair_count > 0 else 0.0

    return {
        "total_nodes": total_nodes,
        "total_statements": total_statements,
        "total_gates": total_gates,
        "max_depth": max_depth,
        "statement_coverage": coverage,
    }
