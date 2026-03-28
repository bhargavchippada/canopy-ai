"""Clustering algorithms for CDT scene-action pair grouping.

Provides KMeans (default, matches CDT paper) and HDBSCAN (density-based,
discovers cluster count from data) strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


class ClusterStrategy(Protocol):
    """Protocol for clustering strategies."""

    def fit_predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings and return (labels, centroids).

        Args:
            embeddings: (N, D) array of document embeddings.

        Returns:
            (labels, centroids) — cluster assignments and centroid vectors.
        """
        ...


class KMeansCluster:
    """KMeans clustering (default — matches CDT paper).

    Automatically determines n_clusters based on data size and n_in_cluster_case.
    """

    def __init__(self, n_in_cluster_case: int = 16, n_max_cluster: int = 8, seed: int = 42) -> None:
        self.n_in_cluster_case = n_in_cluster_case
        self.n_max_cluster = n_max_cluster
        self.seed = seed

    def fit_predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_clusters = min(int(np.ceil(embeddings.shape[0] / self.n_in_cluster_case)), self.n_max_cluster)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        kmeans.fit(embeddings)
        return kmeans.labels_, kmeans.cluster_centers_


class HDBSCANCluster:
    """HDBSCAN density-based clustering (discovers cluster count from data).

    Falls back to KMeans if HDBSCAN is not installed or produces no clusters.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3, n_max_cluster: int = 8) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_max_cluster = n_max_cluster

    def fit_predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            import hdbscan
        except ImportError:
            log.warning("hdbscan not installed, falling back to KMeans")
            return KMeansCluster(n_max_cluster=self.n_max_cluster).fit_predict(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
        )
        labels = clusterer.fit_predict(embeddings)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Noise points

        if len(unique_labels) == 0:
            log.warning("HDBSCAN found no clusters, falling back to KMeans")
            return KMeansCluster(n_max_cluster=self.n_max_cluster).fit_predict(embeddings)

        # Compute centroids for each cluster
        centroids = np.array([embeddings[labels == label].mean(axis=0) for label in sorted(unique_labels)])

        # Limit to n_max_cluster by keeping the largest
        if len(unique_labels) > self.n_max_cluster:
            cluster_sizes = [(label, (labels == label).sum()) for label in sorted(unique_labels)]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_labels = {cs[0] for cs in cluster_sizes[: self.n_max_cluster]}
            # Remap labels
            new_labels = np.full_like(labels, -1)
            centroids_list = []
            for new_idx, old_label in enumerate(sorted(keep_labels)):
                new_labels[labels == old_label] = new_idx
                centroids_list.append(embeddings[labels == old_label].mean(axis=0))
            labels = new_labels
            centroids = np.array(centroids_list)

        return labels, centroids


def select_representative_samples(
    pairs: list[dict[str, Any]],
    embeddings: np.ndarray,
    centroids: np.ndarray,
    n_samples: int = 8,
) -> list[list[dict[str, Any]]]:
    """Select representative samples from each cluster, closest to centroids.

    Args:
        pairs: Original scene-action pairs.
        embeddings: (N, D) document embeddings (same order as pairs).
        centroids: (K, D) cluster centroids.
        n_samples: Number of samples per cluster.

    Returns:
        List of clusters, each containing n_samples representative pairs.
    """
    clusters: list[list[dict[str, Any]]] = []
    for centroid in centroids:
        distances = np.sqrt(((embeddings - centroid) ** 2).sum(axis=-1))
        indices = distances.argsort()[:n_samples]
        clusters.append([pairs[idx] for idx in indices])
    return clusters
