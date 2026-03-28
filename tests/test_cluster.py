"""Tests for canopy.cluster — KMeans and HDBSCAN clustering."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from canopy.cluster import (
    HDBSCANCluster,
    KMeansCluster,
    select_representative_samples,
)

# ---------------------------------------------------------------------------
# KMeansCluster
# ---------------------------------------------------------------------------


class TestKMeansCluster:
    def test_basic_clustering(self) -> None:
        # 20 points in 2D, clearly separable into 2 clusters
        rng = np.random.RandomState(42)
        cluster1 = rng.randn(10, 2) + np.array([5, 5])
        cluster2 = rng.randn(10, 2) + np.array([-5, -5])
        embeddings = np.vstack([cluster1, cluster2])

        km = KMeansCluster(n_in_cluster_case=10, n_max_cluster=8)
        labels, centroids = km.fit_predict(embeddings)
        assert labels.shape == (20,)
        assert centroids.shape[0] == 2  # 20/10 = 2 clusters
        assert centroids.shape[1] == 2  # 2D

    def test_respects_n_max_cluster(self) -> None:
        rng = np.random.RandomState(0)
        embeddings = rng.randn(100, 4)
        km = KMeansCluster(n_in_cluster_case=5, n_max_cluster=3)
        labels, centroids = km.fit_predict(embeddings)
        assert centroids.shape[0] == 3  # Capped at 3

    def test_single_cluster(self) -> None:
        embeddings = np.ones((5, 3))
        km = KMeansCluster(n_in_cluster_case=10)
        labels, centroids = km.fit_predict(embeddings)
        assert centroids.shape[0] == 1

    def test_seed_reproducibility(self) -> None:
        rng = np.random.RandomState(0)
        embeddings = rng.randn(30, 4)
        km = KMeansCluster(seed=123)
        _, c1 = km.fit_predict(embeddings)
        _, c2 = km.fit_predict(embeddings)
        np.testing.assert_array_equal(c1, c2)

    def test_default_params(self) -> None:
        km = KMeansCluster()
        assert km.n_in_cluster_case == 16
        assert km.n_max_cluster == 8
        assert km.seed == 42

    def test_empty_input(self) -> None:
        km = KMeansCluster()
        labels, centroids = km.fit_predict(np.empty((0, 4)))
        assert labels.shape == (0,)
        assert centroids.shape == (0, 4)


# ---------------------------------------------------------------------------
# HDBSCANCluster
# ---------------------------------------------------------------------------


class TestHDBSCANCluster:
    def test_fallback_to_kmeans_when_not_installed(self) -> None:
        rng = np.random.RandomState(0)
        embeddings = rng.randn(20, 4)

        with patch.dict("sys.modules", {"hdbscan": None}):
            hdb = HDBSCANCluster()
            with patch("canopy.cluster.KMeansCluster.fit_predict") as mock_km:
                mock_km.return_value = (np.zeros(20, dtype=int), np.zeros((1, 4)))
                labels, centroids = hdb.fit_predict(embeddings)
                mock_km.assert_called_once()

    def test_fallback_when_import_fails(self) -> None:
        rng = np.random.RandomState(0)
        embeddings = rng.randn(20, 4)

        hdb = HDBSCANCluster()

        # Simulate ImportError by replacing the import mechanism
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "hdbscan":
                raise ImportError("No module named 'hdbscan'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            labels, centroids = hdb.fit_predict(embeddings)
            assert labels.shape == (20,)

    def test_default_params(self) -> None:
        hdb = HDBSCANCluster()
        assert hdb.min_cluster_size == 5
        assert hdb.min_samples == 3
        assert hdb.n_max_cluster == 8

    def test_hdbscan_with_valid_clusters(self) -> None:
        """Test HDBSCAN when it finds valid clusters."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 4)

        mock_hdbscan_module = type("module", (), {})()
        mock_clusterer = type("clusterer", (), {})()
        # 3 clusters: 0, 1, 2 (no noise)
        mock_clusterer.fit_predict = lambda e: np.array([0]*7 + [1]*7 + [2]*6)
        mock_hdbscan_module.HDBSCAN = lambda **kw: mock_clusterer

        import sys
        old = sys.modules.get("hdbscan")
        sys.modules["hdbscan"] = mock_hdbscan_module
        try:
            hdb = HDBSCANCluster(n_max_cluster=8)
            labels, centroids = hdb.fit_predict(embeddings)
            assert centroids.shape[0] == 3
            assert labels.shape == (20,)
        finally:
            if old is not None:
                sys.modules["hdbscan"] = old
            else:
                del sys.modules["hdbscan"]

    def test_hdbscan_no_clusters_fallback(self) -> None:
        """When HDBSCAN finds only noise, fall back to KMeans."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 4)

        mock_hdbscan_module = type("module", (), {})()
        mock_clusterer = type("clusterer", (), {})()
        mock_clusterer.fit_predict = lambda e: np.full(len(e), -1)  # All noise
        mock_hdbscan_module.HDBSCAN = lambda **kw: mock_clusterer

        import sys
        old = sys.modules.get("hdbscan")
        sys.modules["hdbscan"] = mock_hdbscan_module
        try:
            hdb = HDBSCANCluster()
            labels, centroids = hdb.fit_predict(embeddings)
            assert centroids.shape[0] > 0  # KMeans fallback
        finally:
            if old is not None:
                sys.modules["hdbscan"] = old
            else:
                del sys.modules["hdbscan"]

    def test_hdbscan_exceeds_max_clusters(self) -> None:
        """When HDBSCAN finds more clusters than n_max, keep largest."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(50, 4)

        mock_hdbscan_module = type("module", (), {})()
        mock_clusterer = type("clusterer", (), {})()
        # 5 clusters of varying sizes
        mock_clusterer.fit_predict = lambda e: np.array(
            [0]*15 + [1]*12 + [2]*10 + [3]*8 + [4]*5,
        )
        mock_hdbscan_module.HDBSCAN = lambda **kw: mock_clusterer

        import sys
        old = sys.modules.get("hdbscan")
        sys.modules["hdbscan"] = mock_hdbscan_module
        try:
            hdb = HDBSCANCluster(n_max_cluster=3)
            labels, centroids = hdb.fit_predict(embeddings)
            assert centroids.shape[0] == 3  # Capped at 3
            # Noise points (from dropped clusters) should be -1
            assert (labels == -1).sum() > 0
        finally:
            if old is not None:
                sys.modules["hdbscan"] = old
            else:
                del sys.modules["hdbscan"]


# ---------------------------------------------------------------------------
# select_representative_samples
# ---------------------------------------------------------------------------


class TestSelectRepresentativeSamples:
    def test_basic_selection(self) -> None:
        pairs = [{"action": f"a{i}", "scene": f"s{i}"} for i in range(20)]
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 4)
        centroids = embeddings[:2]  # Use first 2 points as centroids

        clusters = select_representative_samples(pairs, embeddings, centroids, n_samples=3)
        assert len(clusters) == 2
        assert len(clusters[0]) == 3
        assert len(clusters[1]) == 3

    def test_samples_closest_to_centroid(self) -> None:
        # Create embeddings where pairs 0-4 are near centroid, rest are far
        embeddings = np.zeros((10, 2))
        embeddings[:5] = np.array([[0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.15, 0.15], [0.05, 0.05]])
        embeddings[5:] = np.array([[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]])
        centroid = np.array([[0.0, 0.0]])

        pairs = [{"id": i} for i in range(10)]
        clusters = select_representative_samples(pairs, embeddings, centroid, n_samples=3)
        # The 3 closest to [0,0] should be from the first 5
        selected_ids = [p["id"] for p in clusters[0]]
        assert all(i < 5 for i in selected_ids)

    def test_n_samples_capped_at_available(self) -> None:
        pairs = [{"id": i} for i in range(3)]
        embeddings = np.eye(3)
        centroids = np.array([[1, 0, 0]])

        clusters = select_representative_samples(pairs, embeddings, centroids, n_samples=10)
        assert len(clusters[0]) == 3  # Only 3 available

    def test_empty_pairs(self) -> None:
        clusters = select_representative_samples(
            [], np.empty((0, 4)), np.array([[0, 0, 0, 0]]), n_samples=5,
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == 0

    def test_length_mismatch_raises(self) -> None:
        pairs = [{"id": i} for i in range(5)]
        embeddings = np.eye(3)  # Only 3 rows
        centroids = np.array([[1, 0, 0]])
        with pytest.raises(ValueError, match="pairs length"):
            select_representative_samples(pairs, embeddings, centroids)

    def test_multiple_centroids(self) -> None:
        pairs = [{"id": i} for i in range(20)]
        embeddings = np.random.RandomState(0).randn(20, 4)
        centroids = embeddings[:4]

        clusters = select_representative_samples(pairs, embeddings, centroids, n_samples=5)
        assert len(clusters) == 4
        for cluster in clusters:
            assert len(cluster) == 5
