"""Tests for canopy.quality — pure metric functions (no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.core import CDTNode
from canopy.quality import (
    compute_clustering_quality,
    compute_data_quality,
    compute_hypothesis_quality,
    compute_tree_quality,
)

# ---------------------------------------------------------------------------
# compute_data_quality
# ---------------------------------------------------------------------------


class TestComputeDataQuality:
    def test_empty_pairs(self):
        result = compute_data_quality([])
        assert result == {
            "pair_count": 0,
            "scene_length_mean": 0.0,
            "action_length_mean": 0.0,
        }

    def test_single_pair(self):
        pairs = [{"scene": "hello", "action": "world!"}]
        result = compute_data_quality(pairs)
        assert result["pair_count"] == 1
        assert result["scene_length_mean"] == pytest.approx(5.0)
        assert result["action_length_mean"] == pytest.approx(6.0)

    def test_multiple_pairs(self):
        pairs = [
            {"scene": "ab", "action": "cdef"},
            {"scene": "ghij", "action": "kl"},
        ]
        result = compute_data_quality(pairs)
        assert result["pair_count"] == 2
        # scene lengths: 2, 4 -> mean 3.0
        assert result["scene_length_mean"] == pytest.approx(3.0)
        # action lengths: 4, 2 -> mean 3.0
        assert result["action_length_mean"] == pytest.approx(3.0)

    def test_missing_keys_default_to_empty(self):
        pairs = [{"scene": "abc"}, {"action": "de"}]
        result = compute_data_quality(pairs)
        assert result["pair_count"] == 2
        # scene lengths: 3, 0 -> mean 1.5
        assert result["scene_length_mean"] == pytest.approx(1.5)
        # action lengths: 0, 2 -> mean 1.0
        assert result["action_length_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_clustering_quality
# ---------------------------------------------------------------------------


class TestComputeClusteringQuality:
    def test_two_well_separated_clusters(self):
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ])
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[0.05, 0.0], [10.05, 0.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert -1.0 <= result["silhouette_score"] <= 1.0
        assert result["silhouette_score"] > 0.9  # well separated
        assert result["cluster_balance"] == pytest.approx(1.0)
        assert result["n_clusters"] == 2

    def test_single_cluster(self):
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 0])
        centroids = np.array([[2.0, 3.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert result["silhouette_score"] == -1.0
        assert result["cluster_balance"] == pytest.approx(1.0)
        assert result["n_clusters"] == 1

    def test_n_samples_equal_n_clusters(self):
        embeddings = np.array([[1.0], [2.0]])
        labels = np.array([0, 1])
        centroids = np.array([[1.0], [2.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert result["silhouette_score"] == -1.0
        assert result["n_clusters"] == 2

    def test_imbalanced_clusters(self):
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 0.0],
        ])
        labels = np.array([0, 0, 0, 1])
        centroids = np.array([[0.1, 0.0], [10.0, 0.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        # min_size=1, max_size=3 -> balance = 1/3
        assert result["cluster_balance"] == pytest.approx(1.0 / 3.0)
        assert result["n_clusters"] == 2

    def test_balanced_three_clusters(self):
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ])
        labels = np.array([0, 0, 1, 1, 2, 2])
        centroids = np.array([[0.05, 0.0], [5.05, 0.0], [10.05, 0.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert result["cluster_balance"] == pytest.approx(1.0)
        assert result["n_clusters"] == 3

    def test_noise_labels_ignored(self):
        """Labels with -1 (HDBSCAN noise) should not count as a cluster."""
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [5.0, 5.0],  # noise point
        ])
        labels = np.array([0, 0, 1, 1, -1])
        centroids = np.array([[0.05, 0.0], [10.05, 0.0]])

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert result["n_clusters"] == 2

    def test_all_noise_zero_clusters(self):
        """All labels are -1 (noise) -> n_clusters=0, balance=0.0."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([-1, -1])
        centroids = np.array([]).reshape(0, 2)

        result = compute_clustering_quality(labels, embeddings, centroids)
        assert result["n_clusters"] == 0
        assert result["silhouette_score"] == -1.0
        assert result["cluster_balance"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_hypothesis_quality
# ---------------------------------------------------------------------------


class TestComputeHypothesisQuality:
    def test_empty_hypotheses(self):
        result = compute_hypothesis_quality([], [])
        assert result == {
            "count": 0,
            "mean_word_count": 0.0,
            "diversity": 0.0,
        }

    def test_single_hypothesis(self):
        hypotheses = ["Alice tends to be cheerful"]
        result = compute_hypothesis_quality(hypotheses, [])
        assert result["count"] == 1
        assert result["mean_word_count"] == pytest.approx(5.0)
        # 5 unique words / 5 total = 1.0
        assert result["diversity"] == pytest.approx(1.0)

    def test_multiple_hypotheses_varying_length(self):
        hypotheses = [
            "short one",          # 2 words
            "this is a longer hypothesis",  # 5 words
        ]
        result = compute_hypothesis_quality(hypotheses, [])
        assert result["count"] == 2
        assert result["mean_word_count"] == pytest.approx(3.5)
        assert 0.0 < result["diversity"] <= 1.0

    def test_all_identical_words(self):
        hypotheses = ["the the the", "the the"]
        result = compute_hypothesis_quality(hypotheses, [])
        assert result["count"] == 2
        # 5 total words, 1 unique -> diversity = 1/5
        assert result["diversity"] == pytest.approx(0.2)

    def test_all_unique_words(self):
        hypotheses = ["alpha beta", "gamma delta"]
        result = compute_hypothesis_quality(hypotheses, [])
        assert result["count"] == 2
        # 4 total words, 4 unique -> diversity = 1.0
        assert result["diversity"] == pytest.approx(1.0)

    def test_pairs_argument_accepted(self):
        """pairs is reserved for future use but should not cause errors."""
        pairs = [{"scene": "s", "action": "a"}]
        result = compute_hypothesis_quality(["hypothesis one"], pairs)
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# compute_tree_quality
# ---------------------------------------------------------------------------


class TestComputeTreeQuality:
    def _make_leaf(self, character, topic, statements, depth=1):
        node = CDTNode(character, topic, None, built_statements=statements, depth=depth)
        return node

    def test_empty_topic2cdt(self):
        result = compute_tree_quality({}, [])
        assert result == {
            "total_nodes": 0,
            "total_statements": 0,
            "total_gates": 0,
            "max_depth": 0,
            "statement_coverage": 0.0,
        }

    def test_single_leaf_node(self):
        node = self._make_leaf("Alice", "identity", ["stmt1", "stmt2"])
        pairs = [{"scene": "s1", "action": "a1"}, {"scene": "s2", "action": "a2"}]

        result = compute_tree_quality({"identity": node}, pairs)
        assert result["total_nodes"] == 1
        assert result["total_statements"] == 2
        assert result["total_gates"] == 0
        assert result["max_depth"] == 1
        # 2 statements / 2 pairs = 1.0
        assert result["statement_coverage"] == pytest.approx(1.0)

    def test_tree_with_children(self):
        parent = self._make_leaf("Alice", "identity", ["parent_stmt"], depth=1)
        child = self._make_leaf("Alice", "identity", ["child_stmt1", "child_stmt2"], depth=2)
        parent.gates = ["Is Alice cheerful?"]
        parent.children = [child]

        pairs = [{"scene": "s", "action": "a"}] * 3

        result = compute_tree_quality({"identity": parent}, pairs)
        assert result["total_nodes"] == 2
        assert result["total_statements"] == 3  # 1 parent + 2 child
        assert result["total_gates"] == 1
        assert result["max_depth"] == 2
        # 3 statements / 3 pairs = 1.0
        assert result["statement_coverage"] == pytest.approx(1.0)

    def test_multiple_topics(self):
        node_a = self._make_leaf("Alice", "identity", ["s1"], depth=1)
        node_b = self._make_leaf("Alice", "emotion", ["s2", "s3"], depth=1)

        pairs = [{"scene": "s", "action": "a"}] * 6

        result = compute_tree_quality(
            {"identity": node_a, "emotion": node_b}, pairs
        )
        assert result["total_nodes"] == 2
        assert result["total_statements"] == 3
        assert result["total_gates"] == 0
        assert result["max_depth"] == 1
        # 3 statements / 6 pairs = 0.5
        assert result["statement_coverage"] == pytest.approx(0.5)

    def test_no_pairs_zero_coverage(self):
        node = self._make_leaf("Alice", "identity", ["stmt"])

        result = compute_tree_quality({"identity": node}, [])
        assert result["total_nodes"] == 1
        assert result["total_statements"] == 1
        assert result["statement_coverage"] == pytest.approx(0.0)

    def test_deep_nested_tree(self):
        leaf = self._make_leaf("Alice", "t", ["leaf_s"], depth=3)
        mid = self._make_leaf("Alice", "t", ["mid_s"], depth=2)
        mid.gates = ["gate_mid"]
        mid.children = [leaf]
        root = self._make_leaf("Alice", "t", ["root_s"], depth=1)
        root.gates = ["gate_root"]
        root.children = [mid]

        pairs = [{"scene": "s", "action": "a"}] * 3

        result = compute_tree_quality({"t": root}, pairs)
        assert result["total_nodes"] == 3
        assert result["total_statements"] == 3
        assert result["total_gates"] == 2
        assert result["max_depth"] == 3
        assert result["statement_coverage"] == pytest.approx(1.0)
