"""Integration tests — require real GPU models and optionally LLM.

Run with: uv run python -m pytest tests/test_integration.py -m integration -v
Skip with: uv run python -m pytest (default — skipped by addopts)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# Model paths — use local if available, else HF hub names
SURFACE_EMBEDDER = os.environ.get(
    "CANOPY_SURFACE_EMBEDDER",
    str(Path.home() / "models" / "Qwen3-Embedding-0.6B"),
)
GENERATOR_EMBEDDER = os.environ.get(
    "CANOPY_GENERATOR_EMBEDDER",
    str(Path.home() / "models" / "Qwen3-0.6B"),
)
DISCRIMINATOR = os.environ.get(
    "CANOPY_DISCRIMINATOR",
    str(Path.home() / "models" / "deberta-v3-base-rp-nli"),
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]


@pytest.fixture(scope="module")
def embedding_models():
    """Load embedding models once for all tests in this module."""
    from canopy.embeddings import init_models

    init_models(SURFACE_EMBEDDER, GENERATOR_EMBEDDER, DEVICE)
    yield
    # No teardown needed — models stay loaded


@pytest.fixture(scope="module")
def validation_model():
    """Load the NLI validation model once for all tests."""
    from canopy.validation import init_models

    init_models(DISCRIMINATOR, DEVICE)
    yield


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_surface_encode(self, embedding_models: None) -> None:
        from canopy.embeddings import surface_encode

        embeddings = surface_encode(["Hello world", "Testing embedding model"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Embedding dimension
        # Should be L2-normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_generative_encode(self, embedding_models: None) -> None:
        from canopy.embeddings import generative_encode

        embeddings = generative_encode(["The cat sat on the mat", "Dogs are loyal"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        # fp16 models have lower precision — use relaxed tolerance
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_select_cluster_centers(self, embedding_models: None) -> None:
        from canopy.embeddings import select_cluster_centers

        pairs = [
            {"scene": f"Scene number {i} where things happen", "action": f"Character does action {i}"}
            for i in range(20)
        ]
        clusters = select_cluster_centers(
            "TestChar", pairs,
            n_in_cluster_case=10, n_in_cluster_sample=3, n_max_cluster=4, bs=4,
        )
        assert len(clusters) == 2  # 20/10 = 2 clusters
        assert len(clusters[0]) == 3  # n_in_cluster_sample
        for pair in clusters[0]:
            assert "scene" in pair
            assert "action" in pair

    def test_different_texts_produce_different_embeddings(self, embedding_models: None) -> None:
        from canopy.embeddings import surface_encode

        emb = surface_encode(["I love pizza", "Quantum mechanics is complex"])
        similarity = np.dot(emb[0], emb[1])
        assert similarity < 0.95  # Different texts should have different embeddings


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_check_scene(self, validation_model: None) -> None:
        from canopy.validation import check_scene

        results = check_scene(
            ["Alice ran to the store to buy milk."],
            ["Did Alice go somewhere?"],
        )
        assert len(results) == 1
        assert results[0] in (True, False, None)

    def test_check_scene_batch(self, validation_model: None) -> None:
        from canopy.validation import check_scene

        results = check_scene(
            ["Bob ate lunch.", "Carol went swimming."],
            ["Did Bob eat?", "Did Carol drive?"],
        )
        assert len(results) == 2
        for r in results:
            assert r in (True, False, None)

    def test_check_statement_probs(self, validation_model: None) -> None:
        from canopy.validation import check_statement_probs

        probs = check_statement_probs(
            "Alice",
            ["Alice helped Bob with his homework."],
            ["Alice tends to help her friends."],
        )
        assert len(probs) == 3  # [false, none, true]
        assert all(p >= 0 for p in probs)

    def test_validate_hypothesis(self, validation_model: None) -> None:
        from canopy.validation import validate_hypothesis

        pairs = [
            {"scene": "At school, Alice saw Bob struggling.", "action": "Alice helped Bob."},
            {"scene": "During lunch, Alice sat alone.", "action": "Alice read a book."},
            {"scene": "After class, Alice met Carol.", "action": "Alice greeted Carol warmly."},
        ]
        res, filtered = validate_hypothesis(
            "Alice", pairs, None, "Alice tends to be kind to others.",
        )
        # Structural assertions — don't assume specific NLI outcomes
        total = sum(res.values())
        assert total > 0, "Expected at least one validated pair"
        assert all(v >= 0 for v in res.values())
        assert len(filtered) == len(pairs)  # No gate → all pairs kept

    def test_validate_hypothesis_with_gate(self, validation_model: None) -> None:
        from canopy.validation import validate_hypothesis

        pairs = [
            {"scene": "Bob asked Alice to help him carry the boxes. Alice, can you help me please?", "action": "Alice immediately dropped what she was doing and helped Bob carry the heavy boxes."},
            {"scene": "Carol was sitting alone in the park reading a book.", "action": "Alice walked past without stopping."},
            {"scene": "Dave called out to Alice for assistance with his project.", "action": "Alice came over and assisted Dave."},
        ]
        res, filtered = validate_hypothesis(
            "Alice", pairs,
            "Did someone ask Alice for help?",
            "Alice tends to help when asked.",
        )
        # The gate should classify some as relevant and some as irrelevant
        total = sum(res.values())
        assert total > 0  # At least some classification happened
        # filtered should have fewer or equal pairs than original
        assert len(filtered) <= len(pairs)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_build_cdt_with_real_models(
        self, embedding_models: None, validation_model: None,
    ) -> None:
        """Build a small CDT with real embeddings and validation.

        Uses max_depth=1 to keep it fast (no recursive subtrees).
        Skips LLM hypothesis gen — uses mocked hypothesize/summarize.
        """
        from canopy.builder import BehavioralObservation, build_cdt
        from canopy.core import CDTConfig

        # Use 7 observations — below MIN_PAIRS_FOR_TREE(8), so _build
        # is NOT called and no LLM is needed. This tests the full pipeline
        # path (observations → pairs → CDTNode) without requiring Claude.
        observations = [
            BehavioralObservation(
                scene=f"Scene {i}: the group is discussing plans",
                action=f"Alice suggests idea number {i}",
                actor="Alice",
            )
            for i in range(7)
        ]

        tree = build_cdt(
            observations,
            character="Alice",
            topic="personality",
            config=CDTConfig(max_depth=1),
        )
        assert tree is not None
        stats = tree.count_stats()
        assert stats["total_nodes"] == 1  # Root only, no subtrees
        assert stats["total_statements"] == 0  # No hypotheses generated

    def test_wikify_integration(
        self, embedding_models: None, validation_model: None,
    ) -> None:
        """Test wikify on a real (but small) CDT."""
        from canopy.core import CDTNode
        from canopy.wikify import wikify_tree

        tree = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        md = wikify_tree(tree, title="Alice's identity")
        assert "Alice is kind" in md
        assert "## Alice's identity" in md
