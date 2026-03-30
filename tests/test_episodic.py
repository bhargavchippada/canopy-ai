"""Tests for canopy.episodic — EpisodicIndex, hybrid_ground, format_grounding."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.builder import BehavioralObservation
from canopy.core import CDTNode
from canopy.embeddings import EmbeddingCache
from canopy.episodic import (
    EpisodicIndex,
    GroundingResult,
    RetrievedObservation,
    _traverse_with_gates,
    format_grounding,
    hybrid_ground,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_observations(n: int = 5) -> list[BehavioralObservation]:
    """Create n simple observations for testing."""
    return [
        BehavioralObservation(
            scene=f"Scene {i}: context about topic {i % 3}",
            action=f"Action {i}: specific behavior observed",
            actor="Alice",
        )
        for i in range(n)
    ]


def _make_embeddings(n: int = 5, dim: int = 8) -> np.ndarray:
    """Create n L2-normalized random embeddings."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


def _make_index(n: int = 5, dim: int = 8) -> EpisodicIndex:
    """Create an EpisodicIndex with n observations."""
    return EpisodicIndex.from_arrays(
        observations=_make_observations(n),
        embeddings=_make_embeddings(n, dim),
    )


def _embed_fn_factory(dim: int = 8) -> callable:
    """Create a deterministic embed function for testing."""
    rng = np.random.default_rng(99)
    cache: dict[str, np.ndarray] = {}

    def embed_fn(text: str) -> np.ndarray:
        if text not in cache:
            vec = rng.standard_normal(dim).astype(np.float32)
            cache[text] = vec / np.linalg.norm(vec)
        return cache[text]

    return embed_fn


# ---------------------------------------------------------------------------
# EpisodicIndex construction
# ---------------------------------------------------------------------------

class TestEpisodicIndexConstruction:
    def test_from_arrays(self) -> None:
        obs = _make_observations(3)
        emb = _make_embeddings(3, 8)
        index = EpisodicIndex.from_arrays(obs, emb)
        assert len(index) == 3
        assert index.embeddings.shape == (3, 8)
        assert len(index.observations) == 3

    def test_from_embedding_cache(self) -> None:
        obs = _make_observations(4)
        surface = _make_embeddings(4, 8)
        generator = _make_embeddings(4, 16)
        cache = EmbeddingCache(surface=surface, generator=generator)
        index = EpisodicIndex.from_embedding_cache(obs, cache)
        assert len(index) == 4
        assert index.embeddings.shape == (4, 8)  # uses surface embeddings

    def test_from_embedding_cache_size_mismatch(self) -> None:
        obs = _make_observations(3)
        surface = _make_embeddings(4, 8)
        generator = _make_embeddings(4, 16)
        cache = EmbeddingCache(surface=surface, generator=generator)
        with pytest.raises(ValueError, match="observations.*!=.*cache"):
            EpisodicIndex.from_embedding_cache(obs, cache)

    def test_embeddings_must_be_2d(self) -> None:
        obs = _make_observations(3)
        with pytest.raises(ValueError, match="2-D"):
            EpisodicIndex.from_arrays(obs, np.zeros(3))

    def test_length_mismatch_raises(self) -> None:
        obs = _make_observations(3)
        emb = _make_embeddings(5, 8)
        with pytest.raises(ValueError, match="observations.*!=.*embeddings"):
            EpisodicIndex.from_arrays(obs, emb)

    def test_embeddings_are_read_only(self) -> None:
        index = _make_index()
        with pytest.raises(ValueError, match="read-only"):
            index.embeddings[0, 0] = 999.0

    def test_observations_are_tuple(self) -> None:
        obs = _make_observations(3)
        emb = _make_embeddings(3, 8)
        index = EpisodicIndex(observations=obs, embeddings=emb)  # type: ignore[arg-type]
        assert isinstance(index.observations, tuple)

    def test_empty_index(self) -> None:
        index = EpisodicIndex.from_arrays([], _make_embeddings(0, 8).reshape(0, 8))
        assert len(index) == 0


# ---------------------------------------------------------------------------
# EpisodicIndex retrieval
# ---------------------------------------------------------------------------

class TestEpisodicIndexRetrieval:
    def test_retrieve_basic(self) -> None:
        index = _make_index(10, 8)
        query = index.embeddings[0]  # query is identical to first observation
        results = index.retrieve(query, top_k=3)
        assert len(results) == 3
        assert results[0].score > results[1].score
        # First result should be the exact match
        assert results[0].observation == index.observations[0]
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_retrieve_top_k_larger_than_index(self) -> None:
        index = _make_index(3, 8)
        query = _make_embeddings(1, 8)[0]
        results = index.retrieve(query, top_k=100)
        assert len(results) == 3  # can't return more than index size

    def test_retrieve_empty_index(self) -> None:
        emb = np.zeros((0, 8), dtype=np.float32)
        index = EpisodicIndex.from_arrays([], emb)
        query = _make_embeddings(1, 8)[0]
        results = index.retrieve(query, top_k=5)
        assert results == []

    def test_retrieve_dimension_mismatch(self) -> None:
        index = _make_index(5, 8)
        bad_query = np.ones(16, dtype=np.float32)
        with pytest.raises(ValueError, match="query dim"):
            index.retrieve(bad_query, top_k=3)

    def test_retrieve_with_gate_filtering(self) -> None:
        # Create index where first 5 obs are "domain A" and last 5 are "domain B"
        n = 10
        dim = 8
        rng = np.random.default_rng(42)

        # Domain A: embeddings near [1,0,0,...] and domain B: near [0,1,0,...]
        emb_a = np.zeros((5, dim), dtype=np.float32)
        emb_a[:, 0] = 1.0
        emb_a += rng.standard_normal((5, dim)).astype(np.float32) * 0.1
        norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
        emb_a /= norms_a

        emb_b = np.zeros((5, dim), dtype=np.float32)
        emb_b[:, 1] = 1.0
        emb_b += rng.standard_normal((5, dim)).astype(np.float32) * 0.1
        norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
        emb_b /= norms_b

        all_emb = np.concatenate([emb_a, emb_b])
        obs = _make_observations(n)
        index = EpisodicIndex.from_arrays(obs, all_emb)

        # Query near domain A
        query = np.zeros(dim, dtype=np.float32)
        query[0] = 1.0

        # Gate condition also near domain A
        gate_emb = np.zeros((1, dim), dtype=np.float32)
        gate_emb[0, 0] = 1.0

        results = index.retrieve(
            query, top_k=5, gate_embeddings=gate_emb, gate_threshold=0.5
        )
        # All results should be from domain A (indices 0-4)
        for r in results:
            idx = obs.index(r.observation)
            assert idx < 5, f"Expected domain A observation, got index {idx}"

    def test_gate_filter_fallback_when_all_eliminated(self) -> None:
        index = _make_index(5, 8)
        query = index.embeddings[0]

        # Gate embedding orthogonal to everything — should eliminate all, then fall back
        gate = np.zeros((1, 8), dtype=np.float32)
        gate[0, 7] = 1.0  # unlikely to match random embeddings well

        results = index.retrieve(
            query, top_k=3, gate_embeddings=gate, gate_threshold=0.99
        )
        # Should fall back to unfiltered and return results
        assert len(results) >= 1

    def test_retrieve_results_sorted_by_score(self) -> None:
        index = _make_index(20, 8)
        query = _make_embeddings(1, 8)[0]
        results = index.retrieve(query, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_nan_query_raises(self) -> None:
        index = _make_index(5, 8)
        bad_query = np.full(8, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            index.retrieve(bad_query, top_k=3)

    def test_retrieve_inf_query_raises(self) -> None:
        index = _make_index(5, 8)
        bad_query = np.full(8, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            index.retrieve(bad_query, top_k=3)

    def test_retrieve_1d_gate_embeddings_raises(self) -> None:
        index = _make_index(5, 8)
        query = index.embeddings[0]
        bad_gate = np.ones(8, dtype=np.float32)  # 1-D, should be 2-D
        with pytest.raises(ValueError, match="2-D"):
            index.retrieve(query, top_k=3, gate_embeddings=bad_gate)

    def test_retrieve_gate_dim_mismatch_raises(self) -> None:
        index = _make_index(5, 8)
        query = index.embeddings[0]
        bad_gate = np.ones((1, 16), dtype=np.float32)  # wrong dim
        with pytest.raises(ValueError, match="gate_embeddings dim"):
            index.retrieve(query, top_k=3, gate_embeddings=bad_gate)

    def test_retrieve_nan_gate_embeddings_raises(self) -> None:
        index = _make_index(5, 8)
        query = index.embeddings[0]
        nan_gate = np.full((1, 8), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            index.retrieve(query, top_k=3, gate_embeddings=nan_gate)

    def test_all_results_have_finite_scores(self) -> None:
        index = _make_index(10, 8)
        query = _make_embeddings(1, 8)[0]
        results = index.retrieve(query, top_k=5)
        for r in results:
            assert np.isfinite(r.score)


# ---------------------------------------------------------------------------
# _traverse_with_gates
# ---------------------------------------------------------------------------

class TestTraverseWithGates:
    def test_leaf_node(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["stmt1", "stmt2"])
        stmts, gates = _traverse_with_gates(node, "some scene")
        assert stmts == ["stmt1", "stmt2"]
        assert gates == []

    def test_empty_node(self) -> None:
        node = CDTNode("Alice", "identity", None)
        stmts, gates = _traverse_with_gates(node, "some scene")
        assert stmts == []
        assert gates == []

    def test_with_gated_children(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test traversal when gates activate — mock check_scene to return True."""
        from canopy import validation as val_mod

        # Build a tree with a gate
        child = CDTNode("Alice", "identity", None, built_statements=["child_stmt"])
        parent = CDTNode("Alice", "identity", None, built_statements=["parent_stmt"])
        parent.gates = ["when stressed"]
        parent.children = [child]

        # Mock check_scene to always return True
        monkeypatch.setattr(val_mod, "check_scene", lambda scenes, gates: [True] * len(scenes))

        stmts, gates = _traverse_with_gates(parent, "stressful situation")
        assert "parent_stmt" in stmts
        assert "child_stmt" in stmts
        assert "when stressed" in gates

    def test_gate_not_activated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test traversal when gates don't activate."""
        from canopy import validation as val_mod
        monkeypatch.setattr(val_mod, "check_scene", lambda scenes, gates: [False] * len(scenes))

        child = CDTNode("Alice", "identity", None, built_statements=["child_stmt"])
        parent = CDTNode("Alice", "identity", None, built_statements=["parent_stmt"])
        parent.gates = ["when happy"]
        parent.children = [child]

        stmts, gates = _traverse_with_gates(parent, "sad scene")
        assert stmts == ["parent_stmt"]
        assert gates == []

    def test_depth_limit_prevents_infinite_recursion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cyclic CDT should hit depth limit instead of stack overflow."""
        from canopy import validation as val_mod
        monkeypatch.setattr(val_mod, "check_scene", lambda scenes, gates: [True] * len(scenes))

        # Create a cycle: node → child → node (via manual assignment)
        node = CDTNode("Alice", "identity", None, built_statements=["stmt"])
        node.gates = ["always true"]
        node.children = [node]  # cycle!

        stmts, gates = _traverse_with_gates(node, "scene")
        # Should terminate due to depth limit, not crash
        assert len(stmts) > 0
        assert len(gates) > 0

    def test_gate_none_result_treated_as_inactive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When check_scene returns None (uncertain), gate should not activate."""
        from canopy import validation as val_mod
        monkeypatch.setattr(val_mod, "check_scene", lambda scenes, gates: [None] * len(scenes))

        child = CDTNode("Alice", "identity", None, built_statements=["child_stmt"])
        parent = CDTNode("Alice", "identity", None, built_statements=["parent_stmt"])
        parent.gates = ["uncertain gate"]
        parent.children = [child]

        stmts, gates = _traverse_with_gates(parent, "ambiguous scene")
        assert stmts == ["parent_stmt"]  # Only parent, child not traversed
        assert gates == []


# ---------------------------------------------------------------------------
# hybrid_ground
# ---------------------------------------------------------------------------

class TestHybridGround:
    def test_basic_grounding(self) -> None:
        # Build a simple CDT with known statements
        cdt = CDTNode("Alice", "identity", None, built_statements=["Alice is brave"])
        topic2cdt = {"identity": cdt}

        index = _make_index(5, 8)
        embed_fn = _embed_fn_factory(8)

        result = hybrid_ground(
            "What would Alice do?",
            topic2cdt,
            index,
            embed_fn=embed_fn,
            top_k=3,
        )

        assert isinstance(result, GroundingResult)
        assert "Alice is brave" in result.behavioral_statements
        assert len(result.factual_observations) <= 3

    def test_grounding_with_empty_cdt(self) -> None:
        cdt = CDTNode("Alice", "identity", None)
        topic2cdt = {"identity": cdt}
        index = _make_index(5, 8)
        embed_fn = _embed_fn_factory(8)

        result = hybrid_ground("query", topic2cdt, index, embed_fn=embed_fn)
        assert result.behavioral_statements == ()
        assert result.active_gates == ()
        assert len(result.factual_observations) <= 10

    def test_grounding_with_multiple_topics(self) -> None:
        cdt1 = CDTNode("Alice", "identity", None, built_statements=["stmt1"])
        cdt2 = CDTNode("Alice", "personality", None, built_statements=["stmt2"])
        topic2cdt = {"identity": cdt1, "personality": cdt2}

        index = _make_index(5, 8)
        embed_fn = _embed_fn_factory(8)

        result = hybrid_ground("query", topic2cdt, index, embed_fn=embed_fn)
        assert "stmt1" in result.behavioral_statements
        assert "stmt2" in result.behavioral_statements

    def test_grounding_with_active_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test hybrid_ground when CDT gates activate — covers gate embedding path."""
        from canopy import validation as val_mod
        monkeypatch.setattr(val_mod, "check_scene", lambda scenes, gates: [True] * len(scenes))

        child = CDTNode("Alice", "identity", None, built_statements=["child_stmt"])
        parent = CDTNode("Alice", "identity", None, built_statements=["parent_stmt"])
        parent.gates = ["when coding"]
        parent.children = [child]
        topic2cdt = {"identity": parent}

        index = _make_index(10, 8)
        embed_fn = _embed_fn_factory(8)

        result = hybrid_ground("query", topic2cdt, index, embed_fn=embed_fn, top_k=5)
        assert "parent_stmt" in result.behavioral_statements
        assert "child_stmt" in result.behavioral_statements
        assert "when coding" in result.active_gates
        assert len(result.factual_observations) <= 5

    def test_grounding_with_empty_index(self) -> None:
        cdt = CDTNode("Alice", "identity", None, built_statements=["stmt"])
        topic2cdt = {"identity": cdt}
        emb = np.zeros((0, 8), dtype=np.float32)
        index = EpisodicIndex.from_arrays([], emb)
        embed_fn = _embed_fn_factory(8)

        result = hybrid_ground("query", topic2cdt, index, embed_fn=embed_fn)
        assert result.behavioral_statements == ("stmt",)
        assert result.factual_observations == ()


# ---------------------------------------------------------------------------
# format_grounding
# ---------------------------------------------------------------------------

class TestFormatGrounding:
    def test_format_with_both(self) -> None:
        obs = BehavioralObservation(scene="meeting context", action="proposed idea", actor="Bob")
        result = GroundingResult(
            behavioral_statements=("Bob is creative",),
            factual_observations=(RetrievedObservation(observation=obs, score=0.9),),
            active_gates=(),
        )
        text = format_grounding(result)
        assert "Behavioral Profile" in text
        assert "Bob is creative" in text
        assert "Relevant Context" in text
        assert "proposed idea" in text

    def test_format_behavioral_only(self) -> None:
        result = GroundingResult(
            behavioral_statements=("stmt1", "stmt2"),
            factual_observations=(),
            active_gates=(),
        )
        text = format_grounding(result)
        assert "Behavioral Profile" in text
        assert "stmt1" in text
        assert "Relevant Context" not in text

    def test_format_factual_only(self) -> None:
        obs = BehavioralObservation(scene="ctx", action="did thing", actor="X")
        result = GroundingResult(
            behavioral_statements=(),
            factual_observations=(RetrievedObservation(observation=obs, score=0.5),),
            active_gates=(),
        )
        text = format_grounding(result)
        assert "Behavioral Profile" not in text
        assert "Relevant Context" in text

    def test_format_empty(self) -> None:
        result = GroundingResult(
            behavioral_statements=(),
            factual_observations=(),
            active_gates=(),
        )
        text = format_grounding(result)
        assert text == ""

    def test_format_respects_limits(self) -> None:
        stmts = tuple(f"stmt{i}" for i in range(30))
        result = GroundingResult(
            behavioral_statements=stmts,
            factual_observations=(),
            active_gates=(),
        )
        text = format_grounding(result, max_behavioral=5)
        assert text.count("- stmt") == 5

    def test_format_truncates_long_scenes(self) -> None:
        obs = BehavioralObservation(
            scene="x" * 500,
            action="short action",
            actor="A",
        )
        result = GroundingResult(
            behavioral_statements=(),
            factual_observations=(RetrievedObservation(observation=obs, score=0.5),),
            active_gates=(),
        )
        text = format_grounding(result)
        # Scene should be truncated to exactly 200 chars
        assert "x" * 200 in text
        assert "x" * 201 not in text

    def test_format_truncates_long_actions(self) -> None:
        obs = BehavioralObservation(
            scene="short scene",
            action="y" * 1000,
            actor="A",
        )
        result = GroundingResult(
            behavioral_statements=(),
            factual_observations=(RetrievedObservation(observation=obs, score=0.5),),
            active_gates=(),
        )
        text = format_grounding(result)
        # Action should be truncated to exactly 500 chars
        assert "y" * 500 in text
        assert "y" * 501 not in text


# ---------------------------------------------------------------------------
# GroundingResult immutability
# ---------------------------------------------------------------------------

class TestGroundingResultImmutability:
    def test_frozen(self) -> None:
        result = GroundingResult(
            behavioral_statements=("a",),
            factual_observations=(),
            active_gates=(),
        )
        with pytest.raises(AttributeError):
            result.behavioral_statements = ("b",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievedObservation
# ---------------------------------------------------------------------------

class TestRetrievedObservation:
    def test_frozen(self) -> None:
        obs = BehavioralObservation(scene="s", action="a", actor="X")
        r = RetrievedObservation(observation=obs, score=0.8)
        assert r.score == 0.8
        with pytest.raises(AttributeError):
            r.score = 0.5  # type: ignore[misc]
