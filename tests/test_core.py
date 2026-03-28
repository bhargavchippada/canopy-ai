"""Tests for canopy.core — CDTConfig, CDTNode, build_character_cdts."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from canopy.core import (
    ATTRIBUTE_TOPICS,
    MIN_PAIRS_FOR_TREE,
    MIN_RELATION_PAIRS,
    CDTConfig,
    CDTNode,
    build_character_cdts,
)

# ---------------------------------------------------------------------------
# CDTConfig
# ---------------------------------------------------------------------------


class TestCDTConfig:
    def test_default_values(self) -> None:
        cfg = CDTConfig()
        assert cfg.max_depth == 3
        assert cfg.threshold_accept == 0.8
        assert cfg.threshold_reject == 0.5
        assert cfg.threshold_filter == 0.8

    def test_custom_values(self) -> None:
        cfg = CDTConfig(max_depth=5, threshold_accept=0.9, threshold_reject=0.3, threshold_filter=0.7)
        assert cfg.max_depth == 5
        assert cfg.threshold_accept == 0.9

    def test_frozen(self) -> None:
        cfg = CDTConfig()
        with pytest.raises(AttributeError):
            cfg.max_depth = 10  # type: ignore[misc]

    def test_equality(self) -> None:
        assert CDTConfig() == CDTConfig()
        assert CDTConfig(max_depth=2) != CDTConfig(max_depth=3)

    def test_hashable(self) -> None:
        d: dict[CDTConfig, str] = {CDTConfig(): "default"}
        assert d[CDTConfig()] == "default"


# ---------------------------------------------------------------------------
# CDTNode — leaf construction (no LLM needed)
# ---------------------------------------------------------------------------


class TestCDTNodeLeaf:
    def test_built_statements(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        assert node.statements == ["Alice is kind"]
        assert node.gates == []
        assert node.children == []
        assert node.depth == 1

    def test_built_statements_is_copy(self) -> None:
        stmts = ["stmt1", "stmt2"]
        node = CDTNode("Bob", "personality", None, built_statements=stmts)
        assert node.statements == stmts
        assert node.statements is not stmts

    def test_built_statements_with_pairs_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            CDTNode("Alice", "identity", [{"action": "a", "scene": "s"}], built_statements=["s"])

    def test_empty_pairs_no_build(self) -> None:
        node = CDTNode("Alice", "identity", [])
        assert node.statements == []

    def test_few_pairs_no_build(self) -> None:
        pairs = [{"action": f"a{i}", "scene": f"s{i}"} for i in range(MIN_PAIRS_FOR_TREE)]
        node = CDTNode("Alice", "identity", pairs)
        assert node.statements == []

    def test_none_pairs_no_build(self) -> None:
        node = CDTNode("Alice", "identity", None)
        assert node.statements == []

    def test_depth_exceeds_max_no_build(self) -> None:
        pairs = [{"action": f"a{i}", "scene": f"s{i}"} for i in range(20)]
        cfg = CDTConfig(max_depth=1)
        node = CDTNode("Alice", "identity", pairs, depth=2, config=cfg)
        assert node.statements == []

    def test_custom_depth(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["x"], depth=5)
        assert node.depth == 5

    def test_default_config_when_none(self) -> None:
        pairs = [{"action": f"a{i}", "scene": f"s{i}"} for i in range(5)]
        node = CDTNode("Alice", "identity", pairs)
        assert node.statements == []


# ---------------------------------------------------------------------------
# Mock helpers for _build testing
# ---------------------------------------------------------------------------


def _mock_cluster_fn(character: str, pairs: list, **kw: Any) -> list[list[dict]]:
    mid = len(pairs) // 2
    return [pairs[:mid], pairs[mid:]]


def _mock_hypothesize(
    clusters: list,
    character: str,
    topic: str,
    est: list,
    gp: list,
    **kw: Any,
) -> tuple[list[str], list[str]]:
    return ([f"{character} is brave", f"{character} helps"], ["Danger?", "Friend nearby?"])


def _mock_summarize(character: str, gates: list, stmts: list, **kw: Any) -> tuple[list[str], list[str]]:
    return gates, stmts


def _mock_validate_accept(character: str, pairs: list, q: str | None, a: str, **kw: Any) -> tuple[dict, list]:
    return {"True": 90.0, "False": 1.0, "None": 5.0, "Irrelevant": 4.0}, pairs


def _mock_validate_gated(character: str, pairs: list, q: str | None, a: str, **kw: Any) -> tuple[dict, list]:
    if q is None:
        return {"True": 5.0, "False": 90.0, "None": 3.0, "Irrelevant": 2.0}, pairs
    return {"True": 85.0, "False": 5.0, "None": 5.0, "Irrelevant": 80.0}, pairs[:5]


def _mock_validate_reject(character: str, pairs: list, q: str | None, a: str, **kw: Any) -> tuple[dict, list]:
    if q is None:
        return {"True": 5.0, "False": 90.0, "None": 3.0, "Irrelevant": 2.0}, pairs
    return {"True": 3.0, "False": 90.0, "None": 5.0, "Irrelevant": 80.0}, pairs


# ---------------------------------------------------------------------------
# CDTNode — _build with mocked dependencies
# ---------------------------------------------------------------------------


class TestCDTNodeBuild:
    def _pairs(self, n: int = 20) -> list[dict[str, Any]]:
        return [{"action": f"action_{i}", "scene": f"scene_{i}"} for i in range(n)]

    def test_build_all_accepted_globally(self) -> None:
        node = CDTNode(
            "Alice",
            "identity",
            self._pairs(),
            config=CDTConfig(max_depth=1),
            _embedder=_mock_cluster_fn,
            _validator=_mock_validate_accept,
            _hypothesis_fn=_mock_hypothesize,
            _summarize_fn=_mock_summarize,
        )
        assert len(node.statements) == 2
        assert "Alice is brave" in node.statements
        assert node.gates == []
        assert node.children == []

    def test_build_gated_creates_children(self) -> None:
        node = CDTNode(
            "Alice",
            "identity",
            self._pairs(),
            config=CDTConfig(max_depth=2),
            _embedder=_mock_cluster_fn,
            _validator=_mock_validate_gated,
            _hypothesis_fn=_mock_hypothesize,
            _summarize_fn=_mock_summarize,
        )
        assert node.statements == []
        assert len(node.gates) == 2
        assert len(node.children) == 2
        for child in node.children:
            assert len(child.statements) == 1

    def test_build_all_rejected_empty(self) -> None:
        node = CDTNode(
            "Alice",
            "identity",
            self._pairs(),
            config=CDTConfig(max_depth=1),
            _embedder=_mock_cluster_fn,
            _validator=_mock_validate_reject,
            _hypothesis_fn=_mock_hypothesize,
            _summarize_fn=_mock_summarize,
        )
        assert node.statements == []
        assert node.gates == []

    def test_build_partially_correct_creates_recursive_child(self) -> None:
        """When correctness is between reject and accept, recurse with filtered_pairs."""

        def partial_validate(character: str, pairs: list, q: str | None, a: str, **kw: Any) -> tuple[dict, list]:
            if q is None:
                # Global check fails
                return {"True": 5.0, "False": 90.0, "None": 3.0, "Irrelevant": 2.0}, pairs
            # Gated: correctness between reject (0.5) and accept (0.8) → recurse
            # broadness = 1 - 80/100 = 0.2, which is <= threshold_filter (0.8) ✓
            # correctness = 60/(60+30) = 0.67, between reject (0.5) and accept (0.8) ✓
            return {"True": 60.0, "False": 30.0, "None": 5.0, "Irrelevant": 80.0}, pairs[:3]

        node = CDTNode(
            "Alice",
            "identity",
            self._pairs(),
            config=CDTConfig(max_depth=2),
            _embedder=_mock_cluster_fn,
            _validator=partial_validate,
            _hypothesis_fn=_mock_hypothesize,
            _summarize_fn=_mock_summarize,
        )
        # Should have created recursive children (not built_statements leaf)
        assert len(node.gates) == 2
        assert len(node.children) == 2
        # Children should be recursive (pairs passed, not built_statements)
        for child in node.children:
            # Children have depth 2 and very few pairs (3), so they're empty leaves
            assert child.statements == []

    def test_deps_forwarded_to_children(self) -> None:
        calls: list[str] = []

        def tracking_cluster(*a: Any, **kw: Any) -> list[list[dict]]:
            calls.append("cluster")
            return _mock_cluster_fn(*a, **kw)

        CDTNode(
            "Alice",
            "identity",
            self._pairs(),
            config=CDTConfig(max_depth=2),
            _embedder=tracking_cluster,
            _validator=_mock_validate_accept,
            _hypothesis_fn=_mock_hypothesize,
            _summarize_fn=_mock_summarize,
        )
        assert "cluster" in calls


# ---------------------------------------------------------------------------
# CDTNode — traversal and verbalization
# ---------------------------------------------------------------------------


class TestCDTNodeTraversal:
    def _make_tree(self) -> CDTNode:
        root = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        child1 = CDTNode("Alice", "identity", None, built_statements=["Alice helps others"], depth=2)
        child2 = CDTNode("Alice", "identity", None, built_statements=["Alice studies hard"], depth=2)
        root.gates = ["Is Alice helping?", "Is Alice at school?"]
        root.children = [child1, child2]
        return root

    def test_verbalize_leaf(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        assert "- Alice is kind" in node.verbalize()

    def test_verbalize_tree(self) -> None:
        text = self._make_tree().verbalize()
        assert "- Alice is kind" in text
        assert 'IF "Is Alice helping?"' in text
        assert "  - Alice helps others" in text

    def test_verbalize_empty(self) -> None:
        assert CDTNode("A", "x", None).verbalize() == "(empty)"

    def test_verbalize_indent(self) -> None:
        node = CDTNode("A", "x", None, built_statements=["s"])
        assert node.verbalize(indent=3).startswith("      - s")

    def test_count_stats_leaf(self) -> None:
        stats = CDTNode("A", "x", None, built_statements=["s1", "s2"]).count_stats()
        assert stats == {
            "statements": 2,
            "gates": 0,
            "max_depth": 1,
            "total_nodes": 1,
            "total_statements": 2,
            "total_gates": 0,
        }

    def test_count_stats_tree(self) -> None:
        stats = self._make_tree().count_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_statements"] == 3
        assert stats["total_gates"] == 2
        assert stats["max_depth"] == 2

    def test_traverse_with_mocked_check_scene(self) -> None:
        """Test traverse() with mocked NLI checker."""
        root = self._make_tree()

        # check_scene is imported inside traverse() via canopy.validation
        with patch("canopy.validation.check_scene") as mock_cs:
            mock_cs.side_effect = lambda texts, questions: [True] if "helping" in questions[0] else [False]
            stmts = root.traverse("Alice is helping someone")

        assert "Alice is kind" in stmts  # Root statement always included
        assert "Alice helps others" in stmts  # First gate matched
        assert "Alice studies hard" not in stmts  # Second gate didn't match

    def test_traverse_no_gates(self) -> None:
        """Traverse on a leaf node returns just its statements."""
        leaf = CDTNode("A", "x", None, built_statements=["stmt1", "stmt2"])
        # Leaf has no gates, so check_scene is never called
        with patch("canopy.validation.check_scene"):
            stmts = leaf.traverse("any scene")
        assert stmts == ["stmt1", "stmt2"]

    def test_count_stats_3_levels(self) -> None:
        leaf = CDTNode("A", "x", None, built_statements=["deep"], depth=3)
        mid = CDTNode("A", "x", None, built_statements=["mid"], depth=2)
        mid.gates, mid.children = ["gate"], [leaf]
        root = CDTNode("A", "x", None, built_statements=["root"], depth=1)
        root.gates, root.children = ["top"], [mid]
        stats = root.count_stats()
        assert stats["total_nodes"] == 3
        assert stats["max_depth"] == 3


# ---------------------------------------------------------------------------
# build_character_cdts
# ---------------------------------------------------------------------------


class TestBuildCharacterCdts:
    def test_attribute_topics(self) -> None:
        assert len(ATTRIBUTE_TOPICS) == 4

    def test_build_empty_pairs(self) -> None:
        t, r = build_character_cdts("Alice", [], ["Bob"])
        assert len(t) == 4
        assert len(r) == 0

    def test_topic_names(self) -> None:
        t, _ = build_character_cdts("Kasumi", [], [])
        assert set(t.keys()) == {f"Kasumi's {a}" for a in ATTRIBUTE_TOPICS}

    def test_relationship_threshold_met(self) -> None:
        """With exactly MIN_RELATION_PAIRS, relationship CDT is created (but empty — too few for tree)."""
        # Use exactly MIN_RELATION_PAIRS so CDTNode gets called but pairs <= MIN_PAIRS_FOR_TREE*2
        # avoids triggering _build which needs models
        n = MIN_RELATION_PAIRS
        pairs = [{"action": f"a{i}", "scene": f"s{i}", "last_character": ["Bob"]} for i in range(n)]
        # max_depth=0 prevents any _build calls
        _, r = build_character_cdts("Alice", pairs, ["Bob"], config=CDTConfig(max_depth=0))
        assert "Alice's interaction with Bob" in r

    def test_relationship_threshold_not_met(self) -> None:
        n = MIN_RELATION_PAIRS - 1
        pairs = [{"action": f"a{i}", "scene": f"s{i}", "last_character": ["Bob"]} for i in range(n)]
        _, r = build_character_cdts("Alice", pairs, ["Bob"], config=CDTConfig(max_depth=0))
        assert len(r) == 0

    def test_multiple_relationships(self) -> None:
        pairs = list(
            [{"action": f"a{i}", "scene": f"s{i}", "last_character": ["Bob"]} for i in range(20)]
            + [{"action": f"b{i}", "scene": f"s{i}", "last_character": ["Charlie"]} for i in range(20)]
            + [{"action": f"c{i}", "scene": f"s{i}", "last_character": ["Dave"]} for i in range(5)]
        )
        _, r = build_character_cdts("Alice", pairs, ["Bob", "Charlie", "Dave"], config=CDTConfig(max_depth=0))
        assert "Alice's interaction with Bob" in r
        assert "Alice's interaction with Charlie" in r
        assert "Alice's interaction with Dave" not in r

    def test_build_character_cdts_parallel(self) -> None:
        """Verify parallel construction populates both attr and rel dicts."""
        pairs = [{"action": f"a{i}", "scene": f"s{i}", "last_character": ["Bob"]} for i in range(20)]
        cfg = CDTConfig(max_depth=0)
        t, r = build_character_cdts("Alice", pairs, ["Bob"], config=cfg, max_parallel=2)
        assert len(t) == 4
        assert set(t.keys()) == {f"Alice's {a}" for a in ATTRIBUTE_TOPICS}
        assert "Alice's interaction with Bob" in r

    def test_build_character_cdts_max_parallel(self) -> None:
        """Verify ThreadPoolExecutor is called with the specified max_parallel value."""
        with patch("canopy.core.ThreadPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool.__exit__ = MagicMock(return_value=False)
            # Make submit return futures that resolve to CDTNode-like objects
            mock_future = MagicMock()
            mock_future.result.return_value = CDTNode("Alice", "x", None)
            mock_pool.submit.return_value = mock_future
            mock_pool_cls.return_value = mock_pool

            # Patch as_completed to return futures in order
            with patch("canopy.core.as_completed") as mock_as_completed:
                # Each submit call creates one future; we need them all returned
                def side_effect_as_completed(futures_dict: dict) -> list:
                    return list(futures_dict.keys())

                mock_as_completed.side_effect = side_effect_as_completed

                build_character_cdts("Alice", [], [], config=CDTConfig(), max_parallel=7)

            mock_pool_cls.assert_called_once_with(max_workers=7)

    def test_max_parallel_validation(self) -> None:
        with pytest.raises(ValueError, match="max_parallel must be >= 1"):
            build_character_cdts("Alice", [], [], max_parallel=0)
        with pytest.raises(ValueError, match="max_parallel must be >= 1"):
            build_character_cdts("Alice", [], [], max_parallel=-1)

    def test_build_character_cdts_handles_failure(self) -> None:
        """A failing CDT build raises RuntimeError reporting the failed topic(s)."""
        call_count = 0

        def _failing_cdtnode(*args: Any, **kwargs: Any) -> CDTNode:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("GPU OOM")
            return CDTNode.__new__(CDTNode)

        with patch("canopy.core.CDTNode", side_effect=_failing_cdtnode):
            with pytest.raises(RuntimeError, match="CDT build failed"):
                build_character_cdts("Alice", [], [], config=CDTConfig())
