"""Tests for canopy.core — CDTConfig, CDTNode, build_character_cdts."""

from __future__ import annotations

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
        """Frozen dataclasses should be usable as dict keys."""
        d: dict[CDTConfig, str] = {CDTConfig(): "default"}
        assert d[CDTConfig()] == "default"


# ---------------------------------------------------------------------------
# CDTNode — construction without LLM (unit testable paths)
# ---------------------------------------------------------------------------


class TestCDTNodeLeaf:
    """Test CDTNode paths that don't require LLM/embedding models."""

    def test_built_statements(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        assert node.statements == ["Alice is kind"]
        assert node.gates == []
        assert node.children == []
        assert node.depth == 1

    def test_built_statements_multiple(self) -> None:
        stmts = ["stmt1", "stmt2", "stmt3"]
        node = CDTNode("Bob", "personality", None, built_statements=stmts)
        assert node.statements == stmts
        # Verify it's a copy, not the same reference
        assert node.statements is not stmts

    def test_built_statements_with_pairs_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            CDTNode("Alice", "identity", [{"action": "a", "scene": "s"}], built_statements=["s"])

    def test_empty_pairs_no_build(self) -> None:
        node = CDTNode("Alice", "identity", [])
        assert node.statements == []
        assert node.gates == []

    def test_few_pairs_no_build(self) -> None:
        """Fewer than MIN_PAIRS_FOR_TREE pairs → no tree construction."""
        pairs = [{"action": f"action_{i}", "scene": f"scene_{i}"} for i in range(MIN_PAIRS_FOR_TREE)]
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


# ---------------------------------------------------------------------------
# CDTNode — traversal and verbalization
# ---------------------------------------------------------------------------


class TestCDTNodeTraversal:
    def _make_tree(self) -> CDTNode:
        """Build a hand-crafted tree for testing (no LLM needed)."""
        root = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        child1 = CDTNode("Alice", "identity", None, built_statements=["Alice helps others"], depth=2)
        child2 = CDTNode("Alice", "identity", None, built_statements=["Alice studies hard"], depth=2)
        root.gates = ["Is Alice helping someone?", "Is Alice at school?"]
        root.children = [child1, child2]
        return root

    def test_verbalize_leaf(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["Alice is kind"])
        text = node.verbalize()
        assert "- Alice is kind" in text

    def test_verbalize_tree(self) -> None:
        root = self._make_tree()
        text = root.verbalize()
        assert "- Alice is kind" in text
        assert 'IF "Is Alice helping someone?"' in text
        assert "  - Alice helps others" in text

    def test_verbalize_empty(self) -> None:
        node = CDTNode("Alice", "identity", None)
        text = node.verbalize()
        assert text == "(empty)"

    def test_count_stats_leaf(self) -> None:
        node = CDTNode("Alice", "identity", None, built_statements=["s1", "s2"])
        stats = node.count_stats()
        assert stats["statements"] == 2
        assert stats["total_statements"] == 2
        assert stats["total_nodes"] == 1
        assert stats["gates"] == 0
        assert stats["total_gates"] == 0

    def test_count_stats_tree(self) -> None:
        root = self._make_tree()
        stats = root.count_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_statements"] == 3  # 1 root + 1 + 1
        assert stats["total_gates"] == 2
        assert stats["max_depth"] == 2


# ---------------------------------------------------------------------------
# build_character_cdts (without models — test structure only)
# ---------------------------------------------------------------------------


class TestBuildCharacterCdts:
    def test_attribute_topics_constant(self) -> None:
        assert len(ATTRIBUTE_TOPICS) == 4
        assert "identity" in ATTRIBUTE_TOPICS
        assert "personality" in ATTRIBUTE_TOPICS

    def test_min_relation_pairs_constant(self) -> None:
        assert MIN_RELATION_PAIRS == 16

    def test_build_with_no_pairs_produces_empty_trees(self) -> None:
        """With empty pairs, all CDTs should be empty (no LLM calls)."""
        topic2cdt, rel_topic2cdt = build_character_cdts(
            "Alice", [], ["Bob", "Charlie"],
        )
        assert len(topic2cdt) == 4
        for cdt in topic2cdt.values():
            assert cdt.statements == []
        assert len(rel_topic2cdt) == 0  # No pairs → no relationships

    def test_build_with_few_pairs_produces_empty_trees(self) -> None:
        pairs = [{"action": f"a{i}", "scene": f"s{i}", "last_character": ["Bob"]} for i in range(5)]
        topic2cdt, rel_topic2cdt = build_character_cdts("Alice", pairs, ["Bob"])
        assert len(topic2cdt) == 4
        for cdt in topic2cdt.values():
            assert cdt.statements == []
        assert len(rel_topic2cdt) == 0  # Only 5 pairs < MIN_RELATION_PAIRS

    def test_topic_names(self) -> None:
        topic2cdt, _ = build_character_cdts("Kasumi", [], [])
        expected = {f"Kasumi's {attr}" for attr in ATTRIBUTE_TOPICS}
        assert set(topic2cdt.keys()) == expected
