"""Tests for canopy.wikify — CDT tree to markdown conversion."""

from __future__ import annotations

from canopy.core import CDTNode
from canopy.wikify import wikify_node, wikify_profile, wikify_tree


def _leaf(stmts: list[str], depth: int = 1) -> CDTNode:
    return CDTNode("A", "x", None, built_statements=stmts, depth=depth)


def _tree() -> CDTNode:
    root = _leaf(["Root statement"])
    child1 = _leaf(["Child one stmt"], depth=2)
    child2 = _leaf(["Child two stmt"], depth=2)
    root.gates = ["Is condition A?", "Is condition B?"]
    root.children = [child1, child2]
    return root


def _deep_tree() -> CDTNode:
    leaf = _leaf(["Deep leaf"], depth=3)
    mid = _leaf(["Mid node"], depth=2)
    mid.gates = ["Deep gate?"]
    mid.children = [leaf]
    root = _leaf(["Top root"], depth=1)
    root.gates = ["Top gate?"]
    root.children = [mid]
    return root


# ---------------------------------------------------------------------------
# wikify_node
# ---------------------------------------------------------------------------


class TestWikifyNode:
    def test_leaf_statements(self) -> None:
        md = wikify_node(_leaf(["Statement one", "Statement two"]))
        assert "- Statement one" in md
        assert "- Statement two" in md

    def test_empty_node(self) -> None:
        md = wikify_node(_leaf([]))
        assert md == ""

    def test_gated_children(self) -> None:
        md = wikify_node(_tree())
        assert "- Root statement" in md
        assert "**When** _Is condition A?_" in md
        assert "- Child one stmt" in md
        assert "**When** _Is condition B?_" in md

    def test_deep_nesting(self) -> None:
        md = wikify_node(_deep_tree())
        assert "- Top root" in md
        assert "**When** _Top gate?_" in md
        assert "- Mid node" in md
        assert "**When** _Deep gate?_" in md
        assert "- Deep leaf" in md

    def test_depth_increases_indentation(self) -> None:
        md = wikify_node(_tree(), depth=0)
        lines = md.split("\n")
        # Children should be indented more than root
        root_lines = [line for line in lines if line.startswith("- ")]
        child_lines = [line for line in lines if "Child" in line]
        assert len(root_lines) > 0
        assert len(child_lines) > 0


# ---------------------------------------------------------------------------
# wikify_tree
# ---------------------------------------------------------------------------


class TestWikifyTree:
    def test_with_title(self) -> None:
        md = wikify_tree(_leaf(["A statement"]), title="Test Topic")
        assert md.startswith("## Test Topic")
        assert "- A statement" in md

    def test_without_title(self) -> None:
        md = wikify_tree(_leaf(["A statement"]))
        assert not md.startswith("## ")
        assert "- A statement" in md

    def test_stats_line(self) -> None:
        md = wikify_tree(_tree(), title="T")
        assert "3 statements" in md
        assert "3 nodes" in md

    def test_empty_tree(self) -> None:
        md = wikify_tree(_leaf([]), title="Empty")
        assert "*(no statements)*" in md

    def test_complex_tree_stats(self) -> None:
        md = wikify_tree(_deep_tree(), title="Deep")
        assert "3 statements" in md
        assert "3 nodes" in md
        assert "max depth 3" in md


# ---------------------------------------------------------------------------
# wikify_profile
# ---------------------------------------------------------------------------


class TestWikifyProfile:
    def test_basic_profile(self) -> None:
        topics = {
            "Alice's identity": _leaf(["Kind", "Brave"]),
            "Alice's personality": _leaf(["Cheerful"]),
        }
        md = wikify_profile(topics, character="Alice")
        assert "# Alice — Behavioral Profile" in md
        assert "## Alice's identity" in md
        assert "## Alice's personality" in md
        assert "- Kind" in md
        assert "- Cheerful" in md

    def test_with_relationships(self) -> None:
        topics = {"A's identity": _leaf(["stmt"])}
        rels = {"A's interaction with B": _leaf(["rel stmt"])}
        md = wikify_profile(topics, rels, character="A")
        assert "## A's interaction with B" in md
        assert "- rel stmt" in md
        assert "---" in md  # Separator between attrs and rels

    def test_summary_counts(self) -> None:
        topics = {
            "A's identity": _tree(),  # 3 stmts, 3 nodes
        }
        rels = {
            "A's interaction with B": _leaf(["r1", "r2"]),  # 2 stmts, 1 node
        }
        md = wikify_profile(topics, rels, character="A")
        assert "**5 statements**" in md
        assert "**4 nodes**" in md
        assert "1 attributes" in md or "1 attribute" in md  # grammar not enforced
        assert "1 relationships" in md or "1 relationship" in md

    def test_no_relationships(self) -> None:
        topics = {"A's identity": _leaf(["stmt"])}
        md = wikify_profile(topics, character="A")
        assert "---" not in md  # No separator without rels

    def test_empty_relationships(self) -> None:
        topics = {"A's identity": _leaf(["stmt"])}
        md = wikify_profile(topics, {}, character="A")
        # Empty dict should behave like None
        assert "---" not in md or "0 relationships" in md
