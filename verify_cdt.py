"""Verify a CDT package: print structure, stats, and test traversal."""

from __future__ import annotations

import pickle
import sys
import types
from copy import deepcopy


class _LegacyCDTNode:
    """Minimal CDT_Node for unpickling old packages (pickled as __main__.CDT_Node)."""

    statements: list
    gates: list
    children: list
    depth: int

    def traverse(self, scene: str) -> list[str]:
        statements = deepcopy(self.statements)
        for gate, child in zip(self.gates, self.children):
            statements.extend(child.traverse(scene))
        return statements

    def verbalize(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines: list[str] = []
        for s in self.statements:
            lines.append(f"{prefix}- {s}")
        for gate, child in zip(self.gates, self.children):
            lines.append(f'{prefix}IF "{gate}":')
            lines.append(child.verbalize(indent + 1))
        return "\n".join(lines) if lines else f"{prefix}(empty)"

    def count_stats(self) -> dict:
        stats = {
            "statements": len(self.statements),
            "gates": len(self.gates),
            "max_depth": getattr(self, "depth", 1),
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


def _register_unpickle_classes() -> None:
    """Register CDTNode classes so pickle can find them regardless of how they were saved."""
    # Old packages saved as __main__.CDT_Node
    mod_main = types.ModuleType("__main__")
    mod_main.CDT_Node = _LegacyCDTNode
    sys.modules["__main__"] = mod_main

    # Old packages saved as codified_decision_tree.CDT_Node
    mod_cdt = types.ModuleType("codified_decision_tree")
    mod_cdt.CDT_Node = _LegacyCDTNode
    sys.modules["codified_decision_tree"] = mod_cdt

    # New packages saved as canopy.core.CDTNode
    try:
        from canopy.core import CDTNode
        # Monkey-patch count_stats and verbalize onto CDTNode if missing
        if not hasattr(CDTNode, "count_stats"):
            CDTNode.count_stats = _LegacyCDTNode.count_stats
        if not hasattr(CDTNode, "verbalize"):
            pass  # CDTNode already has verbalize
    except ImportError:
        pass


def load_package(path: str) -> dict:
    _register_unpickle_classes()
    with open(path, "rb") as f:
        return pickle.load(f)


def verify_package(path: str) -> dict:
    data = load_package(path)
    topic2cdt = data["topic2cdt"]
    rel_topic2cdt = data["rel_topic2cdt"]

    results: dict = {"topics": {}, "relationships": {}, "totals": {}}
    grand_statements = 0
    grand_gates = 0
    grand_nodes = 0

    print("=" * 80)
    print(f"CDT Package: {path}")
    print("=" * 80)

    print("\n## Attribute Topics\n")
    for topic, cdt in topic2cdt.items():
        stats = cdt.count_stats()
        results["topics"][topic] = stats
        grand_statements += stats["total_statements"]
        grand_gates += stats["total_gates"]
        grand_nodes += stats["total_nodes"]

        print(f"### {topic}")
        print(f"  Root statements: {stats['statements']}")
        print(f"  Root gates: {stats['gates']}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total statements: {stats['total_statements']}")
        print(f"  Total gates: {stats['total_gates']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"\n  Tree structure:")
        print(cdt.verbalize(indent=2))
        print()

    print("\n## Relationship Topics\n")
    for topic, cdt in rel_topic2cdt.items():
        stats = cdt.count_stats()
        results["relationships"][topic] = stats
        grand_statements += stats["total_statements"]
        grand_gates += stats["total_gates"]
        grand_nodes += stats["total_nodes"]

        print(f"### {topic}")
        print(f"  Root statements: {stats['statements']}")
        print(f"  Root gates: {stats['gates']}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total statements: {stats['total_statements']}")
        print(f"  Total gates: {stats['total_gates']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"\n  Tree structure:")
        print(cdt.verbalize(indent=2))
        print()

    results["totals"] = {
        "attribute_topics": len(topic2cdt),
        "relationship_topics": len(rel_topic2cdt),
        "total_statements": grand_statements,
        "total_gates": grand_gates,
        "total_nodes": grand_nodes,
    }

    print("=" * 80)
    print("## Grand Totals")
    print(f"  Attribute topics: {len(topic2cdt)}")
    print(f"  Relationship topics: {len(rel_topic2cdt)}")
    print(f"  Total statements: {grand_statements}")
    print(f"  Total gates: {grand_gates}")
    print(f"  Total nodes: {grand_nodes}")
    print("=" * 80)

    # Traversal test
    print("\n## Traversal Test\n")
    test_scene = (
        "Kasumi looked around at the empty stage.\n"
        "Arisa sighed and crossed her arms.\n"
        '"We don\'t have enough members for a real band," Arisa muttered.\n'
        "Kasumi clenched her fists with determination."
    )
    print(f"Test scene:\n{test_scene}\n")
    print("Collected statements (all branches, no gate filtering):")
    for topic, cdt in topic2cdt.items():
        stmts = cdt.traverse(test_scene)
        if stmts:
            print(f"\n  [{topic}] ({len(stmts)} statements)")
            for s in stmts[:5]:
                print(f"    - {s}")
            if len(stmts) > 5:
                print(f"    ... and {len(stmts) - 5} more")

    for topic, cdt in rel_topic2cdt.items():
        stmts = cdt.traverse(test_scene)
        if stmts:
            print(f"\n  [{topic}] ({len(stmts)} statements)")
            for s in stmts[:3]:
                print(f"    - {s}")

    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "packages/Kasumi.cdt.v3.1.package.relation.pkl"
    verify_package(path)
