"""Verify a CDT package: print structure, stats, and test traversal."""

import pickle
import sys
import types
from copy import deepcopy


# Minimal CDT_Node for unpickling (avoids importing the full module)
class CDT_Node:
    statements: list
    gates: list
    children: list
    depth: int

    def traverse(self, scene: str) -> list[str]:
        statements = deepcopy(self.statements)
        for gate, child in zip(self.gates, self.children):
            # Skip gate checking in verification — just collect all
            statements.extend(child.traverse(scene))
        return statements

    def verbalize(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = []
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


# Register the class for pickle
mod = types.ModuleType("__main__")
mod.CDT_Node = CDT_Node
sys.modules["__main__"] = mod

# Also register as codified_decision_tree.CDT_Node
mod2 = types.ModuleType("codified_decision_tree")
mod2.CDT_Node = CDT_Node
sys.modules["codified_decision_tree"] = mod2


def load_package(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def verify_package(path: str) -> dict:
    data = load_package(path)
    topic2cdt = data["topic2cdt"]
    rel_topic2cdt = data["rel_topic2cdt"]

    results = {"topics": {}, "relationships": {}, "totals": {}}
    grand_statements = 0
    grand_gates = 0
    grand_nodes = 0

    print("=" * 80)
    print(f"CDT Package: {path}")
    print("=" * 80)

    # Attribute topics
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
        print()
        print(f"  Tree structure:")
        print(cdt.verbalize(indent=2))
        print()

    # Relationship topics
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
        print()
        print(f"  Tree structure:")
        print(cdt.verbalize(indent=2))
        print()

    # Grand totals
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

    # Test traversal
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
