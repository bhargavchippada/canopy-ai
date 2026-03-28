"""Convert CDT trees to human-readable markdown profiles.

Usage::

    from canopy.wikify import wikify_tree, wikify_profile

    # Single tree
    markdown = wikify_tree(cdt_node, title="Kasumi's identity")

    # Full profile
    markdown = wikify_profile(topic2cdt, rel_topic2cdt, character="Kasumi")
"""

from __future__ import annotations

from canopy.core import CDTNode


def wikify_node(node: CDTNode, depth: int = 0) -> str:
    """Convert a single CDTNode to markdown with nested gates.

    Args:
        node: The CDT node to wikify.
        depth: Current nesting depth (for heading levels).

    Returns:
        Markdown string representing the node's statements and gates.
    """
    lines: list[str] = []

    for stmt in node.statements:
        lines.append(f"- {stmt}")

    for gate, child in zip(node.gates, node.children):
        lines.append("")
        indent = "  " * (depth + 1)
        lines.append(f"{indent}**When** _{gate}_")
        child_md = wikify_node(child, depth + 1)
        if child_md:
            for line in child_md.split("\n"):
                lines.append(f"{indent}{line}" if line.strip() else "")

    return "\n".join(lines)


def wikify_tree(node: CDTNode, *, title: str | None = None) -> str:
    """Convert a CDTNode tree to a markdown section.

    Args:
        node: Root of the CDT tree.
        title: Optional section title (e.g. "Kasumi's identity").

    Returns:
        Markdown string with the full tree structure.
    """
    parts: list[str] = []

    if title:
        parts.append(f"## {title}")
        parts.append("")

    stats = node.count_stats()
    parts.append(f"*{stats['total_statements']} statements, "
                 f"{stats['total_nodes']} nodes, "
                 f"max depth {stats['max_depth']}*")
    parts.append("")

    body = wikify_node(node)
    if body.strip():
        parts.append(body)
    else:
        parts.append("*(no statements)*")

    return "\n".join(parts)


def wikify_profile(
    topic2cdt: dict[str, CDTNode],
    rel_topic2cdt: dict[str, CDTNode] | None = None,
    *,
    character: str,
) -> str:
    """Convert a full character profile to a markdown document.

    Args:
        topic2cdt: Attribute CDT trees (identity, personality, etc.).
        rel_topic2cdt: Relationship CDT trees (optional).
        character: Character name for the document title.

    Returns:
        Complete markdown profile document.
    """
    parts: list[str] = []
    parts.append(f"# {character} — Behavioral Profile")
    parts.append("")

    # Summary
    total_stmts = sum(n.count_stats()["total_statements"] for n in topic2cdt.values())
    total_nodes = sum(n.count_stats()["total_nodes"] for n in topic2cdt.values())
    if rel_topic2cdt:
        total_stmts += sum(n.count_stats()["total_statements"] for n in rel_topic2cdt.values())
        total_nodes += sum(n.count_stats()["total_nodes"] for n in rel_topic2cdt.values())

    parts.append(f"**{total_stmts} statements** across **{total_nodes} nodes** "
                 f"in {len(topic2cdt)} attributes"
                 + (f" and {len(rel_topic2cdt)} relationships" if rel_topic2cdt else ""))
    parts.append("")

    # Attribute topics
    for topic, cdt in topic2cdt.items():
        parts.append(wikify_tree(cdt, title=topic))
        parts.append("")

    # Relationship topics
    if rel_topic2cdt:
        parts.append("---")
        parts.append("")
        for topic, cdt in rel_topic2cdt.items():
            parts.append(wikify_tree(cdt, title=topic))
            parts.append("")

    return "\n".join(parts)
