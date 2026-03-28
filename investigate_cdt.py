"""CDT quality investigation — single topic, fast iteration.

Runs one topic of the CDT pipeline and prints detailed stats at each step:
clustering, hypothesis generation, NLI validation, tree shape.

Usage:
    uv run python investigate_cdt.py --character Kasumi --topic identity --engine claude-sonnet-4-6
    uv run python investigate_cdt.py --character Kasumi --topic identity --engine claude-haiku-4-5 --theta_accept 0.8
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time

import numpy as np
import torch

from canopy.core import CDTConfig, CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.embeddings import EmbeddingCache, precompute_embeddings
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CDT quality investigation — single topic")
    parser.add_argument("--character", default="Kasumi")
    parser.add_argument("--topic", default="identity", choices=["identity", "personality", "ability", "relationship"])
    parser.add_argument("--engine", default="claude-sonnet-4-6")
    from pathlib import Path
    home = str(Path.home())
    parser.add_argument("--surface_embedder_path", default=f"{home}/models/Qwen3-Embedding-8B")
    parser.add_argument("--generator_embedder_path", default=f"{home}/models/Qwen3-8B")
    parser.add_argument("--discriminator_path", default=f"{home}/models/deberta-v3-base-rp-nli")
    parser.add_argument("--theta_accept", type=float, default=0.8)
    parser.add_argument("--theta_reject", type=float, default=0.5)
    parser.add_argument("--theta_filter", type=float, default=0.8)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--compare_pkl", type=str, default=None, help="Compare against an existing CDT .pkl file")
    return parser


def print_tree_stats(node: CDTNode, prefix: str = "") -> None:
    """Print detailed tree statistics."""
    stats = node.count_stats()
    print(f"\n{prefix}Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total statements: {stats['total_statements']}")
    print(f"  Total gates: {stats['total_gates']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Stmts/node: {stats['total_statements'] / max(stats['total_nodes'], 1):.1f}")

    # Print statement details
    print(f"\n{prefix}Statements ({len(node.statements)} root):")
    for i, s in enumerate(node.statements):
        print(f"  [{i+1}] ({len(s)} chars) {s[:120]}{'...' if len(s) > 120 else ''}")

    if node.gates:
        print(f"\n{prefix}Gates ({len(node.gates)}):")
        for i, (gate, child) in enumerate(zip(node.gates, node.children)):
            child_stats = child.count_stats()
            print(f"  [{i+1}] IF: {gate[:100]}{'...' if len(gate) > 100 else ''}")
            print(f"       → {child_stats['total_statements']} stmts, {child_stats['total_nodes']} nodes")


def compare_with_pkl(topic: str, our_node: CDTNode, pkl_path: str) -> None:
    """Compare our CDT with a reference .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    ref_topic2cdt = data.get("topic2cdt", {})
    # Find matching topic
    ref_node = None
    for key, node in ref_topic2cdt.items():
        if topic.lower() in key.lower():
            ref_node = node
            ref_key = key
            break

    if ref_node is None:
        print(f"\nNo matching topic '{topic}' found in {pkl_path}")
        print(f"Available topics: {list(ref_topic2cdt.keys())}")
        return

    ref_stats = ref_node.count_stats()
    our_stats = our_node.count_stats()

    print(f"\n{'='*60}")
    print(f"COMPARISON: {topic}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Ours':>10} {'Reference':>10}")
    print(f"{'-'*45}")
    print(f"{'Nodes':<25} {our_stats['total_nodes']:>10} {ref_stats['total_nodes']:>10}")
    print(f"{'Statements':<25} {our_stats['total_statements']:>10} {ref_stats['total_statements']:>10}")
    print(f"{'Gates':<25} {our_stats['total_gates']:>10} {ref_stats['total_gates']:>10}")
    print(f"{'Max depth':<25} {our_stats['max_depth']:>10} {ref_stats['max_depth']:>10}")
    print(f"{'Stmts/node':<25} {our_stats['total_statements']/max(our_stats['total_nodes'],1):>10.1f} "
          f"{ref_stats['total_statements']/max(ref_stats['total_nodes'],1):>10.1f}")

    print(f"\nReference tree ({ref_key}):")
    print_tree_stats(ref_node, prefix="  REF ")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()
    device = torch.device(f"cuda:{args.device_id}")

    print(f"{'='*60}")
    print(f"CDT Quality Investigation")
    print(f"Character: {args.character}, Topic: {args.topic}")
    print(f"Engine: {args.engine}, θ_accept: {args.theta_accept}")
    print(f"{'='*60}")

    # Configure
    set_adapter(ClaudeCodeAdapter(default_model=args.engine))
    init_validation_models(args.discriminator_path, device)

    # Load data
    all_characters, character2artifact, band2members = load_character_metadata()
    pairs = load_ar_pairs(args.character, character2artifact, band2members)["train"]
    print(f"\nLoaded {len(pairs)} training pairs")

    # Phase A: Embeddings
    t0 = time.time()
    cache = precompute_embeddings(
        character=args.character, pairs=pairs,
        surface_embedder_path=args.surface_embedder_path,
        generator_embedder_path=args.generator_embedder_path,
        device=f"cuda:{args.device_id}",
    )
    t_embed = time.time() - t0
    print(f"Embeddings: surface={cache.surface.shape}, generator={cache.generator.shape} ({t_embed:.1f}s)")

    # Stamp indices
    indexed_pairs = [{**pair, "_embed_idx": idx} for idx, pair in enumerate(pairs)]

    # Build config
    config = CDTConfig(
        max_depth=args.max_depth,
        threshold_accept=args.theta_accept,
        threshold_reject=args.theta_reject,
        threshold_filter=args.theta_filter,
    )

    # Build single topic
    goal_topic = f"{args.character}'s {args.topic}"
    print(f"\nBuilding CDT: {goal_topic}")
    print(f"Config: depth={args.max_depth}, θ_accept={args.theta_accept}, θ_reject={args.theta_reject}")

    t0 = time.time()
    node = CDTNode(
        args.character,
        goal_topic,
        indexed_pairs,
        config=config,
        _embedding_cache=cache,
    )
    t_build = time.time() - t0

    print(f"\nBuild time: {t_build:.1f}s")
    print_tree_stats(node)

    # Full tree verbalization
    print(f"\n{'='*60}")
    print("FULL TREE:")
    print(f"{'='*60}")
    print(node.verbalize())

    # Compare if reference provided
    if args.compare_pkl:
        compare_with_pkl(args.topic, node, args.compare_pkl)

    print(f"\n{'='*60}")
    print(f"DONE — {goal_topic}: {node.count_stats()['total_nodes']} nodes, "
          f"{node.count_stats()['total_statements']} stmts ({t_embed + t_build:.0f}s total)")


if __name__ == "__main__":
    main()
