"""CDT construction CLI — thin wrapper over canopy package.

Usage:
    uv run python codified_decision_tree.py --character Kasumi --engine claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import pickle

import torch

from canopy.core import CDTConfig, CDTNode
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.embeddings import init_models as init_embedding_models
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.validation import init_models as init_validation_models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CDT for a character")

    parser.add_argument("--character", type=str, default="Kasumi")
    parser.add_argument("--engine", type=str, default="claude-haiku-4-5")
    parser.add_argument("--discriminator_path", type=str, default="KomeijiForce/deberta-v3-base-rp-nli")
    parser.add_argument("--surface_embedder_path", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--generator_embedder_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--threshold_accept", type=float, default=0.8)
    parser.add_argument("--threshold_reject", type=float, default=0.5)
    parser.add_argument("--threshold_filter", type=float, default=0.8)
    parser.add_argument("--device_id", type=int, default=0)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device(f"cuda:{args.device_id}")

    # Configure LLM adapter
    set_adapter(ClaudeCodeAdapter(default_model=args.engine))

    # Load models
    print("Loading embedding models...")
    init_embedding_models(args.surface_embedder_path, args.generator_embedder_path, device)

    print("Loading validation model...")
    init_validation_models(args.discriminator_path, device)

    # Load data
    all_characters, character2artifact, band2members = load_character_metadata()
    artifact = character2artifact[args.character]
    artifact_characters = all_characters[artifact]["major"]
    other_characters = [c for c in artifact_characters if c != args.character]

    pairs = load_ar_pairs(args.character, character2artifact, band2members)["train"]

    config = CDTConfig(
        max_depth=args.max_depth,
        threshold_accept=args.threshold_accept,
        threshold_reject=args.threshold_reject,
        threshold_filter=args.threshold_filter,
    )

    # Build attribute CDTs
    topic2cdt: dict[str, CDTNode] = {}
    for attribute in ["identity", "personality", "ability", "relationship"]:
        goal_topic = f"{args.character}'s {attribute}"
        print(f"\n=== Building CDT: {goal_topic} ===")
        topic2cdt[goal_topic] = CDTNode(args.character, goal_topic, pairs, config=config)

    # Build relationship CDTs
    rel_topic2cdt: dict[str, CDTNode] = {}
    for other_character in other_characters:
        goal_topic = f"{args.character}'s interaction with {other_character}"
        relation_pairs = [d for d in pairs if other_character in d["last_character"]]

        if len(relation_pairs) >= 16:
            print(f"\n=== Building CDT: {goal_topic} ({len(relation_pairs)} pairs) ===")
            rel_topic2cdt[goal_topic] = CDTNode(
                args.character, goal_topic, relation_pairs, config=config,
            )

    # Save
    output_path = f"packages/{args.character}.cdt.v3.1.package.relation.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({"topic2cdt": topic2cdt, "rel_topic2cdt": rel_topic2cdt}, f)

    print(f"\nSaved to {output_path}")

    # Print summary
    total_stmts = sum(len(cdt.statements) for cdt in topic2cdt.values())
    total_stmts += sum(len(cdt.statements) for cdt in rel_topic2cdt.values())
    print(f"  Attribute topics: {len(topic2cdt)}")
    print(f"  Relationship topics: {len(rel_topic2cdt)}")
    print(f"  Root statements: {total_stmts}")


if __name__ == "__main__":
    main()
