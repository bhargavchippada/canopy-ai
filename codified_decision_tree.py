"""CDT construction CLI — thin wrapper over canopy package.

Usage:
    uv run python codified_decision_tree.py --character Kasumi --engine claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import datetime
import pickle

import torch

from canopy.core import CDTConfig, build_character_cdts
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.embeddings import init_models as init_embedding_models
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.validation import init_models as init_validation_models

MODEL_SHORT_NAMES: dict[str, str] = {
    "claude-haiku-4-5": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "gpt-4.1": "gpt41",
}

EMBED_SHORT_NAMES: dict[str, str] = {
    "Qwen/Qwen3-Embedding-0.6B": "qwen06b",
    "Qwen/Qwen3-Embedding-8B": "qwen8b",
    "Qwen/Qwen3-0.6B": "qwen06b",
    "Qwen/Qwen3-8B": "qwen8b",
}

NLI_SHORT_NAMES: dict[str, str] = {
    "KomeijiForce/deberta-v3-base-rp-nli": "deberta",
    "deberta-v3-base-rp-nli": "deberta",
}

CLUSTER_SHORT_NAMES: dict[str, str] = {
    "kmeans": "kmeans",
    "hdbscan": "hdbscan",
}


def _short_name(name: str, table: dict[str, str]) -> str:
    """Resolve a model path/ID to a short filename-safe name."""
    # Check full path first, then basename
    if name in table:
        return table[name]
    import os

    base = os.path.basename(os.path.expanduser(name))
    if base in table:
        return table[base]
    # Also check with parent (e.g. "Qwen/Qwen3-0.6B")
    for key, val in table.items():
        if base == os.path.basename(key):
            return val
    return base.lower().replace(".", "").replace("-", "")


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
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "hdbscan"])
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
    other_characters = [c for c in all_characters[artifact]["major"] if c != args.character]
    pairs = load_ar_pairs(args.character, character2artifact, band2members)["train"]

    config = CDTConfig(
        max_depth=args.max_depth,
        threshold_accept=args.threshold_accept,
        threshold_reject=args.threshold_reject,
        threshold_filter=args.threshold_filter,
    )

    # Build CDTs
    topic2cdt, rel_topic2cdt = build_character_cdts(args.character, pairs, other_characters, config)

    # Compute stats
    total_nodes = 0
    total_stmts = 0
    for cdt in [*topic2cdt.values(), *rel_topic2cdt.values()]:
        stats = cdt.count_stats()
        total_nodes += stats["total_nodes"]
        total_stmts += stats["total_statements"]

    # Build filename: Character.gen.embed.nli.cluster.dN.aXX.rYY.relation.pkl
    gen_short = _short_name(args.engine, MODEL_SHORT_NAMES)
    embed_short = _short_name(args.surface_embedder_path, EMBED_SHORT_NAMES)
    nli_short = _short_name(args.discriminator_path, NLI_SHORT_NAMES)
    cluster_short = args.cluster_method
    a_val = int(args.threshold_accept * 100)
    r_val = int(args.threshold_reject * 100)
    rel_suffix = ".relation" if rel_topic2cdt else ""

    output_path = (
        f"packages/{args.character}.{gen_short}.{embed_short}.{nli_short}"
        f".{cluster_short}.d{args.max_depth}.a{a_val}.r{r_val}{rel_suffix}.pkl"
    )

    metadata = {
        "character": args.character,
        "gen_model": args.engine,
        "embed_model": args.surface_embedder_path,
        "nli_model": args.discriminator_path,
        "cluster_method": args.cluster_method,
        "max_depth": args.max_depth,
        "threshold_accept": args.threshold_accept,
        "threshold_reject": args.threshold_reject,
        "threshold_filter": args.threshold_filter,
        "hypotheses_per_cluster": 3,  # k param in make_hypothesis_prompt
        "n_training_pairs": len(pairs),
        "temperature": 0.0,  # deterministic via claude-agent-sdk
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_nodes": total_nodes,
        "total_statements": total_stmts,
        "has_relationships": len(rel_topic2cdt) > 0,
        "attribute_topics": len(topic2cdt),
        "relationship_topics": len(rel_topic2cdt),
    }

    with open(output_path, "wb") as f:
        pickle.dump(
            {"topic2cdt": topic2cdt, "rel_topic2cdt": rel_topic2cdt, "metadata": metadata},
            f,
        )

    print(f"\nSaved to {output_path}")
    print(f"  Attribute topics: {len(topic2cdt)}")
    print(f"  Relationship topics: {len(rel_topic2cdt)}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total statements: {total_stmts}")


if __name__ == "__main__":
    main()
