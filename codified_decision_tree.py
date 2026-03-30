"""CDT construction CLI — thin wrapper over canopy package.

Usage:
    uv run python codified_decision_tree.py --character Kasumi --engine claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pickle

import torch

from canopy.core import CDTConfig, build_character_cdts
from canopy.data import load_ar_pairs, load_character_metadata
from canopy.embeddings import precompute_embeddings
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

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


def _short_name(name: str, table: dict[str, str]) -> str:
    """Resolve a model path/ID to a short filename-safe name."""
    # Check full path first, then basename
    if name in table:
        return table[name]
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
    parser.add_argument("--time_decay", action="store_true", default=False, help="Enable T-CDT temporal weighting")
    parser.add_argument(
        "--half_life_days", type=int, default=5,
        help="T-CDT half-life in days (default: 5 for chapter-based data)",
    )
    parser.add_argument(
        "--discover_topics", action="store_true", default=False,
        help="Discover additional topics from data (beyond the 4 standard ones)",
    )
    parser.add_argument(
        "--n_extra_topics", type=int, default=4,
        help="Number of extra topics to discover (default: 4)",
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()
    device = torch.device(f"cuda:{args.device_id}")

    # Configure LLM adapter
    set_adapter(ClaudeCodeAdapter(default_model=args.engine, system_prompt=None))

    # Load validation model (DeBERTa ~715MB — stays in main process)
    log.info("Loading validation model...")
    init_validation_models(args.discriminator_path, device)

    # Load data
    all_characters, character2artifact, band2members = load_character_metadata()
    artifact = character2artifact[args.character]
    other_characters = [c for c in all_characters[artifact]["major"] if c != args.character]
    pairs = load_ar_pairs(args.character, character2artifact, band2members)["train"]

    if args.cluster_method != "kmeans":
        log.warning(
            "--cluster_method=%s specified but only kmeans is implemented. "
            "Clustering will use kmeans. Filename will reflect '%s'.",
            args.cluster_method, args.cluster_method,
        )

    # Phase A: Pre-compute embeddings (subprocess isolation for VRAM safety)
    # Each model loads once in a subprocess, encodes ALL pairs, exits.
    # OS reclaims VRAM on subprocess exit — no PyTorch leak issues.
    log.info("Phase A: Pre-computing embeddings via subprocess isolation...")
    cache = precompute_embeddings(
        character=args.character,
        pairs=pairs,
        surface_embedder_path=args.surface_embedder_path,
        generator_embedder_path=args.generator_embedder_path,
        device=f"cuda:{args.device_id}",
    )
    log.info("Phase A complete: surface=%s, generator=%s", cache.surface.shape, cache.generator.shape)

    # Stamp _embed_idx on copies (never mutate caller's dicts)
    if any("_embed_idx" in p for p in pairs):
        log.warning("pairs already contain '_embed_idx' — overwriting with sequential indices")
    indexed_pairs = [{**pair, "_embed_idx": idx} for idx, pair in enumerate(pairs)]

    config = CDTConfig(
        max_depth=args.max_depth,
        threshold_accept=args.threshold_accept,
        threshold_reject=args.threshold_reject,
        threshold_filter=args.threshold_filter,
        time_decay_enabled=args.time_decay,
        time_decay_half_life_days=args.half_life_days,
    )
    if args.time_decay:
        log.info("T-CDT enabled: half_life=%d days", args.half_life_days)

    # Phase B: Build CDTs (no GPU model loading, max_parallel=4)
    # Uses pre-computed embeddings for clustering. Only LLM API + DeBERTa.
    log.info("Phase B: Building CDTs with pre-computed embeddings...")
    topic2cdt, rel_topic2cdt = build_character_cdts(
        args.character, indexed_pairs, other_characters, config,
        max_parallel=4, embedding_cache=cache,
        discover_extra_topics=args.discover_topics,
        n_extra_topics=args.n_extra_topics,
    )

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
    tcdt_suffix = f".tcdt{args.half_life_days}d" if args.time_decay else ""
    disc_suffix = f".disc{args.n_extra_topics}" if args.discover_topics else ""

    output_path = (
        f"packages/{args.character}.{gen_short}.{embed_short}.{nli_short}"
        f".{cluster_short}.d{args.max_depth}.a{a_val}.r{r_val}{tcdt_suffix}{disc_suffix}{rel_suffix}.pkl"
    )

    metadata = {
        "character": args.character,
        "gen_model": args.engine,
        "surface_embed_model": args.surface_embedder_path,
        "generator_embed_model": args.generator_embedder_path,
        "nli_model": args.discriminator_path,
        "cluster_method": args.cluster_method,
        "max_depth": args.max_depth,
        "threshold_accept": args.threshold_accept,
        "threshold_reject": args.threshold_reject,
        "threshold_filter": args.threshold_filter,
        "hypotheses_per_cluster": 3,  # k param in make_hypothesis_prompt
        "n_training_pairs": len(pairs),
        "time_decay_enabled": args.time_decay,
        "time_decay_half_life_days": args.half_life_days if args.time_decay else None,
        "temperature": 0.0,  # deterministic via claude-agent-sdk
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_nodes": total_nodes,
        "total_statements": total_stmts,
        "has_relationships": len(rel_topic2cdt) > 0,
        "attribute_topics": len(topic2cdt),
        "relationship_topics": len(rel_topic2cdt),
    }

    os.makedirs("packages", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(
            {"topic2cdt": topic2cdt, "rel_topic2cdt": rel_topic2cdt, "metadata": metadata},
            f,
        )

    log.info("Saved to %s", output_path)
    log.info("  Attribute topics: %d", len(topic2cdt))
    log.info("  Relationship topics: %d", len(rel_topic2cdt))
    log.info("  Total nodes: %d", total_nodes)
    log.info("  Total statements: %d", total_stmts)


if __name__ == "__main__":
    main()
