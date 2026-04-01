#!/usr/bin/env python3
"""CLI that runs CDT pipeline steps individually with caching.

Usage:
    uv run python scripts/cdt_steps.py --step data --character Kasumi
    uv run python scripts/cdt_steps.py --step embedding --character Kasumi --build_id 001
    uv run python scripts/cdt_steps.py --step all --character Kasumi
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import canopy.cluster as canopy_cluster
import canopy.core as canopy_core
import canopy.data as canopy_data
import canopy.embeddings as canopy_embeddings
import canopy.llm as canopy_llm
import canopy.prompts as canopy_prompts
import canopy.quality as canopy_quality
import canopy.validation as canopy_validation
import canopy.wikify as canopy_wikify

log = logging.getLogger(__name__)

STEPS = (
    "data", "embedding", "clustering", "hypothesis_gen",
    "dedup", "compress", "validate", "build_tree", "wikify", "all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CDT pipeline steps individually.")
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--character", default="Kasumi")
    p.add_argument("--build_id", default=None, help="Auto-increments if omitted")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--surface_embedder_path",
                   default=str(Path.home() / "models" / "Qwen3-Embedding-0.6B"))
    p.add_argument("--generator_embedder_path",
                   default=str(Path.home() / "models" / "Qwen3-0.6B"))
    p.add_argument("--discriminator_path",
                   default=str(Path.home() / "models" / "deberta-v3-base-rp-nli"))
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--engine", default="claude-haiku-4-5")
    return p.parse_args(argv)


def next_build_id(cache_dir: Path, character: str) -> str:
    char_dir = cache_dir / character
    if not char_dir.exists():
        return "001"
    numeric = [int(x) for x in (d.name for d in char_dir.iterdir() if d.is_dir()) if x.isdigit()]
    return "001" if not numeric else f"{max(numeric) + 1:03d}"


def atomic_write(path: Path, data: bytes | str, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing upstream cache: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_quality(cache_path: Path, step_name: str, metrics: dict) -> None:
    quality_path = cache_path / "quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = quality_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing: dict[str, Any] = {}
            if quality_path.exists():
                with open(quality_path, encoding="utf-8") as f:
                    existing = json.load(f)
            atomic_write(quality_path, json.dumps({**existing, step_name: metrics}, indent=2))
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def require_upstream(path: Path, step_name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Upstream cache missing for '{step_name}': {path}. Run the preceding step first.")


def _init_llm(engine: str) -> None:
    canopy_llm.set_adapter(canopy_llm.ClaudeCodeAdapter(default_model=engine))


def _load_emb_cache(cache_path: Path) -> canopy_embeddings.EmbeddingCache:
    arrs = np.load(cache_path / "embedding" / "cache.npz")
    return canopy_embeddings.EmbeddingCache(surface=arrs["surface"], generator=arrs["generator"])


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_data(character: str, cache_path: Path, **kwargs: Any) -> dict:
    _, character2artifact, band2members = canopy_data.load_character_metadata()
    pairs = canopy_data.load_ar_pairs(character, character2artifact, band2members)["train"]

    serializable = []
    for p in pairs:
        row = {k: v for k, v in p.items() if k != "_timestamp"}
        if "_timestamp" in p:
            row["_timestamp"] = p["_timestamp"].isoformat()
        serializable.append(row)

    atomic_write(cache_path / "data" / "pairs.json", json.dumps(serializable, indent=2))
    log.info("Saved %d training pairs", len(pairs))

    metrics = canopy_quality.compute_data_quality(pairs)
    update_quality(cache_path, "data", metrics)
    return metrics


def step_embedding(character: str, cache_path: Path, **kwargs: Any) -> dict:
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(pairs_path, "embedding")
    pairs = load_json(pairs_path)

    emb_cache = canopy_embeddings.precompute_embeddings(
        character, pairs,
        surface_embedder_path=kwargs["surface_embedder_path"],
        generator_embedder_path=kwargs["generator_embedder_path"],
        device=f"cuda:{kwargs.get('device_id', 0)}",
    )

    out_dir = cache_path / "embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "cache_tmp.npz"
    np.savez(tmp_path, surface=emb_cache.surface, generator=emb_cache.generator)
    os.replace(tmp_path, out_dir / "cache.npz")
    log.info("Saved embeddings: surface=%s, generator=%s", emb_cache.surface.shape, emb_cache.generator.shape)

    metrics = {"n_pairs": len(pairs), "surface_shape": list(emb_cache.surface.shape),
               "generator_shape": list(emb_cache.generator.shape)}
    update_quality(cache_path, "embedding", metrics)
    return metrics


def step_clustering(character: str, cache_path: Path, **kwargs: Any) -> dict:
    require_upstream(cache_path / "embedding" / "cache.npz", "clustering")
    emb_cache = _load_emb_cache(cache_path)

    labels, centroids = canopy_cluster.KMeansCluster().fit_predict(emb_cache.document)

    out_dir = cache_path / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [("labels", labels), ("centroids", centroids)]:
        fd, tmp_path = tempfile.mkstemp(suffix=".npy", dir=out_dir)
        os.close(fd)
        np.save(tmp_path, arr)
        # np.save may append .npy if missing; handle both cases
        saved = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npy"
        os.replace(saved, out_dir / f"{name}.npy")

    clusters_idx: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        clusters_idx.setdefault(int(lbl), []).append(i)
    atomic_write(out_dir / "clusters.json",
                 json.dumps([clusters_idx.get(k, []) for k in sorted(clusters_idx)], indent=2))
    log.info("Clustering: %d clusters from %d samples", len(centroids), len(labels))

    metrics = canopy_quality.compute_clustering_quality(labels, emb_cache.document, centroids)
    update_quality(cache_path, "clustering", metrics)
    return metrics


def step_hypothesis_gen(character: str, cache_path: Path, **kwargs: Any) -> dict:
    pairs_path = cache_path / "data" / "pairs.json"
    centroids_path = cache_path / "clustering" / "centroids.npy"
    for p in [pairs_path, cache_path / "embedding" / "cache.npz", centroids_path]:
        require_upstream(p, "hypothesis_gen")

    pairs = load_json(pairs_path)
    emb_cache = _load_emb_cache(cache_path)
    centroids = np.load(centroids_path)
    clusters = canopy_cluster.select_representative_samples(pairs, emb_cache.document, centroids)

    _init_llm(kwargs.get("engine", "claude-haiku-4-5"))
    statements, gates = canopy_prompts.make_hypotheses_batch(
        clusters, character, f"{character}'s identity", [], [])

    result = {"gates": gates, "statements": statements}
    atomic_write(cache_path / "hypothesis_gen" / "hypotheses.json", json.dumps(result, indent=2))
    log.info("Generated %d hypothesis pairs", len(statements))

    metrics = canopy_quality.compute_hypothesis_quality(statements, pairs)
    update_quality(cache_path, "hypothesis_gen", metrics)
    return metrics


def step_dedup(character: str, cache_path: Path, **kwargs: Any) -> dict:
    hyp_path = cache_path / "hypothesis_gen" / "hypotheses.json"
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(hyp_path, "dedup")
    require_upstream(pairs_path, "dedup")

    hyp_data = load_json(hyp_path)
    pairs = load_json(pairs_path)
    _init_llm(kwargs.get("engine", "claude-haiku-4-5"))

    gates_out, stmts_out = canopy_prompts.merge_similar_hypotheses(
        hyp_data["gates"], hyp_data["statements"])
    atomic_write(cache_path / "dedup" / "hypotheses.json",
                 json.dumps({"gates": gates_out, "statements": stmts_out}, indent=2))

    n_before, n_after = len(hyp_data["statements"]), len(stmts_out)
    reduction_pct = ((n_before - n_after) / n_before * 100) if n_before > 0 else 0.0
    log.info("Dedup: %d -> %d hypotheses (%.1f%% reduction)", n_before, n_after, reduction_pct)

    metrics = canopy_quality.compute_hypothesis_quality(stmts_out, pairs)
    metrics.update({"reduction_pct": round(reduction_pct, 1), "count_before": n_before})
    update_quality(cache_path, "dedup", metrics)
    return metrics


def step_compress(character: str, cache_path: Path, **kwargs: Any) -> dict:
    hyp_path = cache_path / "dedup" / "hypotheses.json"
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(hyp_path, "compress")
    require_upstream(pairs_path, "compress")

    hyp_data = load_json(hyp_path)
    pairs = load_json(pairs_path)
    _init_llm(kwargs.get("engine", "claude-haiku-4-5"))

    gates_out, stmts_out = canopy_prompts.summarize_triggers(
        character, hyp_data["gates"], hyp_data["statements"])
    atomic_write(cache_path / "compress" / "hypotheses.json",
                 json.dumps({"gates": gates_out, "statements": stmts_out}, indent=2))
    log.info("Compressed to %d hypothesis pairs", len(stmts_out))

    metrics = canopy_quality.compute_hypothesis_quality(stmts_out, pairs)
    update_quality(cache_path, "compress", metrics)
    return metrics


def step_validate(character: str, cache_path: Path, **kwargs: Any) -> dict:
    import torch

    hyp_path = cache_path / "compress" / "hypotheses.json"
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(hyp_path, "validate")
    require_upstream(pairs_path, "validate")

    hyp_data = load_json(hyp_path)
    pairs = load_json(pairs_path)
    device_id = kwargs.get("device_id", 0)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    canopy_validation.init_models(kwargs["discriminator_path"], device)

    results_list: list[dict[str, Any]] = []
    counts = {"accepted": 0, "gated": 0, "rejected": 0}
    correctness_values: list[float] = []

    for gate, stmt in zip(hyp_data["gates"], hyp_data["statements"]):
        res, filtered = canopy_validation.validate_hypothesis(character, pairs, gate, stmt)
        true_score = res.get("True", 0)
        false_score = res.get("False", 0)
        denom = true_score + false_score
        correctness = float(true_score / (denom + 1e-8)) if denom > 0 else 0.0
        total_score = sum(res.values())
        broadness = 1 - res.get("Irrelevant", 0) / (total_score + 1e-8) if total_score > 0 else 0.0
        correctness_values.append(correctness)
        status = "accepted" if correctness >= 0.8 else ("rejected" if correctness < 0.5 else "gated")
        counts[status] += 1
        results_list.append({
            "gate": gate, "statement": stmt,
            "correctness": round(correctness, 4), "broadness": round(broadness, 4),
            "status": status, "n_filtered": len(filtered),
            "scores": {k: round(v, 4) for k, v in res.items()},
        })

    atomic_write(cache_path / "validate" / "results.json", json.dumps(results_list, indent=2))
    log.info("Validation: %d accepted, %d gated, %d rejected",
             counts["accepted"], counts["gated"], counts["rejected"])

    metrics = {
        "mean_correctness": round(float(np.mean(correctness_values)) if correctness_values else 0.0, 4),
        "n_accepted": counts["accepted"], "n_gated": counts["gated"],
        "n_rejected": counts["rejected"], "n_total": len(results_list),
    }
    update_quality(cache_path, "validate", metrics)
    return metrics


def step_build_tree(character: str, cache_path: Path, **kwargs: Any) -> dict:
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(pairs_path, "build_tree")
    require_upstream(cache_path / "embedding" / "cache.npz", "build_tree")

    pairs = load_json(pairs_path)
    emb_cache = _load_emb_cache(cache_path)
    indexed_pairs = [{**p, "_embed_idx": i} for i, p in enumerate(pairs)]

    _, _, band2members = canopy_data.load_character_metadata()
    band_members: list[str] = []
    for members in band2members.values():
        if character in members:
            band_members = [m for m in members if m != character]
            break

    _init_llm(kwargs.get("engine", "claude-haiku-4-5"))
    topic2cdt, rel_topic2cdt = canopy_core.build_character_cdts(
        character, indexed_pairs, band_members, embedding_cache=emb_cache)

    out_dir = cache_path / "build_tree"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("topic2cdt.pkl", topic2cdt), ("rel_topic2cdt.pkl", rel_topic2cdt)]:
        atomic_write(out_dir / name, pickle.dumps(obj), mode="wb")
    log.info("Built CDTs: %d attr, %d rel topics", len(topic2cdt), len(rel_topic2cdt))

    metrics = canopy_quality.compute_tree_quality({**topic2cdt, **rel_topic2cdt}, pairs)
    update_quality(cache_path, "build_tree", metrics)
    return metrics


def step_wikify(character: str, cache_path: Path, **kwargs: Any) -> dict:
    tree_dir = cache_path / "build_tree"
    require_upstream(tree_dir / "topic2cdt.pkl", "wikify")

    with open(tree_dir / "topic2cdt.pkl", "rb") as f:
        topic2cdt = pickle.load(f)
    rel_topic2cdt = None
    rel_path = tree_dir / "rel_topic2cdt.pkl"
    if rel_path.exists():
        with open(rel_path, "rb") as f:
            rel_topic2cdt = pickle.load(f)

    md = canopy_wikify.wikify_profile(topic2cdt, rel_topic2cdt, character=character)
    atomic_write(cache_path / "wikify" / "profile.md", md)
    log.info("Wikified profile: %d words", len(md.split()))

    metrics = {"word_count": len(md.split())}
    update_quality(cache_path, "wikify", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ORDERED_STEPS: list[tuple[str, Any]] = [
    ("data", step_data), ("embedding", step_embedding), ("clustering", step_clustering),
    ("hypothesis_gen", step_hypothesis_gen), ("dedup", step_dedup), ("compress", step_compress),
    ("validate", step_validate), ("build_tree", step_build_tree), ("wikify", step_wikify),
]
STEP_MAP = dict(ORDERED_STEPS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    build_id = args.build_id or next_build_id(cache_dir, args.character)
    cache_path = cache_dir / args.character / build_id
    log.info("Cache path: %s", cache_path)

    kw = {
        "surface_embedder_path": args.surface_embedder_path,
        "generator_embedder_path": args.generator_embedder_path,
        "discriminator_path": args.discriminator_path,
        "device_id": args.device_id, "engine": args.engine,
    }
    if args.step == "all":
        for step_name, step_fn in ORDERED_STEPS:
            log.info("=== Step: %s ===", step_name)
            metrics = step_fn(args.character, cache_path, **kw)
            log.info("Quality [%s]: %s", step_name, json.dumps(metrics, indent=2))
    else:
        metrics = STEP_MAP[args.step](args.character, cache_path, **kw)
        log.info("Quality [%s]: %s", args.step, json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
