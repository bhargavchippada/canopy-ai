#!/usr/bin/env python3
"""Variance study runner — builds 3 CDTs with identical config, benchmarks each.

Handles two bugs in cdt_steps.py:
1. validate step: numpy float32 not JSON serializable
2. build_tree step (standalone): DeBERTa not initialized

Usage:
    uv run python scripts/variance_runner.py --character Kasumi
    uv run python scripts/variance_runner.py --character Kasumi --resume_from validate --build_id 001
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Add scripts dir to path for cdt_steps import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdt_steps import (
    ORDERED_STEPS,
    atomic_write,
    load_json,
    next_build_id,
    require_upstream,
    step_build_tree,
    step_clustering,
    step_compress,
    step_data,
    step_dedup,
    step_embedding,
    step_hypothesis_gen,
    step_wikify,
    update_quality,
)

import canopy.validation as canopy_validation

log = logging.getLogger(__name__)


def patched_step_validate(character: str, cache_path: Path, **kwargs) -> dict:
    """validate step with float32 -> float conversion fix."""
    hyp_path = cache_path / "compress" / "hypotheses.json"
    pairs_path = cache_path / "data" / "pairs.json"
    require_upstream(hyp_path, "validate")
    require_upstream(pairs_path, "validate")

    hyp_data = load_json(hyp_path)
    pairs = load_json(pairs_path)
    device_id = kwargs.get("device_id", 0)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    canopy_validation.init_models(kwargs["discriminator_path"], device)

    results_list = []
    counts = {"accepted": 0, "gated": 0, "rejected": 0}
    correctness_values = []

    for gate, stmt in zip(hyp_data["gates"], hyp_data["statements"]):
        res, filtered = canopy_validation.validate_hypothesis(character, pairs, gate, stmt)
        true_score = float(res.get("True", 0))
        false_score = float(res.get("False", 0))
        denom = true_score + false_score
        correctness = float(true_score / (denom + 1e-8)) if denom > 0 else 0.0
        total_score = float(sum(res.values()))
        broadness = float(1 - float(res.get("Irrelevant", 0)) / (total_score + 1e-8)) if total_score > 0 else 0.0
        correctness_values.append(correctness)
        status = "accepted" if correctness >= 0.8 else ("rejected" if correctness < 0.5 else "gated")
        counts[status] += 1
        results_list.append({
            "gate": gate, "statement": stmt,
            "correctness": round(correctness, 4), "broadness": round(broadness, 4),
            "status": status, "n_filtered": len(filtered),
            "scores": {k: round(float(v), 4) for k, v in res.items()},
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


def patched_step_build_tree(character: str, cache_path: Path, **kwargs) -> dict:
    """build_tree step with DeBERTa pre-initialization."""
    device_id = kwargs.get("device_id", 0)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    canopy_validation.init_models(kwargs["discriminator_path"], device)
    return step_build_tree(character, cache_path, **kwargs)


# Step registry with patches applied
PATCHED_STEPS = [
    ("data", step_data),
    ("embedding", step_embedding),
    ("clustering", step_clustering),
    ("hypothesis_gen", step_hypothesis_gen),
    ("dedup", step_dedup),
    ("compress", step_compress),
    ("validate", patched_step_validate),
    ("build_tree", patched_step_build_tree),
    ("wikify", step_wikify),
]
STEP_NAMES = [name for name, _ in PATCHED_STEPS]


def run_build(character: str, build_id: str, cache_dir: Path, kw: dict,
              resume_from: str | None = None) -> Path:
    """Run a single CDT build, optionally resuming from a step."""
    cache_path = cache_dir / character / build_id
    log.info("=== BUILD %s === Cache: %s", build_id, cache_path)

    start_idx = 0
    if resume_from:
        start_idx = STEP_NAMES.index(resume_from)
        log.info("Resuming from step '%s' (index %d)", resume_from, start_idx)

    for step_name, step_fn in PATCHED_STEPS[start_idx:]:
        log.info("--- Step: %s ---", step_name)
        try:
            metrics = step_fn(character, cache_path, **kw)
            log.info("Quality [%s]: %s", step_name, json.dumps(metrics, indent=2))
        except Exception:
            log.exception("Step '%s' FAILED", step_name)
            raise

    # Combine pickles for benchmark
    tree_dir = cache_path / "build_tree"
    with open(tree_dir / "topic2cdt.pkl", "rb") as f:
        topic2cdt = pickle.load(f)
    with open(tree_dir / "rel_topic2cdt.pkl", "rb") as f:
        rel_topic2cdt = pickle.load(f)
    pkg = {"topic2cdt": topic2cdt, "rel_topic2cdt": rel_topic2cdt}
    pkg_path = tree_dir / "cdt_package.pkl"
    atomic_write(pkg_path, pickle.dumps(pkg), mode="wb")
    log.info("Combined package: %s (%d attr, %d rel topics)",
             pkg_path, len(topic2cdt), len(rel_topic2cdt))

    return cache_path


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="Variance study runner")
    p.add_argument("--character", default="Kasumi")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--build_id", default=None, help="Specific build_id to run/resume")
    p.add_argument("--resume_from", default=None, choices=STEP_NAMES,
                   help="Resume from this step (skip earlier steps)")
    p.add_argument("--n_builds", type=int, default=3)
    p.add_argument("--surface_embedder_path",
                   default=str(Path.home() / "models" / "Qwen3-Embedding-0.6B"))
    p.add_argument("--generator_embedder_path",
                   default=str(Path.home() / "models" / "Qwen3-0.6B"))
    p.add_argument("--discriminator_path",
                   default=str(Path.home() / "models" / "deberta-v3-base-rp-nli"))
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--engine", default="claude-haiku-4-5")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    kw = {
        "surface_embedder_path": args.surface_embedder_path,
        "generator_embedder_path": args.generator_embedder_path,
        "discriminator_path": args.discriminator_path,
        "device_id": args.device_id, "engine": args.engine,
    }

    if args.build_id:
        # Single build (possibly resuming)
        run_build(args.character, args.build_id, cache_dir, kw,
                  resume_from=args.resume_from)
    else:
        # Run N fresh builds
        for i in range(args.n_builds):
            bid = next_build_id(cache_dir, args.character)
            try:
                run_build(args.character, bid, cache_dir, kw)
            except Exception:
                log.exception("Build %s FAILED — continuing to next", bid)
            log.info("=== Build %d/%d complete ===", i + 1, args.n_builds)


if __name__ == "__main__":
    main()
