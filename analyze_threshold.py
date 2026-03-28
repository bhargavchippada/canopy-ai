#!/usr/bin/env python3
"""Analyze how θ_accept affects tree shape for Sonnet vs GPT-4.1 CDTs.

Quick analysis — not production code.
"""

import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from canopy.data import load_character_metadata, load_ar_pairs
from canopy.validation import validate_hypothesis, init_models


def load_cdt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_tree(node, depth=0):
    stats = {
        "root_stmts": len(node.statements) if depth == 0 else 0,
        "total_stmts": len(node.statements),
        "gates": len(node.gates),
        "max_depth": depth,
        "word_counts": [len(s.split()) for s in node.statements],
    }
    for child in node.children:
        cs = count_tree(child, depth + 1)
        stats["total_stmts"] += cs["total_stmts"]
        stats["gates"] += cs["gates"]
        stats["max_depth"] = max(stats["max_depth"], cs["max_depth"])
        stats["word_counts"].extend(cs["word_counts"])
    return stats


def main():
    sonnet_path = "packages/Kasumi.sonnet.qwen8b.deberta.kmeans.d3.a75.r50.relation.pkl"
    paper_path = "packages/Kasumi.cdt.v3.1.package.relation.pkl"

    print("=" * 70)
    print("CDT STRUCTURE COMPARISON: Sonnet vs GPT-4.1 (Paper)")
    print("=" * 70)

    sonnet = load_cdt(sonnet_path)
    paper = load_cdt(paper_path)

    for label, cdt in [("Sonnet (ours)", sonnet), ("GPT-4.1 (paper)", paper)]:
        print(f"\n{'─' * 50}")
        print(f"  {label}")
        print(f"{'─' * 50}")
        total_root = total_stmts = total_gates = 0
        all_words = []
        for topic, node in cdt["topic2cdt"].items():
            s = count_tree(node)
            total_root += s["root_stmts"]
            total_stmts += s["total_stmts"]
            total_gates += s["gates"]
            all_words.extend(s["word_counts"])
            depth_str = f"depth {s['max_depth']}" if s["max_depth"] > 0 else "FLAT"
            print(f"  {topic:30s}  root={s['root_stmts']:2d}  gates={s['gates']:2d}  stmts={s['total_stmts']:2d}  {depth_str}")
        avg_words = sum(all_words) / len(all_words) if all_words else 0
        print(f"\n  TOTAL: {total_root} root stmts, {total_gates} gates, {total_stmts} total stmts")
        print(f"  Avg statement length: {avg_words:.1f} words")
        print(f"  Avg nodes/topic: {total_stmts / len(cdt['topic2cdt']):.1f}")

    # Load training pairs
    print("\n" + "=" * 70)
    print("NLI VALIDATION: Sonnet root statements at θ=0.75 vs θ=0.80")
    print("=" * 70)

    all_chars, char2art, band2mem = load_character_metadata(
        "all_characters.json", "band2members.json"
    )
    pair_data = load_ar_pairs("Kasumi", char2art, band2mem)
    pairs = pair_data["train"] + pair_data["test"]
    print(f"\nLoaded {len(pairs)} pairs for Kasumi")

    # Init DeBERTa
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    nli_path = Path.home() / "models" / "deberta-v3-base-rp-nli"
    init_models(nli_path, device)

    # Validate each Sonnet root statement
    results = []
    for topic, node in sonnet["topic2cdt"].items():
        for stmt in node.statements:
            res, _ = validate_hypothesis("Kasumi", pairs, None, stmt)
            true_c = res.get("True", 0)
            false_c = res.get("False", 0)
            acc = true_c / (true_c + false_c + 1e-8) + 1e-8
            results.append((topic, stmt, acc, true_c, false_c))

    # Analyze at both thresholds
    pass_075 = [r for r in results if r[2] >= 0.75]
    pass_080 = [r for r in results if r[2] >= 0.80]
    borderline = [r for r in results if r[2] >= 0.75 and r[2] < 0.80]
    fail_both = [r for r in results if r[2] < 0.75]

    print(f"\n{'─' * 50}")
    print(f"  Total root statements: {len(results)}")
    print(f"  Pass at θ=0.75: {len(pass_075)}/{len(results)}")
    print(f"  Pass at θ=0.80: {len(pass_080)}/{len(results)}")
    print(f"  BORDERLINE (0.75 ≤ acc < 0.80): {len(borderline)}")
    print(f"  Fail both: {len(fail_both)}")
    print(f"{'─' * 50}")

    # Print all results sorted by accuracy
    print("\n  ALL STATEMENTS (sorted by accuracy):")
    for topic, stmt, acc, t, f_ in sorted(results, key=lambda x: x[2]):
        words = len(stmt.split())
        flag = ""
        if acc < 0.75:
            flag = " ← REJECTED at both"
        elif acc < 0.80:
            flag = " ← BORDERLINE (pass 0.75, fail 0.80)"
        print(f"  acc={acc:.3f} T={t:5.1f} F={f_:5.1f} [{words:2d}w] [{topic}]{flag}")
        print(f"    {stmt[:100]}{'...' if len(stmt) > 100 else ''}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if len(borderline) < 5:
        print(f"\n  Only {len(borderline)} statements are borderline (0.75-0.80).")
        print("  Raising θ from 0.75 → 0.80 would NOT significantly deepen the tree.")
        print("\n  ROOT CAUSE: Hypothesis generation quality, not validation threshold.")
        print("  Sonnet generates long (~25 word), vague statements that NLI can't falsify.")
        print("  GPT-4.1 generates short (~15 word), specific statements with clear conditions.")
        print("\n  FIX OPTIONS:")
        print("  1. Prompt engineering: force shorter, falsifiable hypotheses with gate conditions")
        print("  2. Post-generation filtering: reject statements > 20 words or without conditions")
        print("  3. Two-stage: generate with Sonnet, then gate-split with a second prompt")
    else:
        print(f"\n  {len(borderline)} statements are borderline.")
        print("  Raising θ to 0.80 WOULD push these to gated subtrees.")


if __name__ == "__main__":
    main()
