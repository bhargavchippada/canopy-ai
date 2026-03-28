"""Canopy quickstart — build a CDT from behavioral observations.

Usage:
    uv run python examples/quickstart.py

Requirements:
    - GPU with CUDA (for embedding and NLI models)
    - Claude Max subscription (for hypothesis generation via claude-agent-sdk)
    - Local models: Qwen3-Embedding-0.6B, Qwen3-0.6B, deberta-v3-base-rp-nli
"""

from __future__ import annotations

import torch

from canopy import BehavioralObservation, CDTConfig, build_cdt
from canopy.embeddings import init_models as init_embeddings
from canopy.llm import ClaudeCodeAdapter, set_adapter
from canopy.validation import init_models as init_validation
from canopy.wikify import wikify_tree


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Configure LLM adapter
    set_adapter(ClaudeCodeAdapter(default_model="claude-haiku-4-5"))

    # 2. Load models
    print("Loading models...")
    from pathlib import Path

    models = Path.home() / "models"
    init_embeddings(
        surface_embedder_path=str(models / "Qwen3-Embedding-0.6B"),
        generator_embedder_path=str(models / "Qwen3-0.6B"),
        device=device,
    )
    init_validation(
        discriminator_path=str(models / "deberta-v3-base-rp-nli"),
        device=device,
    )

    # 3. Create observations (normally loaded from data)
    observations = [
        BehavioralObservation(
            scene="The team is stuck on a hard problem. Everyone is quiet.",
            action="Alice breaks the silence with a bold new idea.",
            actor="Alice",
            participants=["Bob", "Carol"],
        ),
        BehavioralObservation(
            scene="Bob presents his work. It has a bug.",
            action="Alice points out the bug directly but offers to help fix it.",
            actor="Alice",
            participants=["Bob"],
        ),
        BehavioralObservation(
            scene="Carol is feeling discouraged after a setback.",
            action="Alice encourages Carol and reframes the setback as a learning opportunity.",
            actor="Alice",
            participants=["Carol"],
        ),
    ]

    # 4. Build a CDT
    print(f"\nBuilding CDT from {len(observations)} observations...")
    tree = build_cdt(
        observations,
        character="Alice",
        topic="leadership",
        config=CDTConfig(max_depth=1),  # Keep shallow for demo
    )

    # 5. View the result
    print("\n" + wikify_tree(tree, title="Alice's leadership"))
    print(f"\nStats: {tree.count_stats()}")


if __name__ == "__main__":
    main()
