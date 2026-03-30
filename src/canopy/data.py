"""Data loading for CDT construction — HuggingFace dataset fetching and pair extraction."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any


def load_character_metadata(
    characters_path: str = "all_characters.json",
    bands_path: str = "band2members.json",
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Load character → artifact mapping and band membership.

    Returns (all_characters, character2artifact, band2members).
    """
    with open(characters_path, encoding="utf-8") as f:
        all_characters = json.load(f)

    character2artifact = {
        character: artifact
        for artifact in all_characters
        for character in all_characters[artifact]["major"]
    }

    with open(bands_path, encoding="utf-8") as f:
        band2members = json.load(f)

    return all_characters, character2artifact, band2members


def load_ar_pairs(
    character: str,
    character2artifact: dict[str, str],
    band2members: dict[str, Any],
    data_dir: str = "data",
    scene_window: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Load action-reaction pairs for a character from HuggingFace datasets.

    Returns {"train": [...], "test": [...]}, split 50/50.
    """
    from datasets import load_dataset

    artifact = character2artifact[character]
    cache_path = os.path.join(data_dir, f"title2action_series.{artifact}.json")

    if not os.path.exists(cache_path):
        title2action_series: dict[str, list] = defaultdict(list)

        if artifact not in band2members:
            hf_path = "KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences"
        else:
            hf_path = "KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences"

        for data in load_dataset(hf_path)["train"]:
            if data["artifact"] == artifact:
                title2action_series[data["title"]].append(data)

        os.makedirs(data_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(title2action_series, f)
    else:
        with open(cache_path, encoding="utf-8") as f:
            title2action_series = json.load(f)

    all_actions: list[str] = []
    pairs: list[dict] = []
    last_character: list[str] = []

    # Assign synthetic timestamps based on title ordering (chapter_1 = epoch, +1 day per title)
    title_order = {title: idx for idx, title in enumerate(title2action_series)}
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for title in title2action_series:
        action_series = title2action_series[title]
        title_idx = title_order[title]
        title_timestamp = epoch + timedelta(days=title_idx)

        for item in action_series:
            all_actions.append(item["action"])
            if "character" in item:
                item["characters"] = [item["character"]]
            if character in item["characters"]:
                scene = "\n".join(all_actions[-1 - scene_window : -1])
                pairs.append({
                    **item,
                    "scene": scene,
                    "last_character": last_character,
                    "_timestamp": title_timestamp,
                    "_title_idx": title_idx,
                })
            last_character = item["characters"]

    mid = len(pairs) // 2
    return {"train": pairs[:mid], "test": pairs[mid:]}
