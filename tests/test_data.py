"""Tests for canopy.data — character metadata and pair loading."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from canopy.data import load_ar_pairs, load_character_metadata


class TestLoadCharacterMetadata:
    def test_loads_from_project_root(self) -> None:
        """Integration test — loads actual project data files."""
        all_chars, char2artifact, band2members = load_character_metadata()

        assert "Kasumi" in char2artifact
        assert char2artifact["Kasumi"] == "Poppin'Party"
        assert "Arisa" in char2artifact
        assert isinstance(band2members, dict)

    def test_kasumi_artifact_has_5_members(self) -> None:
        all_chars, _, _ = load_character_metadata()
        popipa = all_chars["Poppin'Party"]["major"]
        assert len(popipa) == 5
        assert "Kasumi" in popipa
        assert "Arisa" in popipa

    def test_yui_is_in_kon(self) -> None:
        _, char2artifact, _ = load_character_metadata()
        assert char2artifact["Yui"] == "kon"

    def test_custom_paths(self) -> None:
        """Test with custom JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chars_path = os.path.join(tmpdir, "chars.json")
            bands_path = os.path.join(tmpdir, "bands.json")

            chars_data = {"TestArtifact": {"major": ["Alice", "Bob"]}}
            bands_data = {"TestBand": ["Alice", "Bob"]}

            with open(chars_path, "w") as f:
                json.dump(chars_data, f)
            with open(bands_path, "w") as f:
                json.dump(bands_data, f)

            all_chars, char2artifact, band2members = load_character_metadata(chars_path, bands_path)
            assert char2artifact["Alice"] == "TestArtifact"
            assert char2artifact["Bob"] == "TestArtifact"
            assert "TestBand" in band2members

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_character_metadata("nonexistent.json")


class TestLoadArPairs:
    def test_kasumi_pairs_have_train_test_split(self) -> None:
        """Integration test — loads cached Kasumi data."""
        _, char2artifact, band2members = load_character_metadata()

        # Only run if data is cached
        cache_path = "data/title2action_series.Poppin'Party.json"
        if not os.path.exists(cache_path):
            pytest.skip("Kasumi data not cached — run CDT build first")

        result = load_ar_pairs("Kasumi", char2artifact, band2members)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) > 0
        assert len(result["test"]) > 0

    def test_pairs_have_required_keys(self) -> None:
        _, char2artifact, band2members = load_character_metadata()

        cache_path = "data/title2action_series.Poppin'Party.json"
        if not os.path.exists(cache_path):
            pytest.skip("Kasumi data not cached")

        result = load_ar_pairs("Kasumi", char2artifact, band2members)
        for pair in result["train"][:5]:
            assert "scene" in pair
            assert "action" in pair
            assert "last_character" in pair

    def test_train_test_roughly_equal(self) -> None:
        _, char2artifact, band2members = load_character_metadata()

        cache_path = "data/title2action_series.Poppin'Party.json"
        if not os.path.exists(cache_path):
            pytest.skip("Kasumi data not cached")

        result = load_ar_pairs("Kasumi", char2artifact, band2members)
        ratio = len(result["train"]) / max(len(result["test"]), 1)
        assert 0.8 <= ratio <= 1.2, f"Train/test ratio {ratio} is too skewed"
