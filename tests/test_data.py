"""Tests for canopy.data — character metadata and pair loading."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from canopy.data import load_ar_pairs, load_character_metadata

# ---------------------------------------------------------------------------
# load_character_metadata
# ---------------------------------------------------------------------------


class TestLoadCharacterMetadata:
    def test_loads_from_project_root(self) -> None:
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

    def test_yui_is_in_kon(self) -> None:
        _, char2artifact, _ = load_character_metadata()
        assert char2artifact["Yui"] == "kon"

    def test_custom_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chars_path = os.path.join(tmpdir, "chars.json")
            bands_path = os.path.join(tmpdir, "bands.json")

            with open(chars_path, "w") as f:
                json.dump({"TestArt": {"major": ["Alice", "Bob"]}}, f)
            with open(bands_path, "w") as f:
                json.dump({"TestBand": ["Alice"]}, f)

            all_chars, c2a, bm = load_character_metadata(chars_path, bands_path)
            assert c2a["Alice"] == "TestArt"
            assert c2a["Bob"] == "TestArt"
            assert "TestBand" in bm

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_character_metadata("nonexistent.json")


# ---------------------------------------------------------------------------
# load_ar_pairs — cached path
# ---------------------------------------------------------------------------


class TestLoadArPairsCached:
    def test_kasumi_pairs_have_train_test_split(self) -> None:
        _, char2artifact, band2members = load_character_metadata()
        cache_path = "data/title2action_series.Poppin'Party.json"
        if not os.path.exists(cache_path):
            pytest.skip("Kasumi data not cached")

        result = load_ar_pairs("Kasumi", char2artifact, band2members)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) > 0

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
        assert 0.8 <= ratio <= 1.2


# ---------------------------------------------------------------------------
# load_ar_pairs — download path (mocked HF)
# ---------------------------------------------------------------------------


class TestLoadArPairsDownload:
    def test_downloads_and_caches_for_bandori(self) -> None:
        """Test the HF download path with mocked load_dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            char2artifact = {"TestChar": "TestBand"}
            band2members = {"TestBand": ["TestChar", "Other"]}

            mock_data = [
                {"artifact": "TestBand", "title": "ep1", "action": "hello", "characters": ["TestChar"]},
                {"artifact": "TestBand", "title": "ep1", "action": "greet back", "characters": ["Other"]},
                {"artifact": "TestBand", "title": "ep1", "action": "smile", "characters": ["TestChar"]},
                {"artifact": "TestBand", "title": "ep1", "action": "wave", "characters": ["TestChar"]},
                {"artifact": "OtherBand", "title": "ep2", "action": "ignore", "characters": ["Nobody"]},
            ]
            mock_dataset = MagicMock()
            mock_dataset.__getitem__ = MagicMock(return_value=mock_data)

            with patch("datasets.load_dataset", return_value=mock_dataset) as mock_ld:
                result = load_ar_pairs("TestChar", char2artifact, band2members, data_dir=tmpdir)

                mock_ld.assert_called_once_with(
                    "KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences",
                )

            assert "train" in result
            assert "test" in result
            # Cache file should exist
            cache = os.path.join(tmpdir, "title2action_series.TestBand.json")
            assert os.path.exists(cache)

    def test_downloads_fandom_for_non_bandori(self) -> None:
        """Non-bandori artifacts use the Fandom dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            char2artifact = {"Yui": "kon"}
            band2members = {}  # kon not in band2members

            mock_data = [
                {"artifact": "kon", "title": "ep1", "action": "play guitar", "characters": ["Yui"]},
                {"artifact": "kon", "title": "ep1", "action": "eat cake", "characters": ["Yui"]},
            ]
            mock_dataset = MagicMock()
            mock_dataset.__getitem__ = MagicMock(return_value=mock_data)

            with patch("datasets.load_dataset", return_value=mock_dataset) as mock_ld:
                load_ar_pairs("Yui", char2artifact, band2members, data_dir=tmpdir)

                mock_ld.assert_called_once_with(
                    "KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences",
                )

    def test_character_field_compat(self) -> None:
        """Test the 'character' → 'characters' compatibility path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            char2artifact = {"Alice": "art"}
            band2members = {"art": ["Alice"]}

            # Use "character" (singular) instead of "characters" (plural)
            mock_data = [
                {"artifact": "art", "title": "ep1", "action": "act1", "character": "Alice"},
                {"artifact": "art", "title": "ep1", "action": "act2", "character": "Alice"},
                {"artifact": "art", "title": "ep1", "action": "act3", "character": "Alice"},
                {"artifact": "art", "title": "ep1", "action": "act4", "character": "Alice"},
            ]
            mock_dataset = MagicMock()
            mock_dataset.__getitem__ = MagicMock(return_value=mock_data)

            with patch("datasets.load_dataset", return_value=mock_dataset):
                result = load_ar_pairs("Alice", char2artifact, band2members, data_dir=tmpdir)

            total = len(result["train"]) + len(result["test"])
            assert total > 0

    def test_uses_cache_on_second_call(self) -> None:
        """Second call should read from cache, not download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            char2artifact = {"Alice": "art"}
            band2members = {"art": ["Alice"]}

            # Pre-populate cache
            cache = os.path.join(tmpdir, "title2action_series.art.json")
            cached_data = {
                "ep1": [
                    {"action": "cached_action", "characters": ["Alice"], "artifact": "art", "title": "ep1"},
                ],
            }
            with open(cache, "w") as f:
                json.dump(cached_data, f)

            # Should NOT call load_dataset
            with patch("datasets.load_dataset") as mock_ld:
                result = load_ar_pairs("Alice", char2artifact, band2members, data_dir=tmpdir)
                mock_ld.assert_not_called()

            total = len(result["train"]) + len(result["test"])
            assert total > 0
