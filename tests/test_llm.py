"""Tests for canopy.llm — extract_json and adapter protocol."""

from __future__ import annotations

import pytest

from canopy.llm import extract_json


class TestExtractJson:
    def test_fenced_block(self) -> None:
        response = 'Here is the result:\n```json\n{"key": "value"}\n```'
        assert extract_json(response) == {"key": "value"}

    def test_fenced_block_with_surrounding_text(self) -> None:
        response = 'Some preamble\n```json\n[1, 2, 3]\n```\nSome epilogue'
        assert extract_json(response) == [1, 2, 3]

    def test_raw_json_object(self) -> None:
        response = '{"action_hypotheses": ["stmt1"], "scene_check_hypotheses": ["q1"]}'
        result = extract_json(response)
        assert result["action_hypotheses"] == ["stmt1"]

    def test_raw_json_array(self) -> None:
        response = '[1, 2, 3]'
        assert extract_json(response) == [1, 2, 3]

    def test_json_embedded_in_text(self) -> None:
        response = 'The answer is {"x": 42} as expected.'
        assert extract_json(response) == {"x": 42}

    def test_nested_objects(self) -> None:
        response = '{"a": {"b": [1, 2]}}'
        result = extract_json(response)
        assert result["a"]["b"] == [1, 2]

    def test_no_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("This is just plain text with no JSON.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("")

    def test_malformed_fenced_falls_through(self) -> None:
        response = '```json\n{broken json\n```\nBut here: {"fallback": true}'
        result = extract_json(response)
        assert result == {"fallback": True}

    def test_multiple_fenced_blocks_uses_first(self) -> None:
        response = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        assert extract_json(response) == {"first": 1}

    def test_whitespace_padded(self) -> None:
        response = '  \n  {"key": "val"}  \n  '
        assert extract_json(response) == {"key": "val"}
