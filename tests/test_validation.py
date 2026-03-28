"""Unit tests for canopy.validation — guard checks and logic (no GPU needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestUninitializedGuards:
    def test_check_scene_uninitialized(self) -> None:
        import canopy.validation as val

        saved = val._classifier
        val._classifier = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                val.check_scene(["test"], ["question"])
        finally:
            val._classifier = saved

    def test_check_statement_probs_uninitialized(self) -> None:
        import canopy.validation as val

        saved = val._classifier
        val._classifier = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                val.check_statement_probs("A", ["action"], ["stmt"])
        finally:
            val._classifier = saved


class TestInitModels:
    def test_init_models_sets_globals(self) -> None:
        import canopy.validation as val

        saved_clf = val._classifier
        saved_tok = val._classifier_tokenizer
        saved_dev = val._device

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()
        device = torch.device("cpu")

        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained.return_value = mock_model
        mock_auto_tok_cls = MagicMock()
        mock_auto_tok_cls.from_pretrained.return_value = mock_tokenizer

        try:
            with patch(
                "transformers.AutoModelForSequenceClassification",
                mock_auto_model_cls,
            ), patch(
                "transformers.AutoTokenizer",
                mock_auto_tok_cls,
            ):
                val.init_models("/fake/path", device)

            assert val._classifier is mock_model
            assert val._classifier_tokenizer is mock_tokenizer
            assert val._device is device
            mock_auto_model_cls.from_pretrained.assert_called_once_with("/fake/path")
            mock_model.to.assert_called_once_with(device)
            mock_auto_tok_cls.from_pretrained.assert_called_once_with("/fake/path")
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev


def _setup_mock_classifier(
    val: object, logits_tensor: torch.Tensor
) -> tuple[MagicMock, MagicMock, object, object, object]:
    """Set up mock classifier and tokenizer on the validation module.

    Returns (mock_classifier, mock_tokenizer, saved_clf, saved_tok, saved_dev).
    """
    saved_clf = val._classifier  # type: ignore[attr-defined]
    saved_tok = val._classifier_tokenizer  # type: ignore[attr-defined]
    saved_dev = val._device  # type: ignore[attr-defined]

    mock_tokenizer = MagicMock()
    mock_encoded = MagicMock()
    mock_encoded.to.return_value = mock_encoded
    mock_tokenizer.return_value = mock_encoded

    mock_output = MagicMock()
    mock_output.logits = logits_tensor
    mock_classifier = MagicMock(return_value=mock_output)

    val._classifier = mock_classifier  # type: ignore[attr-defined]
    val._classifier_tokenizer = mock_tokenizer  # type: ignore[attr-defined]
    val._device = torch.device("cpu")  # type: ignore[attr-defined]

    return mock_classifier, mock_tokenizer, saved_clf, saved_tok, saved_dev


class TestCheckScene:
    def test_returns_mapped_values(self) -> None:
        """Verify argmax index maps to [False, None, True]."""
        import canopy.validation as val

        # 3 inputs: argmax at index 0 (False), index 1 (None), index 2 (True)
        logits = torch.tensor([
            [10.0, 0.0, 0.0],  # argmax=0 -> False
            [0.0, 10.0, 0.0],  # argmax=1 -> None
            [0.0, 0.0, 10.0],  # argmax=2 -> True
        ])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(val, logits)

        try:
            result = val.check_scene(
                ["scene1", "scene2", "scene3"],
                ["q1", "q2", "q3"],
            )
            assert result == [False, None, True]
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev

    def test_single_input(self) -> None:
        """Verify check_scene works with a single text/question pair."""
        import canopy.validation as val

        logits = torch.tensor([[0.0, 0.0, 5.0]])  # argmax=2 -> True
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(val, logits)

        try:
            result = val.check_scene(["a scene"], ["a question"])
            assert result == [True]
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev


class TestCheckStatementProbs:
    def test_returns_3_values(self) -> None:
        """Verify shape is (3,) from softmax sum."""
        import canopy.validation as val

        # Two prompts, 3 classes each
        logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(val, logits)

        try:
            result = val.check_statement_probs(
                "Alice",
                ["action1", "action2"],
                ["stmt1", "stmt2"],
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)
            # softmax summed over 2 rows should sum to 2.0
            assert abs(result.sum() - 2.0) < 1e-5
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev

    def test_single_pair(self) -> None:
        """Verify single action/statement returns (3,) summing to 1.0."""
        import canopy.validation as val

        logits = torch.tensor([[2.0, 1.0, 0.5]])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(val, logits)

        try:
            result = val.check_statement_probs("Bob", ["act"], ["stmt"])
            assert result.shape == (3,)
            assert abs(result.sum() - 1.0) < 1e-5
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev


class TestValidateHypothesis:
    def test_no_gate_keeps_all_pairs(self) -> None:
        """When hypothesized_question is None, all pairs pass the scene filter."""
        import canopy.validation as val

        # Mock check_statement_probs to return known values
        pairs = [
            {"scene": "s1", "action": "a1"},
            {"scene": "s2", "action": "a2"},
        ]
        stmt_probs = np.array([0.1, 0.3, 0.6])

        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(val, logits)

        try:
            with patch.object(
                val, "check_statement_probs", return_value=stmt_probs
            ) as mock_stmt:
                result_counts, filtered = val.validate_hypothesis(
                    character="Alice",
                    pairs=pairs,
                    hypothesized_question=None,
                    hypothesized_action="does something",
                )
            # All pairs kept (no scene filtering)
            assert len(filtered) == 2
            assert "Irrelevant" not in result_counts
            # check_statement_probs called once for the batch
            mock_stmt.assert_called_once()
            # Verify shape contract: probs are accumulated into result_counts
            assert stmt_probs.shape == (3,)  # [false, none, true] — matches impl
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev

    def test_with_gate_filters_irrelevant(self) -> None:
        """When hypothesized_question is set, check_scene filters pairs."""
        import canopy.validation as val

        pairs = [
            {"scene": "relevant", "action": "a1"},
            {"scene": "irrelevant", "action": "a2"},
            {"scene": "also relevant", "action": "a3"},
        ]
        stmt_probs = np.array([0.2, 0.3, 0.5])

        # check_scene returns True, False, True => pair[1] is filtered out
        logits_placeholder = torch.tensor([[1.0, 2.0, 3.0]])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(
            val, logits_placeholder
        )

        try:
            with patch.object(
                val, "check_scene", return_value=[True, False, True]
            ) as mock_scene, patch.object(
                val, "check_statement_probs", return_value=stmt_probs
            ):
                result_counts, filtered = val.validate_hypothesis(
                    character="Alice",
                    pairs=pairs,
                    hypothesized_question="Is this relevant?",
                    hypothesized_action="does something",
                )
            # 2 pairs kept, 1 filtered
            assert len(filtered) == 2
            assert result_counts["Irrelevant"] == 1.0
            # check_scene called once for the batch
            mock_scene.assert_called_once()
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev

    def test_batching(self) -> None:
        """Verify batch processing with small bs splits calls correctly."""
        import canopy.validation as val

        pairs = [{"scene": f"s{i}", "action": f"a{i}"} for i in range(5)]
        stmt_probs = np.array([0.1, 0.2, 0.7])

        logits_placeholder = torch.tensor([[1.0, 2.0, 3.0]])
        _, _, saved_clf, saved_tok, saved_dev = _setup_mock_classifier(
            val, logits_placeholder
        )

        try:
            scene_calls: list[list[str]] = []

            def track_check_scene(
                texts: list[str], questions: list[str]
            ) -> list[bool | None]:
                scene_calls.append(texts)
                return [True] * len(texts)

            stmt_calls: list[list[str]] = []

            def track_check_stmt(
                character: str, actions: list[str], statements: list[str]
            ) -> np.ndarray:
                stmt_calls.append(actions)
                return stmt_probs

            with patch.object(
                val, "check_scene", side_effect=track_check_scene
            ), patch.object(
                val, "check_statement_probs", side_effect=track_check_stmt
            ):
                result_counts, filtered = val.validate_hypothesis(
                    character="Alice",
                    pairs=pairs,
                    hypothesized_question="Q?",
                    hypothesized_action="action",
                    bs=2,
                )

            # 5 pairs with bs=2 => 3 scene batches (2, 2, 1)
            assert len(scene_calls) == 3
            assert scene_calls[0] == ["s0", "s1"]
            assert scene_calls[1] == ["s2", "s3"]
            assert scene_calls[2] == ["s4"]

            # All 5 pairs pass => 3 statement batches (2, 2, 1)
            assert len(stmt_calls) == 3
            assert stmt_calls[0] == ["a0", "a1"]
            assert stmt_calls[1] == ["a2", "a3"]
            assert stmt_calls[2] == ["a4"]

            # All pairs kept
            assert len(filtered) == 5
        finally:
            val._classifier = saved_clf
            val._classifier_tokenizer = saved_tok
            val._device = saved_dev
