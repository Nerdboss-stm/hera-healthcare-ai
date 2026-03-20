import os
import pandas as pd
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TestNoteGeneration:
    """Tests for the synthetic clinical note generator."""

    def test_notes_file_exists(self):
        path = os.path.join(DATA_DIR, "clinical_notes", "notes_1000.csv")
        assert os.path.exists(path), "notes_1000.csv not found in data/clinical_notes/"

    def test_notes_have_required_columns(self):
        path = os.path.join(DATA_DIR, "clinical_notes", "notes_1000.csv")
        if not os.path.exists(path):
            pytest.skip("notes_1000.csv not available")
        df = pd.read_csv(path)
        assert "original_note" in df.columns, "Missing 'original_note' column"
        assert "target_summary" in df.columns, "Missing 'target_summary' column"

    def test_notes_minimum_count(self):
        path = os.path.join(DATA_DIR, "clinical_notes", "notes_1000.csv")
        if not os.path.exists(path):
            pytest.skip("notes_1000.csv not available")
        df = pd.read_csv(path)
        assert len(df) >= 1000, f"Expected at least 1000 notes, got {len(df)}"

    def test_notes_not_empty(self):
        path = os.path.join(DATA_DIR, "clinical_notes", "notes_1000.csv")
        if not os.path.exists(path):
            pytest.skip("notes_1000.csv not available")
        df = pd.read_csv(path)
        assert df["original_note"].str.len().min() > 50, "Notes are too short"
        assert df["target_summary"].str.len().min() > 10, "Summaries are too short"


class TestDatasetPreprocessing:
    """Tests for T5 dataset preprocessing."""

    def test_preprocess_function_adds_prefix(self):
        from clinical_summarizer.dataset import preprocess_function
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_tokenizer.as_target_tokenizer.return_value.__enter__ = MagicMock(
            return_value=mock_tokenizer
        )
        mock_tokenizer.as_target_tokenizer.return_value.__exit__ = MagicMock(
            return_value=False
        )

        examples = {
            "original_note": ["Patient has chest pain."],
            "target_summary": ["Chest pain noted."],
        }

        preprocess_function(examples, mock_tokenizer, 512, 128)

        call_args = mock_tokenizer.call_args_list[0]
        assert call_args[0][0][0].startswith("summarize: ")
