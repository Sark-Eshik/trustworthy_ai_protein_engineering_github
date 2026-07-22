# tests/datasets/test_loaders.py
"""Unit tests for dataset loading, schema verification, and exceptions checks."""

import os
import pytest
import pandas as pd
from src.datasets.loaders import DatasetLoader


def test_dataset_loader_valid_load(tmp_path):
    """Test loading a clean parquet file passes validation and returns data."""
    s001_path = os.path.join(tmp_path, "test_megascale.parquet")
    df = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "protein_id": ["pA", "pB"],
            "position": [10, 20],
            "wildtype": ["A", "G"],
            "mutant": ["V", "C"],
            "experimental_ddg": [1.5, -0.2],
        }
    )
    df.to_parquet(s001_path)

    loader = DatasetLoader()
    loaded_df = loader.load("S001", file_path_override=s001_path)

    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 2
    assert "experimental_ddg" in loaded_df.columns


def test_dataset_loader_invalid_schema(tmp_path):
    """Test that loading files with missing required variables raises ValueError."""
    s001_bad_path = os.path.join(tmp_path, "test_megascale_bad.parquet")
    df = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "protein_id": ["pA", "pB"],
            # missing position, wildtype, mutant, etc.
        }
    )
    df.to_parquet(s001_bad_path)

    loader = DatasetLoader()
    with pytest.raises(ValueError):
        loader.load("S001", file_path_override=s001_bad_path)


def test_dataset_loader_missing_file():
    """Test loading a nonexistent file raises FileNotFoundError."""
    loader = DatasetLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("S001", file_path_override="nonexistent_file_path.parquet")
