# tests/validation/test_dataset_validator.py
"""Unit tests for dataset values, ranges, and record profiling validations."""

import os
import pytest
import pandas as pd
from src.validation.dataset_validator import DatasetValidator


def test_dataset_validator_valid_ranges(tmp_path):
    """Test standard boundaries checks with correct records."""
    validator = DatasetValidator()

    # Valid data matching standard S001 boundaries
    valid_df = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "protein_id": ["pA", "pB"],
            "position": [10, 20],
            "wildtype": ["A", "G"],
            "mutant": ["V", "C"],
            "experimental_ddg": [1.5, -0.2],
        }
    )

    assert validator.validate_dataset("S001", valid_df)


def test_dataset_validator_invalid_ranges(tmp_path):
    """Test standard boundaries check fails on out-of-bounds dDG stability inputs."""
    validator = DatasetValidator()

    # Out-of-bounds stability (expected bounds [-15, 15] kcal/mol)
    invalid_df = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "protein_id": ["pA", "pB"],
            "position": [10, 20],
            "wildtype": ["A", "G"],
            "mutant": ["V", "C"],
            "experimental_ddg": [34.5, -0.2],  # 34.5 is physically unreasonable
        }
    )

    assert not validator.validate_dataset("S001", invalid_df)
