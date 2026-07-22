# tests/infrastructure/test_validation_engine.py
"""Unit tests for the centralized Validation Engine."""

import pytest
import pandas as pd
from src.infrastructure.validation_engine import ValidationEngine
from src.infrastructure.dataset_registry import DatasetRegistry


def test_validation_engine_schema_checks():
    """Test schema and primary key validation constraints."""
    engine = ValidationEngine()

    valid_data = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "protein_id": ["pA", "pB"],
            "position": [10, 20],
            "wildtype": ["A", "G"],
            "mutant": ["V", "C"],
            "experimental_ddg": [1.5, -0.2],
        }
    )

    report_valid = engine.validate_schema("S001", valid_data)
    assert report_valid["valid"]

    # Trigger columns mismatch and duplicate primary key checks
    invalid_data = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut1"],  # Duplicate
            "protein_id": ["pA", "pB"],
            "position": [10, 20],
            "wildtype": ["A", "G"],
            "mutant": ["V", "C"],
            # missing experimental_ddg column
        }
    )

    report_invalid = engine.validate_schema("S001", invalid_data)
    assert not report_invalid["valid"]
    assert any("Duplicate" in err or "duplicate" in err for err in report_invalid["errors"])
    assert any("Missing" in err or "missing" in err for err in report_invalid["errors"])


def test_validation_engine_range_checks():
    """Test inclusive range bounds constraints validation."""
    engine = ValidationEngine()

    data = pd.DataFrame({"experimental_ddg": [1.2, 0.4, 4.5]})

    # Valid boundaries
    report_valid = engine.validate_ranges("S001", data, {"experimental_ddg": (0.0, 5.0)})
    assert report_valid["valid"]

    # Boundary violation
    report_invalid = engine.validate_ranges("S001", data, {"experimental_ddg": (0.0, 3.0)})
    assert not report_invalid["valid"]
