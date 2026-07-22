# tests/infrastructure/test_dataset_registry.py
"""Unit tests for the centralized Dataset Registry."""

import pytest
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition


def test_registry_contains_official_definitions():
    """Test standard dataset identifiers resolve correctly."""
    registry = DatasetRegistry()

    s001 = registry.get_dataset("S001")
    assert isinstance(s001, DatasetDefinition)
    assert s001.dataset_id == "S001"
    assert s001.primary_key == "mutation_id"
    assert "experimental_ddg" in s001.required_columns


def test_registry_missing_error():
    """Test registry throws appropriate error on unregistered identifiers."""
    registry = DatasetRegistry()
    with pytest.raises(KeyError):
        registry.get_dataset("INVALID_ID")
