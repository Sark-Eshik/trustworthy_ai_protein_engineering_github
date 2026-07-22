# tests/sparsity/test_empirical_sparsity.py
"""Unit and validation tests for the Empirical Sparsity module."""

import os
import pytest
import pandas as pd
import numpy as np

from src.sparsity.empirical.empirical_sparsity import EmpiricalSparsity
from src.sparsity.empirical.empirical_validation import EmpiricalValidation


def test_empirical_sparsity_calculation():
    """Verify that mutation frequencies and empirical sparsities are calculated correctly."""
    # Create simple mutation count DataFrame
    counts_df = pd.DataFrame(
        {
            "mutation_id": ["mutA", "mutB", "mutC"],
            "count": [10, 1000, 100],  # Total observations = 1110, Max count = 1000
        }
    )

    framework = EmpiricalSparsity()
    res = framework.calculate(counts_df)

    assert len(res) == 3
    # Check frequency calculations: count / total observations
    assert np.isclose(res.loc[res["mutation_id"] == "mutA", "frequency"].values[0], 10 / 1110)
    assert np.isclose(res.loc[res["mutation_id"] == "mutB", "frequency"].values[0], 1000 / 1110)
    assert np.isclose(res.loc[res["mutation_id"] == "mutC", "frequency"].values[0], 100 / 1110)

    # Check normalized frequency: count / max count
    assert np.isclose(res.loc[res["mutation_id"] == "mutA", "normalized_frequency"].values[0], 10 / 1000)
    assert np.isclose(res.loc[res["mutation_id"] == "mutB", "normalized_frequency"].values[0], 1000 / 1000)
    assert np.isclose(res.loc[res["mutation_id"] == "mutC", "normalized_frequency"].values[0], 100 / 1000)

    # Check empirical sparsity: 1.0 - normalized frequency
    assert np.isclose(res.loc[res["mutation_id"] == "mutA", "empirical_sparsity"].values[0], 1.0 - 0.01)
    assert np.isclose(res.loc[res["mutation_id"] == "mutB", "empirical_sparsity"].values[0], 0.0)
    assert np.isclose(res.loc[res["mutation_id"] == "mutC", "empirical_sparsity"].values[0], 1.0 - 0.1)


def test_empirical_sparsity_zero_total_observations():
    """Test that zero total observations are handled gracefully."""
    counts_df = pd.DataFrame(
        {
            "mutation_id": ["mutA", "mutB"],
            "count": [0, 0],
        }
    )

    framework = EmpiricalSparsity()
    res = framework.calculate(counts_df)

    assert (res["frequency"] == 0.0).all()
    assert (res["normalized_frequency"] == 0.0).all()
    assert (res["empirical_sparsity"] == 1.0).all()


def test_load_counts_validation_failures(tmp_path):
    """Test various input validation rules for load_counts."""
    framework = EmpiricalSparsity()

    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        framework.load_counts(os.path.join(tmp_path, "nonexistent.csv"))

    # 2. Missing required columns
    bad_cols_path = os.path.join(tmp_path, "bad_cols.csv")
    pd.DataFrame({"mut_id": ["A"], "cnt": [5]}).to_csv(bad_cols_path, index=False)
    with pytest.raises(ValueError, match="Missing required column"):
        framework.load_counts(bad_cols_path)

    # 3. Contains null values
    nulls_path = os.path.join(tmp_path, "nulls.csv")
    pd.DataFrame({"mutation_id": ["A", "B"], "count": [5, None]}).to_csv(nulls_path, index=False)
    with pytest.raises(ValueError, match="contains.*null"):
        framework.load_counts(nulls_path)

    # 4. Duplicate mutation_id keys
    dups_path = os.path.join(tmp_path, "dups.csv")
    pd.DataFrame({"mutation_id": ["A", "A"], "count": [5, 10]}).to_csv(dups_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        framework.load_counts(dups_path)

    # 5. Negative count values
    neg_path = os.path.join(tmp_path, "neg.csv")
    pd.DataFrame({"mutation_id": ["A", "B"], "count": [10, -5]}).to_csv(neg_path, index=False)
    with pytest.raises(ValueError, match="negative mutation count"):
        framework.load_counts(neg_path)


def test_empirical_sparsity_outputs_and_certification(tmp_path):
    """Test output serialization, file generation, and scientific certification."""
    counts_path = os.path.join(tmp_path, "counts.csv")
    pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2", "mut3"],
            "count": [10, 100, 1000],
        }
    ).to_csv(counts_path, index=False)

    # Run complete pipeline
    framework = EmpiricalSparsity()
    counts_df = framework.load_counts(counts_path)
    res_df = framework.calculate(counts_df)

    output_dir = os.path.join(tmp_path, "results")
    parquet_path, summary_path, plot_path = framework.generate_outputs(res_df, output_dir)

    assert os.path.exists(parquet_path)
    assert os.path.exists(summary_path)
    assert os.path.exists(plot_path)

    # Verify column structures of parquet file
    df_loaded = pd.read_parquet(parquet_path)
    assert "mutation_id" in df_loaded.columns
    assert "count" in df_loaded.columns
    assert "frequency" in df_loaded.columns
    assert "mutation_frequency" in df_loaded.columns
    assert "normalized_frequency" in df_loaded.columns
    assert "empirical_sparsity" in df_loaded.columns

    # Verify that validation engine certifies the generated file
    validator = EmpiricalValidation()
    assert validator.validate_dataset(parquet_path) is True


def test_empirical_validation_scientific_anomalies(tmp_path):
    """Test that validation script correctly catches scientific anomalies."""
    validator = EmpiricalValidation()

    # Create dataset violating range constraint [0, 1] on empirical_sparsity
    invalid_range_path = os.path.join(tmp_path, "invalid_range.parquet")
    pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "count": [10, 100],
            "frequency": [0.1, 0.9],
            "mutation_frequency": [0.1, 0.9],
            "normalized_frequency": [0.1, 0.9],
            "empirical_sparsity": [1.2, -0.5],  # Out of bounds!
        }
    ).to_parquet(invalid_range_path)

    assert validator.validate_dataset(invalid_range_path) is False

    # Create dataset violating scientific ordering (lower count should mean higher sparsity)
    invalid_order_path = os.path.join(tmp_path, "invalid_order.parquet")
    pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2"],
            "count": [10, 100],  # mut1 has smaller count, so should have larger sparsity
            "frequency": [0.09, 0.91],
            "mutation_frequency": [0.09, 0.91],
            "normalized_frequency": [0.1, 1.0],
            "empirical_sparsity": [0.1, 0.9],  # Violates ordering! mut1 (smaller count) has smaller sparsity
        }
    ).to_parquet(invalid_order_path)

    assert validator.validate_dataset(invalid_order_path) is False
