# tests/sparsity/test_combined_sparsity.py
"""Unit and validation tests for the Combined Sparsity module."""

import os
import pytest
import pandas as pd
import numpy as np

from src.sparsity.combined.combine_sparsity import CombinedSparsity
from src.sparsity.combined.combined_validation import CombinedValidation


def test_combined_sparsity_megascale_d_mode(tmp_path):
    """Verify combined sparsity average calculation using 3 dimensions (Megascale-D mode)."""
    # Create mock parquets
    emp_path = os.path.join(tmp_path, "emp.parquet")
    evo_path = os.path.join(tmp_path, "evo.parquet")
    struct_path = os.path.join(tmp_path, "struct.parquet")

    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "count": [10, 100],
        "frequency": [0.1, 0.9],
        "mutation_frequency": [0.1, 0.9],
        "normalized_frequency": [0.1, 1.0],
        "empirical_sparsity": [0.9, 0.0],
    }).to_parquet(emp_path)

    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "esm_probability": [0.01, 0.8],
        "log_probability": [4.6, 0.22],
        "evolutionary_sparsity": [0.95, 0.05],
    }).to_parquet(evo_path)

    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "sasa": [5.0, 150.0],
        "normalized_sasa": [0.05, 1.0],
        "structural_sparsity": [0.95, 0.0],
    }).to_parquet(struct_path)

    pipeline = CombinedSparsity()
    out_dir = os.path.join(tmp_path, "results")
    p_path, r_path, plot = pipeline.run_pipeline(
        mode="megascale_d",
        empirical_path_override=emp_path,
        evolutionary_path_override=evo_path,
        structural_path_override=struct_path,
        output_dir=out_dir,
    )

    assert os.path.exists(p_path)
    assert os.path.exists(r_path)
    assert os.path.exists(plot)

    # Load and check arithmetic combination: (Emp + Evo + Struct) / 3.0
    df = pd.read_parquet(p_path)
    assert len(df) == 2
    assert "combined_sparsity" in df.columns

    # mut1: (0.9 + 0.95 + 0.95) / 3 = 2.8 / 3.0 = 0.933333
    # mut2: (0.0 + 0.05 + 0.0) / 3 = 0.05 / 3.0 = 0.016667
    mut1_rec = df.loc[df["mutation_id"] == "mut1"].iloc[0]
    mut2_rec = df.loc[df["mutation_id"] == "mut2"].iloc[0]

    assert np.isclose(mut1_rec["combined_sparsity"], 2.8 / 3.0)
    assert np.isclose(mut2_rec["combined_sparsity"], 0.05 / 3.0)

    # Validate scientifically
    validator = CombinedValidation()
    assert validator.validate_dataset(p_path, mode="megascale_d") is True


def test_combined_sparsity_tkt_mode(tmp_path):
    """Verify combined sparsity average calculation using 2 dimensions (TKT mode)."""
    evo_path = os.path.join(tmp_path, "evo.parquet")
    struct_path = os.path.join(tmp_path, "struct.parquet")

    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "esm_probability": [0.01, 0.8],
        "log_probability": [4.6, 0.22],
        "evolutionary_sparsity": [0.95, 0.05],
    }).to_parquet(evo_path)

    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "sasa": [5.0, 150.0],
        "normalized_sasa": [0.05, 1.0],
        "structural_sparsity": [0.95, 0.0],
    }).to_parquet(struct_path)

    pipeline = CombinedSparsity()
    out_dir = os.path.join(tmp_path, "results2")
    p_path, r_path, plot = pipeline.run_pipeline(
        mode="tkt",
        empirical_path_override=None,
        evolutionary_path_override=evo_path,
        structural_path_override=struct_path,
        output_dir=out_dir,
    )

    assert os.path.exists(p_path)
    assert os.path.exists(r_path)
    assert os.path.exists(plot)

    # Load and check arithmetic combination: (Evo + Struct) / 2.0
    df = pd.read_parquet(p_path)
    assert len(df) == 2
    assert "combined_sparsity" in df.columns
    assert df["empirical_sparsity"].isnull().all()  # Empirical should be null/NaN in TKT mode

    # mut1: (0.95 + 0.95) / 2 = 0.95
    # mut2: (0.05 + 0.0) / 2 = 0.025
    mut1_rec = df.loc[df["mutation_id"] == "mut1"].iloc[0]
    mut2_rec = df.loc[df["mutation_id"] == "mut2"].iloc[0]

    assert np.isclose(mut1_rec["combined_sparsity"], 0.95)
    assert np.isclose(mut2_rec["combined_sparsity"], 0.025)

    # Validate scientifically
    validator = CombinedValidation()
    assert validator.validate_dataset(p_path, mode="tkt") is True


def test_combined_validation_failures(tmp_path):
    """Verify that scientific validator correctly reports calculation audit mismatches."""
    validator = CombinedValidation()

    # Create dataset violating the math: Combined != (Emp + Evo + Struct) / 3
    bad_calc_path = os.path.join(tmp_path, "bad_calc.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1"],
        "empirical_sparsity": [0.5],
        "evolutionary_sparsity": [0.5],
        "structural_sparsity": [0.5],
        "combined_sparsity": [0.8],  # Math audit fail: should be 0.5, is 0.8
    }).to_parquet(bad_calc_path)

    assert validator.validate_dataset(bad_calc_path, mode="megascale_d") is False
