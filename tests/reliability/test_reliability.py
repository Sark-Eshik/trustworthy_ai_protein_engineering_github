# tests/reliability/test_reliability.py
"""Comprehensive unit and integration tests for Phase 7: Reliability Framework."""

import os
import pytest
import pandas as pd
import numpy as np
from src.reliability.reliability_score import ReliabilityFramework
from src.reliability.reliability_validation import ReliabilityValidator
from src.reliability.reliability_classifier import map_score_to_category

def test_athlete_exercises_and_formula():
    """Verify that the official Reliability calculation and category mapping work exactly as required."""
    # Athlete Exercise 1: Combined Sparsity = 0.10 -> Expected: Reliability = 0.90
    assert np.isclose(ReliabilityFramework.calculate_score(0.10), 0.90)
    
    # Athlete Exercise 2: Combined Sparsity = 0.75 -> Expected: Reliability = 0.25
    assert np.isclose(ReliabilityFramework.calculate_score(0.75), 0.25)
    
    # Athlete Exercise 3: Combined Sparsity = 1.00 -> Expected: Reliability = 0.00
    assert np.isclose(ReliabilityFramework.calculate_score(1.00), 0.00)

    # Categories Verification
    # 0.75 - 1.00 -> High Reliability
    assert ReliabilityFramework.classify_category(0.90) == "High"
    assert ReliabilityFramework.classify_category(0.75) == "High"
    
    # 0.50 - 0.75 -> Moderate Reliability
    assert ReliabilityFramework.classify_category(0.74) == "Moderate"
    assert ReliabilityFramework.classify_category(0.50) == "Moderate"
    
    # 0.25 - 0.50 -> Low Reliability
    assert ReliabilityFramework.classify_category(0.49) == "Low"
    assert ReliabilityFramework.classify_category(0.25) == "Low"
    
    # 0.00 - 0.25 -> Very Low Reliability
    assert ReliabilityFramework.classify_category(0.24) == "Very Low"
    assert ReliabilityFramework.classify_category(0.00) == "Very Low"

    # Also test external classifier function
    assert map_score_to_category(0.85) == "High"
    assert map_score_to_category(0.65) == "Moderate"
    assert map_score_to_category(0.35) == "Low"
    assert map_score_to_category(0.15) == "Very Low"

def test_reliability_pipeline(tmp_path):
    """Verify the end-to-end execution of the reliability scores pipeline on synthetic combined sparsity."""
    sp_path = os.path.join(tmp_path, "combined_sparsity.parquet")
    out_dir = os.path.join(tmp_path, "output_reliability")
    
    # Create valid synthetic combined sparsity file (D104 structure)
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2", "mut3"],
        "empirical_sparsity": [0.1, 0.4, np.nan], # TKT mode supports NaN empirical
        "evolutionary_sparsity": [0.2, 0.5, 0.8],
        "structural_sparsity": [0.3, 0.6, 0.9],
        "combined_sparsity": [0.2, 0.5, 0.85]
    }).to_parquet(sp_path)
    
    framework = ReliabilityFramework()
    parquet_path, summary_path, plot_path = framework.process_pipeline(
        sparsity_path=sp_path,
        output_dir=out_dir
    )
    
    assert os.path.exists(parquet_path)
    assert os.path.exists(summary_path)
    assert os.path.exists(plot_path)
    
    # Check loaded output schema and calculations
    df_out = pd.read_parquet(parquet_path)
    assert len(df_out) == 3
    assert list(df_out.columns) == ["mutation_id", "combined_sparsity", "reliability_score", "reliability_category"]
    
    assert np.isclose(df_out.loc[0, "reliability_score"], 0.8)
    assert df_out.loc[0, "reliability_category"] == "High"
    
    assert np.isclose(df_out.loc[1, "reliability_score"], 0.5)
    assert df_out.loc[1, "reliability_category"] == "Moderate"
    
    assert np.isclose(df_out.loc[2, "reliability_score"], 0.15)
    assert df_out.loc[2, "reliability_category"] == "Very Low"

def test_failure_injection(tmp_path):
    """Verify that failure injection (assigning reliability = 1.20) is caught by the validator."""
    sp_path = os.path.join(tmp_path, "combined_sparsity.parquet")
    out_dir = os.path.join(tmp_path, "output_reliability")
    
    pd.DataFrame({
        "mutation_id": ["mut1"],
        "empirical_sparsity": [0.1],
        "evolutionary_sparsity": [0.2],
        "structural_sparsity": [0.3],
        "combined_sparsity": [0.2]
    }).to_parquet(sp_path)
    
    framework = ReliabilityFramework()
    
    # Assert that process_pipeline raises ValueError when inject_invalid_value is True
    with pytest.raises(ValueError, match="Reliability Scores dataset failed validation check!"):
        framework.process_pipeline(
            sparsity_path=sp_path,
            output_dir=out_dir,
            inject_invalid_value=True
        )

def test_real_data_validation():
    """Verify that the real generated reliability scores pass the official validation audit."""
    real_parquet = "data/final/reliability/reliability_scores.parquet"
    if os.path.exists(real_parquet):
        validator = ReliabilityValidator()
        assert validator.validate_dataset(real_parquet) is True
