# tests/industrial_fitness/test_industrial_fitness.py
"""Comprehensive unit and integration tests for the Industrial Fitness Framework."""

import os
import pytest
import pandas as pd
import numpy as np

from src.industrial_fitness.fitness_calculator import FitnessCalculator
from src.industrial_fitness.fitness_validator import FitnessValidator
from src.industrial_fitness.rank_candidates import CandidateRanker

def test_athlete_exercises():
    """Verify that Athlete Exercises execute and match expected values exactly."""
    calculator = FitnessCalculator()
    
    # Run self-validation checks
    assert calculator.run_athlete_exercises() is True

    # Exercise 1 recomputation
    # Stability = 0.80, Reliability = 0.60, Evolutionary = 0.90 -> 0.745
    fit1 = FitnessCalculator.calculate_score(0.80, 0.60, 0.90)
    assert np.isclose(fit1, 0.745)

    # Exercise 2 recomputation
    # Stability = 0.60, Reliability = 1.00, Evolutionary = 1.00 -> 0.80
    fit2 = FitnessCalculator.calculate_score(0.60, 1.00, 1.00)
    assert np.isclose(fit2, 0.80)

def test_fitness_pipeline(tmp_path):
    """Verify the end-to-end execution of the fitness calculator on mock analysis parquet."""
    anal_path = os.path.join(tmp_path, "tkt_mutation_analysis.parquet")
    out_dir = os.path.join(tmp_path, "out_fitness")
    
    # 3 mock rows
    pd.DataFrame({
        "mutation_id": ["TKT_M1A", "TKT_A100C", "TKT_D155V"],
        "evolutionary_sparsity": [0.1, 0.5, 0.9],
        "structural_sparsity": [0.2, 0.6, 0.8],
        "combined_sparsity": [0.15, 0.55, 0.85],
        "reliability_score": [0.85, 0.45, 0.15],
        "predicted_stability": [1.0, -1.0, -3.0] # Range = 4.0, max = 1.0, min = -3.0
    }).to_parquet(anal_path)
    
    calculator = FitnessCalculator()
    parquet_path, plot_path = calculator.compute_fitness(
        analysis_path=anal_path,
        output_dir=out_dir
    )
    
    assert os.path.exists(parquet_path)
    assert os.path.exists(plot_path)
    
    # Read output and verify structure
    df_out = pd.read_parquet(parquet_path)
    assert len(df_out) == 3
    assert list(df_out.columns) == ["mutation_id", "reliability_score", "predicted_stability", "industrial_fitness_score", "rank"]
    
    # TKT_M1A is best: max stability (norm=1.0), max reliability (0.85), max plausibility (1-0.1 = 0.9)
    # Expected fitness: 0.50*1.0 + 0.35*0.85 + 0.15*0.9 = 0.50 + 0.2975 + 0.135 = 0.9325
    best_row = df_out[df_out["mutation_id"] == "TKT_M1A"].iloc[0]
    assert np.isclose(best_row["industrial_fitness_score"], 0.9325)
    assert best_row["rank"] == 1

def test_failure_injection(tmp_path):
    """Verify that failure injection (fitness score > 1.0) is caught by the validator."""
    anal_path = os.path.join(tmp_path, "tkt_mutation_analysis.parquet")
    out_dir = os.path.join(tmp_path, "out_fitness_fault")
    
    pd.DataFrame({
        "mutation_id": ["TKT_M1A"],
        "evolutionary_sparsity": [0.1],
        "structural_sparsity": [0.2],
        "combined_sparsity": [0.15],
        "reliability_score": [0.85],
        "predicted_stability": [1.0]
    }).to_parquet(anal_path)
    
    calculator = FitnessCalculator()
    
    # Running with inject_fault=True should raise a ValueError due to validation failure
    with pytest.raises(ValueError, match="Industrial Fitness dataset failed validation checks!"):
        calculator.compute_fitness(
            analysis_path=anal_path,
            output_dir=out_dir,
            inject_fault=True
        )

def test_candidate_ranker(tmp_path):
    """Verify that candidate ranker correctly ranks scored mutations and outputs D602 CSV."""
    fit_path = os.path.join(tmp_path, "industrial_fitness_scores.parquet")
    out_dir = os.path.join(tmp_path, "out_rank")
    
    pd.DataFrame({
        "mutation_id": ["TKT_M1A", "TKT_A100C", "TKT_D155V"],
        "reliability_score": [0.85, 0.45, 0.15],
        "predicted_stability": [1.0, -1.0, -3.0],
        "industrial_fitness_score": [0.9325, 0.5025, 0.1225],
        "rank": [1, 2, 3]
    }).to_parquet(fit_path)
    
    ranker = CandidateRanker()
    csv_path = ranker.rank_candidates(
        fitness_path=fit_path,
        output_dir=out_dir,
        top_n=2
    )
    
    assert os.path.exists(csv_path)
    df_csv = pd.read_csv(csv_path)
    
    # Check shape & top_n slice limit
    assert len(df_csv) == 2
    assert list(df_csv.columns) == ["rank", "mutation", "reliability_score", "predicted_stability", "industrial_fitness_score"]
    
    # Check descending order
    assert df_csv.loc[0, "industrial_fitness_score"] > df_csv.loc[1, "industrial_fitness_score"]
    assert df_csv.loc[0, "rank"] == 1
    assert df_csv.loc[1, "rank"] == 2
    assert df_csv.loc[0, "mutation"] == "TKT_M1A"

def test_real_data_validation():
    """Verify that real generated industrial fitness files pass validation audits."""
    real_parquet = "data/final/industrial_fitness/industrial_fitness_scores.parquet"
    real_csv = "data/final/industrial_fitness/top_candidate_mutations.csv"
    
    if os.path.exists(real_parquet) and os.path.exists(real_csv):
        validator = FitnessValidator()
        assert validator.validate_scores(real_parquet) is True
        assert validator.validate_ranking_csv(real_csv) is True
