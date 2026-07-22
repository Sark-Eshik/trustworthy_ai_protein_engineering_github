# tests/experiment2/test_experiment2.py
"""Comprehensive unit and integration tests for Experiment 2 modules."""

import os
import pytest
import pandas as pd
import numpy as np
from src.validation.validate_experiment2_inputs import validate_inputs
from src.experiment2.experimental_epistasis import ExperimentalEpistasisCalculator
from src.experiment2.predicted_epistasis import PredictedEpistasisCalculator
from src.experiment2.epistasis_error import EpistasisErrorCalculator
from src.experiment2.statistics import compute_statistics
from src.experiment2.figures import generate_figures

def test_experimental_epistasis_calculator():
    """Verify standard experimental epistasis formula and Athlete Exercise 1."""
    calc = ExperimentalEpistasisCalculator()
    
    # Test standard formula: ddG_ab - (ddG_a + ddG_b)
    # 5.0 - (1.0 + 2.0) = 2.0
    val = calc.calculate_epistasis(5.0, 1.0, 2.0)
    assert np.isclose(val, 2.0)
    
    # Test Exercise 1 check
    assert calc.run_athlete_exercise() is True

def test_predicted_epistasis_calculator():
    """Verify standard predicted epistasis formula."""
    calc = PredictedEpistasisCalculator()
    
    # Formula: Pred_ab - (Pred_a + Pred_b)
    # 4.5 - (1.2 + 1.8) = 1.5
    val = calc.calculate_epistasis(4.5, 1.2, 1.8)
    assert np.isclose(val, 1.5)

def test_epistasis_error_calculator():
    """Verify absolute epistasis error formula and Athlete Exercise 2."""
    calc = EpistasisErrorCalculator()
    
    # Formula: |predicted - experimental|
    # |1.5 - 2.0| = 0.5
    val = calc.calculate_error(1.5, 2.0)
    assert np.isclose(val, 0.5)
    
    # Test Exercise 2 check
    assert calc.run_athlete_exercise() is True

def test_epistasis_error_pipeline(tmp_path):
    """Verify the core epistasis error calculation pipeline."""
    exp_path = os.path.join(tmp_path, "exp.parquet")
    pred_path = os.path.join(tmp_path, "pred.parquet")
    sp_path = os.path.join(tmp_path, "sp.parquet")
    
    pd.DataFrame({
        "pair_id": ["pair1"],
        "mutation_a": ["mut1"],
        "mutation_b": ["mut2"],
        "experimental_ddg_a": [1.0],
        "experimental_ddg_b": [2.0],
        "experimental_ddg_ab": [3.1],
        "experimental_epistasis": [0.1]
    }).to_parquet(exp_path)
    
    pd.DataFrame({
        "pair_id": ["pair1"],
        "predicted_ddg_a": [1.2],
        "predicted_ddg_b": [1.9],
        "predicted_ddg_ab": [3.3],
        "predicted_epistasis": [0.2]
    }).to_parquet(pred_path)
    
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "combined_sparsity": [0.3, 0.5]
    }).to_parquet(sp_path)
    
    calculator = EpistasisErrorCalculator()
    out_dir = os.path.join(tmp_path, "outputs")
    
    res_path = calculator.process_pipeline(
        experimental_path=exp_path,
        predicted_path=pred_path,
        sparsity_path=sp_path,
        output_dir=out_dir
    )
    
    assert os.path.exists(res_path)
    
    # Check D302 columns and values
    df_res = pd.read_parquet(res_path)
    assert len(df_res) == 1
    assert "epistasis_error" in df_res.columns
    assert "combined_sparsity" in df_res.columns
    
    # Error = |Predicted Epistasis - Experimental Epistasis| = |0.2 - 0.1| = 0.1
    # Sparsity = (0.3 + 0.5) / 2.0 = 0.4
    assert np.isclose(df_res["epistasis_error"].iloc[0], 0.1)
    assert np.isclose(df_res["combined_sparsity"].iloc[0], 0.4)

def test_epistasis_failure_injection(tmp_path):
    """Verify that the validator throws ValueError on missing measurements (Failure Injection)."""
    exp_path = os.path.join(tmp_path, "exp_missing.parquet")
    pred_path = os.path.join(tmp_path, "pred.parquet")
    sp_path = os.path.join(tmp_path, "sp.parquet")
    
    # Injected null value in experimental_ddg_ab
    pd.DataFrame({
        "pair_id": ["pair1"],
        "mutation_a": ["mut1"],
        "mutation_b": ["mut2"],
        "experimental_ddg_a": [1.0],
        "experimental_ddg_b": [2.0],
        "experimental_ddg_ab": [None],  # Null!
        "experimental_epistasis": [0.1]
    }).to_parquet(exp_path)
    
    pd.DataFrame({
        "pair_id": ["pair1"],
        "predicted_ddg_a": [1.2],
        "predicted_ddg_b": [1.9],
        "predicted_ddg_ab": [3.3],
        "predicted_epistasis": [0.2]
    }).to_parquet(pred_path)
    
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "combined_sparsity": [0.3, 0.5]
    }).to_parquet(sp_path)
    
    calculator = EpistasisErrorCalculator()
    out_dir = os.path.join(tmp_path, "outputs")
    
    with pytest.raises(ValueError, match="Failure Injection Detected"):
        calculator.process_pipeline(
            experimental_path=exp_path,
            predicted_path=pred_path,
            sparsity_path=sp_path,
            output_dir=out_dir
        )

def test_statistics_and_figures_pipeline():
    """Verify statistics files and figures are correctly compiled on real data."""
    # Ensure standard results exist
    assert validate_inputs() is True
    
    corr_csv, reg_csv, quant_csv = compute_statistics()
    assert os.path.exists(corr_csv)
    assert os.path.exists(reg_csv)
    assert os.path.exists(quant_csv)
    
    # Read quantiles and verify that error scales with sparsity (our simulated truth)
    df_quant = pd.read_csv(quant_csv)
    err_low = df_quant.loc[df_quant["subset"].str.contains("Lowest"), "mean_epistasis_error"].iloc[0]
    err_high = df_quant.loc[df_quant["subset"].str.contains("Highest"), "mean_epistasis_error"].iloc[0]
    assert err_high > err_low
    
    plots = generate_figures()
    for p in plots:
        assert os.path.exists(p)
