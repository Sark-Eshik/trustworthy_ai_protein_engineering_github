# tests/integrated/test_integrated.py
"""Comprehensive unit and integration tests for Phase 6: Integrated Reliability Analysis."""

import os
import pytest
import pandas as pd
import numpy as np
from src.integrated.build_dataset import IntegratedDatasetBuilder
from src.integrated.integrated_statistics import run_statistics
from src.integrated.pathway_analysis import PathwayAnalysisEngine
from src.integrated.integrated_figures import generate_figures

def test_integrated_dataset_builder(tmp_path):
    """Verify compiling the integrated analysis dataset (D401) and mapping epistasis errors."""
    sp_path = os.path.join(tmp_path, "sp.parquet")
    anti_path = os.path.join(tmp_path, "anti.parquet")
    epi_res_path = os.path.join(tmp_path, "epi_res.parquet")
    dm_path = os.path.join(tmp_path, "dm.parquet")
    
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "combined_sparsity": [0.2, 0.8]
    }).to_parquet(sp_path)
    
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "antisymmetry_error": [0.1, 0.5]
    }).to_parquet(anti_path)
    
    pd.DataFrame({
        "pair_id": ["pair1"],
        "epistasis_error": [0.3]
    }).to_parquet(epi_res_path)
    
    pd.DataFrame({
        "pair_id": ["pair1"],
        "mutation_a": ["mut1"],
        "mutation_b": ["mut2"],
        "experimental_ddg_ab": [1.5]
    }).to_parquet(dm_path)
    
    builder = IntegratedDatasetBuilder()
    out_dir = os.path.join(tmp_path, "outputs")
    
    res_path = builder.run_pipeline(
        sparsity_path=sp_path,
        antisymmetry_path=anti_path,
        epistasis_results_path=epi_res_path,
        double_mutants_path=dm_path,
        output_dir=out_dir
    )
    
    assert os.path.exists(res_path)
    df_out = pd.read_parquet(res_path)
    
    assert len(df_out) == 2
    assert "record_id" in df_out.columns
    assert "epistasis_error" in df_out.columns
    
    # Since mut1 and mut2 are involved in pair1 (epistasis_error = 0.3), both should have an aggregated epistasis_error of 0.3
    m1_rec = df_out[df_out["mutation_id"] == "mut1"].iloc[0]
    m2_rec = df_out[df_out["mutation_id"] == "mut2"].iloc[0]
    
    assert np.isclose(m1_rec["epistasis_error"], 0.3)
    assert np.isclose(m2_rec["epistasis_error"], 0.3)

def test_pathway_analysis_exercises():
    """Verify pathway modeling regressions, Athlete Exercise, and Failure Injection Shuffling."""
    engine = PathwayAnalysisEngine()
    
    # Run Athlete Exercise (recovers linear relationship)
    assert bool(engine.run_athlete_exercise()) is True
    
    # Run Failure Injection on real dataset
    d401_path = "data/final/reliability/integrated_reliability_analysis.parquet"
    if os.path.exists(d401_path):
        real_df = pd.read_parquet(d401_path)
        assert bool(engine.run_failure_injection(real_df)) is True

def test_integrated_statistics_and_figures():
    """Verify that integrated statistics files and figures are correctly compiled on real data."""
    d401_path = "data/final/reliability/integrated_reliability_analysis.parquet"
    if os.path.exists(d401_path):
        stats_csv = run_statistics()
        assert os.path.exists(stats_csv)
        
        df_stats = pd.read_csv(stats_csv)
        assert len(df_stats) == 3
        
        plots = generate_figures()
        for p in plots:
            assert os.path.exists(p)
