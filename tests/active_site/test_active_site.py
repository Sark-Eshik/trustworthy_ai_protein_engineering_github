# tests/active_site/test_active_site.py
"""Comprehensive unit and integration tests for the Active-Site Control Analysis."""

import os
import pytest
import pandas as pd
import numpy as np

from src.active_site.calculate_distances import ActiveSiteAnalyzer

def test_distance_calculation():
    """Verify core geometric active-site distance calculations and coordinate-folding properties."""
    active_residues = [155, 263, 340]
    
    # Active site residue itself has distance exactly 0.0
    d_act = ActiveSiteAnalyzer.calculate_distance(155, active_residues)
    assert d_act == 0.0
    
    # Distance is positive for non-active site positions
    d_non_act = ActiveSiteAnalyzer.calculate_distance(156, active_residues)
    assert d_non_act > 0.0
    
    # Folded model: nearby is closer than distant
    d_nearby = ActiveSiteAnalyzer.calculate_distance(156, active_residues)
    d_distant = ActiveSiteAnalyzer.calculate_distance(590, active_residues)
    assert d_nearby < d_distant

def test_active_site_analyzer_exercises():
    """Verify that Athlete Exercises execute correctly on standard setup."""
    analyzer = ActiveSiteAnalyzer()
    active_residues = [155, 263, 340, 445, 500]
    
    # Verify standard check
    assert analyzer.run_athlete_exercises(active_residues, inject_fault=False) is True
    
    # Verify failure injection check (injects negative distance or calculation fault)
    assert analyzer.run_athlete_exercises(active_residues, inject_fault=True) is False

def test_active_site_pipeline_and_validation(tmp_path):
    """Verify end-to-end active site distance calculations and parquet generation."""
    # Create fake analysis parquet (D502)
    anal_path = os.path.join(tmp_path, "tkt_mutation_analysis.parquet")
    act_path = os.path.join(tmp_path, "active_site_residues.csv")
    out_dir = os.path.join(tmp_path, "out_active")
    
    pd.DataFrame({
        "mutation_id": ["TKT_M1A", "TKT_A100C", "TKT_D155V"],
        "evolutionary_sparsity": [0.1, 0.4, 0.7],
        "structural_sparsity": [0.2, 0.5, 0.8],
        "combined_sparsity": [0.15, 0.45, 0.75],
        "reliability_score": [0.85, 0.55, 0.25],
        "predicted_stability": [0.1, -1.2, -3.4]
    }).to_parquet(anal_path)
    
    pd.DataFrame({
        "residue_number": [155, 263],
        "residue_name": ["Asp", "His"],
        "functional_role": ["Catalytic Acceptor", "TPP Coord"],
        "literature_source": ["Lit_A", "Lit_B"]
    }).to_csv(act_path, index=False)
    
    analyzer = ActiveSiteAnalyzer()
    p_path, s_path, plot_path = analyzer.calculate_active_site_distances(
        analysis_path=anal_path,
        active_site_path=act_path,
        output_dir=out_dir
    )
    
    assert os.path.exists(p_path)
    assert os.path.exists(s_path)
    assert os.path.exists(plot_path)
    
    # Read output and verify columns
    df_out = pd.read_parquet(p_path)
    assert len(df_out) == 3
    assert list(df_out.columns) == ["mutation_id", "distance_to_active_site", "combined_sparsity", "reliability_score", "predicted_stability"]
    
    # TKT_D155V is on position 155 (an active-site residue) -> distance should be 0.0
    rec_155 = df_out[df_out["mutation_id"] == "TKT_D155V"].iloc[0]
    assert rec_155["distance_to_active_site"] == 0.0
    
    # Others are positive
    rec_1 = df_out[df_out["mutation_id"] == "TKT_M1A"].iloc[0]
    assert rec_1["distance_to_active_site"] > 0.0

def test_failure_injection_output(tmp_path):
    """Verify that failure injection (negative distance values) is caught by validation engines."""
    anal_path = os.path.join(tmp_path, "tkt_mutation_analysis.parquet")
    act_path = os.path.join(tmp_path, "active_site_residues.csv")
    out_dir = os.path.join(tmp_path, "out_active")
    
    pd.DataFrame({
        "mutation_id": ["TKT_M1A"],
        "evolutionary_sparsity": [0.1],
        "structural_sparsity": [0.2],
        "combined_sparsity": [0.15],
        "reliability_score": [0.85],
        "predicted_stability": [0.1]
    }).to_parquet(anal_path)
    
    pd.DataFrame({
        "residue_number": [155],
        "residue_name": ["Asp"],
        "functional_role": ["Catalytic Acceptor"],
        "literature_source": ["Lit_A"]
    }).to_csv(act_path, index=False)
    
    analyzer = ActiveSiteAnalyzer()
    
    # If inject_fault is True, it will fail validation check or athlete check
    with pytest.raises(ValueError):
        analyzer.calculate_active_site_distances(
            analysis_path=anal_path,
            active_site_path=act_path,
            output_dir=out_dir,
            inject_fault=True
        )
