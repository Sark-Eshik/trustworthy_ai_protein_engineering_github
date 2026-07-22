# tests/tkt/test_tkt.py
"""Comprehensive unit and integration tests for the complete TKT Mutation Landscape."""

import os
import pytest
import pandas as pd
import numpy as np

from src.tkt.mutation_enumerator import MutationEnumerator
from src.tkt.landscape_generator import LandscapeGenerator
from src.tkt.landscape_analysis import LandscapeAnalyzer

def test_mutation_enumerator():
    """Verify that MutationEnumerator generates exactly 19 substitutions per position."""
    enumerator = MutationEnumerator()
    seq = "MAP" # Length 3
    mutations = enumerator.enumerate_mutations(seq, "TEST")
    
    # Expected: 3 positions * 19 = 57 substitutions
    assert len(mutations) == 57
    
    # Verify fields of a single record
    first = mutations[0]
    assert "mutation_id" in first
    assert first["protein_id"] == "TEST"
    assert first["position"] == 1
    assert first["wildtype"] == "M"
    assert first["mutant"] in enumerator.STANDARD_AAS
    assert first["mutant"] != "M"
    assert first["mutation_label"] == f"M1{first['mutant']}"

def test_landscape_generator_and_athlete_exercises(tmp_path):
    """Verify registration, enumeration, and Athlete Exercises for TKT Landscape Generation."""
    # Write a custom dummy config or override paths
    generator = LandscapeGenerator()
    
    # 1. Register TKT assets if missing
    tkt_seq = generator.register_tkt_assets_if_missing()
    assert len(tkt_seq) == 600
    assert tkt_seq[0] == "M"
    assert tkt_seq[99] == "A"
    
    # Check that they exist in our parquet files
    seq_df = pd.read_parquet("data/raw/sequences/protein_sequences.parquet")
    assert "TKT" in seq_df["protein_id"].values
    
    struct_df = pd.read_parquet("data/raw/structures/protein_structures.parquet")
    assert "TKT" in struct_df["protein_id"].values
    assert os.path.exists("data/raw/structures/TKT.pdb")

    # 2. Run landscape generation pipeline into a temp directory
    out_dir = os.path.join(tmp_path, "tkt_out")
    p_path, r_path = generator.generate_landscape(output_dir=out_dir)
    
    assert os.path.exists(p_path)
    assert os.path.exists(r_path)
    
    df_land = pd.read_parquet(p_path)
    assert len(df_land) == 11400  # 600 * 19
    
    # Check columns
    assert list(df_land.columns) == ["mutation_id", "protein_id", "position", "wildtype", "mutant", "mutation_label"]

def test_landscape_generator_failure_injection(tmp_path):
    """Verify that deleting a position triggers an Athlete Exercise validation failure."""
    generator = LandscapeGenerator()
    out_dir = os.path.join(tmp_path, "tkt_out_fault")
    
    # Running with inject_fault=True should raise ValueError
    with pytest.raises(ValueError, match="Mutation landscape failed Athlete Exercise validation checks."):
        generator.generate_landscape(output_dir=out_dir, inject_fault=True)

def test_landscape_analysis(tmp_path):
    """Verify that landscape analysis computes sparsities, reliability scores, and simulated stability."""
    analyzer = LandscapeAnalyzer()
    out_dir = os.path.join(tmp_path, "tkt_out_analysis")
    
    # We can analyze the real landscape we already generated
    real_land = "data/final/tkt/tkt_single_mutation_landscape.parquet"
    if os.path.exists(real_land):
        out_path = analyzer.analyze_landscape(landscape_path=real_land, output_dir=out_dir)
        assert os.path.exists(out_path)
        
        df_out = pd.read_parquet(out_path)
        assert len(df_out) == 11400
        
        # Check required columns (D502 schema)
        required_cols = [
            "mutation_id",
            "evolutionary_sparsity",
            "structural_sparsity",
            "combined_sparsity",
            "reliability_score",
            "predicted_stability",
        ]
        for col in required_cols:
            assert col in df_out.columns
            assert not df_out[col].isnull().any()
            
        # Check range bounds
        assert df_out["evolutionary_sparsity"].min() >= 0.0
        assert df_out["evolutionary_sparsity"].max() <= 1.0
        assert df_out["structural_sparsity"].min() >= 0.0
        assert df_out["structural_sparsity"].max() <= 1.0
        assert df_out["combined_sparsity"].min() >= 0.0
        assert df_out["combined_sparsity"].max() <= 1.0
        assert df_out["reliability_score"].min() >= 0.0
        assert df_out["reliability_score"].max() <= 1.0
        
        # Verify Combined = (Evo + Struct) / 2
        sample = df_out.iloc[0]
        expected_comb = (sample["evolutionary_sparsity"] + sample["structural_sparsity"]) / 2.0
        assert np.isclose(sample["combined_sparsity"], expected_comb)
        
        # Verify Reliability = 1.0 - Combined
        expected_rel = 1.0 - sample["combined_sparsity"]
        assert np.isclose(sample["reliability_score"], expected_rel)
