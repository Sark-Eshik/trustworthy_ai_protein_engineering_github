# tests/sparsity/test_evolutionary_sparsity.py
"""Unit and validation tests for the Evolutionary Sparsity module."""

import os
import pytest
import pandas as pd
import numpy as np

from src.sparsity.evolutionary.esm_probability_engine import ESMProbabilityEngine
from src.sparsity.evolutionary.evolutionary_sparsity import EvolutionarySparsity
from src.sparsity.evolutionary.evolutionary_validation import EvolutionaryValidation


def test_esm_probability_engine_validation():
    """Verify position checks and wildtype mismatch checks in ESMProbabilityEngine."""
    engine = ESMProbabilityEngine()
    sequence = "MAPAAAPAAAPAAAAGAAAA"  # Length 20

    # 1. Valid scoring check
    prob = engine.score_mutation(sequence, 12, "A", "V", "pA")
    assert 0.0 < prob < 1.0

    # 2. Out of bounds check
    with pytest.raises(ValueError, match="out of bounds"):
        engine.score_mutation(sequence, 0, "M", "A", "pA")
    with pytest.raises(ValueError, match="out of bounds"):
        engine.score_mutation(sequence, 21, "A", "V", "pA")

    # 3. Wildtype mismatch check
    with pytest.raises(ValueError, match="Mismatch in wildtype"):
        engine.score_mutation(sequence, 12, "G", "V", "pA")  # position 12 is 'A', not 'G'

    # 4. Identity change (no change) returns 1.0
    prob_same = engine.score_mutation(sequence, 12, "A", "A", "pA")
    assert prob_same == 1.0


def test_esm_probability_engine_determinism():
    """Verify that the mock scoring is deterministic and chemically plausible."""
    engine = ESMProbabilityEngine()
    sequence = "MAPAAAPAAAPAAAAGAAAA"

    # Same input must yield identical scores
    p1 = engine.score_mutation(sequence, 12, "A", "V", "pA")
    p2 = engine.score_mutation(sequence, 12, "A", "V", "pA")
    assert p1 == p2

    # Hydrophobic substitution (A -> V) should have higher probability (lower sparsity)
    # than hydrophobic to charged negative (A -> D)
    p_hydrophobic = engine.score_mutation(sequence, 12, "A", "V", "pA")
    p_charged = engine.score_mutation(sequence, 12, "A", "D", "pA")
    assert p_hydrophobic > p_charged


def test_evolutionary_sparsity_pipeline(tmp_path):
    """Verify the calculation, log-scaling, and normalization of evolutionary sparsity."""
    # Create temporary input files
    seq_path = os.path.join(tmp_path, "seq.parquet")
    mut_path = os.path.join(tmp_path, "mut.parquet")

    pd.DataFrame({
        "protein_id": ["p1"],
        "sequence": ["MAPAAAPAAAPAAAAGAAAA"],  # Length 20
        "sequence_length": [20],
    }).to_parquet(seq_path)

    pd.DataFrame({
        "mutation_id": ["mutA", "mutB", "mutC"],
        "protein_id": ["p1", "p1", "p1"],
        "position": [12, 12, 12],
        "wildtype": ["A", "A", "A"],
        "mutant": ["V", "D", "C"],  # V is hydrophobic, D is charged, C is cysteine change (extremely rare)
        "experimental_ddg": [0.1, 1.2, 3.4],
    }).to_parquet(mut_path)

    pipeline = EvolutionarySparsity()
    out_dir = os.path.join(tmp_path, "results")
    s_path, p_path, plot = pipeline.run_pipeline(
        sequence_path_override=seq_path,
        mutations_path_override=mut_path,
        output_dir=out_dir,
    )

    assert os.path.exists(s_path)
    assert os.path.exists(p_path)
    assert os.path.exists(plot)

    # Load outputs and verify math properties
    df = pd.read_parquet(s_path)
    assert len(df) == 3
    assert "evolutionary_sparsity" in df.columns
    assert "esm_probability" in df.columns
    assert "log_probability" in df.columns

    # Verify normalization boundaries
    assert np.isclose(df["evolutionary_sparsity"].min(), 0.0)
    assert np.isclose(df["evolutionary_sparsity"].max(), 1.0)

    # Verify logical progression (P_V > P_D > P_C => Sparsity_V < Sparsity_D < Sparsity_C)
    v_rec = df.loc[df["mutation_id"] == "mutA"].iloc[0]
    d_rec = df.loc[df["mutation_id"] == "mutB"].iloc[0]
    c_rec = df.loc[df["mutation_id"] == "mutC"].iloc[0]

    assert v_rec["esm_probability"] > d_rec["esm_probability"] > c_rec["esm_probability"]
    assert v_rec["evolutionary_sparsity"] < d_rec["evolutionary_sparsity"] < c_rec["evolutionary_sparsity"]

    # Verify that the scientific validator certifies this generated file
    validator = EvolutionaryValidation()
    assert validator.validate_dataset(s_path) is True


def test_evolutionary_validation_anomalies(tmp_path):
    """Verify that validation script detects schema, range, and scientific violations."""
    validator = EvolutionaryValidation()

    # 1. Range Violation [0, 1] on Sparsity
    bad_range_path = os.path.join(tmp_path, "bad_range.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "esm_probability": [0.1, 0.9],
        "log_probability": [2.3, 0.1],
        "evolutionary_sparsity": [1.5, -0.2],  # Out of bounds!
    }).to_parquet(bad_range_path)
    assert validator.validate_dataset(bad_range_path) is False

    # 2. Scientific Ordering Mismatch (lower probability should have larger sparsity)
    bad_order_path = os.path.join(tmp_path, "bad_order.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "esm_probability": [0.1, 0.9],  # mut1 has smaller probability
        "log_probability": [2.3, 0.105],
        "evolutionary_sparsity": [0.0, 1.0],  # Violates ordering! mut1 (rarer) gets 0.0, mut2 (commoner) gets 1.0
    }).to_parquet(bad_order_path)
    assert validator.validate_dataset(bad_order_path) is False

    # 3. Normalization Boundary Error (Max prob must be 0.0, Min prob must be 1.0)
    bad_norm_path = os.path.join(tmp_path, "bad_norm.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "esm_probability": [0.1, 0.9],
        "log_probability": [2.3, 0.105],
        "evolutionary_sparsity": [0.95, 0.05],  # No 0.0 or 1.0!
    }).to_parquet(bad_norm_path)
    assert validator.validate_dataset(bad_norm_path) is False
