# tests/sparsity/test_structural_sparsity.py
"""Unit and validation tests for the Structural Sparsity module."""

import os
import pytest
import pandas as pd
import numpy as np

from src.sparsity.structural.sasa_engine import SASAEngine
from src.sparsity.structural.structural_sparsity import StructuralSparsity
from src.sparsity.structural.structural_validation import StructuralValidation
from src.validation.validate_structures import StructureValidator


def test_sasa_engine_mock_generation():
    """Verify that SASAEngine creates deteministic and biochemically plausible mock values."""
    engine = SASAEngine()
    engine.is_mock_active = True

    # Same path must yield identical results (determinism)
    res1 = engine.compute_sasa("dummy_path_1.pdb")
    res2 = engine.compute_sasa("dummy_path_1.pdb")
    assert res1 == res2

    # Different paths should yield different results
    res3 = engine.compute_sasa("dummy_path_2.pdb")
    assert res1 != res3

    # Check structural layout (e.g. terminal residues should have higher SASA than core)
    # Inside mock, sequence is 100 long.
    # Residue 1 (terminal) vs Residue 50 (core center)
    sasa_terminal = res1["A"]["1"]
    sasa_core = res1["A"]["50"]
    assert sasa_terminal > sasa_core


def test_sasa_engine_real_freesasa(tmp_path):
    """Test SASAEngine running with a real, valid PDB file."""
    pdb_path = os.path.join(tmp_path, "sample.pdb")
    pdb_lines = [
        "ATOM      1  N   ALA A  12      10.000  10.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A  12      11.000  11.000  11.000  1.00 20.00           C",
        "ATOM      3  C   ALA A  12      12.000  12.000  12.000  1.00 20.00           C",
        "ATOM      4  O   ALA A  12      13.000  13.000  13.000  1.00 20.00           O",
        "ATOM      5  N   GLY A  14      15.000  15.000  15.000  1.00 20.00           N",
        "ATOM      6  CA  GLY A  14      16.000  16.000  16.000  1.00 20.00           C",
        "ATOM      7  C   GLY A  14      17.000  17.000  17.000  1.00 20.00           C",
        "ATOM      8  O   GLY A  14      18.000  18.000  18.000  1.00 20.00           O",
        "END"
    ]
    with open(pdb_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pdb_lines))

    engine = SASAEngine()
    engine.is_mock_active = False

    sasa_dict = engine.compute_sasa(pdb_path)
    assert "A" in sasa_dict
    assert "12" in sasa_dict["A"]
    assert "14" in sasa_dict["A"]
    assert sasa_dict["A"]["12"] > 0.0


def test_structural_sparsity_pipeline(tmp_path):
    """Verify loading metadata, processing SASA, normalizing, and calculating structural sparsity."""
    # Write mock PDBs
    pdb_a = os.path.join(tmp_path, "pA.pdb")
    pdb_b = os.path.join(tmp_path, "pB.pdb")

    # Both write minimal valid PDB content
    pdb_lines = [
        "ATOM      1  N   ALA A  12      10.000  10.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A  12      11.000  11.000  11.000  1.00 20.00           C",
        "ATOM      3  C   ALA A  12      12.000  12.000  12.000  1.00 20.00           C",
        "ATOM      4  O   ALA A  12      13.000  13.000  13.000  1.00 20.00           O",
        "ATOM      5  N   GLY A  14      15.000  15.000  15.000  1.00 20.00           N",
        "ATOM      6  CA  GLY A  14      16.000  16.000  16.000  1.00 20.00           C",
        "ATOM      7  C   GLY A  14      17.000  17.000  17.000  1.00 20.00           C",
        "ATOM      8  O   GLY A  14      18.000  18.000  18.000  1.00 20.00           O",
        "END"
    ]
    with open(pdb_a, "w", encoding="utf-8") as f:
        f.write("\n".join(pdb_lines))
    with open(pdb_b, "w", encoding="utf-8") as f:
        f.write("\n".join(pdb_lines))

    # Write structures metadata S003
    struct_path = os.path.join(tmp_path, "structures.parquet")
    pd.DataFrame({
        "protein_id": ["pA", "pB"],
        "pdb_id": ["pdbA", "pdbB"],
        "chain_id": ["A", "A"],
        "structure_path": [pdb_a, pdb_b],
    }).to_parquet(struct_path)

    # Write mutations dataset S001
    mut_path = os.path.join(tmp_path, "mutations.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "protein_id": ["pA", "pA"],
        "position": [12, 14],
        "wildtype": ["A", "G"],
        "mutant": ["V", "C"],
        "experimental_ddg": [1.2, -0.4],
    }).to_parquet(mut_path)

    pipeline = StructuralSparsity()
    # Force real engine (test_sasa_engine_real_freesasa verified it works)
    pipeline.sasa_engine.is_mock_active = False

    output_dir = os.path.join(tmp_path, "results")
    s_path, v_path, plot = pipeline.run_pipeline(
        structures_path_override=struct_path,
        mutations_path_override=mut_path,
        output_dir=output_dir,
    )

    assert os.path.exists(s_path)
    assert os.path.exists(v_path)
    assert os.path.exists(plot)

    # Verify structural sparsity bounds and columns
    df = pd.read_parquet(s_path)
    assert len(df) == 2
    assert "sasa" in df.columns
    assert "normalized_sasa" in df.columns
    assert "structural_sparsity" in df.columns

    # Verify normalization boundaries
    assert np.isclose(df["normalized_sasa"].max(), 1.0)
    assert np.isclose(df["structural_sparsity"].min(), 0.0)

    # Verify scientific validator certifies this generated file
    validator = StructuralValidation()
    assert validator.validate_dataset(s_path) is True


def test_structural_validation_violations(tmp_path):
    """Verify that scientific validator correctly reports schema, range, and ordering errors."""
    validator = StructuralValidation()

    # 1. Range Violation [0, 1] on Structural Sparsity
    bad_range_path = os.path.join(tmp_path, "bad_range.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "sasa": [10.0, 100.0],
        "normalized_sasa": [0.1, 1.0],
        "structural_sparsity": [1.5, -0.2],  # Out of bounds!
    }).to_parquet(bad_range_path)
    assert validator.validate_dataset(bad_range_path) is False

    # 2. Scientific Ordering Mismatch (higher SASA should yield lower structural sparsity)
    bad_order_path = os.path.join(tmp_path, "bad_order.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "sasa": [10.0, 100.0],  # mut1 has smaller SASA (more buried)
        "normalized_sasa": [0.1, 1.0],
        "structural_sparsity": [0.1, 0.9],  # Violates ordering! mut1 gets smaller sparsity (more exposed)
    }).to_parquet(bad_order_path)
    assert validator.validate_dataset(bad_order_path) is False

    # 3. Normalization Boundary Error (Max SASA must map to 1.0 normalized SASA and 0.0 sparsity)
    bad_norm_path = os.path.join(tmp_path, "bad_norm.parquet")
    pd.DataFrame({
        "mutation_id": ["mut1", "mut2"],
        "sasa": [10.0, 100.0],
        "normalized_sasa": [0.08, 0.95],  # No 1.0 normalized SASA!
        "structural_sparsity": [0.92, 0.05],  # No 0.0 structural sparsity!
    }).to_parquet(bad_norm_path)
    assert validator.validate_dataset(bad_norm_path) is False


def test_structure_validator(tmp_path):
    """Verify that structure validator correctly certifies valid PDB files and rejects invalid ones."""
    # Write a valid PDB
    valid_pdb = os.path.join(tmp_path, "valid.pdb")
    valid_lines = [
        "ATOM      1  CA  ALA A  12      10.000  10.000  10.000  1.00 20.00           C",
        "END"
    ]
    with open(valid_pdb, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines))

    # Write an invalid PDB (missing ATOM lines)
    invalid_pdb_atoms = os.path.join(tmp_path, "invalid_atoms.pdb")
    with open(invalid_pdb_atoms, "w", encoding="utf-8") as f:
        f.write("HEADER    PROTEIN TRANSKETOLASE\nEND")

    # Write S003 structures list parquet
    struct_parquet = os.path.join(tmp_path, "structures.parquet")
    pd.DataFrame({
        "protein_id": ["valid_p", "invalid_p"],
        "pdb_id": ["VAL1", "INV1"],
        "chain_id": ["A", "A"],
        "structure_path": [valid_pdb, invalid_pdb_atoms],
    }).to_parquet(struct_parquet)

    validator = StructureValidator()
    # Should fail due to the invalid structure
    assert validator.validate_structures(struct_parquet) is False

    # Now write a clean parquet with only the valid PDB
    clean_struct_parquet = os.path.join(tmp_path, "clean_structures.parquet")
    pd.DataFrame({
        "protein_id": ["valid_p"],
        "pdb_id": ["VAL1"],
        "chain_id": ["A"],
        "structure_path": [valid_pdb],
    }).to_parquet(clean_struct_parquet)

    # Should pass successfully with only the valid structure
    assert validator.validate_structures(clean_struct_parquet) is True
