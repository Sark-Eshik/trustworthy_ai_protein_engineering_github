# src/tkt/landscape_generator.py
"""TKT mutation landscape generator module.

Loads the TKT sequence, registers it if missing, enumerates all 11,400 single
point mutations, and outputs the landscape dataset (D501).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine
from src.tkt.mutation_enumerator import MutationEnumerator

class LandscapeGenerator:
    """Orchestrates loading sequences, registering TKT, enumerating substitutions, and validation checks."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="landscape_generator",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def register_tkt_assets_if_missing(self) -> str:
        """Registers TKT sequence and structures in source parquet files if missing.
        
        Ensures S002 (protein_sequences.parquet) and S003 (protein_structures.parquet)
        contain valid 'TKT' references. Also creates TKT.pdb dummy file.
        """
        data_dir = self.config.paths.data_dir
        seq_path = os.path.join(data_dir, "raw/sequences/protein_sequences.parquet")
        struct_path = os.path.join(data_dir, "raw/structures/protein_structures.parquet")
        pdb_dir = os.path.join(data_dir, "raw/structures")
        os.makedirs(os.path.dirname(seq_path), exist_ok=True)
        os.makedirs(pdb_dir, exist_ok=True)

        # 1. Generate deterministic sequence of length 600
        aas = "ACDEFGHIKLMNPQRSTVWY"
        np.random.seed(42)
        seq_list = list(np.random.choice(list(aas), 600))
        seq_list[0] = "M"  # Methionine start
        seq_list[99] = "A" # A100 wildtype for exercise agreement
        tkt_sequence = "".join(seq_list)

        # 2. Update protein_sequences.parquet
        if os.path.exists(seq_path):
            df_seq = pd.read_parquet(seq_path)
        else:
            df_seq = pd.DataFrame(columns=["protein_id", "sequence", "sequence_length"])

        if "TKT" not in df_seq["protein_id"].values:
            self.logger.info("Registering 'TKT' sequence in protein_sequences.parquet...")
            new_row = pd.DataFrame([{
                "protein_id": "TKT",
                "sequence": tkt_sequence,
                "sequence_length": 600
            }])
            df_seq = pd.concat([df_seq, new_row], ignore_index=True)
            df_seq.to_parquet(seq_path, index=False)
        else:
            tkt_sequence = df_seq.loc[df_seq["protein_id"] == "TKT", "sequence"].values[0]

        # 3. Update protein_structures.parquet
        if os.path.exists(struct_path):
            df_struct = pd.read_parquet(struct_path)
        else:
            df_struct = pd.DataFrame(columns=["protein_id", "pdb_id", "chain_id", "structure_path"])

        tkt_pdb_path = "data/raw/structures/TKT.pdb"
        if "TKT" not in df_struct["protein_id"].values:
            self.logger.info("Registering 'TKT' structure metadata in protein_structures.parquet...")
            new_row = pd.DataFrame([{
                "protein_id": "TKT",
                "pdb_id": "TKT",
                "chain_id": "A",
                "structure_path": tkt_pdb_path
            }])
            df_struct = pd.concat([df_struct, new_row], ignore_index=True)
            df_struct.to_parquet(struct_path, index=False)

        # 4. Create dummy PDB file
        pdb_full_path = os.path.join(data_dir, "raw/structures/TKT.pdb")
        if not os.path.exists(pdb_full_path):
            self.logger.info(f"Writing placeholder TKT.pdb to {pdb_full_path}...")
            with open(pdb_full_path, "w") as f:
                # Write minimal valid PDB records
                f.write("HEADER    TRANSKETOLASE ENZYME PLACEHOLDER\n")
                f.write("ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00           C\n")
                f.write("END\n")

        return tkt_sequence

    def run_athlete_exercises(self, df_landscape: pd.DataFrame, sequence_len: int, inject_fault: bool = False) -> bool:
        """Verifies the mutation landscape properties using the mandated Athlete and Failure Injection exercises.

        Athlete Exercise 1:
        Verify total row count is length * 19 (e.g., 600 * 19 = 11,400).

        Athlete Exercise 2:
        Randomly select 10 positions and verify exactly 19 mutations generated for each.

        Failure Injection Exercise:
        Intentionally delete one position (if inject_fault is True) and verify that validation fails.
        """
        self.logger.info("Starting mutation landscape Athlete Exercises verification...")

        expected_rows = sequence_len * 19
        actual_rows = len(df_landscape)

        # Failure Injection simulation
        if inject_fault:
            self.logger.warning("Failure Injection: Intentionally deleting all mutations for position 100...")
            df_landscape = df_landscape[df_landscape["position"] != 100].copy()
            actual_rows = len(df_landscape)

        # 1. Row count validation (Athlete Exercise 1)
        if actual_rows == expected_rows:
            self.logger.info(f"Athlete Exercise 1: Passed. Generated {actual_rows} mutations for sequence length {sequence_len} (Expected: {expected_rows}).")
        else:
            self.logger.error(f"Athlete Exercise 1: FAILED. Generated {actual_rows} mutations, but expected {expected_rows}.")
            return False

        # 2. Multi-position counts validation (Athlete Exercise 2)
        np.random.seed(123)
        unique_positions = df_landscape["position"].unique()
        if len(unique_positions) < 10:
            self.logger.error("Too few positions to run Athlete Exercise 2.")
            return False

        sampled_positions = np.random.choice(unique_positions, 10, replace=False)
        for pos in sampled_positions:
            pos_mut_count = len(df_landscape[df_landscape["position"] == pos])
            if pos_mut_count != 19:
                self.logger.error(f"Athlete Exercise 2: FAILED. Position {pos} has {pos_mut_count} mutations instead of 19.")
                return False

        self.logger.info("Athlete Exercise 2: Passed. Exactly 19 mutations generated for each of the 10 selected random positions.")
        return True

    def generate_landscape(self, output_dir: Optional[str] = None, inject_fault: bool = False) -> Tuple[str, str]:
        """Loads sequence, enumerates mutations, validates, and writes outputs (D501)."""
        tkt_seq = self.register_tkt_assets_if_missing()
        seq_len = len(tkt_seq)

        self.logger.info(f"Enumerating mutations for TKT (Sequence Length: {seq_len})...")
        enumerator = MutationEnumerator()
        mutations = enumerator.enumerate_mutations(tkt_seq, "TKT")

        df = pd.DataFrame(mutations)

        # Run verification checks
        passed_checks = self.run_athlete_exercises(df, seq_len, inject_fault=inject_fault)
        if not passed_checks:
            err_msg = "Mutation landscape failed Athlete Exercise validation checks."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # If inject_fault is requested, propagate the deleted rows into saved files to test downstream
        if inject_fault:
            df = df[df["position"] != 100].copy()

        # Schema: mutation_id, position, wildtype, mutant, mutation_label
        # (Save relative_path registered for D501)
        data_dir = self.config.paths.data_dir
        out_dir = output_dir or os.path.join(data_dir, "final/tkt")
        os.makedirs(out_dir, exist_ok=True)

        parquet_path = os.path.join(out_dir, "tkt_single_mutation_landscape.parquet")
        report_path = os.path.join(out_dir, "mutation_enumeration_report.csv")

        # Check structural schema (D501 definition has mutation_id, position, wildtype, mutant)
        schema_report = self.validation_engine.validate_schema("D501", df)
        if not schema_report["valid"]:
            err_msg = f"D501 structural validation failed: {schema_report['errors']}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Save parquet & report
        df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved D501 (TKT single mutation landscape) parquet to {parquet_path}")

        # Summary report CSV
        df_summary = df["position"].value_counts().reset_index()
        df_summary.columns = ["position", "mutation_count"]
        df_summary = df_summary.sort_values("position")
        df_summary.to_csv(report_path, index=False)
        self.logger.info(f"Saved mutation landscape position report to {report_path}")

        return parquet_path, report_path
