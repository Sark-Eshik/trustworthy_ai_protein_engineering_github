# src/sparsity/structural/structural_validation.py
"""Structural Sparsity scientific validation and certification script.

Verifies that the generated structural_sparsity.parquet meets all scientific criteria:
1. Schema matches D103 definition.
2. Value bounds are valid (SASA >= 0, normalized SASA and sparsities in [0, 1]).
3. No duplicate keys or null/missing values.
4. Correct scientific ranking (higher SASA = lower structural sparsity).
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import pandas as pd

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.validation_engine import ValidationEngine


class StructuralValidation:
    """Validator to verify and certify structural sparsity data properties."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the structural sparsity validator.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="structural_validation",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_dataset(self, file_path: str) -> bool:
        """Performs scientific and structural validation checks on structural sparsity output.

        Parameters
        ----------
        file_path : str
            Path to the structural_sparsity.parquet file.

        Returns
        -------
        bool
            True if all checks pass, False otherwise.
        """
        self.logger.info(f"Initiating scientific validation on structural file: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Target file does not exist: {file_path}")
            print(f"Error: Target file does not exist: {file_path}", file=sys.stderr)
            return False

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self.logger.error(f"Failed to read parquet file: {e}")
            print(f"Error: Failed to read parquet file: {e}", file=sys.stderr)
            return False

        # --- Check 1: Structural Schema Validation via ValidationEngine (D103) ---
        schema_report = self.validation_engine.validate_schema("D103", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            print("Validation Failures (Schema):", file=sys.stderr)
            for err in schema_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 2: Value Range Constraints ---
        range_rules = {
            "sasa": (0.0, 5000.0),  # Upper bound of 5000.0 is safe for any single residue total area
            "normalized_sasa": (0.0, 1.0),
            "structural_sparsity": (0.0, 1.0),
        }
        range_report = self.validation_engine.validate_ranges("D103", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            print("Validation Failures (Value Ranges):", file=sys.stderr)
            for err in range_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 3: Scientific Ordering and Sanity Checks ---
        # Verify that as SASA increases, Structural Sparsity decreases (exposed residue = low sparsity)
        sorted_df = df.sort_values(by="sasa")
        sasa_vals = sorted_df["sasa"].values
        sparsities = sorted_df["structural_sparsity"].values

        # If sasa[i] < sasa[j], then sparsity[i] must be >= sparsity[j]
        failures = 0
        for i in range(len(sorted_df) - 1):
            if sasa_vals[i] < sasa_vals[i + 1] and sparsities[i] < sparsities[i + 1]:
                self.logger.error(
                    f"Scientific anomaly: Mutation with lower SASA {sasa_vals[i]} has lower "
                    f"sparsity ({sparsities[i]}) than mutation with higher SASA {sasa_vals[i + 1]} "
                    f"({sparsities[i + 1]})."
                )
                failures += 1

        if failures > 0:
            print("Validation Failures (Scientific Ordering):", file=sys.stderr)
            print("  * Structural sparsity must inversely correlate with SASA.", file=sys.stderr)
            return False

        # --- Check 4: Normalization Boundaries ---
        # The mutation with maximum SASA must have normalized_sasa = 1.0 and structural_sparsity = 0.0
        max_sasa_idx = df["sasa"].idxmax()
        max_sasa_rec = df.loc[max_sasa_idx]

        if abs(max_sasa_rec["normalized_sasa"] - 1.0) > 1e-6 or abs(max_sasa_rec["structural_sparsity"] - 0.0) > 1e-6:
            self.logger.error(
                f"Normalization error: Maximum SASA mutation ({max_sasa_rec['mutation_id']}) "
                f"has normalized_sasa = {max_sasa_rec['normalized_sasa']} "
                f"and structural_sparsity = {max_sasa_rec['structural_sparsity']}. Expected 1.0 and 0.0."
            )
            print("Validation Failures (Normalization Boundary):", file=sys.stderr)
            print("  * Mutation with maximum SASA must have normalized SASA of 1.0 and sparsity of 0.0.", file=sys.stderr)
            return False

        self.logger.info("Structural sparsity scientific validation completed successfully. Certification status: PASSED.")
        print("Structural Sparsity certified successfully.")
        return True


def main() -> None:
    """Main CLI entry point for Structural Sparsity validation."""
    parser = argparse.ArgumentParser(
        description="Verify scientific and structural consistency of structural sparsity output."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="results/structural/structural_sparsity.parquet",
        help="Path to generated structural sparsity parquet file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    validator = StructuralValidation(config_path=args.config)
    success = validator.validate_dataset(args.file)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
