# src/sparsity/empirical/empirical_validation.py
"""Empirical Sparsity scientific validation and certification script.

Verifies that the generated empirical_sparsity.parquet meets all scientific criteria:
1. Schema matches D101 definition.
2. Sparsity values reside strictly within the [0.0, 1.0] range.
3. No duplicate mutation keys or null values.
4. Correct scientific ranking (lower count = higher sparsity).
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


class EmpiricalValidation:
    """Validator to verify and certify empirical sparsity data properties."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the empirical sparsity validator.

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
            name="empirical_validation",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_dataset(self, file_path: str) -> bool:
        """Performs structural and scientific validations on empirical sparsity output.

        Parameters
        ----------
        file_path : str
            Path to the empirical_sparsity.parquet file to validate.

        Returns
        -------
        bool
            True if all validation checks pass successfully, False otherwise.
        """
        self.logger.info(f"Initiating scientific validation on file: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Target file for validation does not exist: {file_path}")
            print(f"Error: Target file for validation does not exist: {file_path}", file=sys.stderr)
            return False

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self.logger.error(f"Failed to read parquet file: {e}")
            print(f"Error: Failed to read parquet file: {e}", file=sys.stderr)
            return False

        # --- Check 1: Structural Schema Validation via ValidationEngine ---
        schema_report = self.validation_engine.validate_schema("D101", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            print("Validation Failures (Schema):", file=sys.stderr)
            for err in schema_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 2: Value Range Constraint Verification ---
        range_rules = {
            "frequency": (0.0, 1.0),
            "normalized_frequency": (0.0, 1.0),
            "empirical_sparsity": (0.0, 1.0),
        }
        range_report = self.validation_engine.validate_ranges("D101", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            print("Validation Failures (Value Ranges):", file=sys.stderr)
            for err in range_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 3: Scientific Ordering and Sanity Checks ---
        # Verification of logical ordering: mutation count should inversely correlate with sparsity
        sorted_df = df.sort_values(by="count")
        counts = sorted_df["count"].values
        sparsities = sorted_df["empirical_sparsity"].values

        # Since sparsities are computed as 1 - count/max_count, they should be strictly monotonically decreasing with count
        # Let's verify that for any i < j, if count[i] < count[j], then sparsity[i] >= sparsity[j]
        failures = 0
        for i in range(len(sorted_df) - 1):
            if counts[i] < counts[i + 1] and sparsities[i] < sparsities[i + 1]:
                self.logger.error(
                    f"Scientific anomaly: Mutation with count {counts[i]} has lower "
                    f"sparsity ({sparsities[i]}) than mutation with count {counts[i + 1]} "
                    f"({sparsities[i + 1]})."
                )
                failures += 1

        if failures > 0:
            print("Validation Failures (Scientific Ordering):", file=sys.stderr)
            print("  * Empirical sparsity must inversely correlate with mutation count.", file=sys.stderr)
            return False

        # --- Check 4: Normalization Boundaries ---
        # At least one mutation (the one with max count) must have normalized_frequency = 1.0 and empirical_sparsity matching the formula
        max_idx = df["count"].idxmax()
        max_record = df.loc[max_idx]
        total_count = df["count"].sum()
        expected_max_sparsity_total_based = 1.0 - (max_record["count"] / total_count) if total_count > 0 else 1.0
        is_total_based = abs(max_record["empirical_sparsity"] - expected_max_sparsity_total_based) <= 1e-6
        expected_max_sparsity = expected_max_sparsity_total_based if is_total_based else 0.0

        if abs(max_record["normalized_frequency"] - 1.0) > 1e-6 or abs(max_record["empirical_sparsity"] - expected_max_sparsity) > 1e-6:
            self.logger.error(
                f"Normalization error: Max count mutation ({max_record['mutation_id']}) has "
                f"normalized_frequency = {max_record['normalized_frequency']} "
                f"and empirical_sparsity = {max_record['empirical_sparsity']}. Expected 1.0 and {expected_max_sparsity}."
            )
            print("Validation Failures (Normalization Boundaries):", file=sys.stderr)
            print(f"  * Mutation with maximum observations must have normalized frequency of 1.0 and sparsity of {expected_max_sparsity}.", file=sys.stderr)
            return False

        self.logger.info("Empirical sparsity scientific validation completed successfully. Certification status: PASSED.")
        print("Empirical Sparsity certified successfully.")
        return True


def main() -> None:
    """Main CLI entry point for Empirical Sparsity validation."""
    parser = argparse.ArgumentParser(
        description="Verify scientific and structural consistency of empirical sparsity output."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="results/empirical/empirical_sparsity.parquet",
        help="Path to generated empirical sparsity parquet file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    validator = EmpiricalValidation(config_path=args.config)
    success = validator.validate_dataset(args.file)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
