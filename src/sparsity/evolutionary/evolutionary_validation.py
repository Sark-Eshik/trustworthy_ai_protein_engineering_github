# src/sparsity/evolutionary/evolutionary_validation.py
"""Evolutionary Sparsity scientific validation and certification script.

Verifies that the generated evolutionary_sparsity.parquet meets all scientific criteria:
1. Schema matches D102 definition.
2. Value bounds are valid (probabilities and sparsities in [0, 1]).
3. No duplicate keys or null/missing values.
4. Correct scientific ranking (higher probability = lower sparsity).
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


class EvolutionaryValidation:
    """Validator to verify and certify evolutionary sparsity data properties."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the evolutionary sparsity validator.

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
            name="evolutionary_validation",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_dataset(self, file_path: str) -> bool:
        """Performs scientific and structural validation checks on evolutionary sparsity output.

        Parameters
        ----------
        file_path : str
            Path to the evolutionary_sparsity.parquet file.

        Returns
        -------
        bool
            True if all checks pass, False otherwise.
        """
        self.logger.info(f"Initiating scientific validation on evolutionary file: {file_path}")

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

        # --- Check 1: Structural Schema Validation via ValidationEngine (D102) ---
        schema_report = self.validation_engine.validate_schema("D102", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            print("Validation Failures (Schema):", file=sys.stderr)
            for err in schema_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 2: Value Range Constraints ---
        range_rules = {
            "esm_probability": (0.0, 1.0),
            "log_probability": (0.0, 100.0),  # Upper bound log parameter to block huge numbers safely
            "evolutionary_sparsity": (0.0, 1.0),
        }
        range_report = self.validation_engine.validate_ranges("D102", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            print("Validation Failures (Value Ranges):", file=sys.stderr)
            for err in range_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 3: Scientific Ordering and Sanity Checks ---
        # Verify that as ESM Probability decreases, Evolutionary Sparsity increases
        sorted_df = df.sort_values(by="esm_probability")
        probs = sorted_df["esm_probability"].values
        sparsities = sorted_df["evolutionary_sparsity"].values

        # If prob[i] < prob[j], then sparsity[i] must be >= sparsity[j]
        failures = 0
        for i in range(len(sorted_df) - 1):
            if probs[i] < probs[i + 1] and sparsities[i] < sparsities[i + 1]:
                self.logger.error(
                    f"Scientific anomaly: Mutation with lower probability {probs[i]} has lower "
                    f"sparsity ({sparsities[i]}) than mutation with higher probability {probs[i + 1]} "
                    f"({sparsities[i + 1]})."
                )
                failures += 1

        if failures > 0:
            print("Validation Failures (Scientific Ordering):", file=sys.stderr)
            print("  * Evolutionary sparsity must inversely correlate with ESM probability.", file=sys.stderr)
            return False

        # --- Check 4: Normalization Boundaries ---
        # The mutation with minimum esm_probability (highest log_probability) must have evolutionary_sparsity = 1.0
        # The mutation with maximum esm_probability (lowest log_probability) must have evolutionary_sparsity = 0.0
        max_prob_idx = df["esm_probability"].idxmax()
        min_prob_idx = df["esm_probability"].idxmin()

        max_prob_rec = df.loc[max_prob_idx]
        min_prob_rec = df.loc[min_prob_idx]

        if abs(max_prob_rec["evolutionary_sparsity"] - 0.0) > 1e-6:
            self.logger.error(
                f"Normalization error: Maximum probability mutation ({max_prob_rec['mutation_id']}) "
                f"has sparsity = {max_prob_rec['evolutionary_sparsity']}. Expected 0.0."
            )
            print("Validation Failures (Normalization Boundary):", file=sys.stderr)
            print("  * Mutation with maximum ESM probability must have evolutionary sparsity of 0.0.", file=sys.stderr)
            return False

        if abs(min_prob_rec["evolutionary_sparsity"] - 1.0) > 1e-6:
            self.logger.error(
                f"Normalization error: Minimum probability mutation ({min_prob_rec['mutation_id']}) "
                f"has sparsity = {min_prob_rec['evolutionary_sparsity']}. Expected 1.0."
            )
            print("Validation Failures (Normalization Boundary):", file=sys.stderr)
            print("  * Mutation with minimum ESM probability must have evolutionary sparsity of 1.0.", file=sys.stderr)
            return False

        self.logger.info("Evolutionary sparsity scientific validation completed successfully. Certification status: PASSED.")
        print("Evolutionary Sparsity certified successfully.")
        return True


def main() -> None:
    """Main CLI entry point for Evolutionary Sparsity validation."""
    parser = argparse.ArgumentParser(
        description="Verify scientific and structural consistency of evolutionary sparsity output."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="results/evolutionary/evolutionary_sparsity.parquet",
        help="Path to generated evolutionary sparsity parquet file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    validator = EvolutionaryValidation(config_path=args.config)
    success = validator.validate_dataset(args.file)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
