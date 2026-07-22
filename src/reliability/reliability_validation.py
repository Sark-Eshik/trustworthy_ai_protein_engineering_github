# src/reliability/reliability_validation.py
"""Scientific and structural validator utility for Reliability Score.

Verifies that the generated reliability_scores.parquet file meets all engineering and schema criteria:
1. Schema matches D402 definition (mutation_id, combined_sparsity, reliability_score, reliability_category).
2. Reliability values reside strictly within the [0.0, 1.0] range.
3. No duplicate keys or unexpected nulls.
4. Value calculations check: Reliability = 1.0 - Combined Sparsity.
5. Category definitions match boundaries exactly.
"""

import os
import sys
import pandas as pd
import numpy as np

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine

class ReliabilityValidator:
    """Scientific validator to verify and certify reliability dataset properties (D402)."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="reliability_validation",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_dataset(self, file_path: str) -> bool:
        """Performs scientific and structural validation checks on reliability scores output.

        Parameters
        ----------
        file_path : str
            Path to the reliability_scores.parquet file (D402).

        Returns
        -------
        bool
            True if all checks pass, False otherwise.
        """
        self.logger.info(f"Initiating validation checks on reliability dataset: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Target file does not exist: {file_path}")
            return False

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self.logger.error(f"Failed to read parquet file: {e}")
            return False

        # --- Check 1: Structural Schema Validation ---
        schema_report = self.validation_engine.validate_schema("D402", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            return False

        # --- Check 2: Value Range Constraints (0.0 to 1.0) ---
        range_rules = {
            "combined_sparsity": (0.0, 1.0),
            "reliability_score": (0.0, 1.0),
        }
        range_report = self.validation_engine.validate_ranges("D402", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            return False

        # --- Check 3: Scientific Calculations & Categorization Audits ---
        failures = 0
        valid_categories = {"High", "Moderate", "Low", "Very Low"}

        for idx, row in df.iterrows():
            mut_id = row["mutation_id"]
            sp = float(row["combined_sparsity"])
            rel = float(row["reliability_score"])
            cat = row["reliability_category"]

            # Calculation verification: Reliability = 1.0 - Sparsity
            expected_rel = 1.0 - sp
            if abs(rel - expected_rel) > 1e-6:
                self.logger.error(
                    f"Calculation audit mismatch for mutation {mut_id}. "
                    f"Expected: {expected_rel:.6f}, Found: {rel:.6f}."
                )
                failures += 1

            # Category validation
            if cat not in valid_categories:
                self.logger.error(f"Invalid category label '{cat}' detected for mutation {mut_id}.")
                failures += 1
            else:
                # Boundary verification
                if rel >= 0.75 and cat != "High":
                    self.logger.error(f"Category mismatch for mutation {mut_id} (Score {rel:.4f} classified as '{cat}', expected 'High').")
                    failures += 1
                elif 0.50 <= rel < 0.75 and cat != "Moderate":
                    self.logger.error(f"Category mismatch for mutation {mut_id} (Score {rel:.4f} classified as '{cat}', expected 'Moderate').")
                    failures += 1
                elif 0.25 <= rel < 0.50 and cat != "Low":
                    self.logger.error(f"Category mismatch for mutation {mut_id} (Score {rel:.4f} classified as '{cat}', expected 'Low').")
                    failures += 1
                elif rel < 0.25 and cat != "Very Low":
                    self.logger.error(f"Category mismatch for mutation {mut_id} (Score {rel:.4f} classified as '{cat}', expected 'Very Low').")
                    failures += 1

        if failures > 0:
            self.logger.error(f"Scientific verification failed with {failures} value/class anomaly failures.")
            return False

        self.logger.info("Reliability dataset validated successfully. Certification status: PASS.")
        return True

if __name__ == "__main__":
    # Command-line entry for independent validation audits
    import argparse
    parser = argparse.ArgumentParser(description="Validate Reliability Scores.")
    parser.add_argument("--file", type=str, default="data/final/reliability/reliability_scores.parquet", help="Path to reliability parquet.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    args = parser.parse_args()

    validator = ReliabilityValidator(config_path=args.config)
    success = validator.validate_dataset(args.file)
    sys.exit(0 if success else 1)
