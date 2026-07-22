# src/sparsity/combined/combined_validation.py
"""Combined Sparsity scientific validation and certification script.

Verifies that the generated combined_sparsity.parquet meets all scientific criteria:
1. Schema matches D104 definition (accounting for allowed null columns like empirical_sparsity in TKT mode).
2. Combined sparsity values reside strictly within the [0.0, 1.0] range.
3. No duplicate mutation keys or unexpected nulls.
4. Value calculations check: Combined = Average(Constituents).
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.validation_engine import ValidationEngine


class CombinedValidation:
    """Validator to verify and certify combined sparsity data properties."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the combined sparsity validator.

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
            name="combined_validation",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_dataset(self, file_path: str, mode: str = "megascale_d") -> bool:
        """Performs scientific and structural validation checks on combined sparsity output.

        Parameters
        ----------
        file_path : str
            Path to the combined_sparsity.parquet file.
        mode : str
            Orchestration mode: 'megascale_d' (averages 3 dimensions) or 'tkt' (averages 2 dimensions).

        Returns
        -------
        bool
            True if all checks pass, False otherwise.
        """
        self.logger.info(f"Initiating scientific validation on combined file: {file_path} (mode: {mode})")

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

        # --- Check 1: Structural Schema Validation via ValidationEngine (D104) ---
        schema_report = self.validation_engine.validate_schema("D104", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            print("Validation Failures (Schema):", file=sys.stderr)
            for err in schema_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 2: Value Range Constraints ---
        range_rules = {
            "evolutionary_sparsity": (0.0, 1.0),
            "structural_sparsity": (0.0, 1.0),
            "combined_sparsity": (0.0, 1.0),
        }
        # In megascale_d mode, empirical_sparsity is not null and should be bounded
        if mode == "megascale_d":
            range_rules["empirical_sparsity"] = (0.0, 1.0)

        range_report = self.validation_engine.validate_ranges("D104", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            print("Validation Failures (Value Ranges):", file=sys.stderr)
            for err in range_report["errors"]:
                print(f"  * {err}", file=sys.stderr)
            return False

        # --- Check 3: Scientific Recomputation Audit (Average of Components) ---
        failures = 0
        for idx, row in df.iterrows():
            evo = float(row["evolutionary_sparsity"])
            struct = float(row["structural_sparsity"])
            comb = float(row["combined_sparsity"])

            if mode == "megascale_d":
                emp = float(row["empirical_sparsity"])
                recomputed = (emp + evo + struct) / 3.0
            else:
                recomputed = (evo + struct) / 2.0

            if abs(comb - recomputed) > 1e-6:
                self.logger.error(
                    f"Calculation audit mismatch for mutation {row['mutation_id']}. "
                    f"Expected: {recomputed:.6f}, Found: {comb:.6f}."
                )
                failures += 1

        if failures > 0:
            print("Validation Failures (Calculation Audit):", file=sys.stderr)
            print("  * Recomputed average from constituent values does not match stored combined_sparsity.", file=sys.stderr)
            return False

        self.logger.info("Combined sparsity scientific validation completed successfully. Certification status: PASSED.")
        print("Combined Sparsity certified successfully.")
        return True


def main() -> None:
    """Main CLI entry point for Combined Sparsity validation."""
    parser = argparse.ArgumentParser(
        description="Verify scientific and structural consistency of combined sparsity output."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="results/combined/combined_sparsity.parquet",
        help="Path to generated combined sparsity parquet file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="megascale_d",
        choices=["megascale_d", "tkt"],
        help="Pipeline execution mode used during calculation: 'megascale_d' or 'tkt'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    validator = CombinedValidation(config_path=args.config)
    success = validator.validate_dataset(args.file, mode=args.mode)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
