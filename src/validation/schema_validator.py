# src/validation/schema_validator.py
"""Centralized schema validator module.

Checks input datasets against official registry specifications, validates primary key
uniqueness, required column presence, and missing values, and writes reports.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd

from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.validation_engine import ValidationEngine
from src.infrastructure.config_loader import ConfigLoader
from src.infrastructure.logger import get_logger


class SchemaValidator:
    """SchemaValidator class that coordinates verification of source and intermediate datasets."""

    def __init__(self, config_path: str = "configs"):
        """Initialize SchemaValidator with required configuration loading.

        Parameters
        ----------
        config_path : str
            Directory path containing the environment config files.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="schema_validator",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Dynamically load CSV or Parquet files into a pandas DataFrame.

        Parameters
        ----------
        file_path : str
            Full path to the dataset file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset contents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path.lower())
        if ext == ".parquet":
            return pd.read_parquet(file_path)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(file_path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only Parquet, CSV, or TSV are supported.")

    def validate(self, dataset_id: str, file_path: str) -> bool:
        """Validate a dataset file against its registered schema.

        Writes a structured JSON report to 'results/validation/schema_report.json'
        and prints the final certification result as specified in the validation handbook.

        Parameters
        ----------
        dataset_id : str
            Registered identifier of the target schema (e.g., 'S001').
        file_path : str
            Path to the physical file to read and validate.

        Returns
        -------
        bool
            True if dataset passes all validation rules, False otherwise.
        """
        self.logger.info(f"Starting schema validation for dataset {dataset_id} at {file_path}")

        try:
            df = self.load_dataset(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load dataset file: {e}")
            print("Validation Failed")
            return False

        # Execute schema validation via foundational engine
        report = self.engine.validate_schema(dataset_id, df)

        # 4. Write results/validation/schema_report.json
        report_dir = os.path.join(self.config.paths.results_dir, "validation")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "schema_report.json")

        report_payload = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "row_count": len(df),
            "valid": report["valid"],
            "errors": report["errors"],
        }

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=4)
            self.logger.info(f"Validation report written to {report_path}")
        except OSError as e:
            self.logger.error(f"Failed to write validation report: {e}")

        # Console outputs as strictly required by system operation manuals
        if report["valid"]:
            print("Schema Valid")
            self.logger.info("Dataset schema certified successfully.")
            return True
        else:
            print("Validation Failed")
            self.logger.warning(f"Dataset schema validation failed with {len(report['errors'])} error(s):")
            for err in report["errors"]:
                self.logger.warning(f"  - {err}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset schemas and produce reports.")
    parser.add_argument("file_path", help="Path to the dataset file to validate.")
    parser.add_argument(
        "--dataset_id",
        default="S001",
        help="Target schema ID from dataset registry (default: S001).",
    )

    args = parser.parse_args()

    validator = SchemaValidator()
    success = validator.validate(args.dataset_id, args.file_path)
    sys.exit(0 if success else 1)
