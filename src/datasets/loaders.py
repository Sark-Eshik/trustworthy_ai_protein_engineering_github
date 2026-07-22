# src/datasets/loaders.py
"""Centralized loaders for ingesting, caching, and validating source datasets.

Handles reading Parquet, CSV, and FASTA formats, automatically validating schemas and
uniqueness constraints via integration with SchemaValidator and ValidationEngine.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.logger import get_logger
from src.validation.schema_validator import SchemaValidator


class DatasetLoader:
    """Class to manage directory paths, schema validation, and loading datasets safely."""

    def __init__(self, config_path: str = "configs"):
        """Initialize DatasetLoader with config properties and validators.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validator = SchemaValidator(config_path=config_path)
        self.logger = get_logger(
            name="dataset_loader",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def load(self, dataset_id: str, file_path_override: Optional[str] = None) -> pd.DataFrame:
        """Ingests and validates a registered dataset file.

        Parameters
        ----------
        dataset_id : str
            Unique registry identifier of the target dataset (e.g., 'S001', 'S002').
        file_path_override : Optional[str]
            Optional custom file path to load from. If not provided, resolves relative path
            from the dataset registry.

        Returns
        -------
        pd.DataFrame
            Validated pandas DataFrame.

        Raises
        ------
        FileNotFoundError
            If the target dataset file does not exist on disk.
        ValueError
            If dataset fails schema validation constraints.
        """
        definition: DatasetDefinition = self.registry.get_dataset(dataset_id)
        file_path = file_path_override or os.path.join(
            os.getcwd(), definition.relative_path
        )

        if not os.path.exists(file_path):
            self.logger.error(f"Failed to find dataset file for {dataset_id} at {file_path}")
            raise FileNotFoundError(f"Registered dataset {dataset_id} file not found: {file_path}")

        # Validate schema before returning (fail-fast architecture rule)
        is_valid = self.validator.validate(dataset_id, file_path)
        if not is_valid:
            self.logger.error(f"Ingested dataset {dataset_id} failed schema validation: {file_path}")
            raise ValueError(f"Dataset {dataset_id} at {file_path} failed schema validation constraints.")

        # Successfully certified, now load and return
        df = self.validator.load_dataset(file_path)
        self.logger.info(f"Successfully loaded and certified dataset {dataset_id} with {len(df)} row(s).")
        return df


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    print("Testing DatasetLoader utilities...")
    loader = DatasetLoader()

    # Verify loading valid S001 dataset
    try:
        s001_path = "data/raw/megascale_d/megascale_d.parquet"
        df_s001 = loader.load("S001", file_path_override=s001_path)
        print("\n--- Manual Validation ---")
        print("S001 Load: SUCCESS")
        print(f"Loaded {len(df_s001)} record(s) from Parquet safely.")
        print(df_s001.head())
        print("-------------------------")
    except Exception as e:
        print(f"FAILED S001 Load: {e}")
