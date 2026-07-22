# src/industrial_fitness/fitness_validator.py
"""Validation utility for Industrial Fitness Scores.

Verifies that:
1. Schema matches D601 (for parquet) and D602 (for top candidate csv).
2. Fitness scores reside strictly within the [0.0, 1.0] range.
3. Ranks are strictly increasing and sorted (Rank 1 > Rank 2 > Rank 3 based on score).
"""

import os
import sys
import pandas as pd
import numpy as np

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine

class FitnessValidator:
    """Validator to verify and certify industrial fitness dataset and ranking properties."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="fitness_validator",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def validate_scores(self, file_path: str) -> bool:
        """Validates the calculated industrial fitness scores parquet file (D601)."""
        self.logger.info(f"Initiating validation checks on fitness scores parquet: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Target file does not exist: {file_path}")
            return False

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self.logger.error(f"Failed to read parquet file: {e}")
            return False

        # --- Check 1: Schema Verification ---
        schema_report = self.validation_engine.validate_schema("D601", df)
        if not schema_report["valid"]:
            self.logger.error(f"Schema validation failed: {schema_report['errors']}")
            return False

        # --- Check 2: Range constraints [0.0, 1.0] ---
        range_rules = {
            "reliability_score": (0.0, 1.0),
            "industrial_fitness_score": (0.0, 1.0),
        }
        range_report = self.validation_engine.validate_ranges("D601", df, range_rules)
        if not range_report["valid"]:
            self.logger.error(f"Value range constraints violated: {range_report['errors']}")
            return False

        # --- Check 3: Rank Ordering (Rank 1 > Rank 2 > Rank 3 based on score) ---
        # The dataframe must be sorted in descending order of score, with ranks 1..N matching that order.
        failures = 0
        prev_score = float("inf")
        prev_rank = 0

        for idx, row in df.iterrows():
            score = float(row["industrial_fitness_score"])
            rank = int(row["rank"])

            # Verify decreasing score
            if score > prev_score:
                self.logger.error(f"Ranking violation at index {idx}: Score {score:.4f} is higher than previous score {prev_score:.4f}.")
                failures += 1
            
            # Verify increasing rank
            if rank <= prev_rank:
                self.logger.error(f"Ranking violation at index {idx}: Rank {rank} is not greater than previous rank {prev_rank}.")
                failures += 1

            prev_score = score
            prev_rank = rank

        if failures > 0:
            self.logger.error(f"Ranking and score alignment verification failed with {failures} anomalies.")
            return False

        self.logger.info("Industrial Fitness Scores dataset validated successfully. Status: PASS.")
        return True

    def validate_ranking_csv(self, file_path: str) -> bool:
        """Validates the ranked candidates CSV file (D602)."""
        self.logger.info(f"Initiating validation checks on top candidate CSV: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Target file does not exist: {file_path}")
            return False

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"Failed to read CSV file: {e}")
            return False

        # --- Check 1: Schema Verification ---
        schema_report = self.validation_engine.validate_schema("D602", df)
        if not schema_report["valid"]:
            self.logger.error(f"D602 schema validation failed: {schema_report['errors']}")
            return False

        # --- Check 2: Decreasing Score Verification ---
        failures = 0
        prev_score = float("inf")
        prev_rank = 0

        for idx, row in df.iterrows():
            score = float(row["industrial_fitness_score"])
            rank = int(row["rank"])

            if score > prev_score:
                self.logger.error(f"D602 Ranking violation at index {idx}: Score {score:.4f} > previous {prev_score:.4f}.")
                failures += 1
            
            if rank <= prev_rank:
                self.logger.error(f"D602 Ranking violation at index {idx}: Rank {rank} <= previous {prev_rank}.")
                failures += 1

            prev_score = score
            prev_rank = rank

        if failures > 0:
            self.logger.error(f"D602 CSV verification failed with {failures} anomalies.")
            return False

        self.logger.info("Top Candidate CSV validated successfully. Status: PASS.")
        return True
