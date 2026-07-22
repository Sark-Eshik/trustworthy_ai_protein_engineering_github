# src/industrial_fitness/fitness_calculator.py
"""Industrial Fitness calculation module for Transketolase.

Implements the multi-criteria Industrial Fitness Score formula:
Fitness = 0.50 * Stability + 0.35 * Reliability + 0.15 * Evolutionary Plausibility
Where:
- Stability is min-max normalized predicted stability benefit [0, 1].
- Reliability is the raw Reliability Score [0, 1].
- Evolutionary Plausibility is (1.0 - Evolutionary Sparsity) [0, 1].
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine

class FitnessCalculator:
    """Calculates Industrial Fitness Scores, evaluates Athlete Exercises, and saves outputs."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="fitness_calculator",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    @staticmethod
    def calculate_score(stability: float, reliability: float, evolutionary_plausibility: float) -> float:
        """Core weighted sum formula for Industrial Fitness."""
        return 0.50 * stability + 0.35 * reliability + 0.15 * evolutionary_plausibility

    def run_athlete_exercises(self) -> bool:
        """Executes the standard Athlete Exercises for Industrial Fitness calculation:
        
        Exercise 1:
        Input: Stability = 0.80, Reliability = 0.60, Evolutionary = 0.90
        Expected: 0.40 + 0.21 + 0.135 = 0.745

        Exercise 2:
        Input: Stability = 0.60, Reliability = 1.00, Evolutionary = 1.00
        Expected: 0.30 + 0.35 + 0.15 = 0.80
        """
        self.logger.info("Running Athlete Exercises for Industrial Fitness scores...")
        
        # Exercise 1
        calc1 = self.calculate_score(0.80, 0.60, 0.90)
        expected1 = 0.745
        if np.isclose(calc1, expected1):
            self.logger.info(f"Athlete Exercise 1: Passed. Score = {calc1:.4f} (Expected: {expected1:.4f})")
        else:
            self.logger.error(f"Athlete Exercise 1: FAILED. Got {calc1:.4f}, expected {expected1:.4f}")
            return False

        # Exercise 2
        calc2 = self.calculate_score(0.60, 1.00, 1.00)
        expected2 = 0.80
        if np.isclose(calc2, expected2):
            self.logger.info(f"Athlete Exercise 2: Passed. Score = {calc2:.4f} (Expected: {expected2:.4f})")
        else:
            self.logger.error(f"Athlete Exercise 2: FAILED. Got {calc2:.4f}, expected {expected2:.4f}")
            return False

        return True

    def compute_fitness(
        self,
        analysis_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        inject_fault: bool = False,
    ) -> Tuple[str, str]:
        """Runs the complete Industrial Fitness calculation pipeline.
        
        Loads D502 (analysis), normalizes stability, computes fitness, 
        sorts and ranks, validates outputs, and saves D601 parquet and distribution plot.
        """
        if not self.run_athlete_exercises():
            raise ValueError("Fitness formula failed validation checks on Athlete Exercises.")

        data_dir = self.config.paths.data_dir
        results_dir = os.path.join(self.config.paths.results_dir, "industrial_fitness")
        os.makedirs(results_dir, exist_ok=True)

        anal_path = analysis_path or os.path.join(data_dir, "final/tkt/tkt_mutation_analysis.parquet")
        self.logger.info(f"Loading TKT analysis from {anal_path}...")
        if not os.path.exists(anal_path):
            raise FileNotFoundError(f"TKT analysis file not found at: {anal_path}")

        df_anal = pd.read_parquet(anal_path)
        self.logger.info(f"Loaded {len(df_anal)} single mutations.")

        # 1. Normalize Stability to [0.0, 1.0] using min-max scaling
        min_stab = df_anal["predicted_stability"].min()
        max_stab = df_anal["predicted_stability"].max()
        stab_range = max_stab - min_stab if max_stab != min_stab else 1.0
        df_anal["stability_norm"] = (df_anal["predicted_stability"] - min_stab) / stab_range

        # 2. Compute Evolutionary Plausibility = 1.0 - Evolutionary Sparsity
        df_anal["evolutionary_plausibility"] = 1.0 - df_anal["evolutionary_sparsity"]

        # 3. Calculate Fitness Score
        fitness_scores = []
        for idx, row in df_anal.iterrows():
            fit = self.calculate_score(
                stability=float(row["stability_norm"]),
                reliability=float(row["reliability_score"]),
                evolutionary_plausibility=float(row["evolutionary_plausibility"])
            )
            fitness_scores.append(fit)

        df_anal["industrial_fitness_score"] = fitness_scores

        # Trigger Failure Injection if requested
        if inject_fault:
            self.logger.warning("Failure Injection: Forcing invalid fitness score of 1.25...")
            df_anal.loc[0, "industrial_fitness_score"] = 1.25

        # 4. Sort and assign ranks (Rank 1 is highest fitness score)
        df_anal = df_anal.sort_values(by="industrial_fitness_score", ascending=False).reset_index(drop=True)
        df_anal["rank"] = np.arange(1, len(df_anal) + 1)

        # Build official D601 dataframe structure
        d601_cols = [
            "mutation_id",
            "reliability_score",
            "predicted_stability",
            "industrial_fitness_score",
            "rank",
        ]
        df_out = df_anal[d601_cols].copy()

        # Save to parquet (D601)
        out_dir = output_dir or os.path.join(data_dir, "final/industrial_fitness")
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, "industrial_fitness_scores.parquet")
        df_out.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved D601 (Industrial Fitness Scores) parquet to {parquet_path}")

        # Run validation check via Validator
        from src.industrial_fitness.fitness_validator import FitnessValidator
        validator = FitnessValidator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        if not validator.validate_scores(parquet_path):
            err_msg = "Industrial Fitness dataset failed validation checks!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Plot distribution of fitness scores
        plot_path = os.path.join(results_dir, "fitness_distribution.png")
        plt.figure(figsize=(8, 6))
        plt.hist(
            df_out["industrial_fitness_score"],
            bins=15,
            color="#2980b9",
            edgecolor="#2c3e50",
            alpha=0.75,
            rwidth=0.85
        )
        plt.title("Distribution of Industrial Fitness Scores", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Industrial Fitness Score (0 = Poor Candidate, 1 = Optimal Candidate)", fontsize=12)
        plt.ylabel("Mutation Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved fitness distribution plot to {plot_path}")

        # Also copy plot to general figures folder for certification/backwards compatibility
        figures_copy_path = os.path.join(self.config.paths.figures_dir, "fitness_distribution.png")
        os.makedirs(os.path.dirname(figures_copy_path), exist_ok=True)
        import shutil
        shutil.copy(plot_path, figures_copy_path)

        print(f"Industrial Fitness Parquet (D601) saved: {parquet_path}")
        print(f"Fitness Distribution plot saved: {plot_path}")

        return parquet_path, plot_path
