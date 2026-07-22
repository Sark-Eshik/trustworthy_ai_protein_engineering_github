# src/reliability/reliability_score.py
"""Reliability score calculation module.

Defines the mathematical transformation of Combined Sparsity into Reliability Score:
Reliability Score = 1.0 - Combined Sparsity
Assigns mutations into defined, interpretable engineering reliability categories:
- High Reliability: 0.75 - 1.00
- Moderate Reliability: 0.50 - 0.75
- Low Reliability: 0.25 - 0.50
- Very Low Reliability: 0.00 - 0.25
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine

class ReliabilityFramework:
    """Orchestrates calculation, classification, plotting, and exports of Reliability Scores."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="reliability_framework",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    @staticmethod
    def calculate_score(combined_sparsity: float) -> float:
        """Calculates Reliability Score: 1.0 - Combined Sparsity."""
        return 1.0 - combined_sparsity

    @staticmethod
    def classify_category(score: float) -> str:
        """Maps a reliability score to its standard engineering category."""
        if score >= 0.75:
            return "High"
        elif score >= 0.50:
            return "Moderate"
        elif score >= 0.25:
            return "Low"
        else:
            return "Very Low"

    def run_athlete_exercises(self) -> bool:
        """Runs the standard Athlete Exercises for Reliability scoring:
        
        1. Sparsity = 0.10 -> Expected: Reliability = 0.90
        2. Sparsity = 0.75 -> Expected: Reliability = 0.25
        3. Sparsity = 1.00 -> Expected: Reliability = 0.00
        """
        self.logger.info("Executing Athlete Exercises validation for Reliability Scores...")
        
        test_cases = [
            {"sparsity": 0.10, "expected": 0.90},
            {"sparsity": 0.75, "expected": 0.25},
            {"sparsity": 1.00, "expected": 0.00},
        ]
        
        passed_all = True
        for idx, tc in enumerate(test_cases, 1):
            sp = tc["sparsity"]
            expected = tc["expected"]
            calc = self.calculate_score(sp)
            
            if np.isclose(calc, expected):
                self.logger.info(f"Athlete Exercise {idx}: Passed. Sparsity={sp:.2f} -> Reliability={calc:.2f} (Expected: {expected:.2f})")
            else:
                self.logger.error(f"Athlete Exercise {idx}: FAILED! Sparsity={sp:.2f} -> Got {calc:.4f}, expected {expected:.4f}")
                passed_all = False
                
        return passed_all

    def process_pipeline(
        self, 
        sparsity_path: str = None, 
        output_dir: str = None,
        inject_invalid_value: bool = False
    ) -> tuple[str, str, str]:
        """Runs the complete Reliability Score compilation pipeline.
        
        Loads Combined Sparsity (D104), calculates scores, groups categories, 
        generates visualizations, and outputs reliability_scores.parquet (D402).
        """
        if not self.run_athlete_exercises():
            raise ValueError("Reliability score calculation failed validation on Athlete Exercises.")

        data_dir = self.config.paths.data_dir
        sp_path = sparsity_path or os.path.join(data_dir, "intermediate/combined/combined_sparsity.parquet")
        
        out_dir = output_dir or os.path.join(data_dir, "final/reliability")
        os.makedirs(out_dir, exist_ok=True)

        results_dir = os.path.join(self.config.paths.results_dir, "reliability")
        os.makedirs(results_dir, exist_ok=True)

        self.logger.info(f"Loading Combined Sparsity dataset from: {sp_path}")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"Combined sparsity file not found at: {sp_path}")

        df_sp = pd.read_parquet(sp_path)
        self.logger.info(f"Loaded {len(df_sp)} single mutations.")

        # Calculate scores and classify categories
        df_sp["reliability_score"] = df_sp["combined_sparsity"].apply(self.calculate_score)
        df_sp["reliability_category"] = df_sp["reliability_score"].apply(self.classify_category)

        # Trigger failure-injection if requested
        if inject_invalid_value:
            self.logger.warning("Failure Injection: Intentionally assigning invalid reliability value of 1.20...")
            df_sp.loc[0, "reliability_score"] = 1.20

        # Build official D402 output structure
        d402_cols = ["mutation_id", "combined_sparsity", "reliability_score", "reliability_category"]
        df_out = df_sp[d402_cols].copy()

        # Save core Parquet output
        parquet_path = os.path.join(out_dir, "reliability_scores.parquet")
        df_out.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved D402 (Reliability Scores) Parquet to {parquet_path}")

        # Run Validation Utility check
        from src.reliability.reliability_validation import ReliabilityValidator
        validator = ReliabilityValidator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        is_valid = validator.validate_dataset(parquet_path)
        if not is_valid:
            err_msg = "Reliability Scores dataset failed validation check!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Generate summary CSV
        summary_path = os.path.join(results_dir, "reliability_summary.csv")
        summary_counts = df_out["reliability_category"].value_counts().reset_index()
        summary_counts.columns = ["category", "count"]
        summary_counts["percentage"] = (summary_counts["count"] / len(df_out)) * 100.0
        
        # Add general stats to summary CSV
        stats_df = pd.DataFrame([
            {"category": "Mean Score", "count": df_out["reliability_score"].mean(), "percentage": np.nan},
            {"category": "Min Score", "count": df_out["reliability_score"].min(), "percentage": np.nan},
            {"category": "Max Score", "count": df_out["reliability_score"].max(), "percentage": np.nan},
        ])
        
        df_summary = pd.concat([summary_counts, stats_df], ignore_index=True)
        df_summary.to_csv(summary_path, index=False)
        self.logger.info(f"Saved summary CSV to {summary_path}")

        # Generate distribution histogram
        plot_path = os.path.join(results_dir, "reliability_distribution.png")
        plt.figure(figsize=(8, 6))
        plt.hist(
            df_out["reliability_score"],
            bins=15,
            color="#27ae60",
            edgecolor="#1e8449",
            alpha=0.75,
            rwidth=0.85
        )
        plt.title("Distribution of Reliability Scores", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Reliability Score (0 = Low Reliability, 1 = High Reliability)", fontsize=12)
        plt.ylabel("Mutation Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved distribution plot to {plot_path}")

        # Backwards compatibility figure copy to figures_dir
        figures_copy_path = os.path.join(self.config.paths.figures_dir, "reliability_distribution.png")
        os.makedirs(os.path.dirname(figures_copy_path), exist_ok=True)
        import shutil
        shutil.copy(plot_path, figures_copy_path)

        print(f"Reliability Scores Parquet (D402) saved: {parquet_path}")
        print(f"Reliability Summary CSV saved: {summary_path}")
        print(f"Reliability Distribution plot saved: {plot_path}")

        return parquet_path, summary_path, plot_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Reliability Scores.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--sparsity", type=str, default=None, help="Path to combined sparsity parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output parquet.")
    parser.add_argument("--inject-fault", action="store_true", help="Inject invalid values to test validator sensitivity.")
    args = parser.parse_args()

    try:
        framework = ReliabilityFramework(config_path=args.config)
        framework.process_pipeline(
            sparsity_path=args.sparsity,
            output_dir=args.output,
            inject_invalid_value=args.inject_fault
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Reliability Framework failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
