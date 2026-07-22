# src/integrated/pathway_analysis.py
"""Module for performing Reliability Pathway Analysis and validation exercises.

Traces the proposed causal chain:
Combined Sparsity -> Antisymmetry Error -> Epistasis Error

Includes implementation for:
1. Pathway mediation regression.
2. Athlete Exercise: Validating the pipeline on a synthetic increasing dataset.
3. Failure Injection Exercise: Demonstrating degradation of correlation when shuffling Combined Sparsity.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class PathwayAnalysisEngine:
    """Performs pathway-level modeling, Athlete Exercises, and Failure Injection verification."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="pathway_analysis",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    def run_pathway_regression(self, df: pd.DataFrame) -> dict:
        """Traces the direct and indirect links in the proposed causal pathway."""
        self.logger.info("Computing pathway-level regressions...")
        
        # Link 1: Sparsity -> Antisymmetry Error
        slope1, intercept1, r1, p1, _ = linregress(df["combined_sparsity"], df["antisymmetry_error"])
        
        # Link 2: Antisymmetry Error -> Epistasis Error
        slope2, intercept2, r2, p2, _ = linregress(df["antisymmetry_error"], df["epistasis_error"])
        
        # Link 3: Sparsity -> Epistasis Error (Direct)
        slope3, intercept3, r3, p3, _ = linregress(df["combined_sparsity"], df["epistasis_error"])

        pathway_summary = {
            "link1_slope": slope1,
            "link1_r_squared": r1**2,
            "link1_p_value": p1,
            "link2_slope": slope2,
            "link2_r_squared": r2**2,
            "link2_p_value": p2,
            "link3_slope": slope3,
            "link3_r_squared": r3**2,
            "link3_p_value": p3,
        }
        return pathway_summary

    def run_athlete_exercise(self) -> bool:
        """Executes Athlete Exercise to verify pipeline recovery capabilities on synthetic data:
        
        Sparsity: [0.1, 0.3, 0.5, 0.7, 0.9]
        Antisymmetry: Increasing linearly [0.12, 0.31, 0.52, 0.69, 0.91]
        Epistasis: Increasing linearly [0.22, 0.41, 0.63, 0.79, 1.01]
        """
        self.logger.info("Executing Pathway Athlete Exercise on synthetic linearly-increasing dataset...")
        
        # Construct synthetic dataset
        synth_df = pd.DataFrame({
            "combined_sparsity": [0.1, 0.3, 0.5, 0.7, 0.9],
            "antisymmetry_error": [0.12, 0.31, 0.52, 0.69, 0.91],
            "epistasis_error": [0.22, 0.41, 0.63, 0.79, 1.01]
        })

        summary = self.run_pathway_regression(synth_df)
        
        # Recovers relationships if slopes are positive and correlations are strong (R^2 > 0.9)
        link1_ok = summary["link1_slope"] > 0 and summary["link1_r_squared"] > 0.9
        link2_ok = summary["link2_slope"] > 0 and summary["link2_r_squared"] > 0.9
        link3_ok = summary["link3_slope"] > 0 and summary["link3_r_squared"] > 0.9

        success = link1_ok and link2_ok and link3_ok
        
        if success:
            self.logger.info("Athlete Exercise: Passed! The integrated analysis engine successfully recovered relationships.")
            print("Athlete Exercise validation: PASSED")
        else:
            self.logger.error(f"Athlete Exercise: FAILED! Pathway summary: {summary}")
            print("Athlete Exercise validation: FAILED")

        return success

    def run_failure_injection(self, real_df: pd.DataFrame) -> bool:
        """Executes the Failure Injection Exercise.
        
        Intentionally shuffles the 'combined_sparsity' column.
        Expected outcome: Relationship degrades (Pearson correlation coefficient r decreases significantly or flips).
        """
        self.logger.info("Executing Failure Injection Exercise by shuffling Combined Sparsity...")
        
        # Get baseline correlation (Sparsity vs Epistasis Error)
        baseline_r, _ = pearsonr(real_df["combined_sparsity"], real_df["epistasis_error"])
        
        # Shuffled copy
        shuffled_df = real_df.copy()
        shuffled_df["combined_sparsity"] = np.random.RandomState(seed=1337).permutation(shuffled_df["combined_sparsity"].values)
        
        shuffled_r, _ = pearsonr(shuffled_df["combined_sparsity"], shuffled_df["epistasis_error"])
        
        # Degrades if absolute correlation drops below baseline (or changes dramatically)
        degraded = abs(shuffled_r) < abs(baseline_r) or np.sign(shuffled_r) != np.sign(baseline_r) or abs(shuffled_r - baseline_r) > 0.15
        
        if degraded:
            self.logger.info(f"Failure Injection: Passed. Baseline r={baseline_r:.4f}, Shuffled r={shuffled_r:.4f}. Relationship successfully degraded.")
            print("Failure Injection validation: PASSED")
        else:
            self.logger.error(f"Failure Injection: FAILED! Shuffled r={shuffled_r:.4f} did not degrade compared to Baseline r={baseline_r:.4f}")
            print("Failure Injection validation: FAILED")

        return degraded

    def process_pipeline(self, d401_path_override: str = None, report_dir_override: str = None) -> tuple[bool, bool]:
        """Runs the end-to-end pathway analysis pipeline and outputs status."""
        data_dir = self.loader.config.paths.data_dir
        d401_path = d401_path_override or os.path.join(data_dir, "final/reliability/integrated_reliability_analysis.parquet")
        
        if not os.path.exists(d401_path):
            raise FileNotFoundError(f"Missing required D401 Integrated dataset for pathway analysis: {d401_path}")

        df = pd.read_parquet(d401_path)
        
        # 1. Execute Pathway Regressions
        summary = self.run_pathway_regression(df)
        self.logger.info(f"Pathway summary completed. direct slope (link3): {summary['link3_slope']:.4f}")

        # 2. Execute Validation Exercises
        athlete_passed = self.run_athlete_exercise()
        shuffling_passed = self.run_failure_injection(df)

        # Write reports
        report_dir = report_dir_override or self.loader.config.paths.reports_dir
        os.makedirs(report_dir, exist_ok=True)
        
        pathway_report_content = f"""# Integrated Reliability Pathway Analysis Report

## 1. Executive Summary
- **Mediation Pathway**: Combined Sparsity -> Antisymmetry Error -> Epistasis Error
- **Validation Exercises Status**: {"PASS" if athlete_passed and shuffling_passed else "FAIL"}

## 2. Regression Coefficient Tracing
- **Link 1 (Sparsity -> Antisymmetry)**: Slope={summary['link1_slope']:.4f}, R-squared={summary['link1_r_squared']:.4f}, p-val={summary['link1_p_value']:.2e}
- **Link 2 (Antisymmetry -> Epistasis)**: Slope={summary['link2_slope']:.4f}, R-squared={summary['link2_r_squared']:.4f}, p-val={summary['link2_p_value']:.2e}
- **Link 3 (Sparsity -> Epistasis, Direct)**: Slope={summary['link3_slope']:.4f}, R-squared={summary['link3_r_squared']:.4f}, p-val={summary['link3_p_value']:.2e}

## 3. Validation Exercises Outcomes
- **Pathway Recovery (Athlete Exercise)**: {"PASSED (Correctly reconstructed positive linear trend across [0.1 -> 0.9] sparsity boundaries)" if athlete_passed else "FAILED"}
- **Sensitivity Audit (Failure Injection)**: {"PASSED (Intentionally shuffling sparsity column successfully degraded correlations)" if shuffling_passed else "FAILED"}
"""
        report_path = os.path.join(report_dir, "pathway_analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(pathway_report_content)
        self.logger.info(f"Saved Pathway Analysis Report to {report_path}")

        return athlete_passed, shuffling_passed

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Integrated Pathway Analysis and Exercises.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--d401", type=str, default=None, help="Path to integrated dataset.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save report.")
    args = parser.parse_args()

    try:
        engine = PathwayAnalysisEngine(config_path=args.config)
        engine.process_pipeline(d401_path_override=args.d401, report_dir_override=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Pathway analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
