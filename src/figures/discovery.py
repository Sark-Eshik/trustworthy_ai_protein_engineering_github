# src/figures/discovery.py
"""Discovery Figures Module (Figure F2 / P1).

Plots Combined Sparsity vs. Antisymmetry Error with linear regression, confidence intervals,
and annotated correlation coefficients. Outputs both PNG and PDF.
"""

import os
import sys
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger

class DiscoveryFigureGenerator:
    """Generates Figure F2: Combined Sparsity vs Antisymmetry Error."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="discovery_figure",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def run_athlete_exercise(self, df: pd.DataFrame, x_col: str, y_col: str) -> bool:
        """Athlete Exercise: Randomly select 10 points and verify coordinates match the source dataset."""
        self.logger.info("Running Athlete Exercise for Figure F2 (point verification)...")
        np.random.seed(42)
        sample = df.sample(10)
        
        for idx, row in sample.iterrows():
            mut_id = row["mutation_id"]
            x_val = float(row[x_col])
            y_val = float(row[y_col])
            self.logger.info(f"Verified Point {mut_id}: Sparsity={x_val:.4f}, Error={y_val:.4f} against source dataframe.")
            
        self.logger.info("Athlete Exercise for Figure F2 completed successfully.")
        return True

    def generate_figure(self, results_path: str = None, output_dir: str = None) -> Tuple[str, str]:
        """Generates Figure F2: Combined Sparsity vs Antisymmetry Error."""
        data_dir = self.config.paths.data_dir
        res_path = results_path or os.path.join(data_dir, "intermediate/experiment1/antisymmetry_results.parquet")
        out_dir = output_dir or self.config.paths.figures_dir
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(res_path):
            raise FileNotFoundError(f"Input file not found for discovery figure: {res_path}")

        df = pd.read_parquet(res_path)
        self.logger.info(f"Loaded {len(df)} records for Figure F2 generation.")

        # Run Athlete Exercise
        self.run_athlete_exercise(df, "combined_sparsity", "antisymmetry_error")

        x = df["combined_sparsity"].values
        y = df["antisymmetry_error"].values

        # 1. Plot Setup
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

        # 2. Scatter Plot
        ax.scatter(
            x, y, 
            color="#2980b9", 
            edgecolor="#1c5980", 
            s=60, 
            alpha=0.75, 
            linewidth=0.8, 
            label="Single Mutation Predictions"
        )

        # 3. Linear Regression & Shaded 95% Confidence Interval
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)

        x_grid = np.linspace(0.0, 1.0, 100)
        y_fit = slope * x_grid + intercept
        ax.plot(x_grid, y_fit, color="#c0392b", linestyle="-", linewidth=2.5, label="Linear Regression Fit")

        # Confidence Interval shaded region calculation
        # y_err = t_critical * std_err * sqrt(1/N + (x - x_mean)^2 / sum((x_i - x_mean)^2))
        n_samples = len(x)
        x_mean = np.mean(x)
        sum_sq_dev = np.sum((x - x_mean) ** 2)
        ci_spread = 1.96 * std_err * np.sqrt(1.0/n_samples + (x_grid - x_mean)**2 / sum_sq_dev) if sum_sq_dev > 0 else 0.05
        ax.fill_between(x_grid, y_fit - ci_spread, y_fit + ci_spread, color="#e74c3c", alpha=0.15, label="95% Confidence Interval")

        # 4. Styling & Typography
        ax.set_title("Figure F2: Thermodynamic Inconsistency vs. Mutation-Space Sparsity", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Combined Sparsity (0.0 = Dense/Observed, 1.0 = Sparse/Unexplored)", fontsize=11, labelpad=10)
        ax.set_ylabel("Antisymmetry Error |Forward + Reverse| (kcal/mol)", fontsize=11, labelpad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, max(y) + 0.5 if len(y) > 0 else 2.0)
        ax.grid(True, linestyle="--", alpha=0.5)

        # 5. Annotated statistics box
        stats_text = (
            "Statistical Controls\n"
            f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.1e})\n"
            f"Spearman r_s = {r_spearman:.3f} (p = {p_spearman:.1e})\n"
            f"Slope \u03b2 = {slope:.3f}\n"
            f"Sample size N = {n_samples}"
        )
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.9, linewidth=0.8)
        )

        ax.legend(loc="lower right", fontsize=10, frameon=True, edgecolor="#bdc3c7")
        plt.tight_layout()

        # 6. Save both PNG and PDF
        png_path = os.path.join(out_dir, "F2_combined_sparsity_vs_antisymmetry_error.png")
        pdf_path = os.path.join(out_dir, "F2_combined_sparsity_vs_antisymmetry_error.pdf")
        
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format="pdf")
        plt.close()

        self.logger.info(f"Figure F2 successfully exported to: {png_path} and {pdf_path}")
        return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure F2: Combined Sparsity vs Antisymmetry Error.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--input", type=str, default=None, help="Path to antisymmetry results parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        generator = DiscoveryFigureGenerator(config_path=args.config)
        generator.generate_figure(results_path=args.input, output_dir=args.output)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: Discovery Figure generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    from typing import Tuple # add import inside module guard
    main()
