# src/figures/reliability.py
"""Reliability Figures Module (Figure F5 / P5).

Plots the Reliability Score Distribution (histogram) categorized by defined engineering
threshold boundaries. Outputs both PNG and PDF.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger

class ReliabilityFigureGenerator:
    """Generates Figure F5: Reliability Distribution."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="reliability_figure",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def generate_figure(self, results_path: str = None, output_dir: str = None) -> Tuple[str, str]:
        """Generates Figure F5: Reliability Score Distribution."""
        data_dir = self.config.paths.data_dir
        res_path = results_path or os.path.join(data_dir, "final/reliability/reliability_scores.parquet")
        out_dir = output_dir or self.config.paths.figures_dir
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(res_path):
            raise FileNotFoundError(f"Input file not found for reliability figure: {res_path}")

        df = pd.read_parquet(res_path)
        self.logger.info(f"Loaded {len(df)} records for Figure F5 generation.")

        scores = df["reliability_score"].values

        # 1. Plot Setup
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

        # 2. Histogram with threshold highlighting
        # Let's draw vertical threshold bins manually to color block regions
        ax.axvspan(0.75, 1.00, color="#27ae60", alpha=0.08, label="High Reliability [0.75, 1.00]")
        ax.axvspan(0.50, 0.75, color="#f1c40f", alpha=0.08, label="Moderate Reliability [0.50, 0.75)")
        ax.axvspan(0.25, 0.50, color="#e67e22", alpha=0.08, label="Low Reliability [0.25, 0.50)")
        ax.axvspan(0.00, 0.25, color="#c0392b", alpha=0.08, label="Very Low Reliability [0.00, 0.25)")

        # Render the histogram on top of regions
        counts, bins, patches = ax.hist(
            scores, 
            bins=np.linspace(0.0, 1.0, 16),
            color="#34495e", 
            edgecolor="#2c3e50", 
            alpha=0.8, 
            rwidth=0.85,
            label="Mutation Frequencies"
        )

        # Add dashed boundary lines
        ax.axvline(0.75, color="#27ae60", linestyle="--", linewidth=1.2)
        ax.axvline(0.50, color="#f1c40f", linestyle="--", linewidth=1.2)
        ax.axvline(0.25, color="#e67e22", linestyle="--", linewidth=1.2)

        # Add labels to regions
        y_max = max(counts) if len(counts) > 0 else 10
        ax.text(0.875, y_max * 0.9, "High", color="#1b5e20", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.625, y_max * 0.9, "Moderate", color="#7f6000", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.375, y_max * 0.9, "Low", color="#a04000", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.125, y_max * 0.9, "Very Low", color="#7b241c", fontsize=11, fontweight="bold", ha="center")

        # 3. Text and labels
        ax.set_title("Figure F5: Engineering Reliability Score Distribution", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Prediction Reliability Score (1.0 = High Confidence, 0.0 = High Uncertainty)", fontsize=11, labelpad=10)
        ax.set_ylabel("Mutation Count", fontsize=11, labelpad=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, y_max + (y_max * 0.15))
        ax.grid(True, linestyle="--", alpha=0.4)

        # Statistical summary text annotation
        mean_rel = np.mean(scores)
        median_rel = np.median(scores)
        stats_text = (
            fr"$\mathbf{{Reliability\ Metrics}}$" "\n"
            f"Mean Score: {mean_rel:.3f}\n"
            f"Median Score: {median_rel:.3f}\n"
            f"Total Mutations $N$: {len(scores)}"
        )
        ax.text(
            0.03, 0.6, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9, linewidth=0.8)
        )

        ax.legend(loc="upper left", fontsize=9.5, frameon=True, edgecolor="#bdc3c7")
        plt.tight_layout()

        # 4. Save PNG and PDF
        png_path = os.path.join(out_dir, "F5_reliability_distribution.png")
        pdf_path = os.path.join(out_dir, "F5_reliability_distribution.pdf")
        
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format="pdf")
        plt.close()

        self.logger.info(f"Figure F5 successfully exported to: {png_path} and {pdf_path}")
        return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure F5: Reliability Distribution.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--input", type=str, default=None, help="Path to reliability scores parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        generator = ReliabilityFigureGenerator(config_path=args.config)
        generator.generate_figure(results_path=args.input, output_dir=args.output)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: Reliability Figure generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
