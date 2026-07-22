# src/figures/candidate_discovery.py
"""Candidate Discovery Figures Module (Figure F8 / P8).

Plots the Top 15 Candidate Mutations as a professional, publication-ready horizontal
bar chart. Shows predicted stability benefit colored by reliability category, displaying values.
Outputs both PNG and PDF.
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

class CandidateDiscoveryFigureGenerator:
    """Generates Figure F8: Top Candidate Mutations visual list."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="candidate_figure",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def run_athlete_exercise(self, df_csv: pd.DataFrame) -> bool:
        """Candidate Validation Exercise: Verify Rank 1 matches CSV and score ordering."""
        self.logger.info("Running Candidate Validation Exercise for Figure F8...")
        
        # Verify Rank 1 has highest fitness
        first_row = df_csv.iloc[0]
        if first_row["rank"] != 1:
            self.logger.error("Rank 1 is not the first row!")
            return False
            
        # Verify top 10 entries order
        for i in range(min(10, len(df_csv) - 1)):
            if df_csv.loc[i, "industrial_fitness_score"] < df_csv.loc[i+1, "industrial_fitness_score"]:
                self.logger.error(f"Ranking score ordering violation at index {i}!")
                return False
                
        self.logger.info("Candidate Validation Exercise for Figure F8 passed successfully.")
        return True

    def generate_figure(self, csv_path: str = None, output_dir: str = None) -> Tuple[str, str]:
        """Generates Figure F8: Top Candidate Mutations horizontal bar plot."""
        data_dir = self.config.paths.data_dir
        top_path = csv_path or os.path.join(data_dir, "final/industrial_fitness/top_candidate_mutations.csv")
        out_dir = output_dir or self.config.paths.figures_dir
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(top_path):
            raise FileNotFoundError(f"Input file not found for candidate figure: {top_path}")

        df = pd.read_csv(top_path)
        self.logger.info(f"Loaded {len(df)} records for Figure F8 generation.")

        # Run Athlete Validation Exercise
        self.run_athlete_exercise(df)

        # Take Top 15 for a clean publication plot
        df_plot = df.head(15).copy()
        # Invert rows so rank 1 is at the top of the horizontal bar plot
        df_plot = df_plot.iloc[::-1].reset_index(drop=True)

        # 1. Plot Setup
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

        # Color code bars by Reliability Category
        # High=Green (#2ecc71), Moderate=Yellow (#f1c40f), Low=Red (#e74c3c)
        colors = []
        for r in df_plot["reliability_score"]:
            if r >= 0.75:
                colors.append("#2ecc71")
            elif r >= 0.50:
                colors.append("#f1c40f")
            else:
                colors.append("#e74c3c")

        # 2. Horizontal Bar Plot of Fitness Scores
        bars = ax.barh(
            df_plot["mutation"], 
            df_plot["industrial_fitness_score"], 
            color=colors, 
            edgecolor="#2c3e50", 
            height=0.6,
            linewidth=0.8
        )

        # 3. Add value annotations next to the bars
        for bar, idx in zip(bars, df_plot.index):
            fitness = df_plot.loc[idx, "industrial_fitness_score"]
            stability = df_plot.loc[idx, "predicted_stability"]
            reliability = df_plot.loc[idx, "reliability_score"]
            
            # Format: Fitness (Stability, Reliability)
            annotation_text = f" {fitness:.3f} (S: {stability:+.2f}, R: {reliability:.2f})"
            ax.text(
                bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2.0, 
                annotation_text, 
                va="center", 
                fontsize=9.5, 
                fontweight="bold"
            )

        # 4. Custom legend for color categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='#2c3e50', label='High Reliability [0.75, 1.0]'),
            Patch(facecolor='#f1c40f', edgecolor='#2c3e50', label='Moderate Reliability [0.50, 0.75)'),
        ]
        ax.legend(handles=legend_elements, loc="lower right", title="Candidate Reliability Classes", fontsize=10)

        # 5. Styling & Labels
        ax.set_title("Figure F8: Top 15 Prioritized Transketolase Engineering Candidates", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Industrial Fitness Score (0 = Poor Candidate, 1 = Optimal Candidate)", fontsize=11, labelpad=10)
        ax.set_ylabel("Mutation ID (WT-Position-Mutant)", fontsize=11, labelpad=10)
        ax.set_xlim(0, 1.35) # leave extra space for the long annotation text
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        # 6. Save PNG and PDF
        png_path = os.path.join(out_dir, "F8_top_candidate_mutations.png")
        pdf_path = os.path.join(out_dir, "F8_top_candidate_mutations.pdf")
        
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format="pdf")
        plt.close()

        self.logger.info(f"Figure F8 successfully exported to: {png_path} and {pdf_path}")
        return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure F8: Top Candidate Mutations.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--input", type=str, default=None, help="Path to top candidate CSV.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        generator = CandidateDiscoveryFigureGenerator(config_path=args.config)
        generator.generate_figure(csv_path=args.input, output_dir=args.output)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: Candidate Figure generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
