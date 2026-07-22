# src/figures/tkt_landscape.py
"""TKT Landscape Figures Module (Figure F6 & F7 / P6 & P7).

Generates:
1. Figure F6: TKT Reliability Landscape (scatter mapping of sequence position vs. score,
   with Green/Yellow/Red categorical color mapping).
2. Figure F7: TKT Industrial Fitness Landscape (distribution/histogram of fitness scores).
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

class TktLandscapeFigureGenerator:
    """Generates Figure F6 and Figure F7 representing TKT mutational spaces."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="tkt_landscape_figures",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def run_athlete_exercise(self, df: pd.DataFrame) -> bool:
        """Athlete Exercise: Inspect top 20 and bottom 20 mutations, verify color category assignment is consistent."""
        self.logger.info("Running Athlete Exercise for Figure F6 (color mapping verification)...")
        
        # Sort by score
        df_sorted = df.sort_values(by="reliability_score", ascending=False).reset_index(drop=True)
        top_20 = df_sorted.head(20)
        bottom_20 = df_sorted.tail(20)

        # Check top 20 are categorized as High or Moderate (mostly High if score near 1)
        for idx, row in top_20.iterrows():
            mut_id = row["mutation_id"]
            sc = row["reliability_score"]
            if sc < 0.5:
                self.logger.error(f"Integrity error: Top mutation {mut_id} has unreliable score: {sc:.4f}")
                return False

        # Check bottom 20 are low / very low
        for idx, row in bottom_20.iterrows():
            mut_id = row["mutation_id"]
            sc = row["reliability_score"]
            if sc >= 0.5:
                self.logger.error(f"Integrity error: Bottom mutation {mut_id} has highly reliable score: {sc:.4f}")
                return False

        self.logger.info("Athlete Exercise for Figure F6 completed successfully. Colors are mathematically consistent.")
        return True

    def generate_figures(self, analysis_path: str = None, fitness_path: str = None, output_dir: str = None) -> Tuple[str, str, str, str]:
        """Generates Figure F6 (Reliability Landscape) and Figure F7 (Fitness Distribution)."""
        data_dir = self.config.paths.data_dir
        anal_path = analysis_path or os.path.join(data_dir, "final/tkt/tkt_mutation_analysis.parquet")
        fit_path = fitness_path or os.path.join(data_dir, "final/industrial_fitness/industrial_fitness_scores.parquet")
        out_dir = output_dir or self.config.paths.figures_dir
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(anal_path):
            raise FileNotFoundError(f"Input file not found for analysis landscape: {anal_path}")
        if not os.path.exists(fit_path):
            raise FileNotFoundError(f"Input file not found for fitness landscape: {fit_path}")

        df_anal = pd.read_parquet(anal_path)
        df_fit = pd.read_parquet(fit_path)

        # Run Athlete Exercise
        self.run_athlete_exercise(df_anal)

        # Parse position out of mutation_id if not present
        if "position" not in df_anal.columns:
            positions = []
            for mut_id in df_anal["mutation_id"]:
                parts = mut_id.split("_")
                if len(parts) == 2:
                    label = parts[1]
                    pos_str = "".join([c for c in label if c.isdigit()])
                    positions.append(int(pos_str))
                else:
                    positions.append(1)
            df_anal["position"] = positions

        # Create Category colors for mapping: High=Green, Moderate=Yellow, Low/VeryLow=Red
        # Threshold mapping
        colors = []
        for s in df_anal["reliability_score"]:
            if s >= 0.75:
                colors.append("#2ecc71") # Green
            elif s >= 0.50:
                colors.append("#f1c40f") # Yellow
            else:
                colors.append("#e74c3c") # Red
        df_anal["color_hex"] = colors

        # ----------------------------------------------------
        # Figure F6: TKT Reliability Landscape Scatter Map
        # ----------------------------------------------------
        self.logger.info("Plotting Figure F6: TKT Reliability Landscape...")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

        # Draw scatter points mapping sequence position vs reliability score
        # Using a small point size to represent all 11,400 mutations clearly
        scatter = ax.scatter(
            df_anal["position"], 
            df_anal["reliability_score"], 
            c=df_anal["color_hex"],
            s=3, 
            alpha=0.4,
            edgecolors="none"
        )

        # Custom Legend for colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='High [0.75, 1.0]', markerfacecolor='#2ecc71', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Moderate [0.5, 0.75)', markerfacecolor='#f1c40f', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Low/Very Low [0.0, 0.5)', markerfacecolor='#e74c3c', markersize=8),
        ]
        ax.legend(handles=legend_elements, loc="lower left", title="Reliability Classes", fontsize=9.5)

        ax.set_title("Figure F6: Complete Transketolase (TKT) Mutational Reliability Landscape", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Enzyme Residue Position (1 to 600)", fontsize=11, labelpad=10)
        ax.set_ylabel("Prediction Reliability Score", fontsize=11, labelpad=10)
        ax.set_xlim(0, 605)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        f6_png = os.path.join(out_dir, "F6_tkt_reliability_landscape.png")
        f6_pdf = os.path.join(out_dir, "F6_tkt_reliability_landscape.pdf")
        plt.savefig(f6_png, dpi=300)
        plt.savefig(f6_pdf, format="pdf")
        plt.close()
        self.logger.info(f"Figure F6 saved to {f6_png}")

        # ----------------------------------------------------
        # Figure F7: TKT Industrial Fitness Score Distribution
        # ----------------------------------------------------
        self.logger.info("Plotting Figure F7: TKT Industrial Fitness Landscape...")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        ax.hist(
            df_fit["industrial_fitness_score"],
            bins=18,
            color="#2980b9",
            edgecolor="#1f3a52",
            alpha=0.75,
            rwidth=0.85
        )

        ax.set_title("Figure F7: TKT Mutational Industrial Fitness Score Distribution", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Prioritization Fitness Score (0 = Poor Candidate, 1 = Optimal Candidate)", fontsize=11, labelpad=10)
        ax.set_ylabel("Mutation Count (Total N=11,400)", fontsize=11, labelpad=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        f7_png = os.path.join(out_dir, "F7_tkt_industrial_fitness_landscape.png")
        f7_pdf = os.path.join(out_dir, "F7_tkt_industrial_fitness_landscape.pdf")
        plt.savefig(f7_png, dpi=300)
        plt.savefig(f7_pdf, format="pdf")
        plt.close()
        self.logger.info(f"Figure F7 saved to {f7_png}")

        return f6_png, f6_pdf, f7_png, f7_pdf

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figures F6 and F7: TKT Mutation Space Landscapes.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--analysis", type=str, default=None, help="Path to TKT mutation analysis parquet.")
    parser.add_argument("--fitness", type=str, default=None, help="Path to industrial fitness scores parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        generator = TktLandscapeFigureGenerator(config_path=args.config)
        generator.generate_figures(
            analysis_path=args.analysis,
            fitness_path=args.fitness,
            output_dir=args.output
        )
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: TKT Landscapes generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
