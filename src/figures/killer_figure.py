# src/figures/killer_figure.py
"""The Killer Figure Module (Figure F9 / P9).

Combines the entire project story into a single, high-impact multi-panel publication dashboard.
The figure maps out the sequential scientific story from foundation to validation to prospective
industrial enzyme application across 8 distinct panels arranged in a 2x4 layout.
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
from scipy.stats import pearsonr, spearmanr, linregress
from typing import Tuple

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger

class KillerFigureGenerator:
    """Generates Figure F9: The flagship 'Killer Figure' for the project."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="killer_figure",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def generate_figure(self, output_dir: str = None) -> Tuple[str, str]:
        """Generates Figure F9: A magnificent 8-panel dashboard (2 rows x 4 columns)
        illustrating the complete project narrative:
        
        Row 1: Discovery & Validation
          - Panel 1: Megascale-D Combined Sparsity Foundation
          - Panel 2: Combined Sparsity vs. Antisymmetry Error (Discovery)
          - Panel 3: Combined Sparsity vs. Epistasis Prediction Error (Validation)
          - Panel 4: Antisymmetry vs. Epistasis Error (Causal Coupling Link)
          
        Row 2: Reliability Framework & Application
          - Panel 5: Reliability Score Framework Distribution with Thresholds
          - Panel 6: Complete Transketolase (TKT) Mutational Reliability Landscape (N=11,400)
          - Panel 7: TKT Industrial Fitness Prioritization Score Distribution
          - Panel 8: Top 10 Prioritized Prospective TKT Engineering Candidates
        """
        data_dir = self.config.paths.data_dir
        out_dir = output_dir or self.config.paths.figures_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. Resolve and Load all master datasets
        anti_path = os.path.join(data_dir, "intermediate/experiment1/antisymmetry_results.parquet")
        epi_path = os.path.join(data_dir, "intermediate/experiment2/epistasis_results.parquet")
        integ_path = os.path.join(data_dir, "final/reliability/integrated_reliability_analysis.parquet")
        rel_path = os.path.join(data_dir, "final/reliability/reliability_scores.parquet")
        tkt_path = os.path.join(data_dir, "final/tkt/tkt_mutation_analysis.parquet")
        fit_path = os.path.join(data_dir, "final/industrial_fitness/industrial_fitness_scores.parquet")
        cand_path = os.path.join(data_dir, "final/industrial_fitness/top_candidate_mutations.csv")

        # Verify all paths exist
        paths_to_verify = [anti_path, epi_path, integ_path, rel_path, tkt_path, fit_path, cand_path]
        if not all(os.path.exists(p) for p in paths_to_verify):
            raise FileNotFoundError("Missing one or more required datasets to compile the 8-panel Killer Figure (F9). Ensure all prior submodules have run successfully.")

        df_anti = pd.read_parquet(anti_path)
        df_epi = pd.read_parquet(epi_path)
        df_integ = pd.read_parquet(integ_path)
        df_rel = pd.read_parquet(rel_path)
        df_tkt = pd.read_parquet(tkt_path)
        df_fit = pd.read_parquet(fit_path)
        df_cand = pd.read_csv(cand_path)

        # 2. Multi-panel Figure Setup (2 rows x 4 columns)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axs = plt.subplots(2, 4, figsize=(22, 11), dpi=300)
        fig.suptitle("Figure F9: Discovering Mutation-Space Reliability & Designing Trustworthy Industrial Enzymes", 
                     fontsize=18, fontweight="bold", y=0.98)

        # Draw connecting arrows/lines conceptually at top to indicate the step-by-step narrative flow
        # But we can represent flow cleanly with numbered panel titles.

        # ----------------------------------------------------
        # Panel 1: Megascale-D Combined Sparsity Foundation
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 1: Combined Sparsity...")
        ax = axs[0, 0]
        sparsity_vals = df_rel["combined_sparsity"].values
        ax.hist(sparsity_vals, bins=12, color="#34495e", edgecolor="#2c3e50", alpha=0.8, rwidth=0.85)
        ax.set_title("1. Combined Sparsity Foundation", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Combined Sparsity (0=Dense, 1=Sparse)", fontsize=9.5)
        ax.set_ylabel("Mutation Count (Megascale-D)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.05, 0.95, f"Mean: {np.mean(sparsity_vals):.3f}\nMedian: {np.median(sparsity_vals):.3f}\nN: {len(sparsity_vals)}", 
                transform=ax.transAxes, verticalalignment="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.85))

        # ----------------------------------------------------
        # Panel 2: Combined Sparsity vs. Antisymmetry Error (Discovery)
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 2: Thermodynamic Discovery...")
        ax = axs[0, 1]
        x_a = df_anti["combined_sparsity"].values
        y_a = df_anti["antisymmetry_error"].values
        ax.scatter(x_a, y_a, color="#2980b9", edgecolor="#1c5980", s=25, alpha=0.7)
        
        slope_a, int_a, _, _, std_a = linregress(x_a, y_a)
        r_a, p_a = pearsonr(x_a, y_a)
        grid_a = np.linspace(0, 1, 100)
        ax.plot(grid_a, slope_a * grid_a + int_a, color="#c0392b", linewidth=1.8)
        ci_a = 1.96 * std_a * np.sqrt(1.0/len(x_a) + (grid_a - np.mean(x_a))**2 / np.sum((x_a - np.mean(x_a))**2))
        ax.fill_between(grid_a, (slope_a * grid_a + int_a) - ci_a, (slope_a * grid_a + int_a) + ci_a, color="#e74c3c", alpha=0.1)

        ax.set_title("2. Discovery: Sparsity vs. Inconsistency", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Combined Sparsity", fontsize=9.5)
        ax.set_ylabel("Antisymmetry Error (kcal/mol)", fontsize=9.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, max(y_a) + 0.3)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.05, 0.95, f"Pearson r: {r_a:.3f}\np: {p_a:.1e}\nSlope: {slope_a:.2f}", 
                transform=ax.transAxes, verticalalignment="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.85))

        # ----------------------------------------------------
        # Panel 3: Combined Sparsity vs. Epistasis Error (Validation)
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 3: Epistasis Validation...")
        ax = axs[0, 2]
        x_b = df_epi["combined_sparsity"].values
        y_b = df_epi["epistasis_error"].values
        ax.scatter(x_b, y_b, color="#e67e22", edgecolor="#b35400", s=20, alpha=0.6)
        
        slope_b, int_b, _, _, std_b = linregress(x_b, y_b)
        r_b, p_b = pearsonr(x_b, y_b)
        grid_b = np.linspace(0, 1, 100)
        ax.plot(grid_b, slope_b * grid_b + int_b, color="#c0392b", linewidth=1.8)
        ci_b = 1.96 * std_b * np.sqrt(1.0/len(x_b) + (grid_b - np.mean(x_b))**2 / np.sum((x_b - np.mean(x_b))**2))
        ax.fill_between(grid_b, (slope_b * grid_b + int_b) - ci_b, (slope_b * grid_b + int_b) + ci_b, color="#e74c3c", alpha=0.1)

        ax.set_title("3. Validation: Sparsity vs. Epistasis Error", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Pair-Level Combined Sparsity", fontsize=9.5)
        ax.set_ylabel("Epistasis Prediction Error (kcal/mol)", fontsize=9.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, max(y_b) + 0.3)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.05, 0.95, f"Pearson r: {r_b:.3f}\np: {p_b:.1e}\nSlope: {slope_b:.2f}", 
                transform=ax.transAxes, verticalalignment="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.85))

        # ----------------------------------------------------
        # Panel 4: Antisymmetry vs. Epistasis Error (Causal Coupling)
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 4: Causal Link...")
        ax = axs[0, 3]
        x_c = df_integ["antisymmetry_error"].values
        y_c = df_integ["epistasis_error"].values
        ax.scatter(x_c, y_c, color="#8e44ad", edgecolor="#5e2273", s=25, alpha=0.7)
        
        slope_c, int_c, _, _, std_c = linregress(x_c, y_c)
        r_c, p_c = pearsonr(x_c, y_c)
        grid_c = np.linspace(0, max(x_c), 100)
        ax.plot(grid_c, slope_c * grid_c + int_c, color="#c0392b", linewidth=1.8)
        ci_c = 1.96 * std_c * np.sqrt(1.0/len(x_c) + (grid_c - np.mean(x_c))**2 / np.sum((x_c - np.mean(x_c))**2))
        ax.fill_between(grid_c, (slope_c * grid_c + int_c) - ci_c, (slope_c * grid_c + int_c) + ci_c, color="#e74c3c", alpha=0.1)

        ax.set_title("4. Causal Link: Error Coupling", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Antisymmetry Error (kcal/mol)", fontsize=9.5)
        ax.set_ylabel("Epistasis Error (kcal/mol)", fontsize=9.5)
        ax.set_xlim(-0.05, max(x_c) + 0.1)
        ax.set_ylim(-0.1, max(y_c) + 0.3)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.05, 0.95, f"Pearson r: {r_c:.3f}\np: {p_c:.1e}\nSlope: {slope_c:.2f}", 
                transform=ax.transAxes, verticalalignment="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.85))

        # ----------------------------------------------------
        # Panel 5: Reliability Score Framework Distribution
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 5: Reliability Framework...")
        ax = axs[1, 0]
        scores = df_rel["reliability_score"].values
        ax.axvspan(0.75, 1.00, color="#27ae60", alpha=0.08)
        ax.axvspan(0.50, 0.75, color="#f1c40f", alpha=0.08)
        ax.axvspan(0.25, 0.50, color="#e67e22", alpha=0.08)
        ax.axvspan(0.00, 0.25, color="#c0392b", alpha=0.08)

        counts, bins, _ = ax.hist(scores, bins=np.linspace(0.0, 1.0, 11), color="#2c3e50", edgecolor="#2c3e50", alpha=0.7, rwidth=0.8)
        ax.axvline(0.75, color="#27ae60", linestyle="--", linewidth=1.0)
        ax.axvline(0.50, color="#f1c40f", linestyle="--", linewidth=1.0)
        ax.axvline(0.25, color="#e67e22", linestyle="--", linewidth=1.0)

        # Highlight regions
        y_max = max(counts) if len(counts) > 0 else 10
        ax.text(0.875, y_max * 0.8, "High", color="#1b5e20", fontsize=8.5, fontweight="bold", ha="center")
        ax.text(0.625, y_max * 0.8, "Mod", color="#7f6000", fontsize=8.5, fontweight="bold", ha="center")
        ax.text(0.375, y_max * 0.8, "Low", color="#a04000", fontsize=8.5, fontweight="bold", ha="center")
        ax.text(0.125, y_max * 0.8, "Very Low", color="#7b241c", fontsize=8.5, fontweight="bold", ha="center")

        ax.set_title("5. Reliability Score Framework", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Reliability Score (1 - Sparsity)", fontsize=9.5)
        ax.set_ylabel("Mutation Count", fontsize=9.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, y_max * 1.1)
        ax.grid(True, linestyle="--", alpha=0.4)

        # ----------------------------------------------------
        # Panel 6: Complete Transketolase (TKT) Mutational Reliability Landscape
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 6: TKT Landscape...")
        ax = axs[1, 1]
        
        # Parse position out of mutation_id if not present
        if "position" not in df_tkt.columns:
            positions = []
            for mut_id in df_tkt["mutation_id"]:
                parts = mut_id.split("_")
                if len(parts) == 2:
                    label = parts[1]
                    pos_str = "".join([c for c in label if c.isdigit()])
                    positions.append(int(pos_str))
                else:
                    positions.append(1)
            df_tkt["position"] = positions

        # Color categories: High=Green, Moderate=Yellow, Low/VeryLow=Red
        colors = []
        for s in df_tkt["reliability_score"]:
            if s >= 0.75:
                colors.append("#2ecc71") # Green
            elif s >= 0.50:
                colors.append("#f1c40f") # Yellow
            else:
                colors.append("#e74c3c") # Red
        df_tkt["color_hex"] = colors

        ax.scatter(df_tkt["position"], df_tkt["reliability_score"], c=df_tkt["color_hex"], s=1, alpha=0.3, edgecolors="none")
        ax.set_title("6. TKT Reliability Landscape", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Enzyme Sequence Position (1-600)", fontsize=9.5)
        ax.set_ylabel("Reliability Score", fontsize=9.5)
        ax.set_xlim(0, 605)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Custom small color boxes legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor='#2ecc71', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Mod', markerfacecolor='#f1c40f', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Low/VL', markerfacecolor='#e74c3c', markersize=6)
        ]
        ax.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=8, edgecolor="#bdc3c7")

        # ----------------------------------------------------
        # Panel 7: TKT Industrial Fitness Prioritization Score Distribution
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 7: Fitness Prioritization...")
        ax = axs[1, 2]
        ax.hist(df_fit["industrial_fitness_score"], bins=12, color="#2980b9", edgecolor="#1f3a52", alpha=0.75, rwidth=0.85)
        ax.set_title("7. Prioritization Fitness Scores", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Industrial Fitness (Stability + Reliability + Plausibility)", fontsize=9.5)
        ax.set_ylabel("Mutation Count (TKT Space)", fontsize=9.5)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)

        # ----------------------------------------------------
        # Panel 8: Top 10 Prioritized Prospective TKT Engineering Candidates
        # ----------------------------------------------------
        self.logger.info("Compiling Panel 8: Candidate Rankings...")
        ax = axs[1, 3]
        df_cand_top10 = df_cand.head(10).copy().iloc[::-1].reset_index(drop=True)
        colors_bar = ["#2ecc71" if r >= 0.75 else "#f1c40f" for r in df_cand_top10["reliability_score"]]
        
        bars = ax.barh(df_cand_top10["mutation"], df_cand_top10["industrial_fitness_score"], color=colors_bar, edgecolor="#2c3e50", height=0.55)
        for bar, idx in zip(bars, df_cand_top10.index):
            fit_val = df_cand_top10.loc[idx, "industrial_fitness_score"]
            stab_val = df_cand_top10.loc[idx, "predicted_stability"]
            ax.text(
                bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2.0, 
                f" {fit_val:.3f} (S: {stab_val:+.2f})", 
                va="center", 
                fontsize=7.5, 
                fontweight="bold"
            )

        ax.set_title("8. Top TKT Candidates", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Industrial Fitness Score", fontsize=9.5)
        ax.set_ylabel("Mutation", fontsize=9.5)
        ax.set_xlim(0, 1.35)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Add panel numbers visually inside each plot's corner
        panel_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for idx, sub_ax in enumerate(axs.flat):
            sub_ax.text(-0.08, 1.05, panel_letters[idx], transform=sub_ax.transAxes, fontsize=14, fontweight="bold", va="top")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save F9 PNG and PDF
        png_path = os.path.join(out_dir, "F9_killer_figure.png")
        pdf_path = os.path.join(out_dir, "F9_killer_figure.pdf")
        
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format="pdf")
        plt.close()

        self.logger.info(f"Flagship Killer Figure successfully generated and saved to {png_path} and {pdf_path}")
        return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure F9: The flagship 8-panel Killer Figure.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        generator = KillerFigureGenerator(config_path=args.config)
        generator.generate_figure(output_dir=args.output)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: Killer Figure generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
