# src/integrated/integrated_figures.py
"""Module for generating publication-quality figures for Integrated Reliability Analysis.

Generates:
1. Figure I-1: Combined Sparsity vs. Antisymmetry Error (Scatter + regression line)
2. Figure I-2: Combined Sparsity vs. Epistasis Error (Scatter + regression line)
3. Figure I-3: Antisymmetry Error vs. Epistasis Error (Scatter + regression line)
4. Figure I-4: Integrated Reliability Model (Path mediation flow diagram)
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

def generate_figures(config_path: str = "configs", output_dir_override: str = None) -> list[str]:
    loader = DatasetLoader(config_path=config_path)
    logger = get_logger(
        name="integrated_figures",
        log_dir=loader.config.paths.logs_dir,
        level=loader.config.logging.level,
    )
    
    logger.info("Initializing Integrated Reliability figure generation...")
    print("Generating Integrated Reliability figures...")

    # Establish directories
    data_dir = loader.config.paths.data_dir
    figures_dir = output_dir_override or os.path.join(os.getcwd(), loader.config.paths.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    d401_path = os.path.join(data_dir, "final/reliability/integrated_reliability_analysis.parquet")
    if not os.path.exists(d401_path):
        raise FileNotFoundError(f"Missing required D401 Integrated dataset for visualization: {d401_path}")

    # Load results
    df = pd.read_parquet(d401_path)
    logger.info(f"Loaded {len(df)} integrated records successfully.")

    x_sp = df["combined_sparsity"].values
    y_anti = df["antisymmetry_error"].values
    z_epi = df["epistasis_error"].values

    generated_plots = []

    # 1. Figure I-1: Combined Sparsity vs. Antisymmetry Error
    logger.info("Plotting Figure I-1: Combined Sparsity vs. Antisymmetry Error...")
    fig, ax = plt.subplots(figsize=(8, 6))
    r_val, p_val = pearsonr(x_sp, y_anti)
    slope, intercept, _, _, _ = linregress(x_sp, y_anti)
    
    ax.scatter(x_sp, y_anti, color="#2980b9", alpha=0.75, edgecolors="#1a5276", s=50, label="Mutations")
    x_line = np.linspace(min(x_sp), max(x_sp), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#c0392b", linestyle="-", linewidth=2.0, label="Linear Regression Fit")
    
    ax.set_title("Figure I-1: Combined Sparsity vs. Thermodynamic Inconsistency", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Combined Sparsity (0.0 = Dense, 1.0 = Sparse)", fontsize=11)
    ax.set_ylabel("Antisymmetry Error |Forward + Reverse| (kcal/mol)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    stats_text = f"Pearson r: {r_val:.3f}\np-value: {p_val:.2e}\nSlope: {slope:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.9),
            fontsize=9, fontfamily="monospace")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "figure_i_1.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    generated_plots.append(fig1_path)

    # 2. Figure I-2: Combined Sparsity vs. Epistasis Error
    logger.info("Plotting Figure I-2: Combined Sparsity vs. Epistasis Error...")
    fig, ax = plt.subplots(figsize=(8, 6))
    r_val, p_val = pearsonr(x_sp, z_epi)
    slope, intercept, _, _, _ = linregress(x_sp, z_epi)
    
    ax.scatter(x_sp, z_epi, color="#e67e22", alpha=0.75, edgecolors="#a04000", s=50, label="Mutations")
    ax.plot(x_line, slope * x_line + intercept, color="#2c3e50", linestyle="-", linewidth=2.0, label="Linear Regression Fit")
    
    ax.set_title("Figure I-2: Combined Sparsity vs. Epistasis Prediction Failure", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Combined Sparsity (0.0 = Dense, 1.0 = Sparse)", fontsize=11)
    ax.set_ylabel("Average Epistasis Error |Predicted - Experimental| (kcal/mol)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    stats_text = f"Pearson r: {r_val:.3f}\np-value: {p_val:.2e}\nSlope: {slope:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdf2e9", edgecolor="#e59866", alpha=0.9),
            fontsize=9, fontfamily="monospace")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "figure_i_2.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    generated_plots.append(fig2_path)

    # 3. Figure I-3: Antisymmetry Error vs. Epistasis Error
    logger.info("Plotting Figure I-3: Antisymmetry Error vs. Epistasis Error...")
    fig, ax = plt.subplots(figsize=(8, 6))
    r_val, p_val = pearsonr(y_anti, z_epi)
    slope, intercept, _, _, _ = linregress(y_anti, z_epi)
    
    ax.scatter(y_anti, z_epi, color="#8e44ad", alpha=0.75, edgecolors="#4a235a", s=50, label="Mutations")
    x_line_anti = np.linspace(min(y_anti), max(y_anti), 100)
    ax.plot(x_line_anti, slope * x_line_anti + intercept, color="#16a085", linestyle="-", linewidth=2.0, label="Linear Regression Fit")
    
    ax.set_title("Figure I-3: Thermodynamic Inconsistency vs. Epistasis Failure", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Antisymmetry Error |Forward + Reverse| (kcal/mol)", fontsize=11)
    ax.set_ylabel("Average Epistasis Error |Predicted - Experimental| (kcal/mol)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    stats_text = f"Pearson r: {r_val:.3f}\np-value: {p_val:.2e}\nSlope: {slope:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5eef8", edgecolor="#bb8fce", alpha=0.9),
            fontsize=9, fontfamily="monospace")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "figure_i_3.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    generated_plots.append(fig3_path)

    # 4. Figure I-4: Integrated Reliability Model
    # We plot a mediation path flow diagram illustrating path coefficients between key elements
    logger.info("Plotting Figure I-4: Integrated Reliability Model flow diagram...")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate path regression coefficients
    slope_sp_anti, _, _, p_sp_anti, _ = linregress(x_sp, y_anti)
    slope_sp_epi, _, _, p_sp_epi, _ = linregress(x_sp, z_epi)
    slope_anti_epi, _, _, p_anti_epi, _ = linregress(y_anti, z_epi)
    
    # Set plot bounds and hide axes to draw a beautiful schematic
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    
    # Draw boxes for variables
    box_sp = dict(boxstyle="round,pad=0.8", facecolor="#f5cba7", edgecolor="#d35400", lw=2)
    box_anti = dict(boxstyle="round,pad=0.8", facecolor="#aed6f1", edgecolor="#2e86c1", lw=2)
    box_epi = dict(boxstyle="round,pad=0.8", facecolor="#d2b4de", edgecolor="#7d3c98", lw=2)
    
    ax.text(1.5, 2.5, "Combined\nSparsity\n(Information)", ha="center", va="center", bbox=box_sp, fontsize=12, fontweight="bold")
    ax.text(5.0, 3.8, "Antisymmetry\nError\n(Inconsistency)", ha="center", va="center", bbox=box_anti, fontsize=12, fontweight="bold")
    ax.text(8.5, 2.5, "Epistasis\nError\n(Failure)", ha="center", va="center", bbox=box_epi, fontsize=12, fontweight="bold")
    
    # Draw arrows and label with coefficients
    # Arrow 1: Sparsity -> Antisymmetry Error
    ax.annotate("", xy=(3.8, 3.6), xytext=(2.2, 2.8), arrowprops=dict(arrowstyle="->", lw=2.5, color="#1abc9c"))
    ax.text(2.6, 3.4, fr"$\beta$ = {slope_sp_anti:.3f}" + f"\n($p$ = {p_sp_anti:.2f})", fontsize=11, fontweight="bold", color="#16a085")
    
    # Arrow 2: Antisymmetry Error -> Epistasis Error
    ax.annotate("", xy=(7.8, 2.8), xytext=(6.2, 3.6), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2ecc71"))
    ax.text(7.4, 3.4, fr"$\beta$ = {slope_anti_epi:.3f}" + f"\n($p$ = {p_anti_epi:.2f})", fontsize=11, fontweight="bold", color="#27ae60")
    
    # Arrow 3: Sparsity -> Epistasis Error (Direct path)
    ax.annotate("", xy=(7.2, 2.5), xytext=(2.8, 2.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
    ax.text(5.0, 2.1, fr"$\beta$ = {slope_sp_epi:.3f}" + f"\n($p$ = {p_sp_epi:.2e})", ha="center", va="top", fontsize=11, fontweight="bold", color="#c0392b")
    
    ax.set_title("Figure I-4: Unified Scientific Reliability Mediation Chain Model", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    
    fig4_path = os.path.join(figures_dir, "figure_i_4.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    generated_plots.append(fig4_path)
    logger.info(f"Saved Figure I-4 to {fig4_path}")

    # Backwards compatibility output - save copies under results/integrated/
    compat_dir = os.path.join(os.getcwd(), "results/integrated")
    os.makedirs(compat_dir, exist_ok=True)
    import shutil
    shutil.copy(fig1_path, os.path.join(compat_dir, "figure_i_1.png"))
    shutil.copy(fig2_path, os.path.join(compat_dir, "figure_i_2.png"))
    shutil.copy(fig3_path, os.path.join(compat_dir, "figure_i_3.png"))
    shutil.copy(fig4_path, os.path.join(compat_dir, "figure_i_4.png"))
    logger.info("Saved copy of all four figures to results/integrated/")

    print(f"Figures saved in: {figures_dir}")
    return generated_plots

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Integrated Figures.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output PNG figures.")
    args = parser.parse_args()

    try:
        generate_figures(config_path=args.config, output_dir_override=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Figure generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
