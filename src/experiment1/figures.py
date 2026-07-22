# src/experiment1/figures.py
"""Module for generating publication-quality figures for Experiment 1.

Generates:
1. Figure E1-1: Combined Sparsity vs Antisymmetry Error (Scatter plot + regression line)
2. Figure E1-2: Antisymmetry Error Distribution (Histogram)
3. Figure E1-3: Quantile Comparison (Bar plot comparing dense vs sparse extremes)
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
        name="experiment1_figures",
        log_dir=loader.config.paths.logs_dir,
        level=loader.config.logging.level,
    )
    
    logger.info("Initializing Experiment 1 figure generation...")
    print("Generating Experiment 1 figures...")

    # Establish directories
    data_dir = loader.config.paths.data_dir
    figures_dir = output_dir_override or os.path.join(os.getcwd(), loader.config.paths.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    results_path = os.path.join(data_dir, "intermediate/experiment1/antisymmetry_results.parquet")
    
    if not os.path.exists(results_path):
        err_msg = f"Missing required input file for figures: {results_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Load results
    df = pd.read_parquet(results_path)
    logger.info(f"Loaded {len(df)} records for visualization successfully.")

    x = df["combined_sparsity"].values
    y = df["antisymmetry_error"].values

    generated_plots = []

    # 1. Figure E1-1: Combined Sparsity vs Antisymmetry Error (Scatter Plot + Regression)
    logger.info("Plotting Figure E1-1: Combined Sparsity vs Antisymmetry Error...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate statistics for annotation
    r_val, p_val = pearsonr(x, y)
    slope, intercept, _, _, _ = linregress(x, y)
    
    # Plot scatter points
    ax.scatter(x, y, color="#2980b9", alpha=0.7, edgecolors="#1f3a52", s=50, label="Mutations")
    
    # Plot linear regression line
    x_line = np.linspace(0, 1.0, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#c0392b", linestyle="-", linewidth=2.5, label="Linear Regression Fit")
    
    # Format labels, title, and grid
    ax.set_title("Figure E1-1: Combined Sparsity vs. Thermodynamic Inconsistency", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Combined Sparsity (0.0 = Dense, 1.0 = Sparse)", fontsize=12)
    ax.set_ylabel("Antisymmetry Error |Forward + Reverse| (kcal/mol)", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, max(y) + 0.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Add statistical text box
    stats_text = f"Pearson r: {r_val:.3f}\np-value: {p_val:.2e}\nSlope: {slope:.3f}"
    ax.text(
        0.05, 0.95, stats_text, 
        transform=ax.transAxes, 
        verticalalignment="top", 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.9),
        fontsize=10,
        fontfamily="monospace"
    )
    
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    # Save Figure E1-1
    fig1_path = os.path.join(figures_dir, "figure_e1_1.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    generated_plots.append(fig1_path)
    logger.info(f"Saved Figure E1-1 to {fig1_path}")

    # 2. Figure E1-2: Antisymmetry Error Distribution (Histogram)
    logger.info("Plotting Figure E1-2: Antisymmetry Error Distribution...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mean_val = np.mean(y)
    median_val = np.median(y)

    # Plot histogram
    ax.hist(y, bins=15, color="#1abc9c", edgecolor="#16a085", alpha=0.7, rwidth=0.85)
    
    # Plot vertical lines for statistics
    ax.axvline(mean_val, color="#c0392b", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.3f} kcal/mol")
    ax.axvline(median_val, color="#e67e22", linestyle=":", linewidth=2.0, label=f"Median: {median_val:.3f} kcal/mol")
    
    # Format labels, title, and grid
    ax.set_title("Figure E1-2: Distribution of Antisymmetry Error", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Antisymmetry Error |Forward + Reverse| (kcal/mol)", fontsize=12)
    ax.set_ylabel("Mutation Count", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    # Save Figure E1-2
    fig2_path = os.path.join(figures_dir, "figure_e1_2.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    generated_plots.append(fig2_path)
    logger.info(f"Saved Figure E1-2 to {fig2_path}")

    # 3. Figure E1-3: Quantile Comparison
    logger.info("Plotting Figure E1-3: Quantile Comparison (Extreme Sparsity)...")
    # Identify top 10% and bottom 10% Combined Sparsity subgroups
    df_sorted = df.sort_values(by="combined_sparsity").reset_index(drop=True)
    n = len(df_sorted)
    k = max(1, int(np.floor(0.10 * n)))
    
    avg_err_low = df_sorted.head(k)["antisymmetry_error"].mean()
    avg_err_high = df_sorted.tail(k)["antisymmetry_error"].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ["Lowest 10% Sparsity\n(Dense Extremes)", "Highest 10% Sparsity\n(Sparse Extremes)"]
    values = [avg_err_low, avg_err_high]
    colors = ["#2ecc71", "#e74c3c"]
    
    # Plot bar chart
    bars = ax.bar(categories, values, color=colors, edgecolor="#2c3e50", width=0.5, alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 0.02, 
            f"{height:.3f} kcal/mol", 
            ha="center", 
            va="bottom", 
            fontsize=11, 
            fontweight="bold"
        )
        
    ax.set_title("Figure E1-3: Thermodynamic Inconsistency in Dense vs. Sparse Extremes", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Average Antisymmetry Error (kcal/mol)", fontsize=12)
    ax.set_ylim(0, max(values) + 0.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save Figure E1-3
    fig3_path = os.path.join(figures_dir, "figure_e1_3.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    generated_plots.append(fig3_path)
    logger.info(f"Saved Figure E1-3 to {fig3_path}")

    # Backwards compatibility output - save copies under results/experiment1/
    compat_dir = os.path.join(os.getcwd(), "results/experiment1")
    os.makedirs(compat_dir, exist_ok=True)
    import shutil
    shutil.copy(fig1_path, os.path.join(compat_dir, "figure_e1_1.png"))
    shutil.copy(fig2_path, os.path.join(compat_dir, "figure_e1_2.png"))
    shutil.copy(fig3_path, os.path.join(compat_dir, "figure_e1_3.png"))
    logger.info("Saved copy of all three figures to results/experiment1/")

    print(f"Figures saved in: {figures_dir}")
    return generated_plots

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Experiment 1 Figures.")
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
