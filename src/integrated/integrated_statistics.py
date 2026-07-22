# src/integrated/integrated_statistics.py
"""Module for computing pairwise statistical relationships for Integrated Reliability Analysis.

Computes Pearson correlation, Spearman rank correlation, and linear regression parameters
for the three key relationships of the scientific centerpiece:
1. Combined Sparsity vs. Antisymmetry Error
2. Combined Sparsity vs. Epistasis Error
3. Antisymmetry Error vs. Epistasis Error
Saves results in results/integrated/integrated_statistics.csv.
"""

import os
import argparse
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

def run_statistics(config_path: str = "configs", output_dir_override: str = None) -> str:
    loader = DatasetLoader(config_path=config_path)
    logger = get_logger(
        name="integrated_statistics",
        log_dir=loader.config.paths.logs_dir,
        level=loader.config.logging.level,
    )

    logger.info("Initializing Integrated Reliability statistical calculations...")
    print("Computing Integrated Reliability statistics...")

    data_dir = loader.config.paths.data_dir
    results_dir = output_dir_override or os.path.join(loader.config.paths.results_dir, "integrated")
    os.makedirs(results_dir, exist_ok=True)

    d401_path = os.path.join(data_dir, "final/reliability/integrated_reliability_analysis.parquet")
    if not os.path.exists(d401_path):
        raise FileNotFoundError(f"Missing required D401 Integrated dataset at {d401_path}")

    df = pd.read_parquet(d401_path)
    logger.info(f"Loaded {len(df)} integrated records successfully.")

    # Relationships to analyze
    relationships = [
        ("Combined Sparsity vs. Antisymmetry Error", "combined_sparsity", "antisymmetry_error"),
        ("Combined Sparsity vs. Epistasis Error", "combined_sparsity", "epistasis_error"),
        ("Antisymmetry Error vs. Epistasis Error", "antisymmetry_error", "epistasis_error")
    ]

    stats_records = []

    for label, x_col, y_col in relationships:
        logger.info(f"Analyzing relationship: {label}")
        x = df[x_col].values
        y = df[y_col].values

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        stats_records.append({
            "relationship": label,
            "x_variable": x_col,
            "y_variable": y_col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "regression_p_value": p_value,
            "std_error": std_err
        })

    df_stats = pd.DataFrame(stats_records)
    
    # Save statistics CSV
    stats_csv_path = os.path.join(results_dir, "integrated_statistics.csv")
    df_stats.to_csv(stats_csv_path, index=False)

    # Backwards compatibility output to top-level folder or relative to results
    compat_csv_path = os.path.join(results_dir, "../../results/integrated_statistics.csv")
    os.makedirs(os.path.dirname(compat_csv_path), exist_ok=True)
    df_stats.to_csv(compat_csv_path, index=False)

    logger.info(f"Wrote integrated statistics CSVs to: {stats_csv_path}")

    print("\n--- Integrated Reliability Statistical Summary ---")
    for rec in stats_records:
        print(f"[{rec['relationship']}]")
        print(f"  - Pearson r   : {rec['pearson_r']:.4f} (p-value: {rec['pearson_p']:.2e})")
        print(f"  - Spearman rho: {rec['spearman_rho']:.4f} (p-value: {rec['spearman_p']:.2e})")
        print(f"  - Regression  : Slope={rec['slope']:.4f}, R-squared={rec['r_squared']:.4f} (p-value: {rec['regression_p_value']:.2e})")
    print("--------------------------------------------------")

    return stats_csv_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Integrated Reliability Statistics.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output.")
    args = parser.parse_args()

    try:
        run_statistics(config_path=args.config, output_dir_override=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Integrated statistics calculation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
