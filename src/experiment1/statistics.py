# src/experiment1/statistics.py
"""Module for computing Experiment 1 statistics.

Performs Pearson and Spearman correlations, linear regression, and quantile comparisons
between Combined Sparsity and Antisymmetry Error. Saves the statistical summaries to CSV files.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

def compute_statistics(config_path: str = "configs", output_dir_override: str = None) -> tuple[str, str, str]:
    loader = DatasetLoader(config_path=config_path)
    logger = get_logger(
        name="experiment1_statistics",
        log_dir=loader.config.paths.logs_dir,
        level=loader.config.logging.level,
    )
    
    logger.info("Initializing Experiment 1 statistical calculations...")
    print("Computing Experiment 1 statistics...")

    # Establish directories
    data_dir = loader.config.paths.data_dir
    results_dir = output_dir_override or os.path.join(loader.config.paths.results_dir, "experiment1")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(data_dir, "intermediate/experiment1/antisymmetry_results.parquet")
    
    if not os.path.exists(results_path):
        err_msg = f"Missing required input file for statistics: {results_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Load results
    df = pd.read_parquet(results_path)
    logger.info(f"Loaded {len(df)} records from {results_path} successfully.")

    x = df["combined_sparsity"].values
    y = df["antisymmetry_error"].values

    # 1. Pearson and Spearman Correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    corr_records = [
        {"metric": "Pearson Correlation (r)", "coefficient": pearson_r, "p_value": pearson_p},
        {"metric": "Spearman Rank Correlation (rho)", "coefficient": spearman_rho, "p_value": spearman_p}
    ]
    df_corr = pd.DataFrame(corr_records)
    corr_csv_path = os.path.join(results_dir, "experiment1_correlations.csv")
    df_corr.to_csv(corr_csv_path, index=False)
    logger.info(f"Wrote correlations to {corr_csv_path}")

    # 2. Linear Regression (y = slope * x + intercept)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    reg_records = {
        "slope": [slope],
        "intercept": [intercept],
        "r_squared": [r_value ** 2],
        "p_value": [p_value],
        "std_error": [std_err]
    }
    df_reg = pd.DataFrame(reg_records)
    reg_csv_path = os.path.join(results_dir, "experiment1_regression_summary.csv")
    df_reg.to_csv(reg_csv_path, index=False)
    logger.info(f"Wrote linear regression summary to {reg_csv_path}")

    # 3. Quantile Analysis (Lowest 10% vs Highest 10% Combined Sparsity)
    # Sort by Combined Sparsity
    df_sorted = df.sort_values(by="combined_sparsity").reset_index(drop=True)
    n = len(df_sorted)
    
    # 10% threshold index (at least 1 element)
    k = max(1, int(np.floor(0.10 * n)))
    
    lowest_10_df = df_sorted.head(k)
    highest_10_df = df_sorted.tail(k)

    avg_err_low = lowest_10_df["antisymmetry_error"].mean()
    avg_err_high = highest_10_df["antisymmetry_error"].mean()
    
    quantile_records = [
        {"subset": "Lowest 10% Sparsity (Dense)", "count": k, "mean_combined_sparsity": lowest_10_df["combined_sparsity"].mean(), "mean_antisymmetry_error": avg_err_low},
        {"subset": "Highest 10% Sparsity (Sparse)", "count": k, "mean_combined_sparsity": highest_10_df["combined_sparsity"].mean(), "mean_antisymmetry_error": avg_err_high},
        {"subset": "Difference (Sparse - Dense)", "count": k, "mean_combined_sparsity": highest_10_df["combined_sparsity"].mean() - lowest_10_df["combined_sparsity"].mean(), "mean_antisymmetry_error": avg_err_high - avg_err_low}
    ]
    df_quantile = pd.DataFrame(quantile_records)
    quantile_csv_path = os.path.join(results_dir, "quantile_analysis.csv")
    df_quantile.to_csv(quantile_csv_path, index=False)
    logger.info(f"Wrote quantile analysis to {quantile_csv_path}")

    print("\n--- Experiment 1 Statistical Summary ---")
    print(f"Pearson r   : {pearson_r:.4f} (p-value: {pearson_p:.2e})")
    print(f"Spearman rho: {spearman_rho:.4f} (p-value: {spearman_p:.2e})")
    print(f"Regression  : Slope={slope:.4f}, R-squared={r_value**2:.4f} (p-value: {p_value:.2e})")
    print(f"Quantiles   : Dense Mean Error={avg_err_low:.4f}, Sparse Mean Error={avg_err_high:.4f}")
    print(f"Hypothesis  : {'SUPPORTED (Average error is higher in sparse region)' if avg_err_high > avg_err_low else 'NOT SUPPORTED'}")
    print("-----------------------------------------")

    print(f"Statistics saved in: {results_dir}")
    return corr_csv_path, reg_csv_path, quantile_csv_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 statistics.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output CSVs.")
    args = parser.parse_args()

    try:
        compute_statistics(config_path=args.config, output_dir_override=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Statistics calculation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
