# src/sparsity/empirical/empirical_sparsity.py
"""Empirical Sparsity calculation module.

Calculates mutation frequency, normalized frequency, and empirical sparsity
using raw experimental mutation counts.
"""

import argparse
import os
import sys
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for headless systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.validation_engine import ValidationEngine


class EmpiricalSparsity:
    """Computes empirical sparsity for a set of mutations based on experimental counts."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize EmpiricalSparsity with central config loader and registries.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="empirical_sparsity",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def load_counts(self, input_path: str) -> pd.DataFrame:
        """Loads and validates the raw mutation counts.

        Parameters
        ----------
        input_path : str
            Absolute or relative path to the mutation_counts.csv file.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame containing columns: mutation_id, count.

        Raises
        ------
        FileNotFoundError
            If input_path does not exist.
        ValueError
            If validation fails (missing columns, duplicate keys, nulls, negative counts).
        """
        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found at: {input_path}")
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        self.logger.info(f"Loading mutation counts from: {input_path}")
        df = pd.read_csv(input_path)

        # 1. Check Required Columns
        required_cols = ["mutation_id", "count"]
        for col in required_cols:
            if col not in df.columns:
                err_msg = f"Missing required column in input counts: '{col}'"
                self.logger.error(err_msg)
                raise ValueError(err_msg)

        # 2. Check for Nulls
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                err_msg = f"Column '{col}' contains {null_count} null/missing value(s)."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

        # 3. Check for Duplicate Mutation IDs
        if df["mutation_id"].duplicated().any():
            num_duplicates = df["mutation_id"].duplicated().sum()
            err_msg = f"Column 'mutation_id' contains {num_duplicates} duplicate mutation ID(s)."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # 4. Check for Negative Counts
        negative_counts = df[df["count"] < 0]
        if not negative_counts.empty:
            num_violations = len(negative_counts)
            min_observed = negative_counts["count"].min()
            err_msg = (
                f"Validation failure: Found {num_violations} negative mutation count(s). "
                f"Minimum observed: {min_observed}."
            )
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        self.logger.info(f"Successfully loaded and validated {len(df)} mutation records.")
        return df

    def calculate(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates mutation frequency, normalized frequency, and empirical sparsity.

        Formulas
        --------
        1. Frequency = count / Total Observations
        2. Normalized Frequency = Frequency / max(Frequency) = count / max(count)
        3. Empirical Sparsity = 1 - Normalized Frequency

        Parameters
        ----------
        counts_df : pd.DataFrame
            DataFrame containing verified mutation counts.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all required fields:
            mutation_id, count, frequency, mutation_frequency, normalized_frequency, empirical_sparsity.
        """
        self.logger.info("Calculating empirical sparsity...")
        df = counts_df.copy()

        total_observations = df["count"].sum()
        max_count = df["count"].max()

        # Handle edge case where total_observations is zero
        if total_observations == 0:
            self.logger.warning("Total observations count is zero. Defaulting frequencies to 0.0.")
            df["frequency"] = 0.0
            df["mutation_frequency"] = 0.0
            df["normalized_frequency"] = 0.0
            df["empirical_sparsity"] = 1.0
            return df

        # Calculate standard frequency
        df["frequency"] = df["count"] / total_observations
        # Provide both aliases to comply with different documentation files
        df["mutation_frequency"] = df["frequency"]

        # Handle edge case where max_count is zero (though theoretically handled if total_observations > 0)
        if max_count == 0:
            df["normalized_frequency"] = 0.0
        else:
            df["normalized_frequency"] = df["count"] / max_count

        # Empirical Sparsity = 1 - Normalized Frequency
        df["empirical_sparsity"] = 1.0 - df["normalized_frequency"]

        self.logger.info("Empirical sparsity calculation completed successfully.")
        return df

    def generate_outputs(
        self, df: pd.DataFrame, output_dir: str
    ) -> Tuple[str, str, str]:
        """Saves output parquet, summary CSV, and distribution plot.

        Parameters
        ----------
        df : pd.DataFrame
            Calculated empirical sparsity DataFrame.
        output_dir : str
            Directory path to save results.

        Returns
        -------
        Tuple[str, str, str]
            Paths to generated parquet, summary CSV, and distribution plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Saving outputs to directory: {output_dir}")

        parquet_path = os.path.join(output_dir, "empirical_sparsity.parquet")
        summary_path = os.path.join(output_dir, "empirical_sparsity_summary.csv")
        plot_path = os.path.join(output_dir, "empirical_distribution.png")

        # 1. Save Core Parquet (D101)
        # Select exact required columns to guarantee schema conformity
        d101_cols = [
            "mutation_id",
            "count",
            "frequency",
            "mutation_frequency",
            "normalized_frequency",
            "empirical_sparsity",
        ]
        df[d101_cols].to_parquet(parquet_path, index=False)
        self.logger.info(f"Wrote core parquet output: {parquet_path}")

        # 2. Save Summary CSV
        summary_stats = {
            "total_mutations": len(df),
            "total_observations": int(df["count"].sum()),
            "min_count": int(df["count"].min()),
            "max_count": int(df["count"].max()),
            "mean_count": float(df["count"].mean()),
            "mean_sparsity": float(df["empirical_sparsity"].mean()),
            "median_sparsity": float(df["empirical_sparsity"].median()),
            "min_sparsity": float(df["empirical_sparsity"].min()),
            "max_sparsity": float(df["empirical_sparsity"].max()),
        }
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=["metric", "value"])
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Wrote summary CSV: {summary_path}")

        # 3. Generate Distribution Plot
        plt.figure(figsize=(8, 6))
        plt.hist(
            df["empirical_sparsity"],
            bins=min(15, len(df)),
            edgecolor="black",
            color="#3498db",
            alpha=0.7,
        )
        plt.title("Distribution of Empirical Sparsity", fontsize=14)
        plt.xlabel("Empirical Sparsity (0 = Dense, 1 = Sparse)", fontsize=12)
        plt.ylabel("Mutation Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Wrote distribution plot: {plot_path}")

        return parquet_path, summary_path, plot_path


def main() -> None:
    """Main command-line entry point for Empirical Sparsity framework."""
    parser = argparse.ArgumentParser(
        description="Compute empirical sparsity metrics from experimental counts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/empirical/mutation_counts.csv",
        help="Path to raw mutation count CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/empirical/",
        help="Directory to save generated parquet, summary, and plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    print("Loading empirical sparsity calculation framework...")
    try:
        framework = EmpiricalSparsity(config_path=args.config)
        counts_df = framework.load_counts(args.input)

        print("Calculating frequencies and empirical sparsity...")
        calculated_df = framework.calculate(counts_df)

        print(f"Writing outputs to {args.output}...")
        parquet, summary, plot = framework.generate_outputs(calculated_df, args.output)

        print("\n--- Empirical Sparsity Executed Successfully ---")
        print(f"Core Parquet: {parquet}")
        print(f"Summary CSV:  {summary}")
        print(f"Distribution: {plot}")
        print("-------------------------------------------------")
        print("Done")
    except Exception as e:
        print(f"ERROR: Empirical Sparsity Execution Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
