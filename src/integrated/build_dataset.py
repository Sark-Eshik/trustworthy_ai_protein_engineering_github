# src/integrated/build_dataset.py
"""Integrated Reliability Analysis dataset compiler (D401).

Merges single-point Combined Sparsity (D104), Antisymmetry Error (D202), and pair-level
Epistasis Error (D302) into a cohesive mutation-level integrated dataset (D401) for
downstream pathway analysis.
"""

import os
import argparse
import sys
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class IntegratedDatasetBuilder:
    """Class to compile and validate the integrated reliability analysis dataset (D401)."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="build_integrated_dataset",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    def run_pipeline(
        self,
        sparsity_path: str = None,
        antisymmetry_path: str = None,
        epistasis_results_path: str = None,
        double_mutants_path: str = None,
        output_dir: str = None
    ) -> str:
        """Loads inputs, aggregates epistasis error to the single-mutation level, merges streams, and saves D401."""
        self.logger.info("Initializing Integrated Dataset compilation pipeline...")
        print("Building Integrated Dataset...")

        data_dir = self.loader.config.paths.data_dir
        sp_p = sparsity_path or os.path.join(data_dir, "intermediate/combined/combined_sparsity.parquet")
        anti_p = antisymmetry_path or os.path.join(data_dir, "intermediate/experiment1/antisymmetry_results.parquet")
        epi_res_p = epistasis_results_path or os.path.join(data_dir, "intermediate/experiment2/epistasis_results.parquet")
        dm_p = double_mutants_path or os.path.join(data_dir, "raw/epistasis/double_mutants.parquet")

        out_dir = output_dir or os.path.join(data_dir, "final/reliability")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Loading inputs. Sparsity: {sp_p}, Antisymmetry: {anti_p}, Epistasis Results: {epi_res_p}, Double mutants: {dm_p}")
        if not all(os.path.exists(p) for p in [sp_p, anti_p, epi_res_p, dm_p]):
            raise FileNotFoundError("One or more required input parquets are missing.")

        df_sp = pd.read_parquet(sp_p)
        df_anti = pd.read_parquet(anti_p)
        df_epi_res = pd.read_parquet(epi_res_p)
        df_dm = pd.read_parquet(dm_p)

        # 1. Aggregate epistasis error to the single mutation level
        # Load double mutant definitions to map pair_id to constituent mutations
        df_ep_mapped = pd.merge(
            df_dm[["pair_id", "mutation_a", "mutation_b"]],
            df_epi_res[["pair_id", "epistasis_error"]],
            on="pair_id",
            how="inner"
        )

        # Stack mutation_a and mutation_b to group by individual mutation_id
        df_a = df_ep_mapped[["mutation_a", "epistasis_error"]].rename(columns={"mutation_a": "mutation_id"})
        df_b = df_ep_mapped[["mutation_b", "epistasis_error"]].rename(columns={"mutation_b": "mutation_id"})
        df_stacked = pd.concat([df_a, df_b], ignore_index=True)

        # Compute average epistasis error for each single mutation
        df_mut_epi = df_stacked.groupby("mutation_id")["epistasis_error"].mean().reset_index()
        self.logger.info(f"Grouped epistasis errors across {len(df_mut_epi)} single mutations.")

        # 2. Merge single-mutation datasets
        # We start with the single mutations that are characterized in both single and double mutant contexts (inner join)
        df_merged = pd.merge(
            df_sp[["mutation_id", "combined_sparsity"]],
            df_anti[["mutation_id", "antisymmetry_error"]],
            on="mutation_id",
            how="inner"
        )
        
        df_final = pd.merge(
            df_merged,
            df_mut_epi,
            on="mutation_id",
            how="inner"
        )

        # 3. Create primary key `record_id`
        df_final["record_id"] = df_final["mutation_id"].apply(lambda m_id: f"rec_{m_id}")

        # Reorder columns to align with D401 schema specification
        d401_cols = [
            "record_id",
            "mutation_id",
            "combined_sparsity",
            "antisymmetry_error",
            "epistasis_error"
        ]
        df_out = df_final[d401_cols].copy()

        # 4. Perform validation checks
        if df_out["record_id"].duplicated().any():
            err_msg = "Duplicate record_id detected in the compiled integrated dataset!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        if df_out.isnull().any().any():
            err_msg = "Integrated dataset contains null/missing values in required columns!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        out_path = os.path.join(out_dir, "integrated_reliability_analysis.parquet")
        df_out.to_parquet(out_path, index=False)

        self.logger.info(f"Successfully compiled D401 Parquet containing {len(df_out)} certified records at {out_path}")
        print(f"Integrated dataset compiled and validated with {len(df_out)} rows: {out_path}")

        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Integrated Reliability Analysis Dataset.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--sparsity", type=str, default=None, help="Path to combined sparsity parquet.")
    parser.add_argument("--antisymmetry", type=str, default=None, help="Path to antisymmetry results parquet.")
    parser.add_argument("--epistasis-results", type=str, default=None, help="Path to epistasis results parquet.")
    parser.add_argument("--double-mutants", type=str, default=None, help="Path to double mutants parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output.")
    args = parser.parse_args()

    try:
        builder = IntegratedDatasetBuilder(config_path=args.config)
        builder.run_pipeline(
            sparsity_path=args.sparsity,
            antisymmetry_path=args.antisymmetry,
            epistasis_results_path=args.epistasis_results,
            double_mutants_path=args.double_mutants,
            output_dir=args.output
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Build integrated dataset failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
