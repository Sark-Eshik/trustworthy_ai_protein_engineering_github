# src/experiment2/epistasis_error.py
"""Module for calculating epistasis error.

Formula: Epistasis Error = |Predicted Epistasis - Experimental Epistasis|
Loads experimental epistasis and predicted epistasis, merges them with combined sparsity,
and verifies consistency against Athlete Exercise 2 and schema definitions.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class EpistasisErrorCalculator:
    """Calculates epistasis prediction errors and performs scientific validations."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="epistasis_error",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    @staticmethod
    def calculate_error(predicted: float, experimental: float) -> float:
        """Calculates absolute epistasis error: |predicted - experimental|."""
        return abs(predicted - experimental)

    def run_athlete_exercise(self) -> bool:
        """Verifies Athlete Exercise 2:
        
        Experimental = 2, Predicted = 1.5
        Expected: Error = 0.5
        """
        self.logger.info("Verifying Athlete Exercise 2...")
        calc = self.calculate_error(1.5, 2.0)
        expected = 0.5
        if np.isclose(calc, expected):
            self.logger.info(f"Athlete Exercise 2: Passed. Experimental=2, Predicted=1.5 -> Error={calc:.2f} (Expected: 0.5)")
            return True
        else:
            self.logger.error(f"Athlete Exercise 2: FAILED! Got {calc:.4f}, expected {expected:.4f}")
            return False

    def process_pipeline(
        self, 
        experimental_path: str = None, 
        predicted_path: str = None, 
        sparsity_path: str = None,
        output_dir: str = None
    ) -> str:
        """Runs the epistasis error pipeline.
        
        Merges experimental epistasis, predicted epistasis, and combined sparsity to save epistasis_results.parquet (D302).
        """
        if not self.run_athlete_exercise():
            raise ValueError("Epistasis Error Calculator failed validation on Athlete Exercise 2.")

        data_dir = self.loader.config.paths.data_dir
        exp_path = experimental_path or os.path.join(data_dir, "intermediate/experiment2/experimental_epistasis.parquet")
        pred_path = predicted_path or os.path.join(data_dir, "intermediate/experiment2/predicted_epistasis.parquet")
        sp_path = sparsity_path or os.path.join(data_dir, "intermediate/combined/combined_sparsity.parquet")
        
        out_dir = output_dir or os.path.join(data_dir, "intermediate/experiment2")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Loading files. Experimental: {exp_path}, Predicted: {pred_path}, Sparsity: {sp_path}")
        if not os.path.exists(exp_path) or not os.path.exists(pred_path) or not os.path.exists(sp_path):
            raise FileNotFoundError("Required input parquets for epistasis error calculation are missing.")

        df_exp = pd.read_parquet(exp_path)
        df_pred = pd.read_parquet(pred_path)
        df_sp = pd.read_parquet(sp_path)

        # 1. Failure Injection Check: Assure no missing AB measurements
        # (Throw error if missing measurements exist, complying with Failure Injection requirements)
        if df_exp["experimental_ddg_ab"].isnull().any():
            err_msg = "Failure Injection Detected: Missing experimental_ddg_ab measurement in double mutant input!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
            
        if df_pred["predicted_ddg_ab"].isnull().any():
            err_msg = "Failure Injection Detected: Missing predicted_ddg_ab prediction in input!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Merge datasets based on pair_id
        df_merged = pd.merge(
            df_exp[["pair_id", "mutation_a", "mutation_b", "experimental_epistasis"]],
            df_pred[["pair_id", "predicted_epistasis"]],
            on="pair_id",
            how="inner"
        )

        # Calculate epistasis error
        df_merged["epistasis_error"] = df_merged.apply(
            lambda row: self.calculate_error(row["predicted_epistasis"], row["experimental_epistasis"]),
            axis=1
        )

        # Compute pair-level combined sparsity (average of combined sparsity of A and B)
        sp_lookup = dict(zip(df_sp["mutation_id"], df_sp["combined_sparsity"]))
        
        pair_sparsity = []
        for idx, row in df_merged.iterrows():
            mut_a = row["mutation_a"]
            mut_b = row["mutation_b"]
            
            sp_a = sp_lookup.get(mut_a)
            sp_b = sp_lookup.get(mut_b)

            if sp_a is None or sp_b is None:
                err_msg = f"Combined sparsity not found for constituent single mutation '{mut_a}' or '{mut_b}'."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            pair_sparsity.append((sp_a + sp_b) / 2.0)

        df_merged["combined_sparsity"] = pair_sparsity

        # Match columns of official Dataset D302
        d302_cols = [
            "pair_id",
            "experimental_epistasis",
            "predicted_epistasis",
            "epistasis_error",
            "combined_sparsity"
        ]
        df_out = df_merged[d302_cols].copy()

        out_path = os.path.join(out_dir, "epistasis_results.parquet")
        df_out.to_parquet(out_path, index=False)
        
        self.logger.info(f"Successfully saved D302 (Epistasis Results) Parquet with {len(df_out)} pairs to {out_path}")
        print(f"Epistasis results written to {out_path}")

        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate epistasis error.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--experimental", type=str, default=None, help="Path to experimental epistasis parquet.")
    parser.add_argument("--predicted", type=str, default=None, help="Path to predicted epistasis parquet.")
    parser.add_argument("--sparsity", type=str, default=None, help="Path to combined sparsity parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output.")
    args = parser.parse_args()

    try:
        calculator = EpistasisErrorCalculator(config_path=args.config)
        calculator.process_pipeline(
            experimental_path=args.experimental,
            predicted_path=args.predicted,
            sparsity_path=args.sparsity,
            output_dir=args.output
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Epistasis error failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
