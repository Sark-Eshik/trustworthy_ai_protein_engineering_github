# src/experiment2/experimental_epistasis.py
"""Module for calculating experimental epistasis.

Formula: Experimental Epistasis = ddG_ab - (ddG_a + ddG_b)
Loads the double mutant dataset (D301) and single mutant experimental values (S001)
to compute experimental epistasis.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class ExperimentalEpistasisCalculator:
    """Computes experimental epistasis and executes scientific validation checks."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="experimental_epistasis",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    @staticmethod
    def calculate_epistasis(ddg_ab: float, ddg_a: float, ddg_b: float) -> float:
        """Calculates epistatic coupling: ddG_ab - (ddG_a + ddG_b)."""
        return ddg_ab - (ddg_a + ddg_b)

    def run_athlete_exercise(self) -> bool:
        """Verifies Athlete Exercise 1:
        
        Input: A = 1, B = 2, AB = 5
        Expected: Epistasis = 2
        """
        self.logger.info("Verifying Athlete Exercise 1...")
        calc = self.calculate_epistasis(5.0, 1.0, 2.0)
        expected = 2.0
        if np.isclose(calc, expected):
            self.logger.info(f"Athlete Exercise 1: Passed. Input A=1, B=2, AB=5 -> Epistasis={calc:.2f} (Expected: 2.0)")
            return True
        else:
            self.logger.error(f"Athlete Exercise 1: FAILED! Got {calc:.4f}, expected {expected:.4f}")
            return False

    def process_pipeline(
        self, 
        double_mutants_path: str = None, 
        megascale_path: str = None, 
        output_dir: str = None
    ) -> str:
        """Runs the experimental epistasis pipeline.
        
        Loads D301 and S001, calculates experimental epistasis, and saves output.
        """
        if not self.run_athlete_exercise():
            raise ValueError("Experimental Epistasis Calculator failed validation on Athlete Exercise 1.")

        data_dir = self.loader.config.paths.data_dir
        dm_path = double_mutants_path or os.path.join(data_dir, "raw/epistasis/double_mutants.parquet")
        mega_path = megascale_path or os.path.join(data_dir, "raw/megascale_d/megascale_d.parquet")
        
        out_dir = output_dir or os.path.join(data_dir, "intermediate/experiment2")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Loading files. Double mutants: {dm_path}, Megascale: {mega_path}")
        if not os.path.exists(dm_path) or not os.path.exists(mega_path):
            raise FileNotFoundError("Required input parquets are missing.")

        # Load safely using DatasetLoader or direct read
        df_dm = pd.read_parquet(dm_path)
        df_mega = pd.read_parquet(mega_path)

        # Build S001 lookup for O(1) experimental ddG lookup
        mega_lookup = dict(zip(df_mega["mutation_id"], df_mega["experimental_ddg"]))

        self.logger.info(f"Processing {len(df_dm)} double mutant pairs...")
        
        exp_epistasis_values = []
        for idx, row in df_dm.iterrows():
            pair_id = row["pair_id"]
            mut_a = row["mutation_a"]
            mut_b = row["mutation_b"]
            ddg_ab = row["experimental_ddg_ab"]

            ddg_a = mega_lookup.get(mut_a)
            ddg_b = mega_lookup.get(mut_b)

            if ddg_a is None or ddg_b is None:
                err_msg = f"Missing single-mutant experimental ddG for constituent '{mut_a}' or '{mut_b}' in S001."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            val = self.calculate_epistasis(ddg_ab, ddg_a, ddg_b)
            exp_epistasis_values.append({
                "pair_id": pair_id,
                "mutation_a": mut_a,
                "mutation_b": mut_b,
                "experimental_ddg_a": ddg_a,
                "experimental_ddg_b": ddg_b,
                "experimental_ddg_ab": ddg_ab,
                "experimental_epistasis": round(val, 3)
            })

        df_out = pd.DataFrame(exp_epistasis_values)
        out_path = os.path.join(out_dir, "experimental_epistasis.parquet")
        df_out.to_parquet(out_path, index=False)
        
        self.logger.info(f"Successfully wrote experimental epistasis results to {out_path}")
        print(f"Experimental epistasis compiled successfully: {out_path}")
        
        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate experimental epistasis.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--double-mutants", type=str, default=None, help="Path to double mutants parquet.")
    parser.add_argument("--megascale", type=str, default=None, help="Path to megascale parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output.")
    args = parser.parse_args()

    try:
        calculator = ExperimentalEpistasisCalculator(config_path=args.config)
        calculator.process_pipeline(
            double_mutants_path=args.double_mutants,
            megascale_path=args.megascale,
            output_dir=args.output
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Experimental epistasis failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
