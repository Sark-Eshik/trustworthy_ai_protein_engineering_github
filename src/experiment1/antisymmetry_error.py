# src/experiment1/antisymmetry_error.py
"""Core module for calculating Antisymmetry Error and verifying scientific consistency.

Defines the mathematical relationship: Antisymmetry Error = |Forward + Reverse|.
Includes validation routines to verify calculation accuracy against explicit standard exercises,
and supports failure-injection simulation to verify validation sensitivity.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class AntisymmetryErrorCalculator:
    """Calculates thermodynamic inconsistency (antisymmetry error) and executes validations."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="antisymmetry_calculation",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    @staticmethod
    def calculate_error(forward: float, reverse: float, use_incorrect_formula: bool = False) -> float:
        """Calculates antisymmetry error using standard or intentionally modified formula.
        
        Standard Formula: |Forward + Reverse|
        Injected Fault Formula: |Forward - Reverse|
        """
        if use_incorrect_formula:
            return abs(forward - reverse)
        return abs(forward + reverse)

    def run_athlete_exercises(self, use_incorrect_formula: bool = False) -> bool:
        """Executes verification on explicitly defined scientific test cases.
        
        Exercises:
        1. Forward = 2.5, Reverse = -2.5 -> Error = 0.0
        2. Forward = 2.0, Reverse = -1.4 -> Error = 0.6
        3. Forward = 0.9, Reverse = -0.2 -> Error = 0.7
        """
        self.logger.info("Executing Athlete Exercises verification tests...")
        
        test_cases = [
            {"forward": 2.5, "reverse": -2.5, "expected": 0.0},
            {"forward": 2.0, "reverse": -1.4, "expected": 0.6},
            {"forward": 0.9, "reverse": -0.2, "expected": 0.7},
        ]

        passed_all = True
        for idx, tc in enumerate(test_cases, 1):
            f, r, exp = tc["forward"], tc["reverse"], tc["expected"]
            calc = self.calculate_error(f, r, use_incorrect_formula=use_incorrect_formula)
            # Use np.isclose to avoid floating-point representation anomalies
            if np.isclose(calc, exp):
                self.logger.info(f"Athlete Exercise {idx}: Passed. Forward={f}, Reverse={r}, Calc={calc:.2f}, Exp={exp}")
            else:
                self.logger.error(f"Athlete Exercise {idx}: FAILED! Forward={f}, Reverse={r}, Calc={calc:.4f}, Exp={exp}")
                passed_all = False

        return passed_all

    def process_pipeline(
        self, 
        forward_path: str = None, 
        reverse_path: str = None, 
        sparsity_path: str = None,
        output_dir: str = None,
        use_incorrect_formula: bool = False
    ) -> tuple[str, str]:
        """Runs the core antisymmetry error compilation pipeline.
        
        Parameters
        ----------
        forward_path : str, optional
            Path to forward predictions.
        reverse_path : str, optional
            Path to reverse predictions.
        sparsity_path : str, optional
            Path to combined sparsity dataset.
        output_dir : str, optional
            Target output directory.
        use_incorrect_formula : bool, default False
            If True, intentionally applies faulty calculation to trigger validation failure.
            
        Returns
        -------
        tuple of str
            Paths to the generated predictions (D201) and results (D202) parquet files.
        """
        # Validate calculations first
        exercises_passed = self.run_athlete_exercises(use_incorrect_formula=use_incorrect_formula)
        if not exercises_passed:
            err_msg = "Scientific validation failed! The calculation formula produces incorrect results."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        data_dir = self.loader.config.paths.data_dir
        f_path = forward_path or os.path.join(data_dir, "intermediate/experiment1/forward_predictions.parquet")
        r_path = reverse_path or os.path.join(data_dir, "intermediate/experiment1/reverse_predictions.parquet")
        s_path = sparsity_path or os.path.join(data_dir, "intermediate/combined/combined_sparsity.parquet")
        
        out_dir = output_dir or os.path.join(data_dir, "intermediate/experiment1")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Loading predictions and sparsity. Forward: {f_path}, Reverse: {r_path}, Sparsity: {s_path}")
        
        if not os.path.exists(f_path) or not os.path.exists(r_path) or not os.path.exists(s_path):
            raise FileNotFoundError("One or more of the required input parquets for calculation are missing.")

        df_f = pd.read_parquet(f_path)
        df_r = pd.read_parquet(r_path)
        df_s = pd.read_parquet(s_path)

        # Merge datasets based on mutation_id
        # D201 predictions: mutation_id, forward_ddg, reverse_ddg, predictor
        df_pred = pd.merge(df_f, df_r, on="mutation_id", how="inner")
        # In case predictor names differ or to maintain schema, we keep 'predictor' column using the first available or default name
        if "predictor_name_x" in df_pred.columns:
            df_pred["predictor"] = df_pred["predictor_name_x"]
        elif "predictor_name" in df_pred.columns:
            df_pred["predictor"] = df_pred["predictor_name"]
        else:
            df_pred["predictor"] = "ThermoNet-v1"

        d201_cols = ["mutation_id", "forward_ddg", "reverse_ddg", "predictor"]
        d201_df = df_pred[d201_cols].copy()

        # Save D201
        d201_path = os.path.join(out_dir, "antisymmetry_predictions.parquet")
        d201_df.to_parquet(d201_path, index=False)
        self.logger.info(f"Saved D201 (Antisymmetry Predictions) Parquet to {d201_path}")

        # Compute D202 results: mutation_id, forward_ddg, reverse_ddg, antisymmetry_error, combined_sparsity
        df_results = pd.merge(df_pred, df_s[["mutation_id", "combined_sparsity"]], on="mutation_id", how="inner")
        
        # Apply the thermodynamic consistency formula
        df_results["antisymmetry_error"] = df_results.apply(
            lambda row: self.calculate_error(row["forward_ddg"], row["reverse_ddg"], use_incorrect_formula=use_incorrect_formula),
            axis=1
        )

        d202_cols = ["mutation_id", "forward_ddg", "reverse_ddg", "antisymmetry_error", "combined_sparsity"]
        d202_df = df_results[d202_cols].copy()

        # Save D202 and copy as antisymmetry_results.parquet to support legacy scripts
        d202_path = os.path.join(out_dir, "antisymmetry_results.parquet")
        d202_df.to_parquet(d202_path, index=False)
        self.logger.info(f"Saved D202 (Antisymmetry Results) Parquet to {d202_path}")

        print(f"Antisymmetry Predictions (D201) written to {d201_path}")
        print(f"Antisymmetry Results (D202) written to {d202_path}")
        
        return d201_path, d202_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Antisymmetry Error.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--forward", type=str, default=None, help="Path to forward predictions.")
    parser.add_argument("--reverse", type=str, default=None, help="Path to reverse predictions.")
    parser.add_argument("--sparsity", type=str, default=None, help="Path to combined sparsity dataset.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output parquets.")
    parser.add_argument("--inject-fault", action="store_true", help="Simulate a calculation error to test validation sensitivity.")
    args = parser.parse_args()

    try:
        calculator = AntisymmetryErrorCalculator(config_path=args.config)
        calculator.process_pipeline(
            forward_path=args.forward,
            reverse_path=args.reverse,
            sparsity_path=args.sparsity,
            output_dir=args.output,
            use_incorrect_formula=args.inject_fault
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Antisymmetry compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
