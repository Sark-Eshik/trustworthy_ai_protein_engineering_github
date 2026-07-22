# src/experiment2/predicted_epistasis.py
"""Module for calculating predicted epistasis.

Formula: Predicted Epistasis = Pred_ab - (Pred_a + Pred_b)
Loads the double mutation predictions (from double_mutation_predictions.parquet) and
single mutation predictions (from single_mutation_predictions.parquet) to compute
predicted epistasis.
"""

import os
import argparse
import sys
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

class PredictedEpistasisCalculator:
    """Computes predicted epistasis across double mutant pairs."""

    def __init__(self, config_path: str = "configs") -> None:
        self.loader = DatasetLoader(config_path=config_path)
        self.logger = get_logger(
            name="predicted_epistasis",
            log_dir=self.loader.config.paths.logs_dir,
            level=self.loader.config.logging.level,
        )

    @staticmethod
    def calculate_epistasis(pred_ab: float, pred_a: float, pred_b: float) -> float:
        """Calculates predicted epistatic coupling: Pred_ab - (Pred_a + Pred_b)."""
        return pred_ab - (pred_a + pred_b)

    def process_pipeline(
        self, 
        double_pred_path: str = None, 
        single_pred_path: str = None, 
        double_mutants_path: str = None,
        output_dir: str = None
    ) -> str:
        """Runs the predicted epistasis pipeline.
        
        Loads predictions and double mutants metadata, calculates predicted epistasis, and saves output.
        """
        data_dir = self.loader.config.paths.data_dir
        dpred_p = double_pred_path or os.path.join(data_dir, "raw/epistasis/double_mutation_predictions.parquet")
        spred_p = single_pred_path or os.path.join(data_dir, "raw/epistasis/single_mutation_predictions.parquet")
        dm_p = double_mutants_path or os.path.join(data_dir, "raw/epistasis/double_mutants.parquet")
        
        out_dir = output_dir or os.path.join(data_dir, "intermediate/experiment2")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Loading files. Double preds: {dpred_p}, Single preds: {spred_p}, Double mutants: {dm_p}")
        if not os.path.exists(dpred_p) or not os.path.exists(spred_p) or not os.path.exists(dm_p):
            raise FileNotFoundError("Required prediction input parquets are missing.")

        df_dpred = pd.read_parquet(dpred_p)
        df_spred = pd.read_parquet(spred_p)
        df_dm = pd.read_parquet(dm_p)

        # Build lookups
        spred_lookup = dict(zip(df_spred["mutation_id"], df_spred["predicted_ddg"]))
        dpred_lookup = dict(zip(df_dpred["pair_id"], df_dpred["predicted_ddg_ab"]))

        self.logger.info(f"Processing {len(df_dm)} pairs for predicted epistasis...")
        
        pred_epistasis_values = []
        for idx, row in df_dm.iterrows():
            pair_id = row["pair_id"]
            mut_a = row["mutation_a"]
            mut_b = row["mutation_b"]

            pred_ab = dpred_lookup.get(pair_id)
            pred_a = spred_lookup.get(mut_a)
            pred_b = spred_lookup.get(mut_b)

            if pred_ab is None or pred_a is None or pred_b is None:
                err_msg = f"Missing prediction values for pair '{pair_id}' or constituents '{mut_a}', '{mut_b}'."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            val = self.calculate_epistasis(pred_ab, pred_a, pred_b)
            pred_epistasis_values.append({
                "pair_id": pair_id,
                "predicted_ddg_a": pred_a,
                "predicted_ddg_b": pred_b,
                "predicted_ddg_ab": pred_ab,
                "predicted_epistasis": round(val, 3)
            })

        df_out = pd.DataFrame(pred_epistasis_values)
        out_path = os.path.join(out_dir, "predicted_epistasis.parquet")
        df_out.to_parquet(out_path, index=False)
        
        self.logger.info(f"Successfully wrote predicted epistasis results to {out_path}")
        print(f"Predicted epistasis compiled successfully: {out_path}")
        
        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate predicted epistasis.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--double-pred", type=str, default=None, help="Path to double mutation predictions parquet.")
    parser.add_argument("--single-pred", type=str, default=None, help="Path to single mutation predictions parquet.")
    parser.add_argument("--double-mutants", type=str, default=None, help="Path to double mutants parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output.")
    args = parser.parse_args()

    try:
        calculator = PredictedEpistasisCalculator(config_path=args.config)
        calculator.process_pipeline(
            double_pred_path=args.double_pred,
            single_pred_path=args.single_pred,
            double_mutants_path=args.double_mutants,
            output_dir=args.output
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Predicted epistasis failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
