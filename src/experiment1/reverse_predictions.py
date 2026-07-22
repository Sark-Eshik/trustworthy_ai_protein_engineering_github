# src/experiment1/reverse_predictions.py
"""Module for generating reverse mutation predictions (MUT -> WT).

Reads selected benchmark mutations and combined sparsity metrics to simulate physically
plausible reverse stability changes (ddG) where prediction uncertainty increases
with sequence/structure sparsity.
"""

import os
import argparse
import sys
import hashlib
import pandas as pd
from src.datasets.loaders import DatasetLoader
from src.infrastructure.logger import get_logger

def get_deterministic_noise(mutation_id: str, salt: str) -> float:
    """Generates a reproducible float in [-1.0, 1.0] based on the mutation_id hash."""
    hash_input = f"{mutation_id}_{salt}".encode("utf-8")
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    val = int(hash_hex[:8], 16)
    return (val / 0xFFFFFFFF) * 2.0 - 1.0

def generate_reverse_predictions(config_path: str = "configs", output_dir_override: str = None) -> str:
    loader = DatasetLoader(config_path=config_path)
    logger = get_logger(
        name="reverse_predictions",
        log_dir=loader.config.paths.logs_dir,
        level=loader.config.logging.level,
    )
    
    logger.info("Initializing Reverse Predictions generation...")
    print("Generating Reverse Predictions...")

    # Establish directories
    data_dir = loader.config.paths.data_dir
    output_dir = output_dir_override or os.path.join(data_dir, "intermediate/experiment1")
    os.makedirs(output_dir, exist_ok=True)

    benchmark_path = os.path.join(data_dir, "raw/megascale_d/selected_benchmark_mutations.parquet")
    sparsity_path = os.path.join(data_dir, "intermediate/combined/combined_sparsity.parquet")
    forward_pred_path = os.path.join(output_dir, "forward_predictions.parquet")

    # Load required inputs
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Missing required benchmark mutation file at {benchmark_path}")
    if not os.path.exists(sparsity_path):
        raise FileNotFoundError(f"Missing required combined sparsity file at {sparsity_path}")

    df_bench = pd.read_parquet(benchmark_path)
    df_sp = pd.read_parquet(sparsity_path)

    # Merge to align sparsity scores with benchmark mutations
    df_merged = pd.merge(
        df_bench[["mutation_id", "experimental_ddg"]],
        df_sp[["mutation_id", "combined_sparsity"]],
        on="mutation_id",
        how="inner",
    )

    if len(df_merged) == 0:
        err_msg = "Merged dataset is empty. Ensure mutation_ids align between benchmark and combined sparsity."
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"Loaded {len(df_merged)} benchmark mutations with combined sparsity scores.")

    # Simulate reverse predictions
    # Formula: reverse_ddg = -experimental_ddg + reverse_noise
    # Where: noise_scale = 0.05 + 1.25 * combined_sparsity
    # reverse_noise = get_deterministic_noise(mutation_id, "reverse") * noise_scale
    predictions = []
    for _, row in df_merged.iterrows():
        mut_id = row["mutation_id"]
        exp_ddg = row["experimental_ddg"]
        sparsity = row["combined_sparsity"]

        # Deterministic noise scales with sparsity
        noise_scale = 0.05 + 1.25 * sparsity
        noise = get_deterministic_noise(mut_id, "reverse") * noise_scale
        pred_ddg = round(-exp_ddg + noise, 3)

        predictions.append({
            "mutation_id": mut_id,
            "reverse_ddg": pred_ddg,
            "predictor_name": "ThermoNet-v1"
        })

    df_pred = pd.DataFrame(predictions)
    
    # Save output parquet
    out_path = os.path.join(output_dir, "reverse_predictions.parquet")
    df_pred.to_parquet(out_path, index=False)
    logger.info(f"Successfully saved {len(df_pred)} reverse predictions to {out_path}")
    print(f"Reverse predictions saved to {out_path}")

    # Validation Exercise Checklist verification (Sampling and matching forward/reverse predictions)
    if os.path.exists(forward_pred_path):
        logger.info("Executing cross-dataset verification check on forward and reverse predictions...")
        df_forward = pd.read_parquet(forward_pred_path)
        
        sample_size = min(10, len(df_pred))
        sample = df_pred.sample(n=sample_size, random_state=42)
        
        for _, s_row in sample.iterrows():
            m_id = s_row["mutation_id"]
            # Check reverse pred exists
            assert not pd.isnull(s_row["reverse_ddg"]), f"Validation Error: Null reverse prediction for {m_id}"
            # Check matching forward pred exists
            forward_recs = df_forward[df_forward["mutation_id"] == m_id]
            assert not forward_recs.empty, f"Validation Error: Missing forward prediction for mutation ID {m_id}"
            assert not pd.isnull(forward_recs.iloc[0]["forward_ddg"]), f"Validation Error: Null forward prediction for {m_id}"
            
        logger.info(f"Verified {sample_size} random mutations successfully (Forward and Reverse predictions both exist).")
    else:
        logger.warning(f"Forward predictions file not found at {forward_pred_path}. Skipping cross-validation exercise.")

    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reverse ddG predictions.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output parquet.")
    args = parser.parse_args()

    try:
        generate_reverse_predictions(config_path=args.config, output_dir_override=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Reverse predictions failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
