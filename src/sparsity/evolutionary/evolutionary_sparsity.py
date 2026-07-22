# src/sparsity/evolutionary/evolutionary_sparsity.py
"""Evolutionary Sparsity calculation submodule.

Loads protein sequences and mutation datasets, runs the ESM2 probability engine,
calculates -log(P), normalizes the values, and generates evolutionary sparsity datasets.
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.infrastructure.hardware_detection import detect_hardware
from src.datasets.loaders import DatasetLoader
from src.sparsity.evolutionary.esm_probability_engine import ESMProbabilityEngine


class EvolutionarySparsity:
    """Calculates evolutionary sparsity using ESM2 probability engine."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the EvolutionarySparsity submodule.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.logger = get_logger(
            name="evolutionary_sparsity",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )
        self.loader = DatasetLoader(config_path=config_path)

    def run_pipeline(
        self,
        sequence_path_override: Optional[str] = None,
        mutations_path_override: Optional[str] = None,
        output_dir: str = "results/evolutionary/",
    ) -> Tuple[str, str, str]:
        """Runs the complete Evolutionary Sparsity calculation pipeline.

        Parameters
        ----------
        sequence_path_override : Optional[str]
            Optional path to protein sequences parquet (S002).
        mutations_path_override : Optional[str]
            Optional path to Megascale-D mutation parquet (S001).
        output_dir : str
            Directory to save generated parquets and plots.

        Returns
        -------
        Tuple[str, str, str]
            Paths to the three generated output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        reports_dir = self.config.paths.reports_dir
        os.makedirs(reports_dir, exist_ok=True)

        self.logger.info("Initializing ESM model loading and profiling hardware...")
        t_start = time.time()

        # Gather baseline hardware details
        hw_before = detect_hardware()

        # Instantiate ESM2 Engine
        # We check configs to see if GPU is enabled
        use_gpu = self.config.hardware.gpu_enabled
        engine = ESMProbabilityEngine(use_gpu=use_gpu, logger=self.logger)

        t_end_load = time.time()
        elapsed_load = t_end_load - t_start

        # Capture post-loading hardware details
        hw_after = detect_hardware()

        # Save esm_hardware_profile.md report
        self.save_hardware_report(engine.is_mock_active, hw_before, hw_after, elapsed_load, os.path.join(reports_dir, "esm_hardware_profile.md"))

        # Load input datasets safely using DatasetLoader (ensures schema validation and fail-fast)
        self.logger.info("Loading sequences and mutations datasets...")
        seq_df = self.loader.load("S002", file_path_override=sequence_path_override)
        mut_df = self.loader.load("S001", file_path_override=mutations_path_override)

        # Index sequences by protein_id for O(1) lookup
        seq_lookup = dict(zip(seq_df["protein_id"], seq_df["sequence"]))

        self.logger.info(f"Processing {len(mut_df)} mutations for evolutionary scoring...")

        probs: List[float] = []
        logs: List[float] = []

        for idx, row in mut_df.iterrows():
            protein_id = row["protein_id"]
            pos = int(row["position"])
            wt = row["wildtype"]
            mut = row["mutant"]
            mut_id = row["mutation_id"]

            sequence = seq_lookup.get(protein_id)
            if not sequence:
                err_msg = f"Sequence for protein_id '{protein_id}' not found in the sequences dataset."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            # Score mutation probability
            p = engine.score_mutation(
                sequence=sequence,
                position=pos,
                wildtype=wt,
                mutant=mut,
                protein_id=protein_id,
            )

            # Calculate -log(P)
            # Clip probability to avoid division by zero or log of zero
            p_clipped = max(1e-12, p)
            log_p = -float(np.log(p_clipped))

            probs.append(p)
            logs.append(log_p)

        # Build output dataframe
        out_df = pd.DataFrame({
            "mutation_id": mut_df["mutation_id"],
            "esm_probability": probs,
            "log_probability": logs,
        })

        # Calculate normalized evolutionary sparsity: (log_p - min) / (max - min)
        min_log = out_df["log_probability"].min()
        max_log = out_df["log_probability"].max()

        if np.isclose(max_log, min_log):
            self.logger.warning("All log probability values are identical. Normalizing sparsity to 0.0.")
            out_df["evolutionary_sparsity"] = 0.0
        else:
            out_df["evolutionary_sparsity"] = (out_df["log_probability"] - min_log) / (max_log - min_log)

        # Generate outputs
        parquet_sparsity_path = os.path.join(output_dir, "evolutionary_sparsity.parquet")
        parquet_probs_path = os.path.join(output_dir, "esm_mutation_probabilities.parquet")
        plot_path = os.path.join(output_dir, "evolutionary_distribution.png")

        # Save core evolutionary_sparsity dataset (D102)
        d102_cols = ["mutation_id", "esm_probability", "log_probability", "evolutionary_sparsity"]
        out_df[d102_cols].to_parquet(parquet_sparsity_path, index=False)
        self.logger.info(f"Wrote D102 Parquet: {parquet_sparsity_path}")

        # Save esm_mutation_probabilities.parquet as requested in specifications
        out_df.to_parquet(parquet_probs_path, index=False)
        self.logger.info(f"Wrote probabilities Parquet: {parquet_probs_path}")

        # Generate Distribution Plot
        self.logger.info("Generating distribution histogram plot...")
        plt.figure(figsize=(8, 6))
        plt.hist(
            out_df["evolutionary_sparsity"],
            bins=min(15, len(out_df)),
            edgecolor="black",
            color="#2ecc71",
            alpha=0.7,
        )
        plt.title("Distribution of Evolutionary Sparsity", fontsize=14)
        plt.xlabel("Evolutionary Sparsity (0 = Common, 1 = Rare)", fontsize=12)
        plt.ylabel("Mutation Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Wrote distribution plot: {plot_path}")

        return parquet_sparsity_path, parquet_probs_path, plot_path

    def save_hardware_report(
        self, is_mock: bool, hw_before: Dict[str, Any], hw_after: Dict[str, Any], elapsed: float, path: str
    ) -> None:
        """Saves hardware footprint details recorded during ESM engine loading."""
        report = f"""# ESM Model Loading Hardware Profile

## Execution Summary
- **Model Load Time**: {elapsed:.4f} seconds
- **Engine Mode**: {"Simulated (Fallback)" if is_mock else "Real ESM2 model"}

## System Resource Deltas
### CPU Usage
- **Logical CPU Cores**: {hw_after['cpu_count_logical']}
- **Physical CPU Cores**: {hw_after['cpu_count_physical']}

### RAM Footprint
- **Baseline RAM Available**: {hw_before['available_ram_gb']:.2f} GB
- **Post-Loading RAM Available**: {hw_after['available_ram_gb']:.2f} GB
- **Approximate RAM Delta**: {max(0, hw_before['available_ram_gb'] - hw_after['available_ram_gb']):.2f} GB

### GPU Resources
- **GPU Available**: {hw_after['gpu_available']}
- **GPU Device Name**: {hw_after['gpu_name']}
- **GPU Memory Allocation**: {hw_after['gpu_memory_gb']} GB
"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info(f"Wrote ESM hardware report: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save ESM hardware report: {e}")


def main() -> None:
    """Main command-line entry point for Evolutionary Sparsity pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute ESM2-based evolutionary sparsity metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw sequences parquet (overrides S002).",
    )
    parser.add_argument(
        "--mutations",
        type=str,
        default=None,
        help="Path to Megascale-D mutations parquet (overrides S001).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evolutionary/",
        help="Directory to save calculated parquet files and distribution plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    print("Loading evolutionary sparsity pipeline framework...")
    try:
        pipeline = EvolutionarySparsity(config_path=args.config)
        s_path, p_path, plot = pipeline.run_pipeline(
            sequence_path_override=args.input,
            mutations_path_override=args.mutations,
            output_dir=args.output,
        )

        print("\n--- Evolutionary Sparsity Pipeline Executed Successfully ---")
        print(f"Core Parquet:        {s_path}")
        print(f"Probability Parquet: {p_path}")
        print(f"Distribution Plot:   {plot}")
        print("------------------------------------------------------------")
        print("Evolutionary sparsity values successfully generated.")
    except Exception as e:
        print(f"ERROR: Evolutionary Sparsity Pipeline Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
