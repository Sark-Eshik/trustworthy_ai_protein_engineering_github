# src/sparsity/structural/structural_sparsity.py
"""Structural Sparsity calculation submodule.

Loads protein structures and mutation datasets, runs the SASA calculation engine,
normalizes SASA measurements, and calculates structural sparsity (1 - Normalized SASA).
"""

import argparse
import os
import sys
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry, DatasetDefinition
from src.datasets.loaders import DatasetLoader
from src.sparsity.structural.sasa_engine import SASAEngine


class StructuralSparsity:
    """Calculates structural sparsity for single point mutations using residue Solvent Accessible Surface Area."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the StructuralSparsity submodule.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.logger = get_logger(
            name="structural_sparsity",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )
        self.loader = DatasetLoader(config_path=config_path)
        self.sasa_engine = SASAEngine(logger=self.logger)

    def run_pipeline(
        self,
        structures_path_override: Optional[str] = None,
        mutations_path_override: Optional[str] = None,
        output_dir: str = "results/structural/",
    ) -> Tuple[str, str, str]:
        """Runs the complete Structural Sparsity calculation pipeline.

        Parameters
        ----------
        structures_path_override : Optional[str]
            Optional path to protein structures parquet (S003).
        mutations_path_override : Optional[str]
            Optional path to Megascale-D mutations parquet (S001).
        output_dir : str
            Directory to save generated parquets and plots.

        Returns
        -------
        Tuple[str, str, str]
            Paths to the three generated output files:
            structural_sparsity.parquet, sasa_values.parquet, structural_distribution.png.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info("Starting structural sparsity pipeline...")

        # Load S003 (structures metadata) and S001 (mutations) safely via DatasetLoader (fail-fast rule)
        struct_df = self.loader.load("S003", file_path_override=structures_path_override)
        mut_df = self.loader.load("S001", file_path_override=mutations_path_override)

        # Index structure metadata by protein_id for O(1) lookup
        # S003 expected columns: protein_id, pdb_id, chain_id, structure_path
        struct_lookup = {}
        for _, row in struct_df.iterrows():
            struct_lookup[row["protein_id"]] = {
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "structure_path": row["structure_path"],
            }

        # Cache computed SASA dictionary per protein/PDB path to prevent redundant calculations
        sasa_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

        self.logger.info(f"Processing {len(mut_df)} mutations for structural SASA scoring...")

        sasas: List[float] = []

        for idx, row in mut_df.iterrows():
            protein_id = row["protein_id"]
            pos = str(row["position"])
            mut_id = row["mutation_id"]

            metadata = struct_lookup.get(protein_id)
            if not metadata:
                err_msg = f"Structure metadata for protein_id '{protein_id}' not found in registered S003 dataset."
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            pdb_path = metadata["structure_path"]
            chain_id = metadata["chain_id"]

            # Load/Compute SASA values for this structure
            if pdb_path not in sasa_cache:
                sasa_cache[pdb_path] = self.sasa_engine.compute_sasa(pdb_path)

            sasa_dict = sasa_cache[pdb_path]

            # Look up SASA value for the specified chain and residue position
            chain_sasa = sasa_dict.get(chain_id, {})
            sasa_val = chain_sasa.get(pos)

            if sasa_val is None:
                # If residue is not found, log warning and fallback to a default/simulated value (usually 0.0 or average)
                self.logger.warning(
                    f"Residue position {pos} in chain {chain_id} was not found in "
                    f"computed SASA for structure {pdb_path}. Defaulting to 0.0."
                )
                sasa_val = 0.0

            sasas.append(sasa_val)

        # Build output dataframe
        out_df = pd.DataFrame({
            "mutation_id": mut_df["mutation_id"],
            "sasa": sasas,
        })

        # Calculate normalized SASA: sasa / max(sasa)
        max_sasa = out_df["sasa"].max()
        if max_sasa == 0 or np.isnan(max_sasa):
            self.logger.warning("Maximum SASA is zero or NaN. Normalizing all SASA values to 0.0.")
            out_df["normalized_sasa"] = 0.0
        else:
            out_df["normalized_sasa"] = out_df["sasa"] / max_sasa

        # Structural Sparsity = 1.0 - Normalized SASA
        out_df["structural_sparsity"] = 1.0 - out_df["normalized_sasa"]

        # Define file outputs
        parquet_sparsity_path = os.path.join(output_dir, "structural_sparsity.parquet")
        parquet_values_path = os.path.join(output_dir, "sasa_values.parquet")
        plot_path = os.path.join(output_dir, "structural_distribution.png")

        # Save core structural_sparsity dataset (D103)
        d103_cols = ["mutation_id", "sasa", "normalized_sasa", "structural_sparsity"]
        out_df[d103_cols].to_parquet(parquet_sparsity_path, index=False)
        self.logger.info(f"Wrote D103 Parquet: {parquet_sparsity_path}")

        # Save sasa_values.parquet as requested in specifications
        out_df.to_parquet(parquet_values_path, index=False)
        self.logger.info(f"Wrote values Parquet: {parquet_values_path}")

        # Generate Distribution Plot
        self.logger.info("Generating distribution histogram plot...")
        plt.figure(figsize=(8, 6))
        plt.hist(
            out_df["structural_sparsity"],
            bins=min(15, len(out_df)),
            edgecolor="black",
            color="#9b59b6",
            alpha=0.7,
        )
        plt.title("Distribution of Structural Sparsity", fontsize=14)
        plt.xlabel("Structural Sparsity (0 = Exposed, 1 = Buried)", fontsize=12)
        plt.ylabel("Mutation Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Wrote distribution plot: {plot_path}")

        return parquet_sparsity_path, parquet_values_path, plot_path


def main() -> None:
    """Main command-line entry point for Structural Sparsity pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute FreeSASA-based structural sparsity metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw structures parquet (overrides S003).",
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
        default="results/structural/",
        help="Directory to save calculated parquet files and distribution plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    print("Loading structural sparsity pipeline framework...")
    try:
        pipeline = StructuralSparsity(config_path=args.config)
        s_path, v_path, plot = pipeline.run_pipeline(
            structures_path_override=args.input,
            mutations_path_override=args.mutations,
            output_dir=args.output,
        )

        print("\n--- Structural Sparsity Pipeline Executed Successfully ---")
        print(f"Core Parquet:   {s_path}")
        print(f"Values Parquet: {v_path}")
        print(f"Distribution:   {plot}")
        print("----------------------------------------------------------")
        print("Structural sparsity values successfully generated.")
    except Exception as e:
        print(f"ERROR: Structural Sparsity Pipeline Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
