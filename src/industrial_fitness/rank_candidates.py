# src/industrial_fitness/rank_candidates.py
"""Candidate ranking module for Transketolase.

Ranks mutations based on Industrial Fitness Score and outputs the top N candidate
mutations as `top_candidate_mutations.csv` (D602) for wet-lab screening.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.industrial_fitness.fitness_validator import FitnessValidator

class CandidateRanker:
    """Ranks candidates by fitness, outputs D602 CSV, and verifies score ordering."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.logger = get_logger(
            name="candidate_ranker",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def rank_candidates(
        self,
        fitness_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        top_n: int = 25,
    ) -> str:
        """Loads D601, extracts the top N candidates, formats columns, and writes D602 CSV."""
        data_dir = self.config.paths.data_dir
        fit_path = fitness_path or os.path.join(data_dir, "final/industrial_fitness/industrial_fitness_scores.parquet")

        self.logger.info(f"Loading fitness scores from {fit_path}...")
        if not os.path.exists(fit_path):
            raise FileNotFoundError(f"Fitness scores parquet file not found at: {fit_path}")

        df_fit = pd.read_parquet(fit_path)
        self.logger.info(f"Loaded {len(df_fit)} scored mutations. Extracting top {top_n} candidates...")

        # 1. Sort and slice
        df_sorted = df_fit.sort_values(by="industrial_fitness_score", ascending=False).reset_index(drop=True)
        df_top = df_sorted.head(top_n).copy()

        # Update ranks for the top N candidates to make sure they are 1-based sequential integers
        df_top["rank"] = np.arange(1, len(df_top) + 1)

        # 2. Map and rename columns to match D602 specifications:
        # Schema: rank, mutation, reliability_score, predicted_stability, industrial_fitness_score
        df_top = df_top.rename(columns={"mutation_id": "mutation"})
        
        d602_cols = ["rank", "mutation", "reliability_score", "predicted_stability", "industrial_fitness_score"]
        df_csv = df_top[d602_cols].copy()

        # 3. Save D602 CSV
        out_dir = output_dir or os.path.join(data_dir, "final/industrial_fitness")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "top_candidate_mutations.csv")
        df_csv.to_csv(csv_path, index=False)
        self.logger.info(f"Saved D602 (Top Candidate Mutations) CSV to {csv_path}")

        # 4. Verify ordering of top candidates (Rank 1 > Rank 2 > Rank 3 based on score)
        validator = FitnessValidator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        if not validator.validate_ranking_csv(csv_path):
            err_msg = "Top Candidate CSV failed ranking and score alignment checks!"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Print the top candidates to stdout/logs for manual inspection
        print("\n=== TOP 25 CANDIDATE MUTATIONS FOR TRANSKETOLASE ===")
        print(df_csv.to_string(index=False))
        print("====================================================")

        return csv_path

def main() -> None:
    import numpy as np # import inside main since class-level might miss it
    parser = argparse.ArgumentParser(description="Rank engineering candidates based on fitness.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--fitness", type=str, default=None, help="Path to industrial fitness scores parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated candidate CSV.")
    parser.add_argument("--top", type=int, default=25, help="Number of top candidates to output.")
    args = parser.parse_args()

    try:
        ranker = CandidateRanker(config_path=args.config)
        ranker.rank_candidates(
            fitness_path=args.fitness,
            output_dir=args.output,
            top_n=args.top
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Candidate Ranking failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import numpy as np # guarantee numpy import in module
    main()
