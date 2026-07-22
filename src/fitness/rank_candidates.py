# src/fitness/rank_candidates.py
"""Ranks engineering candidates.

Execution command wrapper script mapping to the industrial_fitness module.
"""

import sys
import argparse
from src.industrial_fitness.rank_candidates import CandidateRanker

def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate mutations based on Industrial Fitness.")
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
    main()
