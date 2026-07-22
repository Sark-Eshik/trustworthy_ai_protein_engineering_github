# src/fitness/calculate.py
"""Calculates Industrial Fitness scores.

Execution command wrapper script mapping to the industrial_fitness module.
"""

import sys
import argparse
from src.industrial_fitness.fitness_calculator import FitnessCalculator

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Industrial Fitness Scores.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--analysis", type=str, default=None, help="Path to TKT mutation analysis parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated parquet file.")
    parser.add_argument("--inject-fault", action="store_true", help="Inject invalid values to test validator sensitivity.")
    args = parser.parse_args()

    try:
        calculator = FitnessCalculator(config_path=args.config)
        calculator.compute_fitness(
            analysis_path=args.analysis,
            output_dir=args.output,
            inject_fault=args.inject_fault
        )
        print("Done")
    except Exception as e:
        print(f"ERROR: Fitness calculation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
