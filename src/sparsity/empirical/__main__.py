# src/sparsity/empirical/__main__.py
"""Module execution entry point for the empirical sparsity module.

Allows executing the submodule directly via:
python -m src.sparsity.empirical
"""

import argparse
import sys
from .empirical_sparsity import EmpiricalSparsity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute empirical sparsity metrics from experimental counts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/megascale_d/megascale_d.parquet",
        help="Path to Megascale-D mutation dataset parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sparsity/empirical",
        help="Directory to save generated parquet, summary, and plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    try:
        es = EmpiricalSparsity(config_path=args.config)
        df = es.load_counts(args.input)
        result = es.calculate(df)
        es.write_output(result, args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
