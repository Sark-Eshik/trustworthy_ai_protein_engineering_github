# src/tkt/analyze_landscape.py
"""Command-line script to analyze the complete single-mutation landscape of Transketolase.

Executes:
python -m src.tkt.analyze_landscape
Outputs:
- data/final/tkt/tkt_mutation_analysis.parquet (D502)
"""

import sys
import argparse
from src.tkt.landscape_analysis import LandscapeAnalyzer

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the complete TKT single-point mutation landscape.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--landscape", type=str, default=None, help="Path to mutation landscape parquet.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated analysis parquet.")
    args = parser.parse_args()

    try:
        analyzer = LandscapeAnalyzer(config_path=args.config)
        out_path = analyzer.analyze_landscape(landscape_path=args.landscape, output_dir=args.output)
        print(f"SUCCESS: Analysis parquet generated at: {out_path}")
    except Exception as e:
        print(f"ERROR: TKT landscape analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
