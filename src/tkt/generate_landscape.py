# src/tkt/generate_landscape.py
"""Command-line script to generate the complete single-mutation landscape of Transketolase.

Executes:
python -m src.tkt.generate_landscape
Outputs:
- data/final/tkt/tkt_single_mutation_landscape.parquet (D501)
- data/final/tkt/mutation_enumeration_report.csv
"""

import sys
import argparse
from src.tkt.landscape_generator import LandscapeGenerator

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete TKT single-point mutation landscape.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated files.")
    parser.add_argument("--inject-fault", action="store_true", help="Delete a position to trigger Athlete validation failure.")
    args = parser.parse_args()

    try:
        generator = LandscapeGenerator(config_path=args.config)
        p_path, r_path = generator.generate_landscape(output_dir=args.output, inject_fault=args.inject_fault)
        print(f"SUCCESS: Landscape parquet generated at: {p_path}")
        print(f"SUCCESS: Position report CSV generated at: {r_path}")
    except Exception as e:
        print(f"ERROR: TKT landscape generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
