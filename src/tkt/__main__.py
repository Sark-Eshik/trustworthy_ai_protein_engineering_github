# src/tkt/__main__.py
"""Main package entry point for the TKT application submodule.

Provides guidance on running landscape generation and analysis.
"""

import sys

def main() -> None:
    print("Trustworthy AI Protein Engineering - TKT Submodule")
    print("=================================================")
    print("To generate the complete single-mutation landscape, run:")
    print("  python -m src.tkt.generate_landscape")
    print("")
    print("To analyze the single-mutation landscape (calculate sparsities & scores), run:")
    print("  python -m src.tkt.analyze_landscape")
    print("=================================================")

if __name__ == "__main__":
    main()
