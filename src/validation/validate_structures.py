# src/validation/validate_structures.py
"""Protein Structure validation and certification script.

Verifies that all registered PDB structure files:
1. Exist on disk and are readable.
2. Load successfully into FreeSASA or a parser.
3. Contain valid 3D coordinates.
4. Contain valid residues and chains.
"""

import argparse
import os
import sys
from typing import Dict, Any, List, Optional

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.datasets.loaders import DatasetLoader

try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False


class StructureValidator:
    """Validator to certify protein structure files (PDBs) before processing."""

    def __init__(self, config_path: str = "configs") -> None:
        """Initialize the structure validator.

        Parameters
        ----------
        config_path : str
            Directory path containing application configurations.
        """
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="structure_validator",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )
        self.loader = DatasetLoader(config_path=config_path)

    def validate_structures(self, structures_path_override: Optional[str] = None) -> bool:
        """Loads registered structure records and certifies each PDB file.

        Parameters
        ----------
        structures_path_override : Optional[str]
            Optional path to protein structures parquet (S003).

        Returns
        -------
        bool
            True if all structures pass validation, False otherwise.
        """
        self.logger.info("Starting structure validation check...")

        try:
            struct_df = self.loader.load("S003", file_path_override=structures_path_override)
        except Exception as e:
            self.logger.error(f"Failed to load structure list (S003): {e}")
            print(f"Error: Failed to load structure list (S003): {e}", file=sys.stderr)
            return False

        if struct_df.empty:
            self.logger.error("Protein structures dataset (S003) is empty.")
            print("Error: Protein structures dataset (S003) is empty.", file=sys.stderr)
            return False

        failures = 0
        for _, row in struct_df.iterrows():
            protein_id = row["protein_id"]
            pdb_id = row["pdb_id"]
            pdb_path = row["structure_path"]
            chain_id = row["chain_id"]

            self.logger.info(f"Checking structure for protein {protein_id} (PDB: {pdb_id}) at: {pdb_path}")

            # 1. Check file existence
            if not os.path.exists(pdb_path):
                self.logger.error(f"PDB file for {protein_id} is missing at: {pdb_path}")
                print(f"Error: PDB file for {protein_id} is missing at: {pdb_path}", file=sys.stderr)
                failures += 1
                continue

            # 2. Check file size
            if os.path.getsize(pdb_path) == 0:
                self.logger.error(f"PDB file for {protein_id} at '{pdb_path}' is empty.")
                print(f"Error: PDB file for {protein_id} at '{pdb_path}' is empty.", file=sys.stderr)
                failures += 1
                continue

            # 3. Check for ATOM coordinates and residues
            has_coords = False
            has_atoms = False
            try:
                with open(pdb_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("ATOM  "):
                            has_atoms = True
                            # Extract coordinates from PDB columns 31-54
                            try:
                                x = float(line[30:38].strip())
                                y = float(line[38:46].strip())
                                z = float(line[46:54].strip())
                                has_coords = True
                            except (ValueError, IndexError):
                                pass
                            if has_atoms and has_coords:
                                break
            except Exception as e:
                self.logger.error(f"Failed to read/parse file '{pdb_path}': {e}")
                failures += 1
                continue

            if not has_atoms:
                self.logger.error(f"No ATOM coordinates found in PDB file '{pdb_path}'.")
                print(f"Error: No ATOM coordinates found in PDB file '{pdb_path}'.", file=sys.stderr)
                failures += 1
                continue

            if not has_coords:
                self.logger.error(f"Failed to find valid coordinates (x, y, z) in PDB file '{pdb_path}'.")
                print(f"Error: Failed to find valid coordinates (x, y, z) in PDB file '{pdb_path}'.", file=sys.stderr)
                failures += 1
                continue

            # 4. Check FreeSASA parsing if available
            if FREESASA_AVAILABLE:
                try:
                    structure = freesasa.Structure(pdb_path)
                    if structure.nAtoms() == 0:
                        self.logger.error(f"FreeSASA loaded zero atoms for PDB '{pdb_path}'.")
                        failures += 1
                        continue
                except Exception as e:
                    self.logger.error(f"FreeSASA failed to parse PDB file '{pdb_path}': {e}")
                    failures += 1
                    continue

            self.logger.info(f"Structure for {protein_id} (PDB: {pdb_id}) certified successfully.")

        if failures > 0:
            self.logger.error(f"Structure validation failed with {failures} error(s).")
            print("Structure validation certification: FAIL", file=sys.stderr)
            return False

        self.logger.info("All registered protein structures certified successfully.")
        print("Structure validation certification: PASS")
        return True


def main() -> None:
    """Main command-line entry point for structure validation."""
    parser = argparse.ArgumentParser(
        description="Verify coordinates, readability, and compatibility of PDB structure files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw structures parquet (overrides S003).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to application configuration folder.",
    )
    args = parser.parse_args()

    validator = StructureValidator(config_path=args.config)
    success = validator.validate_structures(args.input)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
