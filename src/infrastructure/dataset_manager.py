import os
import pandas as pd
from typing import Optional, Dict, Any


class DatasetManager:
    """
    Centralized loader for datasets used throughout the protein engineering pipeline.
    Supports CSV, TSV, and basic FASTA formats.
    """

    def __init__(self, base_path: str = "data"):
        self.base_path = base_path

    def _full_path(self, filename: str) -> str:
        """Construct full path to a dataset file."""
        full = os.path.join(self.base_path, filename)
        if not os.path.exists(full):
            raise FileNotFoundError(f"Dataset not found: {full}")
        return full

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Parameters
        ----------
        filename : str
            CSV file inside the data directory.

        Returns
        -------
        pd.DataFrame
        """
        path = self._full_path(filename)
        return pd.read_csv(path, **kwargs)

    def load_tsv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a TSV file into a DataFrame.
        """
        path = self._full_path(filename)
        return pd.read_csv(path, sep="\t", **kwargs)

    def load_fasta(self, filename: str) -> Dict[str, str]:
        """
        Load a FASTA file into a dictionary: {header: sequence}.
        """
        path = self._full_path(filename)
        sequences = {}
        header = None
        seq_chunks = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header:
                        sequences[header] = "".join(seq_chunks)
                    header = line[1:]
                    seq_chunks = []
                else:
                    seq_chunks.append(line)

        if header:
            sequences[header] = "".join(seq_chunks)

        return sequences

