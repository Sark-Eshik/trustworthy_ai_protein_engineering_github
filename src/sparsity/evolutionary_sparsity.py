import pandas as pd
from typing import Dict, Any, List


class EvolutionarySparsity:
    """
    Computes evolutionary sparsity using mutation frequency tables
    (e.g., Megascale-D or other evolutionary datasets).
    """

    def __init__(self, frequency_table_path: str):
        """
        Load a CSV containing mutation frequencies.
        Expected columns: position, wt, mut, frequency
        """
        self.freq_table = pd.read_csv(frequency_table_path)
        self._index_table()

    def _index_table(self):
        """
        Create a fast lookup dictionary:
        (position, wt, mut) -> frequency
        """
        self.lookup = {}
        for _, row in self.freq_table.iterrows():
            key = (int(row["position"]), row["wt"], row["mut"])
            self.lookup[key] = float(row["frequency"])

    def sparsity_score(self, mutation: Dict[str, Any]) -> float:
        """
        Compute evolutionary sparsity for a single mutation.
        Lower frequency → higher sparsity.
        """
        key = (mutation["position"], mutation["wt"], mutation["mut"])
        freq = self.lookup.get(key, None)

        if freq is None:
            raise ValueError(f"No evolutionary frequency for mutation {mutation}")

        # Normalize frequency to [0,1]
        max_freq = max(self.lookup.values())
        normalized = freq / max_freq

        # Evolutionary sparsity = 1 - normalized frequency
        return 1.0 - normalized

    def sparsity_scores(self, mutations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute evolutionary sparsity for a list of mutations.
        """
        results = []
        for m in mutations:
            score = self.sparsity_score(m)
            out = m.copy()
            out["evolutionary_sparsity"] = score
            results.append(out)
        return results

