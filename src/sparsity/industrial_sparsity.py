import pandas as pd
from typing import Dict, Any, List


class IndustrialSparsity:
    """
    Computes industrial sparsity using industrial-condition scoring tables.
    Expected columns: position, wt, mut, industrial_score
    """

    def __init__(self, industrial_table_path: str):
        """
        Load a CSV containing industrial-condition scores.
        """
        self.table = pd.read_csv(industrial_table_path)
        self._index_table()

    def _index_table(self):
        """
        Create a fast lookup dictionary:
        (position, wt, mut) -> industrial_score
        """
        self.lookup = {}
        for _, row in self.table.iterrows():
            key = (int(row["position"]), row["wt"], row["mut"])
            self.lookup[key] = float(row["industrial_score"])

    def sparsity_score(self, mutation: Dict[str, Any]) -> float:
        """
        Compute industrial sparsity for a single mutation.
        Lower industrial score → higher sparsity.
        """
        key = (mutation["position"], mutation["wt"], mutation["mut"])
        score = self.lookup.get(key, None)

        if score is None:
            raise ValueError(f"No industrial score for mutation {mutation}")

        # Normalize industrial score to [0,1]
        max_score = max(self.lookup.values())
        normalized = score / max_score

        # Industrial sparsity = 1 - normalized industrial score
        return 1.0 - normalized

    def sparsity_scores(self, mutations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute industrial sparsity for a list of mutations.
        """
        results = []
        for m in mutations:
            s = self.sparsity_score(m)
            out = m.copy()
            out["industrial_sparsity"] = s
            results.append(out)
        return results

