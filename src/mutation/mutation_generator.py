import itertools
from typing import List, Dict


class MutationGenerator:
    """
    Generates single, double, or custom mutations for protein sequences.
    """

    def __init__(self, amino_acids: List[str] = None):
        if amino_acids is None:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.amino_acids = amino_acids

    def single_mutants(self, sequence: str) -> List[Dict[str, str]]:
        """
        Generate all single mutants of a sequence.
        Returns a list of dicts: {"position": int, "wt": str, "mut": str}
        """
        mutants = []
        for i, wt in enumerate(sequence):
            for aa in self.amino_acids:
                if aa != wt:
                    mutants.append({"position": i, "wt": wt, "mut": aa})
        return mutants

    def double_mutants(self, sequence: str) -> List[Dict[str, str]]:
        """
        Generate all double mutants of a sequence.
        Returns a list of dicts with two mutation entries.
        """
        mutants = []
        positions = range(len(sequence))

        for (i, j) in itertools.combinations(positions, 2):
            wt_i, wt_j = sequence[i], sequence[j]
            for aa_i in self.amino_acids:
                if aa_i == wt_i:
                    continue
                for aa_j in self.amino_acids:
                    if aa_j == wt_j:
                        continue
                    mutants.append({
                        "pos1": i, "wt1": wt_i, "mut1": aa_i,
                        "pos2": j, "wt2": wt_j, "mut2": aa_j
                    })
        return mutants

    def custom_mutants(self, sequence: str, positions: List[int]) -> List[Dict[str, str]]:
        """
        Generate mutants only at specified positions.
        """
        mutants = []
        for pos in positions:
            wt = sequence[pos]
            for aa in self.amino_acids:
                if aa != wt:
                    mutants.append({"position": pos, "wt": wt, "mut": aa})
        return mutants

