# src/tkt/mutation_enumerator.py
"""Mutation enumeration module for Transketolase.

Enumerates all possible single amino acid mutations for a given wildtype sequence.
For a sequence of length L, generates L * 19 alternative amino acids.
"""

from typing import List, Dict, Any

class MutationEnumerator:
    """Enumerates single amino acid substitutions across a protein sequence."""

    # Standard 20 amino acids
    STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

    def enumerate_mutations(self, sequence: str, protein_id: str = "TKT") -> List[Dict[str, Any]]:
        """Generates all possible single point mutations for the given sequence.

        Parameters
        ----------
        sequence : str
            Wildtype amino acid sequence.
        protein_id : str
            Identifier for the protein (default: 'TKT').

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries representing each single point mutation.
        """
        records = []
        for pos_idx, wt in enumerate(sequence):
            position = pos_idx + 1  # 1-based indexing
            for mut in self.STANDARD_AAS:
                if mut == wt:
                    continue  # skip wildtype self-identity

                mutation_id = f"{protein_id}_{wt}{position}{mut}"
                mutation_label = f"{wt}{position}{mut}"
                records.append({
                    "mutation_id": mutation_id,
                    "protein_id": protein_id,
                    "position": position,
                    "wildtype": wt,
                    "mutant": mut,
                    "mutation_label": mutation_label,
                })
        return records
