"""
pega.operators.mutation
=======================
Mutation operators for the PEGA genetic algorithm.

Three operations are available:

substitute
    Replace one or more residues using BLOSUM62-guided substitution.
    At each selected position, the amino acid with the highest BLOSUM62
    score relative to the current residue is chosen (conservative).

insert
    Insert a random canonical amino acid at a random position.

delete
    Remove one residue at a random position (minimum length: 2).

Usage
-----
>>> from pega.operators.mutation import mutate, insert_aa, delete_aa
>>> mutate("ACDEFGHIKL", mutation_rate=0.2)
'ACDEFGHIKV'
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# BLOSUM62 substitution matrix
# ---------------------------------------------------------------------------

BLOSUM62: dict[str, dict[str, int]] = {
    "A": {"A": 4,"R":-1,"N":-2,"D":-2,"C": 0,"Q":-1,"E":-1,"G": 0,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 0,"W":-3,"Y":-2,"V": 0},
    "R": {"A":-1,"R": 5,"N": 0,"D":-2,"C":-3,"Q": 1,"E": 0,"G":-2,"H": 0,"I":-3,"L":-2,"K": 2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
    "N": {"A":-2,"R": 0,"N": 6,"D": 1,"C":-3,"Q": 0,"E": 0,"G": 0,"H": 1,"I":-3,"L":-3,"K": 0,"M":-2,"F":-3,"P":-2,"S": 1,"T": 0,"W":-4,"Y":-2,"V":-3},
    "D": {"A":-2,"R":-2,"N": 1,"D": 6,"C":-3,"Q": 0,"E": 2,"G":-1,"H":-1,"I":-3,"L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S": 0,"T":-1,"W":-4,"Y":-3,"V":-3},
    "C": {"A": 0,"R":-3,"N":-3,"D":-3,"C": 9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,"L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
    "Q": {"A":-1,"R": 1,"N": 0,"D": 0,"C":-3,"Q": 5,"E": 2,"G":-2,"H": 0,"I":-3,"L":-2,"K": 1,"M": 0,"F":-3,"P":-1,"S": 0,"T":-1,"W":-2,"Y":-1,"V":-2},
    "E": {"A":-1,"R": 0,"N": 0,"D": 2,"C":-4,"Q": 2,"E": 5,"G":-2,"H": 0,"I":-3,"L":-3,"K": 1,"M":-2,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "G": {"A": 0,"R":-2,"N": 0,"D":-1,"C":-3,"Q":-2,"E":-2,"G": 6,"H":-2,"I":-4,"L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S": 0,"T":-2,"W":-2,"Y":-3,"V":-3},
    "H": {"A":-2,"R": 0,"N": 1,"D":-1,"C":-3,"Q": 0,"E": 0,"G":-2,"H": 8,"I":-3,"L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y": 2,"V":-3},
    "I": {"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I": 4,"L": 2,"K":-3,"M": 1,"F": 0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V": 3},
    "L": {"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I": 2,"L": 4,"K":-2,"M": 2,"F": 0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V": 1},
    "K": {"A":-1,"R": 2,"N": 0,"D":-1,"C":-3,"Q": 1,"E": 1,"G":-2,"H":-1,"I":-3,"L":-2,"K": 5,"M":-1,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "M": {"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q": 0,"E":-2,"G":-3,"H":-2,"I": 1,"L": 2,"K":-1,"M": 5,"F": 0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V": 1},
    "F": {"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I": 0,"L": 0,"K":-3,"M": 0,"F": 6,"P":-4,"S":-2,"T":-2,"W": 1,"Y": 3,"V":-1},
    "P": {"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,"L":-3,"K":-1,"M":-2,"F":-4,"P": 7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
    "S": {"A": 1,"R":-1,"N": 1,"D": 0,"C":-1,"Q": 0,"E": 0,"G": 0,"H":-1,"I":-2,"L":-2,"K": 0,"M":-1,"F":-2,"P":-1,"S": 4,"T": 1,"W":-3,"Y":-2,"V":-2},
    "T": {"A": 0,"R":-1,"N": 0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 5,"W":-2,"Y":-2,"V": 0},
    "W": {"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,"L":-2,"K":-3,"M":-1,"F": 1,"P":-4,"S":-3,"T":-2,"W":11,"Y": 2,"V":-3},
    "Y": {"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H": 2,"I":-1,"L":-1,"K":-2,"M":-1,"F": 3,"P":-3,"S":-2,"T":-2,"W": 2,"Y": 7,"V":-1},
    "V": {"A": 0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I": 3,"L": 1,"K":-2,"M": 1,"F":-1,"P":-2,"S":-2,"T": 0,"W":-3,"Y":-1,"V": 4},
}

_CANONICAL: list[str] = sorted(BLOSUM62)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def mutate(sequence: str, mutation_rate: float = 0.1) -> str:
    """BLOSUM62-guided point substitution.

    Randomly selects positions to mutate (at least one, at most
    ``ceil(len * mutation_rate)``), then replaces each residue with the
    non-identical amino acid that has the highest BLOSUM62 score against it
    (most conservative substitution).

    Parameters
    ----------
    sequence : str
        Input amino acid sequence.
    mutation_rate : float
        Fraction of positions that may be mutated (0 < rate ≤ 1).
        Default: 0.1.

    Returns
    -------
    str
        Mutated sequence (same length as input).

    Raises
    ------
    ValueError
        If the sequence is empty or ``mutation_rate`` is out of range.
    """
    _validate_sequence(sequence)
    if not 0 < mutation_rate <= 1:
        raise ValueError(f"mutation_rate must be in (0, 1], got {mutation_rate}.")

    max_muts = max(1, int(len(sequence) * mutation_rate))
    n_muts   = np.random.randint(1, max_muts + 1)
    positions = np.random.choice(len(sequence), n_muts, replace=False)

    seq = list(sequence)
    for pos in positions:
        aa = seq[pos]
        candidates = [a for a in BLOSUM62.get(aa, {}) if a != aa]
        if candidates:
            seq[pos] = max(candidates, key=lambda a: BLOSUM62[aa][a])
    return "".join(seq)


def insert_aa(sequence: str) -> str:
    """Insert a random canonical amino acid at a random position.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence.

    Returns
    -------
    str
        Sequence one residue longer than the input.
    """
    _validate_sequence(sequence)
    aa  = random.choice(_CANONICAL)
    pos = random.randint(0, len(sequence))
    return sequence[:pos] + aa + sequence[pos:]


def delete_aa(sequence: str) -> str:
    """Remove one residue at a random position.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence (must be at least 2 residues).

    Returns
    -------
    str
        Sequence one residue shorter than the input.

    Raises
    ------
    ValueError
        If the sequence is shorter than 2 residues.
    """
    if len(sequence) < 2:
        raise ValueError("Sequence must be at least 2 residues to delete one.")
    pos = random.randint(0, len(sequence) - 1)
    return sequence[:pos] + sequence[pos + 1:]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("Sequence must not be empty.")
    invalid = set(sequence.upper()) - set(_CANONICAL)
    if invalid:
        raise ValueError(f"Non-canonical amino acids found: {sorted(invalid)}")
