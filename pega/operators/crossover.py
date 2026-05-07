"""
pega.operators.crossover
========================
Crossover operators for the PEGA genetic algorithm.

All operators take two parent sequences (strings of amino acids) of equal
length and return two offspring strings.

Available methods
-----------------
single_point   — one random cut point
two_point      — two random cut points
uniform        — per-position coin flip

Usage
-----
>>> from pega.operators.crossover import crossover
>>> c1, c2 = crossover("ACDEFGHIKL", "LKIHGFEDCA", method="single_point")
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Individual operators
# ---------------------------------------------------------------------------


def single_point_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """One-point crossover.

    Selects a random cut point and swaps the tails of the two parents.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of the same length.

    Returns
    -------
    tuple[str, str]
        Two offspring sequences.

    Raises
    ------
    ValueError
        If the parents have different lengths or are shorter than 2 residues.
    """
    _validate(parent1, parent2)
    cut = random.randint(1, len(parent1) - 1)
    return (parent1[:cut] + parent2[cut:],
            parent2[:cut] + parent1[cut:])


def two_point_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """Two-point crossover.

    Selects two random cut points and swaps the middle segment.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of the same length.

    Returns
    -------
    tuple[str, str]
        Two offspring sequences.
    """
    _validate(parent1, parent2)
    p1 = random.randint(1, len(parent1) - 2)
    p2 = random.randint(p1 + 1, len(parent1) - 1)
    return (parent1[:p1] + parent2[p1:p2] + parent1[p2:],
            parent2[:p1] + parent1[p1:p2] + parent2[p2:])


def uniform_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """Uniform crossover.

    At each position, randomly decides which parent contributes the residue
    (50/50 probability).

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of the same length.

    Returns
    -------
    tuple[str, str]
        Two offspring sequences.
    """
    _validate(parent1, parent2)
    c1, c2 = [], []
    for a, b in zip(parent1, parent2):
        if random.random() < 0.5:
            c1.append(a); c2.append(b)
        else:
            c1.append(b); c2.append(a)
    return "".join(c1), "".join(c2)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METHODS: dict[str, callable] = {
    "single_point": single_point_crossover,
    "two_point":    two_point_crossover,
    "uniform":      uniform_crossover,
}


def crossover(
    parent1: str,
    parent2: str,
    method: str = "single_point",
) -> tuple[str, str]:
    """Apply a crossover operator to two parent sequences.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of equal length.
    method : {"single_point", "two_point", "uniform"}
        Crossover method to apply. Default: ``"single_point"``.

    Returns
    -------
    tuple[str, str]
        Two offspring sequences.

    Raises
    ------
    ValueError
        If ``method`` is not recognised.

    Examples
    --------
    >>> crossover("ACDEFGHIKL", "LKIHGFEDCA", method="single_point")
    ('ACDEFEDCA', 'LKIHGGHIKL')   # example — actual output is random
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown crossover method '{method}'. "
            f"Choose from: {list(_METHODS)}"
        )
    return _METHODS[method](parent1, parent2)


def available_methods() -> list[str]:
    """Return the list of available crossover method names."""
    return list(_METHODS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate(parent1: str, parent2: str) -> None:
    if len(parent1) != len(parent2):
        raise ValueError(
            f"Parents must have equal length "
            f"(got {len(parent1)} and {len(parent2)})."
        )
    if len(parent1) < 2:
        raise ValueError("Parents must be at least 2 residues long.")
