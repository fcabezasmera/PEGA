"""
PEGA — Peptide Evolution via Genetic Algorithm
===============================================
Multi-predictor ensemble scoring for antimicrobial peptide research.

https://github.com/fcabezasmera/PEGA
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "fcabezasmera"
__license__ = "MIT"

__all__ = [
    "score",
    "list_predictors",
    "__version__",
]


def score(
    fasta_path: str,
    predictors: list[str] | None = None,
    export_tsv: str | None = None,
):
    """Score sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.
    predictors:
        List of predictor names to use (e.g. ``["ampnet", "modlamp_rf"]``).
        If ``None``, all available predictors are used automatically.
    export_tsv:
        Optional file path. When provided, results are written as TSV.

    Returns
    -------
    pandas.DataFrame
        One row per sequence, one column per predictor score.
    """
    from pega.utils import calculate_scores

    return calculate_scores(
        fasta_path=fasta_path,
        predictor_names=predictors,
        export_tsv=export_tsv,
    )


def list_predictors(available_only: bool = False) -> list[dict]:
    """Return metadata for all registered predictors."""
    from pega.registry import registry

    predictors = (
        registry.list_available() if available_only else registry.list_all()
    )
    return [
        {
            "name": p.name,
            "predictor_id": p.predictor_id,
            "description": p.description,
            "category": p.category,
            "available": p.is_available(),
        }
        for p in predictors
    ]
