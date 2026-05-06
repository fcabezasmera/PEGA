"""
pega.utils
==========
Main scoring orchestrator for PEGA.

The central function is :func:`calculate_scores`, which:

1. Resolves which predictors to run (all available, or a user-specified
   subset).
2. Runs each predictor on the input FASTA file and collects the score
   DataFrames.
3. Merges all results on ``seq_name``.
4. Optionally appends ensemble summary columns via :mod:`pega.ensemble`.
5. Optionally exports the result to a TSV file.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from pega.ensemble import apply_ensemble
from pega.registry import registry


def calculate_scores(
    fasta_path: str | Path,
    predictor_names: list[str] | None = None,
    apply_ensemble: bool = True,
    membership: bool = False,
    export_tsv: str | Path | None = None,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.  All sequences must be composed of
        standard single-letter amino acid codes.
    predictor_names:
        Names of the predictors to run (e.g. ``["ampnet", "modlamp_rf"]``).
        When ``None``, all available predictors are used automatically.
    apply_ensemble:
        If ``True`` (default), six ensemble summary columns are appended
        to the output DataFrame.
    membership:
        If ``True``, fuzzy membership columns are appended for each
        predictor score.  Requires ``apply_ensemble=True``.
    export_tsv:
        Optional output file path.  When provided, the results DataFrame
        is written as a tab-separated values file.

    Returns
    -------
    pandas.DataFrame
        One row per sequence.  Columns:

        * ``seq_name`` — sequence identifier from the FASTA header.
        * ``<name>_score`` — one column per predictor that was run.
        * ``arithmetic_mean``, ``geometric_mean``, ``weighted_voting``,
          ``rank_aggregation``, ``majority_voting``, ``stacking_ensemble``
          — present when ``apply_ensemble=True``.

    Raises
    ------
    FileNotFoundError
        If ``fasta_path`` does not exist.
    ValueError
        If no predictors are available or all requested predictors are
        unavailable.
    RuntimeError
        If every predictor fails during scoring.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}\n"
            "Please verify the path and try again."
        )

    # ------------------------------------------------------------------
    # Resolve predictor set
    # ------------------------------------------------------------------
    if predictor_names is not None:
        predictor_classes = []
        for name in predictor_names:
            cls = registry.get(name)
            if not cls.is_available():
                print(
                    f"  [warning] Predictor '{name}' is not available in this "
                    "environment and will be skipped.",
                    file=sys.stderr,
                )
                continue
            predictor_classes.append(cls)
    else:
        predictor_classes = registry.list_available()

    if not predictor_classes:
        raise ValueError(
            "No predictors are available.\n"
            "Run 'pega list' to see which predictors are installed and "
            "what is required to enable them."
        )

    # ------------------------------------------------------------------
    # Run predictors
    # ------------------------------------------------------------------
    dfs: list[pd.DataFrame] = []

    for cls in predictor_classes:
        predictor = cls()
        print(f"  Running {cls.name} (predictor {cls.predictor_id})...")
        try:
            result = predictor.score(fasta_path)
            dfs.append(result)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [warning] {cls.name} failed and will be excluded: {exc}",
                file=sys.stderr,
            )

    if not dfs:
        raise RuntimeError(
            "All predictors failed.  Check the error messages above."
        )

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="seq_name", how="outer")

    score_columns = [c for c in merged.columns if c.endswith("_score")]

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    if apply_ensemble and len(score_columns) >= 2:
        from pega.ensemble import apply_ensemble as _apply

        merged = _apply(merged, score_columns, membership=membership)
    elif apply_ensemble and len(score_columns) == 1:
        print(
            "  [info] Ensemble methods require at least two predictors. "
            "Skipping ensemble columns.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if export_tsv is not None:
        out_path = Path(export_tsv)
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"  Results saved to: {out_path}")

    return merged
