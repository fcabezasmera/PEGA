"""
pega.utils
==========
Main scoring orchestrator for PEGA.

Runs available predictors on a FASTA file and returns a merged DataFrame
with one score column per predictor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pega.registry import registry


def calculate_scores(
    fasta_path: str | Path,
    predictor_names: list[str] | None = None,
    export_tsv: str | Path | None = None,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.  Must contain only standard ASCII
        characters and 20 standard amino acid codes.
    predictor_names:
        Names of the predictors to run (e.g. ``["ampnet", "modlamp_rf"]``).
        When ``None``, all available predictors are used automatically.
    export_tsv:
        Optional output file path.  When provided, the results DataFrame
        is written as a tab-separated values file.

    Returns
    -------
    pandas.DataFrame
        One row per sequence.  Columns: ``seq_name`` plus one
        ``<name>_score`` column per predictor that ran successfully.

    Raises
    ------
    FileNotFoundError
        If ``fasta_path`` does not exist.
    ValueError
        If no predictors are available or the FASTA contains non-ASCII text.
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
                    f"  [warning] Predictor '{name}' is not available "
                    "and will be skipped.",
                    file=sys.stderr,
                )
                continue
            predictor_classes.append(cls)
    else:
        predictor_classes = registry.list_available()

    if not predictor_classes:
        raise ValueError(
            "No predictors are available.\n"
            "Run 'pega list' to see which predictors are installed."
        )

    # ------------------------------------------------------------------
    # Run predictors
    # ------------------------------------------------------------------
    dfs: list[pd.DataFrame] = []

    for cls in predictor_classes:
        print(f"  Running {cls.name} (predictor {cls.predictor_id})...")
        try:
            result = cls().score(fasta_path)
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
    # Merge results on seq_name
    # ------------------------------------------------------------------
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="seq_name", how="outer")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if export_tsv is not None:
        out_path = Path(export_tsv)
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"  Results saved to: {out_path}")

    return merged
