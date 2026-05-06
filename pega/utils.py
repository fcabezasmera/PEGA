"""
pega.utils
==========
Main scoring orchestrator for PEGA.

Runs available predictors on a FASTA file, optionally in parallel,
and returns a merged DataFrame with one score column per predictor.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pega.registry import registry


def calculate_scores(
    fasta_path: str | Path,
    predictor_names: list[str] | None = None,
    export_tsv: str | Path | None = None,
    jobs: int = 1,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.  Must contain only ASCII characters
        and standard amino acid codes.
    predictor_names:
        Names of the predictors to run (e.g. ``["ampnet", "modlamp_rf"]``).
        When ``None``, all available predictors are used automatically.
    export_tsv:
        Optional output file path.  Results are written as TSV.
    jobs:
        Number of predictors to run in parallel (default 1 = sequential).
        Use ``-1`` to use all available CPU threads.

    Returns
    -------
    pandas.DataFrame
        One row per sequence.  Columns: ``seq_name`` plus one
        ``<name>_score`` column per predictor that ran successfully.
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
                    f"  [warning] '{cls.display_name}' is not available "
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
            "Run 'PEGA list' to see which predictors are installed."
        )

    # ------------------------------------------------------------------
    # Run predictors — sequential or parallel
    # ------------------------------------------------------------------
    import os
    max_workers = os.cpu_count() if jobs == -1 else max(1, jobs)

    dfs: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    def _run(cls):
        print(f"  Running {cls.display_name} (predictor {cls.predictor_id})...")
        return cls.name, cls().score(fasta_path)

    if max_workers == 1:
        # Sequential — simpler output, easier to debug
        for cls in predictor_classes:
            try:
                name, df = _run(cls)
                dfs[name] = df
            except Exception as exc:  # noqa: BLE001
                errors[cls.display_name] = str(exc)
                print(
                    f"  [warning] {cls.display_name} failed and will be "
                    f"excluded: {exc}",
                    file=sys.stderr,
                )
    else:
        # Parallel — predictors run concurrently
        print(f"  Running {len(predictor_classes)} predictors "
              f"with {max_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run, cls): cls for cls in predictor_classes}
            for future in as_completed(futures):
                cls = futures[future]
                try:
                    name, df = future.result()
                    dfs[name] = df
                except Exception as exc:  # noqa: BLE001
                    errors[cls.display_name] = str(exc)
                    print(
                        f"  [warning] {cls.display_name} failed: {exc}",
                        file=sys.stderr,
                    )

    if not dfs:
        raise RuntimeError(
            "All predictors failed.  Check the error messages above."
        )

    # ------------------------------------------------------------------
    # Merge in original predictor order
    # ------------------------------------------------------------------
    ordered = [dfs[cls.name] for cls in predictor_classes if cls.name in dfs]
    merged = ordered[0]
    for df in ordered[1:]:
        merged = pd.merge(merged, df, on="seq_name", how="outer")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if export_tsv is not None:
        out_path = Path(export_tsv)
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"  Results saved to: {out_path}")

    if errors:
        print(f"\n  Predictors excluded due to errors: {', '.join(errors)}")

    return merged
