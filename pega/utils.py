"""
pega.utils
==========
Main scoring orchestrator for PEGA.
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
    validate: bool = True,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.
    predictor_names:
        Predictor names to run. ``None`` uses all available.
    export_tsv:
        Optional path to save results as TSV.
    jobs:
        Parallel workers (``-1`` = all CPU threads, default ``1``).
    validate:
        If ``True``, validate sequences before scoring and warn about
        non-canonical amino acids, short sequences, etc.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}"
        )

    # ------------------------------------------------------------------
    # Optional sequence validation
    # ------------------------------------------------------------------
    if validate:
        from pega.preprocess.validator import validate_fasta
        ok = validate_fasta(fasta_path, verbose=True)
        if not ok:
            raise ValueError(
                "FASTA file contains invalid sequences. "
                "Run 'PEGA validate --fasta ...' for details."
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
    # Run predictors
    # ------------------------------------------------------------------
    import os
    max_workers = os.cpu_count() if jobs == -1 else max(1, jobs)

    dfs: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    def _run(cls):
        print(f"  Running {cls.display_name} (predictor {cls.predictor_id})...")
        return cls.name, cls().score(fasta_path)

    if max_workers == 1:
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
                    print(f"  [warning] {cls.display_name} failed: {exc}",
                          file=sys.stderr)

    if not dfs:
        raise RuntimeError("All predictors failed. Check the error messages above.")

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
