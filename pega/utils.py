"""
pega.utils
==========
Main scoring orchestrator for PEGA.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pega.registry import registry


def _progress(current: int, total: int, name: str, elapsed: float | None = None) -> None:
    """Print a simple [X/Y] progress line."""
    width = len(str(total))
    time_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
    print(f"  [{current:{width}}/{total}]  {name}{time_str}")


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
        If ``True``, validate sequences before scoring.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

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
                print(f"  [skip] '{cls.display_name}' is not available.", file=sys.stderr)
                continue
            predictor_classes.append(cls)
    else:
        predictor_classes = registry.list_available()

    if not predictor_classes:
        raise ValueError(
            "No predictors are available.\n"
            "Run 'PEGA list' to see which predictors are installed."
        )

    total = len(predictor_classes)

    # ------------------------------------------------------------------
    # Run predictors
    # ------------------------------------------------------------------
    import os
    max_workers = os.cpu_count() if jobs == -1 else max(1, jobs)

    dfs: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    if max_workers == 1:
        # Sequential — clean ordered output
        for i, cls in enumerate(predictor_classes, 1):
            _progress(i, total, cls.display_name)
            t0 = time.perf_counter()
            try:
                dfs[cls.name] = cls().score(fasta_path)
            except Exception as exc:  # noqa: BLE001
                errors[cls.display_name] = str(exc)
                print(
                    f"  [warning] {cls.display_name} failed "
                    f"({time.perf_counter() - t0:.1f}s): {exc}",
                    file=sys.stderr,
                )

    else:
        # Parallel — disable individual tqdm bars to keep output clean
        import os
        os.environ["TQDM_DISABLE"] = "1"

        print(f"  Running {total} predictors with {max_workers} parallel workers...")
        print()

        completed = 0

        def _run(cls):
            t0 = time.perf_counter()
            result = cls().score(fasta_path)
            return cls.name, result, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run, cls): cls for cls in predictor_classes}
            for future in as_completed(futures):
                cls = futures[future]
                completed += 1
                try:
                    name, df, elapsed = future.result()
                    dfs[name] = df
                    _progress(completed, total, f"{cls.display_name} ✓", elapsed)
                except Exception as exc:  # noqa: BLE001
                    errors[cls.display_name] = str(exc)
                    _progress(completed, total, f"{cls.display_name} ✗")
                    print(f"       {cls.display_name} failed: {exc}", file=sys.stderr)

        os.environ.pop("TQDM_DISABLE", None)

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
        print(f"\n  Saved → {out_path}")

    if errors:
        print(f"\n  Failed: {', '.join(errors)}")

    return merged
