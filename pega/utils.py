"""
pega.utils
==========
Main scoring orchestrator for PEGA.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pega.registry import registry


def _progress(current: int, total: int, name: str, elapsed: float | None = None) -> None:
    width = len(str(total))
    time_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
    print(f"  [{current:{width}}/{total}]  {name}{time_str}")


def calculate_scores(
    fasta_path: str | Path,
    predictor_names: list[str] | None = None,
    export_tsv: str | Path | None = None,
    jobs: int = 1,
    validate: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors."""
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # ------------------------------------------------------------------
    # Unified header + validation
    # ------------------------------------------------------------------
    if validate:
        from pega.preprocess.validator import validate_fasta, _fasta_stats_from_path
        stats = _fasta_stats_from_path(fasta_path)
        predictor_label = "all available" if predictor_names is None else ", ".join(predictor_names)

        print(f"  Input      : {fasta_path.name}")
        print(f"  Sequences  : {stats['n']}  "
              f"(len: {stats['min_len']}–{stats['max_len']}, avg {stats['avg_len']})")
        print(f"  Predictors : {predictor_label}")
        print(f"  Jobs       : {jobs}")
        out_label = export_tsv if export_tsv else f"PEGA_results_<timestamp>.tsv"
        print(f"  Output     : {out_label}")

        ok, status_msg = validate_fasta(fasta_path, verbose=False)
        print(f"  Status     : {status_msg}")
        print()

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
        for i, cls in enumerate(predictor_classes, 1):
            _progress(i, total, cls.display_name)
            t0 = time.perf_counter()
            try:
                dfs[cls.name] = cls().score(fasta_path)
            except Exception as exc:  # noqa: BLE001
                errors[cls.display_name] = str(exc)
                print(f"  [warning] {cls.display_name} failed "
                      f"({time.perf_counter() - t0:.1f}s): {exc}", file=sys.stderr)
    else:
        import tqdm as _tqdm_module
        _orig_tqdm_init = _tqdm_module.tqdm.__init__

        def _silent_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _orig_tqdm_init(self, *args, **kwargs)

        _tqdm_module.tqdm.__init__ = _silent_init
        print(f"  Running {total} predictors with {max_workers} parallel workers...")
        print()

        completed = 0

        def _run(cls):
            t0 = time.perf_counter()
            result = cls().score(fasta_path)
            return cls.name, result, time.perf_counter() - t0

        try:
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
                        print(f"       failed: {exc}", file=sys.stderr)
        finally:
            _tqdm_module.tqdm.__init__ = _orig_tqdm_init

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
    # Summary statistics across predictor scores
    # ------------------------------------------------------------------
    score_cols = [c for c in merged.columns if c.endswith("_score")]
    if score_cols:
        scores = merged[score_cols].fillna(0)
        merged["mean_score"]      = scores.mean(axis=1)
        merged["geomean_score"]   = np.exp(np.log(scores.clip(lower=0) + 1e-9).mean(axis=1))
        merged["median_score"]    = scores.median(axis=1)
        merged["min_score"]       = scores.min(axis=1)
        merged["std_score"]       = scores.std(axis=1, ddof=0)  # population std (all predictors measured, not a sample)
        merged["consensus_score"] = 1 / (1 + merged["std_score"])

    # ------------------------------------------------------------------
    # Export — always save to TSV (default name if not specified)
    # ------------------------------------------------------------------
    if export_tsv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_tsv = f"PEGA_results_{timestamp}.tsv"

    out_path = Path(export_tsv)
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved → {out_path}  ({len(merged)} sequences × {len(merged.columns)-1} predictors)")

    if errors:
        print(f"\n  Failed: {', '.join(errors)}")

    return merged
