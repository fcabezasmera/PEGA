"""
pega.base
=========
Abstract base class that every PEGA predictor must implement.

Adding a new predictor
----------------------
1. Create a module in ``pega/predictors/``.
2. Define a class that inherits from ``BasePredictor``.
3. Set the five required class attributes.
4. Implement ``is_available()`` and ``score()``.
5. The registry discovers it automatically.
"""

from __future__ import annotations

import abc
import functools
import shutil
import subprocess
import threading
from pathlib import Path
from typing import ClassVar

import pandas as pd


class BasePredictor(abc.ABC):
    """Abstract base class for all PEGA AMP predictors.

    Class attributes
    ----------------
    name : str
        Short lowercase identifier used in CLI and output column headers.
        Must be unique (e.g. ``"ampnet"``).
    display_name : str
        Human-readable name shown in ``PEGA list`` and log messages
        (e.g. ``"AMPnet"``).
    predictor_id : int
        Stable integer identifier (1–99).
    description : str
        One-sentence description shown by ``PEGA list``.
    category : {"pip", "r", "conda"}
        How the predictor's external dependency is installed.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    predictor_id: ClassVar[int]
    description: ClassVar[str]
    category: ClassVar[str]

    @classmethod
    def score_column(cls) -> str:
        """Return the score column name: ``"{name}_score"``."""
        return f"{cls.name}_score"

    @staticmethod
    def models_dir() -> Path:
        """Return the path to the bundled ``pega/models/`` directory."""
        return Path(__file__).resolve().parent / "models"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if all runtime dependencies are satisfied.

        Must never raise. Must be fast (no network access, no large imports).
        """

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score all sequences in a FASTA file.

        Returns
        -------
        pandas.DataFrame
            Two columns: ``["seq_name", "<name>_score"]``.
            Scores are in ``[0, 1]``.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _executable_on_path(name: str) -> bool:
        """Return ``True`` if ``name`` is found on the system PATH."""
        return shutil.which(name) is not None

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _r_package_installed(package: str) -> bool:
        """Return ``True`` if an R package is installed inside pega_env.

        Cached — this spawns an Rscript subprocess, and availability can't
        change mid-process, so repeated calls (e.g. once per chunk in
        ``screen_sequences``) would otherwise re-spawn it needlessly.
        """
        # Use the conda environment's Rscript if available
        import os
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        rscript = os.path.join(conda_prefix, "bin", "Rscript") if conda_prefix else ""
        if not rscript or not Path(rscript).exists():
            rscript = shutil.which("Rscript") or ""
        if not rscript:
            return False
        result = subprocess.run(
            [rscript, "-e",
             f'if (!requireNamespace("{package}", quietly=TRUE)) quit(status=1)'],
            capture_output=True,
        )
        return result.returncode == 0

    @staticmethod
    def _run_subprocess_with_progress(
        cmd: list[str],
        desc: str,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd`` to completion, showing a cosmetic tqdm progress bar.

        Runs ``subprocess.run(cmd, capture_output=True, ...)`` on a
        background thread; the polling loop driving the progress bar only
        checks ``thread.is_alive()`` and never touches the child's stdout/
        stderr pipes directly. This matters: a naive
        ``Popen(...); while process.poll() is None: sleep()`` loop never
        drains those pipes, so a child that writes more than the OS pipe
        buffer (commonly 64KB) blocks forever on write() with nothing
        reading it — a silent deadlock discovered in production when
        amPEPpy hung a multi-day cluster screening run. ``subprocess.run``
        drains both pipes concurrently internally, so this can't happen
        here regardless of how much output ``cmd`` produces.
        """
        result_holder: dict[str, subprocess.CompletedProcess] = {}

        def _run() -> None:
            result_holder["result"] = subprocess.run(
                cmd, capture_output=True, text=True, cwd=cwd, env=env,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        from tqdm import tqdm
        with tqdm(total=100, desc=desc, unit="%", leave=False) as pbar:
            while thread.is_alive():
                thread.join(timeout=0.5)
                if pbar.n < 80:
                    pbar.update(2)
            pbar.update(100 - pbar.n)

        return result_holder["result"]

    @staticmethod
    def _validate_fasta(fasta_path: str | Path) -> Path:
        """Validate that a FASTA file exists, is non-empty, and is pure ASCII."""
        path = Path(fasta_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {path}\n"
                "Please verify the path and try again."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"FASTA file is empty: {path}")
        try:
            path.read_text(encoding="ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"FASTA file contains non-ASCII characters: {path}\n"
                f"Detail: {exc}\n"
                "All headers and sequences must use standard ASCII characters."
            ) from exc
        return path

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"<{self.display_name} id={self.predictor_id} [{status}]>"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if abc.ABC in cls.__bases__:
            return
        required = ("name", "display_name", "predictor_id", "description", "category")
        missing = [a for a in required if not hasattr(cls, a)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is missing required class attributes: "
                + ", ".join(missing)
            )
        valid = {"pip", "r", "conda"}
        if cls.category not in valid:
            raise TypeError(
                f"{cls.__name__}.category must be one of {valid}, "
                f"got '{cls.category}'."
            )
