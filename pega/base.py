"""
pega.base
=========
Abstract base class that every PEGA predictor must implement.

Adding a new predictor
----------------------
1. Create a module in ``pega/predictors/``.
2. Define a class that inherits from ``BasePredictor``.
3. Set the four required class attributes.
4. Implement ``is_available()`` and ``score()``.
5. The registry discovers it automatically — no other changes needed.
"""

from __future__ import annotations

import abc
import shutil
import subprocess
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
    predictor_id : int
        Stable integer identifier (1–99) used for sorting and references.
    description : str
        One-sentence description shown by ``pega list``.
    category : {"pip", "r", "conda"}
        How the predictor's external dependency is installed.
    """

    name: ClassVar[str]
    predictor_id: ClassVar[int]
    description: ClassVar[str]
    category: ClassVar[str]

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @classmethod
    def score_column(cls) -> str:
        """Return the score column name produced by this predictor.

        By convention: ``"{name}_score"`` (e.g. ``"ampnet_score"``).
        """
        return f"{cls.name}_score"

    @staticmethod
    def models_dir() -> Path:
        """Return the absolute path to the bundled ``pega/models/`` directory."""
        return Path(__file__).resolve().parent / "models"

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if all runtime dependencies are satisfied.

        Must never raise — return ``False`` on any missing dependency.
        Must be fast: no network access, no large imports.

        Recommended pattern for pip dependencies::

            @classmethod
            def is_available(cls) -> bool:
                try:
                    import tensorflow  # noqa: F401
                    return True
                except ImportError:
                    return False

        For executables (R, conda tools) use :meth:`_executable_on_path`.
        For R packages use :meth:`_r_package_installed`.
        """

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score all sequences in a FASTA file.

        Parameters
        ----------
        fasta_path:
            Path to a FASTA file with standard single-letter amino acid codes.

        Returns
        -------
        pandas.DataFrame
            Two columns: ``["seq_name", "<name>_score"]``.
            Scores are in ``[0, 1]`` — higher means more likely AMP.

        Raises
        ------
        FileNotFoundError
            If ``fasta_path`` does not exist.
        ValueError
            If the file is empty or contains no valid sequences.
        RuntimeError
            If the underlying model or external tool fails.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _executable_on_path(name: str) -> bool:
        """Return ``True`` if ``name`` is found on the system PATH."""
        return shutil.which(name) is not None

    @staticmethod
    def _r_package_installed(package: str) -> bool:
        """Return ``True`` if an R package is installed.

        Parameters
        ----------
        package:
            Name of the R package (e.g. ``"ampir"``).
        """
        if shutil.which("Rscript") is None:
            return False
        result = subprocess.run(
            [
                "Rscript", "-e",
                f'if (!requireNamespace("{package}", quietly=TRUE)) quit(status=1)',
            ],
            capture_output=True,
        )
        return result.returncode == 0

    @staticmethod
    def _validate_fasta(fasta_path: str | Path) -> Path:
        """Check that a FASTA file exists and is non-empty.

        Returns the resolved ``Path`` or raises ``FileNotFoundError`` /
        ``ValueError``.
        """
        path = Path(fasta_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {path}\n"
                "Please verify the path and try again."
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"FASTA file is empty: {path}\n"
                "The file must contain at least one sequence."
            )
        return path

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"<{self.__class__.__name__} id={self.predictor_id} [{status}]>"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate required class attributes on concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if abc.ABC in cls.__bases__:
            return
        required = ("name", "predictor_id", "description", "category")
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
