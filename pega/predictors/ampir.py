"""
pega.predictors.ampir
=====================
ampir predictors — mature and precursor models (R package).

Both predictors write a temporary R script, invoke ``Rscript``, and parse
the CSV output.  Two separate classes are registered so that users can
enable each model independently.

Installation
------------
    install.packages("ampir")   # inside an R session

Reference
---------
Fingerhut L. et al. (2021).  ampir: an R package for fast genome-wide
prediction of antimicrobial peptides.  *Bioinformatics*, 36(21), 5262–5263.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor


# ---------------------------------------------------------------------------
# Shared logic
# ---------------------------------------------------------------------------


def _run_ampir(fasta_path: Path, model_type: str) -> pd.DataFrame:
    """Execute ampir in R and return the parsed results.

    Parameters
    ----------
    fasta_path:
        Validated path to the FASTA file.
    model_type:
        Either ``"mature"`` or ``"precursor"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["seq_name", "ampir_{model_type}_score"]``.
    """
    score_col = f"ampir_{model_type}_score"

    with tempfile.TemporaryDirectory() as tmp_dir:
        r_script_path = os.path.join(tmp_dir, f"ampir_{model_type}.R")
        csv_path = os.path.join(tmp_dir, f"ampir_{model_type}.csv")

        r_code = f"""\
library(ampir)
seqs <- read_faa("{fasta_path}")
preds <- predict_amps(seqs, model = "{model_type}")
df <- as.data.frame(preds)
df${score_col} <- df[, 3]
df <- df[, c("seq_name", "{score_col}")]
write.csv(df, file = "{csv_path}", row.names = FALSE)
"""

        with open(r_script_path, "w") as f:
            f.write(r_code)

        process = subprocess.Popen(
            ["Rscript", r_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        with tqdm(
            total=100,
            desc=f"ampir ({model_type})",
            unit="%",
            leave=False,
        ) as pbar:
            while process.poll() is None:
                time.sleep(0.5)
                if pbar.n < 90:
                    pbar.update(2)
            pbar.update(100 - pbar.n)

        returncode = process.wait()
        if returncode != 0:
            stderr = process.stderr.read()
            raise RuntimeError(
                f"ampir ({model_type}) failed with exit code {returncode}.\n"
                f"R stderr: {stderr}"
            )

        if not os.path.exists(csv_path):
            raise RuntimeError(
                f"ampir ({model_type}) did not produce output.  "
                "Check your R and ampir installation."
            )

        return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Mature model
# ---------------------------------------------------------------------------


class AmpirMaturePredictor(BasePredictor):
    """ampir mature-peptide model (logistic regression, R).

    Requires R (>= 4.0) with the ``ampir`` package installed.
    """

    name = "ampir_mature"
    predictor_id = 3
    description = "Logistic regression AMP predictor — mature peptide model (R/ampir)."
    category = "r"

    @classmethod
    def is_available(cls) -> bool:
        return cls._r_package_installed("ampir")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with the ampir mature model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "ampir_mature_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        return _run_ampir(fasta_path, model_type="mature")


# ---------------------------------------------------------------------------
# Precursor model
# ---------------------------------------------------------------------------


class AmpirPrecursorPredictor(BasePredictor):
    """ampir precursor-peptide model (logistic regression, R).

    Requires R (>= 4.0) with the ``ampir`` package installed.
    """

    name = "ampir_precursor"
    predictor_id = 4
    description = "Logistic regression AMP predictor — precursor peptide model (R/ampir)."
    category = "r"

    @classmethod
    def is_available(cls) -> bool:
        return cls._r_package_installed("ampir")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with the ampir precursor model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "ampir_precursor_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        return _run_ampir(fasta_path, model_type="precursor")
