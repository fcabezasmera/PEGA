"""
pega.predictors.ampir
=====================
ampir predictors — mature and precursor SVM models (R package).

ampir uses two support vector machine classifiers:

* ``"mature"``    — optimised for short mature peptides (< 60 amino acids).
* ``"precursor"`` — optimised for full-length precursor proteins.

Both predictors write a temporary R script, invoke ``Rscript``, and parse
the CSV output.  The output column ``prob_AMP`` is used as the score.

Note: sequences shorter than 10 amino acids or containing non-standard
amino acids receive ``NA`` for ``prob_AMP`` and are excluded from results.

Installation
------------
    # inside pega_env (R is bundled via r-base):
    Rscript -e 'install.packages("ampir", repos="https://cloud.r-project.org")'

Reference
---------
Fingerhut L. et al. (2021). ampir: an R package for fast genome-wide
prediction of antimicrobial peptides. *Bioinformatics*, 36(21), 5262–5263.
https://github.com/Legana/ampir
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
# Shared R execution logic
# ---------------------------------------------------------------------------


def _run_ampir(fasta_path: Path, model_type: str) -> pd.DataFrame:
    """Run ampir in R and return parsed results.

    Parameters
    ----------
    fasta_path:
        Validated path to the FASTA file.
    model_type:
        ``"mature"`` or ``"precursor"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["seq_name", "ampir_{model_type}_score"]``.
    """
    score_col = f"ampir_{model_type}_score"

    with tempfile.TemporaryDirectory() as tmp_dir:
        r_script_path = os.path.join(tmp_dir, f"ampir_{model_type}.R")
        csv_path = os.path.join(tmp_dir, f"ampir_{model_type}.csv")

        # predict_amps() output columns: seq_name, seq_aa, prob_AMP
        r_code = f"""\
library(ampir)
seqs <- read_faa("{fasta_path}")
preds <- predict_amps(seqs, model = "{model_type}")
df <- data.frame(
  seq_name = preds$seq_name,
  {score_col} = preds$prob_AMP
)
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
            total=100, desc=f"ampir ({model_type})", unit="%", leave=False
        ) as pbar:
            while process.poll() is None:
                time.sleep(0.5)
                if pbar.n < 90:
                    pbar.update(2)
            pbar.update(100 - pbar.n)

        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ampir ({model_type}) failed (exit {returncode}).\n"
                f"R stderr: {process.stderr.read()}\n"
                "Check your R installation: "
                "Rscript -e 'install.packages(\"ampir\")'"
            )

        if not os.path.exists(csv_path):
            raise RuntimeError(
                f"ampir ({model_type}) did not produce output. "
                "Check your R and ampir installation."
            )

        df = pd.read_csv(csv_path)

    # Drop sequences where ampir returned NA (< 10 aa or non-standard aa).
    return df.dropna(subset=[score_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mature model
# ---------------------------------------------------------------------------


class AmpirMaturePredictor(BasePredictor):
    """ampir SVM predictor — mature peptide model.

    Best suited for short mature peptides (< 60 amino acids).
    Requires R with the ``ampir`` package installed inside ``pega_env``.
    """

    name = "ampir_mature"
    predictor_id = 3
    description = "SVM AMP predictor — mature peptide model (R/ampir)."
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
        return _run_ampir(self._validate_fasta(fasta_path), model_type="mature")


# ---------------------------------------------------------------------------
# Precursor model
# ---------------------------------------------------------------------------


class AmpirPrecursorPredictor(BasePredictor):
    """ampir SVM predictor — precursor protein model.

    Best suited for full-length precursor protein sequences.
    Requires R with the ``ampir`` package installed inside ``pega_env``.
    """

    name = "ampir_precursor"
    predictor_id = 4
    description = "SVM AMP predictor — precursor protein model (R/ampir)."
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
        return _run_ampir(self._validate_fasta(fasta_path), model_type="precursor")
