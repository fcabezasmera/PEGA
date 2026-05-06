"""
pega.predictors.macrel
======================
Macrel predictor — SVM with physico-chemical and structural features (conda).

The predictor invokes the ``macrel`` command-line tool inside its conda
environment and parses the output prediction file.

Installation
------------
    conda create -n macrel_env -c bioconda macrel

Reference
---------
Santos-Júnior C.D. et al. (2020).  Macrel: antimicrobial peptide screening
in genomes and metagenomes.  *PeerJ*, 8, e10555.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor

_CONDA_ENV = "macrel_env"


class MacrelPredictor(BasePredictor):
    """Macrel SVM AMP predictor (conda environment).

    Requires a conda environment named ``macrel_env`` with Macrel installed.
    Create it with::

        conda create -n macrel_env -c bioconda macrel
    """

    name = "macrel"
    predictor_id = 6
    description = "SVM with physico-chemical features via the Macrel tool (conda)."
    category = "conda"

    @classmethod
    def is_available(cls) -> bool:
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "macrel", "--version"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with Macrel.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "macrel_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "macrel", "peptides",
                "--fasta", str(fasta_path),
                "--output", tmp_dir,
                "--keep-negatives",
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with tqdm(
                total=100, desc="Macrel", unit="%", leave=False
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
                    f"Macrel failed with exit code {returncode}.\n"
                    f"stderr: {stderr}"
                )

            # Macrel writes prediction.prediction inside the output dir.
            pred_files = list(Path(tmp_dir).glob("*.prediction"))
            if not pred_files:
                raise RuntimeError(
                    "Macrel did not produce a prediction file.  "
                    "Check your Macrel installation."
                )

            raw = pd.read_csv(pred_files[0], sep="\t", comment="#")

        # Normalise columns — Macrel output has 'Access' and 'AMP_probability'.
        raw.columns = [c.strip() for c in raw.columns]
        name_col = next(
            (c for c in raw.columns if "access" in c.lower() or "name" in c.lower()),
            raw.columns[0],
        )
        score_col = next(
            (c for c in raw.columns if "prob" in c.lower() or "score" in c.lower()),
            raw.columns[-1],
        )

        return pd.DataFrame(
            {
                "seq_name": raw[name_col].astype(str),
                "macrel_score": pd.to_numeric(raw[score_col], errors="coerce"),
            }
        ).dropna(subset=["macrel_score"]).reset_index(drop=True)
