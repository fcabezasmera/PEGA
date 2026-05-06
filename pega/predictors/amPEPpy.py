"""
pega.predictors.amPEPpy
=======================
amPEPpy predictor — Random Forest trained on sequence composition features.

The predictor invokes the ``ampep`` command-line tool (amPEPpy) and parses
its output.  The pre-trained model file is bundled with PEGA.

Pre-trained weights
-------------------
File: ``pega/models/amPEP.model``
Download: ``pega download-models``

Installation
------------
    pip install ampep
    # or
    conda install -c bioconda ampep

Reference
---------
Yan J. et al. (2022).  amPEPpy 1.0: A portable and accurate antimicrobial
peptide prediction tool.  *Bioinformatics*, 38(4), 1162–1163.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor


class AmPEPpyPredictor(BasePredictor):
    """Random Forest AMP predictor via the amPEPpy command-line tool.

    Requires the ``ampep`` executable and the pre-trained model file
    ``pega/models/amPEP.model``.  Run ``pega download-models`` to obtain
    the model weights.
    """

    name = "ampep"
    predictor_id = 2
    description = "Random Forest on sequence composition features via amPEPpy."
    category = "pip"

    @classmethod
    def is_available(cls) -> bool:
        return cls._executable_on_path("ampep")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with amPEPpy.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "ampep_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        model_path = self.models_dir() / "amPEP.model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"amPEP model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_pattern = os.path.join(tmp_dir, "ampep_results_*.tsv")

            cmd = [
                "ampep", "score",
                "--fasta", str(fasta_path),
                "--model", str(model_path),
                "--outdir", tmp_dir,
            ]

            print("  Running amPEPpy...")
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"amPEPpy failed with exit code {exc.returncode}.\n"
                    f"stderr: {exc.stderr.decode(errors='replace')}"
                ) from exc

            result_files = glob.glob(result_pattern)
            if not result_files:
                raise RuntimeError(
                    "amPEPpy did not produce an output file.  "
                    "Check your amPEPpy installation."
                )

            result_file = max(result_files, key=os.path.getctime)
            raw = pd.read_csv(result_file, sep="\t")

        # Normalise column names across amPEPpy versions.
        raw.columns = [c.strip().lower() for c in raw.columns]
        name_col = next(
            (c for c in raw.columns if "name" in c or "id" in c), raw.columns[0]
        )
        score_col = next(
            (c for c in raw.columns if "score" in c or "prob" in c), raw.columns[1]
        )

        df = pd.DataFrame(
            {
                "seq_name": raw[name_col].astype(str),
                "ampep_score": pd.to_numeric(raw[score_col], errors="coerce"),
            }
        )
        return df.dropna(subset=["ampep_score"]).reset_index(drop=True)
