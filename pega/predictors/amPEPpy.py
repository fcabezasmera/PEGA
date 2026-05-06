"""
pega.predictors.amPEPpy
=======================
amPEPpy predictor — Random Forest on sequence composition features.

The predictor invokes the ``ampep predict`` command and parses its output.
The pre-trained model file is bundled with PEGA (``pega/models/amPEP.model``).

Installation
------------
    pip install git+https://github.com/tlawrence3/amPEPpy.git

Usage (standalone)
------------------
    ampep predict -m pega/models/amPEP.model -s sequences.fasta

Reference
---------
Lawrence T.J. et al. (2021).  amPEPpy 1.0: A portable and accurate
antimicrobial peptide prediction tool.  *Bioinformatics*, 37(18), 2979–2981.
https://github.com/tlawrence3/amPEPpy
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor


class AmPEPpyPredictor(BasePredictor):
    """Random Forest AMP predictor via the amPEPpy ``ampep`` command.

    Requires the ``ampep`` executable::

        pip install git+https://github.com/tlawrence3/amPEPpy.git

    and the pre-trained model ``pega/models/amPEP.model``.
    """

    name = "ampep"
    predictor_id = 2
    description = "Random Forest on sequence composition features via amPEPpy."
    category = "pip"

    @classmethod
    def is_available(cls) -> bool:
        return cls._executable_on_path("ampep")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``ampep predict``.

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
            out_file = Path(tmp_dir) / "ampep_results.tsv"

            cmd = [
                "ampep", "predict",
                "-m", str(model_path),
                "-s", str(fasta_path),
                "-o", str(out_file),
            ]

            print("  Running amPEPpy...")
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"ampep predict failed (exit {exc.returncode}).\n"
                    f"stderr: {exc.stderr}\n"
                    "Check your amPEPpy installation: "
                    "pip install git+https://github.com/tlawrence3/amPEPpy.git"
                ) from exc

            if not out_file.exists():
                raise RuntimeError(
                    "ampep predict did not produce an output file. "
                    "Check your amPEPpy installation."
                )

            raw = pd.read_csv(out_file, sep="\t")
            raw.columns = [c.strip() for c in raw.columns]

        # Locate sequence name and score columns robustly.
        name_col = next(
            (c for c in raw.columns
             if c.lower() in ("sequence_id", "seq_name", "name", "id")),
            raw.columns[0],
        )
        score_col = next(
            (c for c in raw.columns
             if "prob" in c.lower() or "score" in c.lower() or "amp" in c.lower()),
            raw.columns[-1],
        )

        return pd.DataFrame({
            "seq_name": raw[name_col].astype(str),
            "ampep_score": pd.to_numeric(raw[score_col], errors="coerce"),
        }).dropna(subset=["ampep_score"]).reset_index(drop=True)
