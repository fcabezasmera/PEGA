"""
pega.predictors.amPEPpy
=======================
amPEPpy predictor — Random Forest on sequence composition features.

Command: ``ampep predict -m <model> -i <fasta> -o <output>``

Output TSV columns (0-indexed)
-------------------------------
0  features...
1  score        ← used by PEGA
2  ...
3  seq_name     ← used by PEGA

Installation
------------
    pip install git+https://github.com/tlawrence3/amPEPpy.git

Reference
---------
Lawrence T.J. et al. (2021). *Bioinformatics*, 37(18), 2979–2981.
https://github.com/tlawrence3/amPEPpy
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor


class AmPEPpyPredictor(BasePredictor):
    """Random Forest AMP predictor via the amPEPpy ``ampep`` command."""

    name = "ampep"
    display_name = "amPEPpy"
    predictor_id = 2
    description = "Random Forest on sequence composition features via amPEPpy."
    category = "pip"

    @classmethod
    def is_available(cls) -> bool:
        return cls._executable_on_path("ampep")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``ampep predict``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "amPEPpy_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        model_path = self.models_dir() / "amPEP.model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"amPEP model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )

        tmpdir = tempfile.mkdtemp(prefix="ampep_tmp_")
        results_file = os.path.join(tmpdir, "results.tsv")

        try:
            cmd = [
                "ampep", "predict",
                "-m", str(model_path),
                "-i", str(fasta_path),
                "-o", results_file,
            ]

            with tqdm(total=100, desc="amPEPpy", unit="%", leave=False) as pbar:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                while process.poll() is None:
                    time.sleep(0.5)
                    if pbar.n < 80:
                        pbar.update(2)
                returncode = process.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode, cmd, stderr=process.stderr.read()
                    )
                pbar.update(100 - pbar.n)

            if not os.path.exists(results_file):
                raise RuntimeError("amPEPpy did not produce the expected results file.")

            df_raw = pd.read_csv(results_file, sep="\t", header=0, dtype=str)

            # Column 3 = seq_name, column 1 = score
            if df_raw.shape[1] >= 4:
                seq_col   = df_raw.columns[3]
                score_col = df_raw.columns[1]
                df = pd.DataFrame({
                    "seq_name":   df_raw[seq_col].astype(str),
                    "amPEPpy_score": pd.to_numeric(df_raw[score_col], errors="coerce"),
                })
            else:
                # Fallback: find columns by name
                seq_cols   = [c for c in df_raw.columns if "seq" in c.lower() or "name" in c.lower()]
                score_cols = [c for c in df_raw.columns if "score" in c.lower()]
                df = pd.DataFrame({
                    "seq_name":   df_raw[seq_cols[0]].astype(str),
                    "amPEPpy_score": pd.to_numeric(df_raw[score_cols[0]], errors="coerce"),
                })

            # amPEPpy leaves *features.csv files in cwd — clean them up
            for f in glob.glob(os.path.join(os.getcwd(), "*features.csv")):
                try:
                    os.remove(f)
                except Exception:
                    pass

            return df.dropna(subset=["amPEPpy_score"]).reset_index(drop=True)

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ampep predict failed (exit {exc.returncode}).\n"
                f"stderr: {exc.stderr}\n"
                "Check: pip install git+https://github.com/tlawrence3/amPEPpy.git"
            ) from exc

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
