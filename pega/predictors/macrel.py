"""
pega.predictors.macrel
======================
Macrel predictor — SVM with physico-chemical and structural features (conda).

Output file
-----------
Macrel writes ``macrel.out.prediction.gz`` inside the output directory.
PEGA decompresses it and reads it with ``skiprows=2, header=None``.

Output columns (0-indexed, after skiprows=2)
--------------------------------------------
0  seq_name     ← used by PEGA
1  sequence
2  AMP / non-AMP label
3  hemolytic label
4  AMP_probability  ← used by PEGA

Installation
------------
    conda create --name macrel_env -c bioconda macrel

Reference
---------
Santos-Júnior C.D. et al. (2020). PeerJ 8:e10555.
https://github.com/BigDataBiology/macrel
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor

_CONDA_ENV = "macrel_env"


class MacrelPredictor(BasePredictor):
    """Macrel SVM AMP predictor (conda).

    Requires::

        conda create --name macrel_env -c bioconda macrel
    """

    name = "macrel"
    display_name = "Macrel"
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
        """Score sequences with ``macrel peptides``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "Macrel_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        output_dir = "temp_macrel"

        try:
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "macrel", "peptides",
                "--fasta", str(fasta_path),
                "--output", output_dir,
                "--keep-negatives",
            ]

            with tqdm(total=100, desc="Macrel", unit="%", leave=False) as pbar:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                while process.poll() is None:
                    time.sleep(1)
                    if pbar.n < 70:
                        pbar.update(2)
                returncode = process.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode, cmd, stderr=process.stderr.read()
                    )
                pbar.update(100 - pbar.n)

            # Locate and decompress the output file
            gz_file = os.path.join(output_dir, "macrel.out.prediction.gz")
            if not os.path.exists(gz_file):
                raise FileNotFoundError(
                    f"Macrel output not found: {gz_file}\n"
                    "Check your Macrel installation."
                )

            subprocess.run(["gzip", "-d", gz_file], check=True)

            result_file = os.path.join(output_dir, "macrel.out.prediction")
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"Decompressed file not found: {result_file}")

            # skiprows=2 skips the comment line and column header
            # Columns: 0=seq_name, 1=sequence, 2=AMP label, 3=hemolytic, 4=AMP_prob
            raw = pd.read_csv(result_file, sep="\t", skiprows=2, header=None)
            scores = raw.iloc[:, [0, 4]].rename(columns={0: "seq_name", 4: "Macrel_score"})
            scores["Macrel_score"] = pd.to_numeric(scores["Macrel_score"], errors="coerce")

            return scores.dropna(subset=["Macrel_score"]).reset_index(drop=True)

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Macrel failed (exit {exc.returncode}).\n"
                f"{exc.stderr}\n"
                "Check: conda create --name macrel_env -c bioconda macrel"
            ) from exc

        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
