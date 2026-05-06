"""
pega.predictors.amplify
=======================
AMPlify predictors — balanced and imbalanced deep learning models (conda).

Both predictors invoke the ``AMPlify`` command-line tool inside its conda
environment and parse the TSV output file.

Installation
------------
    conda create -n amplify_env -c bioconda amplify

Reference
---------
Li C. et al. (2022).  AMPlify: attentive deep learning model for discovery
of novel antimicrobial peptides effective against WHO priority pathogens.
*BMC Genomics*, 23, 77.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pega.base import BasePredictor

_CONDA_ENV = "amplify_env"


# ---------------------------------------------------------------------------
# Shared logic
# ---------------------------------------------------------------------------


def _run_amplify(fasta_path: Path, model_type: str) -> pd.DataFrame:
    """Execute AMPlify and return parsed results.

    Parameters
    ----------
    fasta_path:
        Validated path to the FASTA file.
    model_type:
        Either ``"balanced"`` or ``"imbalanced"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["seq_name", "amplify_{model_type}_score"]``.
    """
    score_col = f"amplify_{model_type}_score"

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "conda", "run", "-n", _CONDA_ENV,
            "AMPlify", "-s", str(fasta_path),
        ]
        if model_type == "imbalanced":
            cmd += ["-m", "imbalanced"]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_dir,
        )

        with tqdm(
            total=100,
            desc=f"AMPlify ({model_type})",
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
                f"AMPlify ({model_type}) failed with exit code {returncode}.\n"
                f"stderr: {stderr}"
            )

        pattern = os.path.join(tmp_dir, f"AMPlify_{model_type}_results_*.tsv")
        result_files = glob.glob(pattern)
        if not result_files:
            # Fallback: any TSV in the directory.
            result_files = glob.glob(os.path.join(tmp_dir, "*.tsv"))

        if not result_files:
            raise RuntimeError(
                f"AMPlify ({model_type}) did not produce an output file.  "
                "Check your AMPlify installation."
            )

        result_file = max(result_files, key=os.path.getctime)
        raw = pd.read_csv(result_file, sep="\t", skiprows=1, header=None)

    # AMPlify output columns: sequence_ID, sequence, score, prediction.
    raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
    scores = pd.to_numeric(raw["col_2"], errors="coerce")
    valid = scores.notna()

    return pd.DataFrame(
        {
            "seq_name": raw.loc[valid, "col_0"].astype(str).values,
            score_col: scores[valid].values,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Balanced model
# ---------------------------------------------------------------------------


class AmplifyBalancedPredictor(BasePredictor):
    """AMPlify deep learning predictor — balanced training dataset (conda).

    Requires a conda environment named ``amplify_env`` with AMPlify installed.
    """

    name = "amplify_balanced"
    predictor_id = 7
    description = "Deep learning AMP predictor — balanced model (AMPlify / conda)."
    category = "conda"

    @classmethod
    def is_available(cls) -> bool:
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "AMPlify", "--help"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with AMPlify (balanced model).

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "amplify_balanced_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        return _run_amplify(fasta_path, model_type="balanced")


# ---------------------------------------------------------------------------
# Imbalanced model
# ---------------------------------------------------------------------------


class AmplifyImbalancedPredictor(BasePredictor):
    """AMPlify deep learning predictor — imbalanced training dataset (conda).

    Requires a conda environment named ``amplify_env`` with AMPlify installed.
    """

    name = "amplify_imbalanced"
    predictor_id = 8
    description = "Deep learning AMP predictor — imbalanced model (AMPlify / conda)."
    category = "conda"

    @classmethod
    def is_available(cls) -> bool:
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "AMPlify", "--help"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with AMPlify (imbalanced model).

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "amplify_imbalanced_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        return _run_amplify(fasta_path, model_type="imbalanced")
