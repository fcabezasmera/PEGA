"""
pega.predictors.amplify
=======================
AMPlify predictors — balanced and imbalanced models (conda).

AMPlify writes output TSV files to the current working directory.
PEGA runs it in a temporary directory and parses column 4 (0-indexed)
as the score, matching the original PEGA implementation.

Output file columns (0-indexed)
--------------------------------
0  seq_name
1  sequence
2  log-scaled score
3  AMP / non-AMP label
4  probability score   ← used by PEGA

Installation
------------
    conda create -n amplify_env python=3.6 -y
    conda activate amplify_env
    mamba install bioconda::amplify

Reference
---------
Li C. et al. (2022). AMPlify. *BMC Genomics*, 23, 77.
https://github.com/BirolLab/AMPlify
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "amplify_env"


def _run_amplify(fasta_path: Path, model_type: str) -> pd.DataFrame:
    """Run AMPlify in a temporary directory and return parsed results."""
    score_col = f"AMPlify_{model_type}_score"

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "conda", "run", "-n", _CONDA_ENV,
            "AMPlify", "-s", str(fasta_path),
        ]
        if model_type == "imbalanced":
            cmd += ["-m", "imbalanced"]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp_dir,           # AMPlify writes output files here
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"AMPlify ({model_type}) failed (exit {exc.returncode}).\n"
                f"{exc.stderr.decode(errors='replace')}"
            ) from exc

        # AMPlify writes: AMPlify_{model_type}_results_<timestamp>.tsv
        pattern = os.path.join(tmp_dir, f"AMPlify_{model_type}_results_*.tsv")
        matching = glob.glob(pattern)
        if not matching:
            # Fallback: balanced uses 'balanced' in name regardless
            pattern = os.path.join(tmp_dir, "AMPlify_balanced_results_*.tsv")
            matching = glob.glob(pattern)

        if not matching:
            raise FileNotFoundError(
                f"AMPlify ({model_type}) produced no output file. "
                "Check your AMPlify installation."
            )

        output_file = max(matching, key=os.path.getctime)

        # Columns (0-indexed): 0=seq_name, 1=seq, 2=log_score, 3=label, 4=prob
        raw = pd.read_csv(output_file, sep="\t", skiprows=1, header=None)
        raw[4] = pd.to_numeric(raw[4], errors="coerce")
        raw = raw[raw[4].notnull()]

    return pd.DataFrame({
        "seq_name": raw[0].astype(str).values,
        score_col: raw[4].values,
    }).reset_index(drop=True)


class AmplifyBalancedPredictor(BasePredictor):
    """AMPlify balanced model (conda)."""

    name = "amplify_balanced"
    display_name = "AMPlify (balanced)"
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
        """Score with AMPlify balanced model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPlify_balanced_score"]``.
        """
        return _run_amplify(self._validate_fasta(fasta_path), "balanced")


class AmplifyImbalancedPredictor(BasePredictor):
    """AMPlify imbalanced model (conda)."""

    name = "amplify_imbalanced"
    display_name = "AMPlify (imbalanced)"
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
        """Score with AMPlify imbalanced model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPlify_imbalanced_score"]``.
        """
        return _run_amplify(self._validate_fasta(fasta_path), "imbalanced")
