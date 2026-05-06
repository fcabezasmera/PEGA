"""
pega.predictors.ampnet
======================
AMPnet predictor — convolutional neural network (TensorFlow).

Model
-----
A 1-D CNN trained on a benchmark AMP / non-AMP dataset.  Input features
are a one-hot encoding of the amino acid sequence (padded to 198 residues)
combined with Kyte-Doolittle hydrophobicity scores.

Pre-trained weights
-------------------
File: ``pega/models/convolutional_nn_1.h5``
Download: ``pega download-models``

Reference
---------
Please cite the original AMPnet publication when using this predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor

# Suppress TensorFlow informational messages.
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}
_MAX_LEN = 198

_KD_SCORES: dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class AMPnetPredictor(BasePredictor):
    """Convolutional neural network AMP predictor (TensorFlow).

    Requires the TensorFlow package and the pre-trained model file
    ``pega/models/convolutional_nn_1.h5``.  Run ``pega download-models``
    to obtain the model weights.
    """

    name = "ampnet"
    predictor_id = 1
    description = "Convolutional neural network trained on AMP/non-AMP sequences (TensorFlow)."
    category = "pip"

    # Model is loaded once per class and shared across instances.
    _model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load_model(cls):
        """Load the Keras model from disk (once per process)."""
        if cls._model is not None:
            return cls._model

        import tensorflow as tf

        model_path = cls.models_dir() / "convolutional_nn_1.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AMPnet model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )
        cls._model = tf.keras.models.load_model(str(model_path))
        return cls._model

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _one_hot_pad(sequence: str) -> np.ndarray:
        """One-hot encode a sequence and pad/truncate to ``_MAX_LEN``."""
        matrix = np.zeros((_MAX_LEN, len(_AA_ORDER)), dtype=np.float32)
        for i, aa in enumerate(sequence[:_MAX_LEN]):
            if aa in _AA_INDEX:
                matrix[i, _AA_INDEX[aa]] = 1.0
        return matrix

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences in ``fasta_path`` with the AMPnet CNN.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "ampnet_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        model = self._load_model()

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        names: list[str] = []
        scores: list[float] = []

        with tqdm(total=len(records), desc="AMPnet", unit="seq") as pbar:
            for record in records:
                seq = str(record.seq).upper()
                x = self._one_hot_pad(seq)[np.newaxis, ...]
                pred = model.predict(x, verbose=0)
                scores.append(float(pred.ravel()[0]))
                names.append(record.id)
                pbar.update(1)

        return pd.DataFrame({"seq_name": names, "ampnet_score": scores})
