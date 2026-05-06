"""
pega.predictors.modlamp
=======================
modlAMP-based predictors — Random Forest and SVM (scikit-learn).

Both predictors use PepCATS cross-correlation descriptors computed with
the ``modlamp`` library.  Pre-trained models are loaded from disk.

Pre-trained weights
-------------------
* ``pega/models/modlamp_RF.joblib``  — Random Forest model.
* ``pega/models/modlamp_SVM.joblib`` — Support Vector Machine model.

Run ``pega download-models`` to obtain the model weights.

Reference
---------
Müller A.T. et al. (2017).  modlAMP: Python for antimicrobial peptides.
*Bioinformatics*, 33(17), 2753–2755.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor


# ---------------------------------------------------------------------------
# Shared feature extraction
# ---------------------------------------------------------------------------


def _compute_descriptors(sequences: list[str], corr_window: int = 5) -> np.ndarray:
    """Compute PepCATS cross-correlation descriptors via modlAMP.

    Parameters
    ----------
    sequences:
        List of amino acid sequences (uppercase, standard codes).
    corr_window:
        Window size for cross-correlation (default 5).

    Returns
    -------
    numpy.ndarray
        Feature matrix of shape ``(n_sequences, n_features)``.
    """
    from modlamp.descriptors import PeptideDescriptor

    descr = PeptideDescriptor(sequences, "pepcats")
    descr.calculate_crosscorr(corr_window)
    return descr.descriptor


# ---------------------------------------------------------------------------
# Random Forest predictor
# ---------------------------------------------------------------------------


class ModlampRFPredictor(BasePredictor):
    """Random Forest AMP predictor using modlAMP PepCATS descriptors.

    Requires the ``modlamp``, ``scikit-learn``, and ``joblib`` packages,
    and the pre-trained model file ``pega/models/modlamp_RF.joblib``.
    """

    name = "modlamp_rf"
    predictor_id = 9
    description = "Random Forest on PepCATS cross-correlation descriptors (modlAMP)."
    category = "pip"

    _model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import modlamp   # noqa: F401
            import joblib    # noqa: F401
            import sklearn   # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model
        import joblib

        model_path = cls.models_dir() / "modlamp_RF.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"modlAMP RF model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )
        cls._model = joblib.load(str(model_path))
        return cls._model

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with the modlAMP Random Forest model.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "modlamp_rf_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        model = self._load_model()

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        names = [r.id for r in records]
        sequences = [str(r.seq).upper() for r in records]

        print("  Computing PepCATS descriptors (RF)...")
        features = _compute_descriptors(sequences)

        scores: list[float] = []
        with tqdm(total=len(sequences), desc="modlAMP RF", unit="seq") as pbar:
            for i in range(len(sequences)):
                prob = model.predict_proba(features[i].reshape(1, -1))[0]
                # Index 1 corresponds to the AMP class.
                scores.append(float(prob[1]) if len(prob) > 1 else float(prob[0]))
                pbar.update(1)

        return pd.DataFrame({"seq_name": names, "modlamp_rf_score": scores})


# ---------------------------------------------------------------------------
# SVM predictor
# ---------------------------------------------------------------------------


class ModlampSVMPredictor(BasePredictor):
    """Support Vector Machine AMP predictor using modlAMP PepCATS descriptors.

    Requires the ``modlamp``, ``scikit-learn``, and ``joblib`` packages,
    and the pre-trained model file ``pega/models/modlamp_SVM.joblib``.
    """

    name = "modlamp_svm"
    predictor_id = 10
    description = "Support Vector Machine on PepCATS cross-correlation descriptors (modlAMP)."
    category = "pip"

    _model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import modlamp   # noqa: F401
            import joblib    # noqa: F401
            import sklearn   # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model
        import joblib

        model_path = cls.models_dir() / "modlamp_SVM.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"modlAMP SVM model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )
        cls._model = joblib.load(str(model_path))
        return cls._model

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with the modlAMP SVM model.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "modlamp_svm_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        model = self._load_model()

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        names = [r.id for r in records]
        sequences = [str(r.seq).upper() for r in records]

        print("  Computing PepCATS descriptors (SVM)...")
        features = _compute_descriptors(sequences)

        scores: list[float] = []
        with tqdm(total=len(sequences), desc="modlAMP SVM", unit="seq") as pbar:
            for i in range(len(sequences)):
                # SVMs with probability=True expose predict_proba.
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features[i].reshape(1, -1))[0]
                    scores.append(float(prob[1]) if len(prob) > 1 else float(prob[0]))
                else:
                    # Fallback: decision function normalised to [0, 1].
                    decision = model.decision_function(features[i].reshape(1, -1))[0]
                    scores.append(float(1 / (1 + np.exp(-decision))))
                pbar.update(1)

        return pd.DataFrame({"seq_name": names, "modlamp_svm_score": scores})
