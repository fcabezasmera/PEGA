"""
pega.predictors.amp_cg
======================
AMP-CG predictor — ESM-2 protein language model fine-tuned for AMP classification
(PyTorch + HuggingFace Transformers).

Model
-----
A custom classification head on top of the ESM-2 (8M parameter) encoder
from Meta AI.  The backbone is loaded from HuggingFace Hub on first use;
the classification head weights are loaded from disk.

Pre-trained weights
-------------------
File: ``pega/models/best_model.pth``
Download: ``pega download-models``

Reference
---------
Please cite the original AMP-CG publication when using this predictor.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor

warnings.filterwarnings("ignore")

_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Model definition (mirrors the original architecture)
# ---------------------------------------------------------------------------


def _build_model():
    """Instantiate the AMP-CG model architecture."""
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification

    class AMPCGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                _CHECKPOINT, num_labels=320
            )
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(320, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.output_layer = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.0)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = out.logits
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            return self.output_layer(x)

    return AMPCGModel()


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class AMPCGPredictor(BasePredictor):
    """ESM-2 transformer-based AMP predictor (PyTorch + HuggingFace).

    Requires ``torch`` and ``transformers`` packages, and the pre-trained
    classification head ``pega/models/best_model.pth``.
    Run ``pega download-models`` to obtain the model weights.
    """

    name = "amp_cg"
    predictor_id = 5
    description = "ESM-2 protein language model fine-tuned for AMP classification (PyTorch)."
    category = "pip"

    _model = None
    _tokenizer = None
    _device = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load(cls):
        """Load tokenizer and model from disk (once per process)."""
        if cls._model is not None:
            return cls._model, cls._tokenizer, cls._device

        import torch
        from transformers import AutoTokenizer

        model_path = cls.models_dir() / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AMP-CG model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT)

        model = _build_model()
        state = torch.load(str(model_path), map_location=cls._device)
        model.load_state_dict(state)
        model.to(cls._device)
        model.eval()
        cls._model = model

        return cls._model, cls._tokenizer, cls._device

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with the AMP-CG ESM-2 model.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "amp_cg_score"]``.
        """
        import torch
        import torch.nn.functional as F

        fasta_path = self._validate_fasta(fasta_path)
        model, tokenizer, device = self._load()

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        names: list[str] = []
        scores: list[float] = []

        with tqdm(total=len(records), desc="AMP-CG", unit="seq") as pbar:
            for record in records:
                seq = str(record.seq).upper()
                inputs = tokenizer(
                    seq,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    prob = F.softmax(logits, dim=1)
                    # Index 1 = AMP class
                    scores.append(float(prob[0, 1].cpu()))

                names.append(record.id)
                pbar.update(1)

        return pd.DataFrame({"seq_name": names, "amp_cg_score": scores})
