"""
pega.ensemble
=============
Validated ensemble scores for antimicrobial peptide prediction.

Each ensemble was identified by stochastic search over predictor subsets,
score transformations, and combination methods using dataset_d1
(n = 40,348 sequences). Normalisation parameters (min, max, mean, std)
are fixed from that dataset so that ensemble scores are consistent and
reproducible for any input size — including single sequences.

Transformations
---------------
raw         : f(x) = x
minmax      : f(x) = (x - min) / (max - min)   — fixed from dataset_d1
zsigmoid    : f(x) = sigmoid((x - mean) / std) — fixed from dataset_d1
gamma2      : f(x) = x²                         — no dataset parameters needed
gamma0.5    : f(x) = √x                         — no dataset parameters needed
logistic10  : f(x) = sigmoid(10 × (x − 0.5))   — no dataset parameters needed

Combination methods
-------------------
weighted_mean : ensemble = Σ wᵢ × transform(scoreᵢ),  Σwᵢ = 1
mean          : ensemble = (1/n) × Σ transform(scoreᵢ)

Usage
-----
>>> from pega.ensemble import compute_ensembles
>>> df = compute_ensembles(df)   # df is the output of calculate_scores()

Available ensemble columns added
---------------------------------
ensemble_AMP_score   — weighted_mean of 5 predictors
                       MCC=0.767, AUC=0.932  (dataset_d1, n=40,348)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Normalisation parameters — fixed from dataset_d1 (n = 40,348 sequences)
# ---------------------------------------------------------------------------

_PARAMS: dict[str, dict[str, float]] = {
    "AMP_CG_score": {
        "min":  0.0000547703,
        "max":  0.9986024499,
    },
    "ampir_mature_score": {
        "min":  0.0011548338,
        "max":  0.9994789187,
        "mean": 0.5027099536,
        "std":  0.3431127575,
    },
    "AMPlify_balanced_score": {
        "min":  0.0,
        "max":  1.0,
    },
    # AMPnet_score and modlAMP_RF_score use gamma2 — no parameters needed
}

# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------


def _raw(x: np.ndarray, **_) -> np.ndarray:
    return x.copy()


def _minmax(x: np.ndarray, min: float, max: float, **_) -> np.ndarray:
    rng = max - min
    if rng == 0:
        return np.full_like(x, 0.5)
    return (x - min) / rng


def _zsigmoid(x: np.ndarray, mean: float, std: float, **_) -> np.ndarray:
    if std == 0:
        return np.full_like(x, 0.5)
    z = (x - mean) / std
    return 1.0 / (1.0 + np.exp(-z))


def _gamma2(x: np.ndarray, **_) -> np.ndarray:
    return np.clip(x, 0, 1) ** 2


def _gamma05(x: np.ndarray, **_) -> np.ndarray:
    return np.sqrt(np.clip(x, 0, 1))


def _logistic10(x: np.ndarray, **_) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


_TRANSFORMS = {
    "raw":        _raw,
    "minmax":     _minmax,
    "zsigmoid":   _zsigmoid,
    "gamma2":     _gamma2,
    "gamma0.5":   _gamma05,
    "logistic10": _logistic10,
}

# ---------------------------------------------------------------------------
# Ensemble definitions
# ---------------------------------------------------------------------------
# Each ensemble is a dict with:
#   "method"      : "weighted_mean" | "mean"
#   "components"  : list of (score_col, transform_name, weight)
#   "description" : short human-readable description
#   "metrics"     : performance metrics from dataset_d1
# ---------------------------------------------------------------------------

ENSEMBLES: dict[str, dict] = {
    "ensemble_AMP_score": {
        "method": "weighted_mean",
        "components": [
            # (score_column,            transform,    weight)
            ("AMP_CG_score",          "minmax",    0.383),
            ("ampir_mature_score",     "zsigmoid",  0.225),
            ("AMPlify_balanced_score", "minmax",    0.141),
            ("AMPnet_score",           "gamma2",    0.167),
            ("modlamp_RF_score",       "gamma2",    0.084),
        ],
        "description": (
            "Weighted mean ensemble (5 predictors). "
            "Normalisation parameters fixed from dataset_d1 (n=40,348)."
        ),
        "metrics": {
            "accuracy":    0.8827,
            "precision":   0.9364,
            "f1":          0.8581,
            "sensitivity": 0.7918,
            "specificity": 0.9563,
            "mcc":         0.7670,
            "auc":         0.9318,
            "kappa":       0.7592,
        },
    },
    # Additional ensembles will be added here
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _apply_transform(
    values: np.ndarray,
    transform: str,
    col: str,
) -> np.ndarray:
    """Apply a named transformation to an array of scores.

    Parameters
    ----------
    values : np.ndarray
        Raw predictor scores.
    transform : str
        Transformation name (e.g. ``"minmax"``, ``"zsigmoid"``).
    col : str
        Score column name — used to look up fixed normalisation parameters.

    Returns
    -------
    np.ndarray
        Transformed scores.
    """
    fn = _TRANSFORMS.get(transform)
    if fn is None:
        raise ValueError(
            f"Unknown transformation '{transform}'. "
            f"Available: {list(_TRANSFORMS)}"
        )
    params = _PARAMS.get(col, {})
    return fn(values, **params)


def compute_ensembles(
    df: pd.DataFrame,
    ensemble_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute validated ensemble scores and append them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`pega.utils.calculate_scores`.  Must contain the
        predictor score columns referenced by the requested ensembles.
    ensemble_names : list[str] or None
        Names of ensembles to compute.  ``None`` computes all defined
        ensembles.  Available: ``list(pega.ensemble.ENSEMBLES)``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one new column per ensemble.

    Raises
    ------
    KeyError
        If a requested ensemble name is not defined.
    """
    names = ensemble_names or list(ENSEMBLES)

    for name in names:
        if name not in ENSEMBLES:
            raise KeyError(
                f"Ensemble '{name}' is not defined. "
                f"Available: {list(ENSEMBLES)}"
            )

        cfg = ENSEMBLES[name]
        components = cfg["components"]
        method     = cfg["method"]

        # Verify all required columns are present
        missing = [col for col, *_ in components if col not in df.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Ensemble '{name}' skipped — missing columns: {missing}",
                stacklevel=2,
            )
            continue

        # Apply transformations
        transformed: list[np.ndarray] = []
        weights:     list[float]      = []

        for col, transform, weight in components:
            t = _apply_transform(df[col].fillna(0).values, transform, col)
            transformed.append(t)
            weights.append(weight)

        # Combine
        stack = np.vstack(transformed)   # shape: (n_predictors, n_sequences)
        w     = np.array(weights)

        if method == "weighted_mean":
            w = w / w.sum()              # re-normalise to guard against rounding
            df[name] = (w[:, None] * stack).sum(axis=0)
        elif method == "mean":
            df[name] = stack.mean(axis=0)
        else:
            raise ValueError(f"Unknown ensemble method '{method}'.")

    return df


def list_ensembles() -> list[dict]:
    """Return metadata for all defined ensembles.

    Returns
    -------
    list[dict]
        One dict per ensemble with keys ``name``, ``method``,
        ``n_predictors``, ``description``, and ``metrics``.
    """
    return [
        {
            "name":          name,
            "method":        cfg["method"],
            "n_predictors":  len(cfg["components"]),
            "description":   cfg["description"],
            "metrics":       cfg["metrics"],
        }
        for name, cfg in ENSEMBLES.items()
    ]
