"""
pega.ensemble
=============
Ensemble aggregation methods and fuzzy membership functions.

All public functions in this module operate on a ``pandas.DataFrame``
that contains one or more predictor score columns (columns whose names
end in ``_score``).  They return either a ``pandas.Series`` (for single
aggregation methods) or an augmented ``DataFrame`` (for the convenience
wrappers).

Ensemble methods
----------------
- :func:`arithmetic_mean`   — simple average across predictors.
- :func:`geometric_mean`    — geometric mean; penalises low scores more.
- :func:`weighted_voting`   — weights each predictor by its correlation
                              with the column mean.
- :func:`rank_aggregation`  — averages normalised rank positions.
- :func:`majority_voting`   — fraction of predictors above a threshold.
- :func:`stacking_ensemble` — correlation-matrix-weighted sum.

Fuzzy membership functions
--------------------------
Applied per predictor score column when requested.  Each function maps
a score in ``[0, 1]`` to a membership degree in ``[0, 1]``.

- :func:`sigmoid_membership`
- :func:`gaussian_membership`
- :func:`triangular_membership`
- :func:`trapezoidal_membership`
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fuzzy membership functions
# ---------------------------------------------------------------------------


def sigmoid_membership(
    x: np.ndarray | pd.Series,
    center: float = 0.5,
    steepness: float = 10.0,
) -> np.ndarray:
    """Sigmoid membership function.

    Parameters
    ----------
    x:
        Input values in ``[0, 1]``.
    center:
        Inflection point of the sigmoid (default 0.5).
    steepness:
        Controls the slope at the inflection point (default 10).

    Returns
    -------
    numpy.ndarray
        Membership degrees in ``[0, 1]``.
    """
    return 1.0 / (1.0 + np.exp(-steepness * (np.asarray(x) - center)))


def gaussian_membership(
    x: np.ndarray | pd.Series,
    center: float = 0.5,
    width: float = 0.2,
) -> np.ndarray:
    """Gaussian membership function.

    Parameters
    ----------
    x:
        Input values in ``[0, 1]``.
    center:
        Peak of the Gaussian (default 0.5).
    width:
        Standard deviation controlling spread (default 0.2).

    Returns
    -------
    numpy.ndarray
        Membership degrees in ``[0, 1]``.
    """
    return np.exp(-0.5 * ((np.asarray(x) - center) / width) ** 2)


def triangular_membership(
    x: np.ndarray | pd.Series,
    a: float = 0.2,
    b: float = 0.5,
    c: float = 0.8,
) -> np.ndarray:
    """Triangular membership function.

    Parameters
    ----------
    x:
        Input values in ``[0, 1]``.
    a:
        Left foot of the triangle (membership = 0).
    b:
        Peak of the triangle (membership = 1).
    c:
        Right foot of the triangle (membership = 0).

    Returns
    -------
    numpy.ndarray
        Membership degrees in ``[0, 1]``.
    """
    x = np.asarray(x)
    return np.maximum(
        0,
        np.minimum((x - a) / (b - a), (c - x) / (c - b)),
    )


def trapezoidal_membership(
    x: np.ndarray | pd.Series,
    a: float = 0.1,
    b: float = 0.3,
    c: float = 0.7,
    d: float = 0.9,
) -> np.ndarray:
    """Trapezoidal membership function.

    Parameters
    ----------
    x:
        Input values in ``[0, 1]``.
    a, b:
        Left shoulder: membership rises from 0 at ``a`` to 1 at ``b``.
    c, d:
        Right shoulder: membership falls from 1 at ``c`` to 0 at ``d``.

    Returns
    -------
    numpy.ndarray
        Membership degrees in ``[0, 1]``.
    """
    x = np.asarray(x)
    return np.maximum(
        0,
        np.minimum(1, np.minimum((x - a) / (b - a), (d - x) / (d - c))),
    )


# ---------------------------------------------------------------------------
# Per-column membership wrapper
# ---------------------------------------------------------------------------


def add_membership_columns(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Append four fuzzy membership columns for one predictor score column.

    Columns added (where ``<base>`` is ``score_column`` with ``_score``
    removed):

    * ``<base>_sigmoid``
    * ``<base>_gaussian``
    * ``<base>_triangular``
    * ``<base>_trapezoidal``

    Parameters
    ----------
    df:
        DataFrame containing ``score_column``.
    score_column:
        Name of the predictor score column to transform.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with four new columns appended (in-place copy).
    """
    base = score_column.replace("_score", "")
    scores = df[score_column].fillna(0).to_numpy()

    df = df.copy()
    df[f"{base}_sigmoid"] = sigmoid_membership(scores, center=0.5, steepness=12)
    df[f"{base}_gaussian"] = gaussian_membership(scores, center=0.8, width=0.15)
    df[f"{base}_triangular"] = triangular_membership(scores, a=0.5, b=0.75, c=0.9)
    df[f"{base}_trapezoidal"] = trapezoidal_membership(
        scores, a=0.4, b=0.6, c=0.85, d=0.95
    )
    return df


# ---------------------------------------------------------------------------
# Ensemble aggregation methods
# ---------------------------------------------------------------------------


def arithmetic_mean(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Simple arithmetic mean across predictor scores.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.

    Returns
    -------
    pandas.Series
    """
    return df[score_columns].mean(axis=1)


def geometric_mean(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Geometric mean across predictor scores.

    A small epsilon is added before taking the log to avoid
    ``log(0)`` errors.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.

    Returns
    -------
    pandas.Series
    """
    log_mean = np.log(df[score_columns].fillna(0) + 1e-9).mean(axis=1)
    return np.exp(log_mean)


def weighted_voting(
    df: pd.DataFrame,
    score_columns: list[str],
    weights: list[float] | None = None,
) -> pd.Series:
    """Weighted sum where each predictor is weighted by its correlation
    with the column-wise mean score.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.
    weights:
        Optional explicit weights (must sum to 1).  When ``None``,
        weights are derived from Pearson correlation with the mean.

    Returns
    -------
    pandas.Series
    """
    if weights is None:
        col_mean = df[score_columns].mean(axis=1)
        corr = np.array([df[c].fillna(0).corr(col_mean) for c in score_columns])
        corr = np.nan_to_num(corr, nan=0.0)
        total = corr.sum()
        weights = corr / total if total != 0 else np.ones(len(corr)) / len(corr)

    return sum(w * df[c].fillna(0) for w, c in zip(weights, score_columns))


def rank_aggregation(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Average of normalised rank positions across predictors.

    Each predictor's scores are ranked in descending order and
    normalised to ``[0, 1]``.  The final score is the mean of those
    normalised ranks.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.

    Returns
    -------
    pandas.Series
    """
    ranks = pd.DataFrame(
        {
            col: df[col].fillna(0).rank(ascending=False, method="average")
            for col in score_columns
        }
    )
    return 1.0 - ranks.div(len(df)).mean(axis=1)


def majority_voting(
    df: pd.DataFrame,
    score_columns: list[str],
    threshold: float = 0.5,
) -> pd.Series:
    """Fraction of predictors that classify a sequence as AMP.

    A predictor votes "AMP" if its score exceeds ``threshold``.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.
    threshold:
        Decision boundary (default 0.5).

    Returns
    -------
    pandas.Series
        Values in ``[0, 1]``.
    """
    votes = pd.DataFrame(
        {col: (df[col].fillna(0) > threshold).astype(int) for col in score_columns}
    )
    return votes.sum(axis=1) / len(score_columns)


def stacking_ensemble(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Correlation-matrix-weighted sum across predictors.

    Each predictor is weighted by the sum of its pairwise correlations
    with all other predictors, rewarding predictors that agree with the
    consensus.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the columns to aggregate.

    Returns
    -------
    pandas.Series
    """
    corr_matrix = df[score_columns].corr()
    weights = corr_matrix.sum().values
    total = weights.sum()
    weights = weights / total if total != 0 else np.ones(len(weights)) / len(weights)
    return sum(w * df[col].fillna(0) for w, col in zip(weights, score_columns))


# ---------------------------------------------------------------------------
# Convenience wrapper — applies all ensemble methods at once
# ---------------------------------------------------------------------------


def apply_ensemble(
    df: pd.DataFrame,
    score_columns: list[str],
    membership: bool = False,
) -> pd.DataFrame:
    """Append all ensemble summary columns to ``df``.

    Columns added
    -------------
    * ``arithmetic_mean``
    * ``geometric_mean``
    * ``weighted_voting``
    * ``rank_aggregation``
    * ``majority_voting``
    * ``stacking_ensemble``

    And, when ``membership=True``, four fuzzy membership columns per
    predictor score column.

    Parameters
    ----------
    df:
        DataFrame with predictor score columns.
    score_columns:
        Names of the predictor score columns.
    membership:
        If ``True``, add fuzzy membership columns for each predictor.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with new columns appended.
    """
    df = df.copy()

    df["arithmetic_mean"] = arithmetic_mean(df, score_columns)
    df["geometric_mean"] = geometric_mean(df, score_columns)
    df["weighted_voting"] = weighted_voting(df, score_columns)
    df["rank_aggregation"] = rank_aggregation(df, score_columns)
    df["majority_voting"] = majority_voting(df, score_columns)
    df["stacking_ensemble"] = stacking_ensemble(df, score_columns)

    if membership:
        for col in score_columns:
            df = add_membership_columns(df, col)

    return df
