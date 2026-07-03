"""
pega.ensemble
=============
Validated ensemble scores for antimicrobial peptide prediction.

Each ensemble was identified by stochastic search over predictor subsets,
score transformations, and combination methods on benchmark datasets.
Normalisation parameters are stored per-component so that each ensemble
is self-contained and reproducible for any input size, including single
sequences.

Transformations
---------------
raw         : f(x) = x
gamma2      : f(x) = x²
gamma0.5    : f(x) = √x
logistic10  : f(x) = sigmoid(10 × (x − 0.5))
minmax      : f(x) = (x − min) / (max − min)        — needs min, max
zsigmoid    : f(x) = sigmoid((x − mean) / std)       — needs mean, std
rank_pct    : f(x) = percentile rank in reference CDF — needs quantiles[101]

Combination methods
-------------------
weighted_mean : ensemble = Σ wᵢ × transform(scoreᵢ),  Σwᵢ = 1
mean          : ensemble = (1/n) × Σ transform(scoreᵢ)

Component format
----------------
Each component is a tuple: (column, transform, weight, params)
where params is a dict with the parameters required by the transform.
For transforms that need no parameters (raw, gamma2, gamma0.5,
logistic10) params can be omitted or set to {}.

Usage
-----
>>> from pega.ensemble import compute_ensembles
>>> df = compute_ensembles(df)

Available ensemble columns
--------------------------
ensemble_AMP_score — weighted_mean, 5 predictors, MCC=0.767, AUC=0.932
ensemble_AVP_score — weighted_mean, 3 predictors, MCC=0.456, AUC=0.785
ensemble_AFP_score — weighted_mean, 3 predictors, MCC=0.757, AUC=0.929
ensemble_ABP_score — weighted_mean, 6 predictors, MCC=0.875, AUC=0.975
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transformation functions
# Each accepts (x: np.ndarray, **params) and returns np.ndarray
# ---------------------------------------------------------------------------

def _raw(x, **_):
    return x.copy()

def _gamma2(x, **_):
    return np.clip(x, 0, 1) ** 2

def _gamma05(x, **_):
    return np.sqrt(np.clip(x, 0, 1))

def _logistic10(x, **_):
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))

def _minmax(x, min, max, **_):
    rng = max - min
    return (x - min) / rng if rng != 0 else np.full_like(x, 0.5)

def _zsigmoid(x, mean, std, **_):
    if std == 0:
        return np.full_like(x, 0.5)
    return 1.0 / (1.0 + np.exp(-(x - mean) / std))

def _rank_pct(x, quantiles, **_):
    """Percentile rank using fixed reference CDF (101 quantiles, p0–p100)."""
    q = np.asarray(quantiles)
    return np.clip(np.searchsorted(q, x, side="left") / (len(q) - 1), 0.0, 1.0)


_TRANSFORMS = {
    "raw":        _raw,
    "gamma2":     _gamma2,
    "gamma0.5":   _gamma05,
    "logistic10": _logistic10,
    "minmax":     _minmax,
    "zsigmoid":   _zsigmoid,
    "rank_pct":   _rank_pct,
}


# ---------------------------------------------------------------------------
# Ensemble definitions
# Each component: (column, transform, weight, params_dict)
# params_dict can be omitted for transforms that need no parameters.
# ---------------------------------------------------------------------------

ENSEMBLES: dict[str, dict] = {

    # ── ensemble_AMP_score ──────────────────────────────────────────────────
    # Antimicrobial peptides — dataset_d1, len 10-25, n=18,255
    "ensemble_AMP_score": {
        "method": "weighted_mean",
        "components": [
            ("AMP_CG_score",          "minmax",   0.383, {"min": 0.0025211964, "max": 0.9986024499}),
            ("ampir_mature_score",    "zsigmoid", 0.225, {"mean": 0.4420336992, "std": 0.3666449020}),
            ("AMPlify_balanced_score","minmax",   0.141, {"min": 0.0, "max": 1.0}),
            ("AMPnet_score",          "gamma2",   0.167, {}),
            ("modlAMP_RF_score",      "gamma2",   0.084, {}),
        ],
        "description": "Weighted mean, 5 predictors, antimicrobial. Params from dataset_d1 (len 10-25, n=18,255).",
        "metrics": {"accuracy": 0.8827, "precision": 0.9364, "f1": 0.8581,
                    "sensitivity": 0.7918, "specificity": 0.9563,
                    "mcc": 0.7670, "auc": 0.9318, "kappa": 0.7592},
    },

    # ── ensemble_AVP_score ──────────────────────────────────────────────────
    # Antiviral peptides — dataset_d2, len 10-25, n=3,298
    "ensemble_AVP_score": {
        "method": "weighted_mean",
        "components": [
            ("AMP_CG_score",      "raw",      0.685, {}),
            ("ampir_mature_score","rank_pct", 0.185, {"quantiles": [
                0.0038545681, 0.0138418805, 0.0163696048, 0.0188735696, 0.0229932250,
                0.0257760160, 0.0285391005, 0.0306958031, 0.0333397284, 0.0353221711,
                0.0374720658, 0.0399922286, 0.0426523356, 0.0443161645, 0.0468063454,
                0.0491664996, 0.0515378643, 0.0530864224, 0.0561241939, 0.0583080777,
                0.0603928434, 0.0629327750, 0.0657750503, 0.0694185788, 0.0715574117,
                0.0739998801, 0.0765043283, 0.0802849046, 0.0837245372, 0.0868669510,
                0.0896355488, 0.0921601826, 0.0958501916, 0.0994152852, 0.1023342542,
                0.1064046713, 0.1099172410, 0.1141335839, 0.1194669882, 0.1232139195,
                0.1278957416, 0.1313730181, 0.1368130216, 0.1421774921, 0.1467093361,
                0.1503763081, 0.1557196053, 0.1599669965, 0.1652375929, 0.1715950286,
                0.1776244607, 0.1839629464, 0.1901654258, 0.1966884061, 0.2049427999,
                0.2122800725, 0.2226894216, 0.2334969580, 0.2417093103, 0.2524666309,
                0.2613721770, 0.2710924944, 0.2801859750, 0.2902455768, 0.3034546284,
                0.3158887468, 0.3262203255, 0.3438129538, 0.3563552817, 0.3724954028,
                0.3888862770, 0.4006334430, 0.4188344269, 0.4364492666, 0.4547857253,
                0.4763237816, 0.4970222774, 0.5193380958, 0.5406513381, 0.5680269265,
                0.5902325609, 0.6040498216, 0.6233047093, 0.6442860133, 0.6588480525,
                0.6758827195, 0.6983522876, 0.7211454367, 0.7397908058, 0.7610891274,
                0.7776425309, 0.8028544823, 0.8155412306, 0.8347696481, 0.8543742858,
                0.8761595587, 0.8948862852, 0.9079873216, 0.9404037877, 0.9597116835,
                0.9931267161,
            ]}),
            ("modlAMP_SVM_score", "rank_pct", 0.130, {"quantiles": [
                0.0000347759, 0.0018244984, 0.0031266204, 0.0040991757, 0.0052848232,
                0.0064056495, 0.0077834227, 0.0096041955, 0.0110789066, 0.0128592029,
                0.0144719219, 0.0163412755, 0.0183715621, 0.0194507982, 0.0209914672,
                0.0229027826, 0.0251484524, 0.0275750366, 0.0303581614, 0.0330264649,
                0.0362090472, 0.0392361203, 0.0427315572, 0.0451785350, 0.0476079954,
                0.0506618749, 0.0550201570, 0.0585456024, 0.0607340371, 0.0641639669,
                0.0682683829, 0.0725559264, 0.0783435641, 0.0854915009, 0.0911572719,
                0.0949969820, 0.1009144454, 0.1061402461, 0.1127825001, 0.1186387482,
                0.1284538317, 0.1365025169, 0.1456604056, 0.1522216180, 0.1612323914,
                0.1708641759, 0.1781002032, 0.1943455477, 0.2072003406, 0.2178012598,
                0.2262461021, 0.2336899781, 0.2455051538, 0.2578988245, 0.2687733110,
                0.2809166900, 0.2905573935, 0.3027521793, 0.3179496476, 0.3324916078,
                0.3441135425, 0.3603364167, 0.3736176007, 0.3898113760, 0.4090647989,
                0.4211151383, 0.4409698003, 0.4556800222, 0.4728164740, 0.4940929396,
                0.5147090064, 0.5417840675, 0.5585514054, 0.5777711660, 0.5975120451,
                0.6161665473, 0.6373198107, 0.6571570851, 0.6769534915, 0.6910328476,
                0.7085384579, 0.7291186257, 0.7524602701, 0.7706999228, 0.7900068212,
                0.8074453595, 0.8248172113, 0.8439292290, 0.8609798230, 0.8754929142,
                0.8856651805, 0.8948926908, 0.9042832643, 0.9134355877, 0.9189432603,
                0.9300623420, 0.9504310940, 0.9647362585, 0.9789717453, 0.9934308496,
                0.9999975747,
            ]}),
        ],
        "description": "Weighted mean, 3 predictors, antiviral. Params from dataset_d2 (len 10-25, n=3,298).",
        "metrics": {"accuracy": 0.7113, "precision": 0.8418, "f1": 0.6355,
                    "sensitivity": 0.5105, "specificity": 0.9067,
                    "mcc": 0.4556, "auc": 0.7851, "kappa": 0.4194},
    },

    # ── ensemble_AFP_score ──────────────────────────────────────────────────
    # Antifungal peptides — dataset_d3, len 10-25, n=378
    "ensemble_AFP_score": {
        "method": "weighted_mean",
        "components": [
            ("AMP_CG_score", "gamma0.5", 0.481, {}),
            ("AMPnet_score", "raw",       0.089, {}),
            ("Macrel_score", "zsigmoid",  0.430, {"mean": 0.3275925926, "std": 0.2649254126}),
        ],
        "description": "Weighted mean, 3 predictors, antifungal. Params from dataset_d3 (len 10-25, n=378).",
        "metrics": {"accuracy": 0.8783, "precision": 0.8901, "f1": 0.8757,
                    "sensitivity": 0.8617, "specificity": 0.8947,
                    "mcc": 0.7569, "auc": 0.9291, "kappa": 0.7566},
    },

    # ── ensemble_ABP_score ──────────────────────────────────────────────────
    # Antibiofilm peptides — dataset_d5, len 10-25, n=7,858
    "ensemble_ABP_score": {
        "method": "weighted_mean",
        "components": [
            ("AMP_CG_score",           "gamma2",   0.304, {}),
            ("ampir_mature_score",     "rank_pct", 0.220, {"quantiles": [
                0.0027675091, 0.0139371120, 0.0179184687, 0.0224400443, 0.0250277168,
                0.0279649506, 0.0313734234, 0.0352544447, 0.0383400373, 0.0418566148,
                0.0452991845, 0.0490109517, 0.0526701653, 0.0570347602, 0.0615378688,
                0.0656069023, 0.0701545017, 0.0743257400, 0.0786006045, 0.0826532482,
                0.0887765095, 0.0933023689, 0.0973756618, 0.1024156339, 0.1075746225,
                0.1128060459, 0.1203740853, 0.1271807591, 0.1345570509, 0.1416684635,
                0.1483499768, 0.1548532276, 0.1642759560, 0.1723767891, 0.1829429903,
                0.1923930198, 0.2021415809, 0.2139370873, 0.2276560337, 0.2404269704,
                0.2537505253, 0.2693489358, 0.2873680574, 0.3043663242, 0.3237646359,
                0.3415576466, 0.3631658708, 0.3851342755, 0.4087922161, 0.4322079911,
                0.4601737276, 0.4947436889, 0.5261185884, 0.5604721517, 0.5950825096,
                0.6194910365, 0.6387035415, 0.6686216660, 0.6963549572, 0.7199464446,
                0.7487200659, 0.7726457845, 0.7904044026, 0.8067198946, 0.8213104658,
                0.8344575629, 0.8483844088, 0.8611871350, 0.8724573691, 0.8808452687,
                0.8901730476, 0.8980459960, 0.9038567312, 0.9049960452, 0.9107241532,
                0.9150672230, 0.9200564905, 0.9242433245, 0.9291679034, 0.9338722895,
                0.9380545245, 0.9419077961, 0.9457272097, 0.9489368309, 0.9521253922,
                0.9550723435, 0.9578470809, 0.9602291350, 0.9626073567, 0.9643870191,
                0.9666786974, 0.9691376964, 0.9713719311, 0.9733996751, 0.9762893174,
                0.9782727178, 0.9812168946, 0.9838509959, 0.9872349126, 0.9903307488,
                0.9981153546,
            ]}),
            ("AMPlify_balanced_score", "zsigmoid", 0.133, {"mean": 0.5058931490, "std": 0.4304681870}),
            ("AMPnet_score",           "gamma0.5", 0.145, {}),
            ("Macrel_score",           "gamma0.5", 0.015, {}),
            ("modlAMP_RF_score",       "raw",      0.183, {}),
        ],
        "description": "Weighted mean, 6 predictors, antibiofilm. Params from dataset_d5 (len 10-25, n=7,858).",
        "metrics": {"accuracy": 0.9368, "precision": 0.9665, "f1": 0.9335,
                    "sensitivity": 0.9026, "specificity": 0.9697,
                    "mcc": 0.8752, "auc": 0.9750, "kappa": 0.8733},
    },
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _apply_transform(
    values: np.ndarray,
    transform: str,
    params: dict,
) -> np.ndarray:
    """Apply a named transformation with the given parameters."""
    fn = _TRANSFORMS.get(transform)
    if fn is None:
        raise ValueError(
            f"Unknown transformation '{transform}'. "
            f"Available: {list(_TRANSFORMS)}"
        )
    return fn(values, **params)


def compute_ensembles(
    df: pd.DataFrame,
    ensemble_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute validated ensemble scores and append them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`pega.utils.calculate_scores`.
    ensemble_names : list[str] or None
        Ensembles to compute. ``None`` computes all defined ensembles.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one new column per ensemble.
    """
    import warnings
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

        # Normalise: component may be 3-tuple or 4-tuple
        resolved = []
        for comp in components:
            if len(comp) == 3:
                col, transform, weight = comp
                params = {}
            else:
                col, transform, weight, params = comp
            resolved.append((col, transform, weight, params))

        # Check all required columns exist
        missing = [col for col, *_ in resolved if col not in df.columns]
        if missing:
            warnings.warn(
                f"Ensemble '{name}' skipped — missing columns: {missing}",
                stacklevel=2,
            )
            continue

        # Apply transforms and combine
        weights      = np.array([w for _, _, w, _ in resolved])
        weights      = weights / weights.sum()   # re-normalise (guard rounding)
        transformed  = np.vstack([
            _apply_transform(df[col].fillna(0).values, transform, params)
            for col, transform, _, params in resolved
        ])

        if method == "weighted_mean":
            df[name] = (weights[:, None] * transformed).sum(axis=0)
        elif method == "mean":
            df[name] = transformed.mean(axis=0)
        else:
            raise ValueError(f"Unknown ensemble method '{method}'.")

    return df


def required_predictors(ensemble_names: list[str]) -> set[str]:
    """Return the predictor score-column names needed for the given ensembles.

    Used to auto-restrict which predictors are run when the caller asks for
    specific ensembles only (e.g. ``PEGA score --ensembles AMP``). Names not
    found in :data:`ENSEMBLES` are ignored here — callers that need to warn
    about unknown ensemble names already validate them separately.

    Parameters
    ----------
    ensemble_names : list[str]
        Ensemble names, e.g. ``["ensemble_AMP_score"]``.

    Returns
    -------
    set[str]
        Score-column names (e.g. ``{"AMP_CG_score", "ampir_mature_score"}``)
        required by at least one of the given ensembles.
    """
    needed: set[str] = set()
    for name in ensemble_names:
        cfg = ENSEMBLES.get(name)
        if cfg is None:
            continue
        needed.update(col for col, *_ in cfg["components"])
    return needed


def list_ensembles() -> list[dict]:
    """Return metadata for all defined ensembles."""
    return [
        {
            "name":         name,
            "method":       cfg["method"],
            "n_predictors": len(cfg["components"]),
            "description":  cfg["description"],
            "metrics":      cfg["metrics"],
        }
        for name, cfg in ENSEMBLES.items()
    ]
