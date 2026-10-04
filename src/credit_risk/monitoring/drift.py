"""PSI-based drift monitoring for inputs and the model score."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(e, a, eps=1e-4) -> float:
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    if not pd.api.types.is_numeric_dtype(expected):
        cats = sorted(set(expected.dropna()) | set(actual.dropna()))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        return _psi(e, a)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    # missing values get their own bin: a jump in missingness IS drift
    e = np.append(np.histogram(expected.dropna(), edges)[0], expected.isna().sum()) / len(expected)
    a = np.append(np.histogram(actual.dropna(), edges)[0], actual.isna().sum()) / len(actual)
    return _psi(e, a)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str],
                 warn: float, alert: float) -> pd.DataFrame:
    rep = pd.DataFrame({"feature": columns,
                        "psi": [psi(reference[c], current[c]) for c in columns]})
    rep["status"] = np.select([rep["psi"] >= alert, rep["psi"] >= warn], ["ALERT", "WARN"], "OK")
    return rep.sort_values("psi", ascending=False).reset_index(drop=True)
