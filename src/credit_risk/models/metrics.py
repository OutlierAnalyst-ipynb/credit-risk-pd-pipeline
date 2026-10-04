"""Credit-risk model validation metrics (what a Model Risk Management team asks for)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y, p) -> float:
    """Max separation between cumulative bad and good distributions."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    cum_bad = df["y"].cumsum() / df["y"].sum()
    cum_good = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return float((cum_bad - cum_good).abs().max())


def decile_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Rank-ordering + calibration check: predicted vs. actual default rate by risk band."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("decile").agg(n=("y", "size"), bads=("y", "sum"),
                                 avg_pd=("p", "mean"), actual_dr=("y", "mean"))
    t["lift"] = t["actual_dr"] / df["y"].mean()
    return t.sort_index(ascending=False).round(4)


def classification_report(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)), "default_rate": float(np.mean(y)),
        "roc_auc": float(auc), "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p)),
        "avg_predicted_pd": float(np.mean(p)),
        "calibration_ratio": float(np.mean(p) / max(np.mean(y), 1e-9)),   # ~1.0 is good
    }
