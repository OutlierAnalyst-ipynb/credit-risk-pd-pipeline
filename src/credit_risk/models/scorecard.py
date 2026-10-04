"""Convert probability of default into a points-based credit score.

score = offset - factor * ln(PD / (1 - PD))
factor = PDO / ln(2);   offset = base_score + factor * ln(1 / base_odds)
(Odds expressed as bad:good, so higher score = lower risk.)
"""
from __future__ import annotations

import numpy as np


def pd_to_score(pd_: np.ndarray, base_score: float, base_odds: float, pdo: float) -> np.ndarray:
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(1 / base_odds)
    return np.round(offset - factor * np.log(pd_ / (1 - pd_))).astype(int)


def decide(score: np.ndarray, approve_min: int, decline_max: int) -> np.ndarray:
    score = np.asarray(score)
    return np.where(score >= approve_min, "APPROVE",
                    np.where(score <= decline_max, "DECLINE", "REVIEW"))
